from grandnexus.core.errorRecoveryManager import ErrorRecoveryManager  # fix 6.5: previously provided by the notebook's shared namespace
import logging
import threading
import time
import uuid
import sqlite3
import json  # For serializing/deserializing module configurations
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set  # Importer Set
import heapq  # Correctly manage priorities


class NexusCore:
    """
    Central class orchestrating the GrandNexus system.
    Provides a unified access point for module management,
    lifecycle control, and inter-module communication.

    Features:
        - Registration of modules with optional dependencies
        - Topological sorting of dependencies for startup and shutdown
        - Hooks for lifecycle events (pre_start, post_start, pre_stop, post_stop)
        - Priority-based message queue with optional background processing
        - Persistence of module state, dependencies, and configuration using SQLite
        - Extensive logging and error checks
        - Convenience methods to add or list modules, get module status and configuration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_async_messaging: bool = True, db_file: str = "nexuscore.db"):
        """
        Initializes the NexusCore with an optional configuration.

        Args:
            config (dict, optional):  Global configuration dictionary. Defaults to None.
            enable_async_messaging (bool): If True, use a background thread to process the message queue. Defaults to True.
            db_file (str): Path to the SQLite database file. Defaults to "nexuscore.db".
        """
        self.instance_id: str = str(uuid.uuid4())[:8]
        self.logger: logging.Logger = self._setup_logger()
        self.config: Dict[str, Any] = config or {}
        self.modules: Dict[str, Dict[str, Any]] = {}  # {module_name: {instance, dependencies, status, config}}
        self.running: bool = False
        self.start_time: float = 0.0
        self.stop_event: threading.Event = threading.Event()

        # Hooks for lifecycle customization
        self.pre_start_hooks: List[Callable[['NexusCore'], None]] = []
        self.post_start_hooks: List[Callable[['NexusCore'], None]] = []
        self.pre_stop_hooks: List[Callable[['NexusCore'], None]] = []
        self.post_stop_hooks: List[Callable[['NexusCore'], None]] = []

        # Priority-based Message queue
        self.message_queue: List[Tuple[int, Dict[str, Any]]] = []  # (priority, message)
        self.lock: threading.RLock = threading.RLock()

        self.enable_async_messaging: bool = enable_async_messaging
        self.message_thread: Optional[threading.Thread] = None

        self.db_file: str = db_file
        self._init_db()
        self._load_modules_from_db()

        self.logger.info(f"NexusCore initialized with ID={self.instance_id}, Async Messaging: {enable_async_messaging}")

        # Initialize error recovery manager
        self.error_recovery = ErrorRecoveryManager(self)

        self.logger.info(f"NexusCore initialized with ID={self.instance_id}, Async Messaging: {enable_async_messaging}")


    def _setup_logger(self) -> logging.Logger:
        """Configures and returns a logger for the core."""
        logger = logging.getLogger("NexusCore")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _init_db(self):
        """Initializes the SQLite database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS modules (
                        name TEXT PRIMARY KEY,
                        status TEXT,
                        configuration TEXT,  -- Store configuration as JSON
                        last_updated REAL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dependencies (
                        module_name TEXT,
                        dependency_name TEXT,
                        FOREIGN KEY (module_name) REFERENCES modules(name) ON DELETE CASCADE,
                        FOREIGN KEY (dependency_name) REFERENCES modules(name) ON DELETE CASCADE,
                        PRIMARY KEY (module_name, dependency_name)
                    )
                """)
                # You could add a 'metadata' table here, similar to nsytz's version.
                conn.commit()
            self.logger.info("Database initialized successfully.")

        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise  # Re-raise to prevent startup if DB init fails

    def _load_modules_from_db(self):
        """Loads module information from the database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, status, configuration FROM modules")
                for name, status, configuration in cursor.fetchall():
                    self.modules[name] = {
                        "instance": None,  # Placeholder, will be filled on register
                        "dependencies": [],  # Will be filled below
                        "status": status,
                        "config": json.loads(configuration) if configuration else {}  # Load config
                    }

                cursor.execute("SELECT module_name, dependency_name FROM dependencies")
                for module_name, dependency_name in cursor.fetchall():
                    if module_name in self.modules:
                        self.modules[module_name]["dependencies"].append(dependency_name)
        except sqlite3.Error as e:
            self.logger.error(f"Error loading modules from database: {e}")
            #  Don't re-raise here.  Allow registration, but log the error

    def _persist_module(self, name: str):
        """Persists module information to the database."""
        with self.lock:
            if name not in self.modules:
                return # Nothing to persist.

            module_info = self.modules[name]
            status = module_info["status"]
            config = module_info.get("config", {})  # Get config, default to empty dict
            config_json = json.dumps(config)       # Serialize configuration
            dependencies = module_info["dependencies"]

            try:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO modules (name, status, configuration, last_updated)
                        VALUES (?, ?, ?, ?)
                    """, (name, status, config_json, time.time()))

                    # Clear existing dependencies
                    cursor.execute("DELETE FROM dependencies WHERE module_name = ?", (name,))
                    # Insert new dependencies
                    for dep in dependencies:
                        cursor.execute("INSERT INTO dependencies (module_name, dependency_name) VALUES (?, ?)", (name, dep))
                    conn.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Error persisting module '{name}' to database: {e}")
                # Consider if you want to re-raise, or handle the error here

    def register_module(self, name: str, module: Any, dependencies: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Registers a module in the system.

        Args:
            name (str): Unique identifier for the module.
            module (Any): Instance of the module to register.
            dependencies (List[str], optional): List of module names this module depends on. Defaults to None.
            config (Dict[str, Any], optional): Initial configuration for the module.
        """
        with self.lock:
            if name in self.modules:
                self.logger.warning(f"Module '{name}' is already registered. Overwriting the existing module entry.")

            # Check dependencies
            if dependencies:
                for dep in dependencies:
                    if dep not in self.modules and dep != name: # Allow self-dependency.
                        self.logger.error(f"Dependency '{dep}' not found for module '{name}'.  Registration may fail.")
                        #  Don't raise an exception here; allow the system to try to continue.

            self.modules[name] = {
                "instance": module,
                "dependencies": dependencies or [],
                "status": "registered",
                "config": config or {}  # Store initial configuration
            }
            self.logger.info(f"Module '{name}' registered successfully with status='registered'.")
            self._persist_module(name)  # Persist after *successful* registration

    def get_module(self, name: str) -> Optional[Any]:
        """Retrieves a module by its name."""
        with self.lock:
            module_info = self.modules.get(name)
            if module_info:
                return module_info["instance"]
            return None

    def get_module_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves the configuration of a module."""
        with self.lock:
            module_info = self.modules.get(name)
            if module_info:
                return module_info.get("config")  # Return config, or None
            return None

    def set_module_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Sets or updates the configuration for a module.

        Args:
             name (str): The name of the module
             config (Dict[str, Any]):  The configuration dictionary
        """
        with self.lock:
            if name not in self.modules:
                self.logger.error(f"Cannot set configuration: Module '{name}' not found.")
                return

            module_info = self.modules[name]
            module_info["config"] = config

            # Call set_configuration if available.
            instance = module_info['instance']
            if instance and hasattr(instance, 'set_configuration') and callable(getattr(instance, 'set_configuration')):
                try:
                    instance.set_configuration(config)
                    self.logger.info(f"Module '{name}' configuration updated and applied.")
                except Exception as e:
                     self.logger.error(f"Error applying configuration to module '{name}': {e}")
                     # Don't change the module status here, as the module might still be functional

            self._persist_module(name) # Persist new config
            self.logger.info(f"Module '{name}' configuration updated in registry.")


    def list_modules(self) -> List[str]:
        """Lists all registered modules by name."""
        with self.lock:
            return list(self.modules.keys())

    def start(self) -> bool:
      """
      Starts the NexusCore and all registered modules in the order of their dependencies.

      Returns:
          bool: True if started successfully, False otherwise
      """
      if self.running:
          self.logger.warning("NexusCore is already running.")
          return True

      self.logger.info(f"Starting NexusCore [{self.instance_id}]")

      # Execute pre-start hooks
      pre_hooks_success = self._execute_hooks('pre_start')
      if not pre_hooks_success:
          self.logger.warning("Some pre-start hooks failed, but continuing with startup")

      self.stop_event.clear()
      self.start_time = time.time()
      self.running = True

      # Determine startup order based on dependencies
      try:
          start_order = self._resolve_dependencies()
          self.logger.info(f"Module startup order resolved to: {start_order}")
      except ValueError as e:
          self.logger.error(f"Failed to start NexusCore: {e}")
          self.running = False  # Reset running flag
          return False  # Exit if dependency resolution fails

      # Start each module in the resolved order
      start_success = True
      for module_name in start_order:
          if not self._start_module(module_name):
              start_success = False
              # Continue starting other modules unless configured otherwise
              if self.config.get("fail_fast", False):
                  break

      # Start the background thread for message processing if enabled
      if self.enable_async_messaging:
          self._start_async_message_processing()

      # Execute post-start hooks
      post_hooks_success = self._execute_hooks('post_start')
      if not post_hooks_success:
          self.logger.warning("Some post-start hooks failed")

      if start_success:
          self.logger.info(f"NexusCore [{self.instance_id}] started successfully.")
      else:
          self.logger.warning(f"NexusCore [{self.instance_id}] started with some module failures.")

      return start_success

    def stop(self) -> bool:
      """
      Gracefully stops NexusCore and all modules.

      Returns:
          bool: True if stopped successfully, False otherwise
      """
      if not self.running:
          self.logger.warning("NexusCore is not running.")
          return True

      self.logger.info(f"Stopping NexusCore [{self.instance_id}]")

      # Execute pre-stop hooks
      pre_hooks_success = self._execute_hooks('pre_stop')
      if not pre_hooks_success:
          self.logger.warning("Some pre-stop hooks failed, but continuing with shutdown")

      self.running = False
      self.stop_event.set()

      # Stop the asynchronous message processing thread if running
      if self.message_thread and self.message_thread.is_alive():
          self.logger.info("Waiting for the message processing thread to finish...")
          self.message_thread.join(timeout=5.0)  # Give it some time to exit
          if self.message_thread.is_alive():
              self.logger.warning("Message processing thread did not exit cleanly within 5 seconds.")

      # Determine the order to stop modules (reverse of the startup order)
      stop_success = True
      try:
          stop_order = list(reversed(self._resolve_dependencies()))
          self.logger.info(f"Module shutdown order resolved to: {stop_order}")
      except ValueError as e:
          self.logger.error(f"Error during shutdown dependency resolution: {e}. Forcing shutdown of known modules.")
          stop_order = list(self.modules.keys())  # Attempt to stop all registered, even with bad dependency graph.

      # Stop modules in reverse order
      for module_name in stop_order:
          if not self._stop_module(module_name):
              stop_success = False
              # Continue stopping other modules

      # Execute post-stop hooks
      post_hooks_success = self._execute_hooks('post_stop')
      if not post_hooks_success:
          self.logger.warning("Some post-stop hooks failed")

      uptime = time.time() - self.start_time
      self.logger.info(f"NexusCore [{self.instance_id}] stopped. Total uptime: {uptime:.2f}s")

      # Stop the error handler (fix 6.5: was after the return)
      if hasattr(self, "error_recovery"):
          try:
              self.error_recovery.shutdown()
          except Exception as e:
              self.logger.error(f"ErrorRecoveryManager.shutdown: {e}")

      return stop_success

    # Enhancement for NexusCore._resolve_dependencies()
    def _resolve_dependencies(self) -> List[str]:
        """
        Determines the startup (and shutdown) order of modules based on dependencies.
        Uses topological sorting (Kahn's Algorithm) with improved cycle detection and error handling.

        Returns:
            List[str]: Names of modules in the correct startup order.
        Raises:
            ValueError: If a circular dependency is detected, with detailed information about the cycle.
        """
        with self.lock:
            # Build the dependency graph and track reverse dependencies for cycle detection
            graph = {name: set(info["dependencies"]) for name, info in self.modules.items()}
            reverse_graph = {name: set() for name in self.modules}

            # Populate reverse graph for cycle detection
            for node, deps in graph.items():
                for dep in deps:
                    if dep in reverse_graph:  # Check if dependency is registered
                        reverse_graph[dep].add(node)
                    else:
                        self.logger.warning(f"Module '{node}' depends on unknown module '{dep}'")

            # Kahn's Algorithm with cycle detection
            in_degree = {node: 0 for node in graph}
            for node in graph:
                for dep in graph.get(node, []):
                    if dep in in_degree:  # Check if dependency is registered
                        in_degree[dep] += 1

            # Create priority queue for processing nodes
            queue = [(0, node) for node, degree in in_degree.items() if degree == 0]
            heapq.heapify(queue)
            order = []

            # Process nodes in priority order
            while queue:
                _, node = heapq.heappop(queue)
                order.append(node)

                for neighbor in list(graph.get(node, [])):
                    if neighbor in in_degree:  # Check if neighbor is registered
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            # Use module priority in heap if available, default to 0
                            priority = self.modules.get(neighbor, {}).get("priority", 0)
                            heapq.heappush(queue, (priority, neighbor))

            # Check for circular dependencies
            if len(order) != len(self.modules):
                # Find and report the cycle
                remaining = set(self.modules.keys()) - set(order)
                cycles = self._find_dependency_cycles(remaining, graph, reverse_graph)
                cycle_str = '; '.join([' -> '.join(cycle) for cycle in cycles])
                raise ValueError(f"Circular dependency detected: {cycle_str}")

            return order

    # New helper method for cycle detection
    def _find_dependency_cycles(self, nodes: Set[str], graph: Dict[str, Set[str]],
                              reverse_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Helper method to find and report cycles in the dependency graph.

        Args:
            nodes: Set of nodes that are part of a cycle
            graph: Forward dependency graph
            reverse_graph: Reverse dependency graph for cycle detection

        Returns:
            List of cycles, where each cycle is a list of module names
        """
        cycles = []
        visited = set()

        def dfs(node, path, visited_in_path):
            if node in visited_in_path:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            visited.add(node)
            visited_in_path.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor in nodes and neighbor not in visited:
                    dfs(neighbor, path, visited_in_path.copy())

            path.pop()
            visited_in_path.remove(node)

        # Try to find cycles starting from each remaining node
        for node in nodes:
            if node not in visited:
                dfs(node, [], set())

        return cycles


    def _start_module(self, name: str) -> bool:
      """
      Starts a specific module by name, calling its start() method if available.

      Args:
          name (str): Name of the module to start.

      Returns:
          bool: True if module started successfully, False otherwise
      """
      with self.lock:
          if name not in self.modules:
              self.logger.error(f"Cannot start '{name}': Module not found.")
              return False

          module_info = self.modules[name]
          if module_info["status"] == "running":
              self.logger.debug(f"Module '{name}' is already running.")
              return True

          instance = module_info["instance"]
          try:
              start_method = getattr(instance, "start", None)
              if callable(start_method):
                  start_method()
                  self.logger.info(f"Module '{name}' started successfully.")
              else:
                  self.logger.debug(f"Module '{name}' has no start() method. Marking as 'running' anyway.")
              module_info["status"] = "running"
              self._persist_module(name)  # Persist new status
              return True
          except Exception as e:
              # Appeler error_recovery
              recovery_result = self.error_recovery.handle_error(name, e, "start")
              if not recovery_result["success"]:
                  self.logger.error(f"Error while starting module '{name}': {e}", exc_info=True)
                  module_info["status"] = "error"
                  module_info.setdefault("error_info", {})
                  module_info["error_info"]["last_error"] = str(e)
                  module_info["error_info"]["last_error_time"] = time.time()
                  self._persist_module(name)  # Persist error status
              return recovery_result["success"]


    def _stop_module(self, name: str) -> bool:
      """
      Stops a specific module by name, calling its stop() method if available.

      Args:
          name (str): Name of the module to stop.

      Returns:
          bool: True if module stopped successfully, False otherwise
      """
      with self.lock:
          if name not in self.modules:
              self.logger.error(f"Cannot stop '{name}': Module not found.")
              return False

          module_info = self.modules[name]
          if module_info["status"] != "running":
              self.logger.debug(f"Module '{name}' is not currently running. Status is '{module_info['status']}'.")
              return True  # Consider it a success if it's already not running

          instance = module_info["instance"]
          try:
              stop_method = getattr(instance, "stop", None)
              if callable(stop_method):
                  stop_method()
                  self.logger.info(f"Module '{name}' stopped successfully.")
              else:
                  self.logger.debug(f"Module '{name}' has no stop() method. Marking as 'stopped'.")
              module_info["status"] = "stopped"
              self._persist_module(name)  # Persist
              return True
          except Exception as e:
              self.logger.error(f"Error while stopping module '{name}': {e}", exc_info=True)
              module_info["status"] = "error"
              module_info.setdefault("error_info", {})
              module_info["error_info"]["last_error"] = str(e)
              module_info["error_info"]["last_error_time"] = time.time()
              self._persist_module(name)  # Persist
              return False

    # Enhancements for lifecycle hooks

    def add_hook(self, hook_type: str, hook_func: Callable[['NexusCore'], None]) -> bool:
        """
        Adds a hook to the core lifecycle.

        Args:
            hook_type (str): Type of hook ('pre_start', 'post_start', 'pre_stop', 'post_stop').
            hook_func (Callable[['NexusCore'], None]): Function to execute during the hook.

        Returns:
            bool: True if the hook was added successfully, False otherwise.
        """
        valid_hooks = {'pre_start', 'post_start', 'pre_stop', 'post_stop'}

        if hook_type not in valid_hooks:
            self.logger.warning(f"Unknown hook type: {hook_type}. Valid types are: {valid_hooks}")
            return False

        if not callable(hook_func):
            self.logger.warning(f"Hook function must be callable")
            return False

        hook_list = getattr(self, f"{hook_type}_hooks")

        # Avoid duplicate hooks
        if hook_func in hook_list:
            self.logger.debug(f"Hook function already registered for {hook_type}")
            return True

        hook_list.append(hook_func)
        self.logger.debug(f"Added {hook_type} hook: {hook_func.__name__}")
        return True

    def remove_hook(self, hook_type: str, hook_func: Callable[['NexusCore'], None]) -> bool:
        """
        Removes a previously added hook from the core lifecycle.

        Args:
            hook_type (str): Type of hook ('pre_start', 'post_start', 'pre_stop', 'post_stop').
            hook_func (Callable[['NexusCore'], None]): Function to remove.

        Returns:
            bool: True if the hook was removed successfully, False if not found.
        """
        valid_hooks = {'pre_start', 'post_start', 'pre_stop', 'post_stop'}

        if hook_type not in valid_hooks:
            self.logger.warning(f"Unknown hook type: {hook_type}. Valid types are: {valid_hooks}")
            return False

        hook_list = getattr(self, f"{hook_type}_hooks")

        if hook_func not in hook_list:
            self.logger.debug(f"Hook function not found in {hook_type} hooks")
            return False

        hook_list.remove(hook_func)
        self.logger.debug(f"Removed {hook_type} hook: {hook_func.__name__}")
        return True

    def _execute_hooks(self, hook_type: str) -> bool:
        """
        Internal method to execute all hooks of a given type.

        Args:
            hook_type (str): Type of hooks to execute

        Returns:
            bool: True if all hooks executed successfully, False if any hooks failed
        """
        hook_list = getattr(self, f"{hook_type}_hooks")
        all_succeeded = True

        for i, hook in enumerate(hook_list):
            try:
                self.logger.debug(f"Executing {hook_type} hook {i+1}/{len(hook_list)}: {hook.__name__}")
                hook(self)
            except Exception as e:
                all_succeeded = False
                self.logger.error(f"Error in {hook_type} hook {hook.__name__}: {e}", exc_info=True)
                # Continue with other hooks even if one fails

        return all_succeeded

    def send_message(self, source: str, target: str, message_type: str, content: Any, priority: int = 1) -> None:
        """
        Sends a message from one module to another.  Adds the message to the priority queue.

        Args:
            source (str): Source module name.
            target (str): Target module name.
            message_type (str): Type/category of the message.
            content (Any): The message payload.
            priority (int):  Message priority (lower is higher).
        """

        message = {
            "source": source,
            "target": target,
            "type": message_type,
            "content": content,
            "timestamp": time.time()
        }

        with self.lock:
            heapq.heappush(self.message_queue, (priority, message))  # Use heapq
            self.logger.debug(f"Message from '{source}' to '{target}' (type: {message_type}, priority: {priority}) queued.")

        # If asynchronous messaging is disabled, process messages immediately
        if not self.enable_async_messaging:
            self._process_messages()


    def _process_messages(self):
        """
        Process messages from the queue.  This is either called immediately after
        send_message (if async is disabled) or runs in a separate thread (if async is enabled).
        """
        with self.lock:
            while self.message_queue:
                priority, message = heapq.heappop(self.message_queue) # Use heappop
                self._deliver_message(message)



    def _deliver_message(self, message: Dict[str, Any]) -> None:
        """
        Deliver the message directly to the target's handle_message method, if it exists.
        This is used both in synchronous mode or by the background thread in asynchronous mode.

        Args:
            message (dict): The message to be delivered.
        """
        target = message["target"]
        with self.lock:
            if target in self.modules:
                target_module = self.modules[target]["instance"]
                handle_method = getattr(target_module, "handle_message", None)
                if callable(handle_method):
                    try:
                        handle_method(message)
                    except Exception as e:
                        self.logger.error(f"Error processing message by '{target}': {e}")

    def _start_async_message_processing(self) -> None:
        """
        Spawns a background thread that continuously processes messages from the queue
        until the core is stopped.  The thread calls _deliver_message for each message.
        """

        def message_loop():
            self.logger.info("Asynchronous message processing thread started.")
            while self.running and not self.stop_event.is_set():
                self._process_messages()  # Process all available messages
                time.sleep(0.01)  # Avoid busy-waiting if queue is empty, check frequently
            self.logger.info("Asynchronous message processing thread exiting.")

        self.message_thread = threading.Thread(target=message_loop, daemon=True)
        self.message_thread.start()

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of NexusCore and its modules,
        including health information from ErrorRecoveryManager.

        Returns:
            Dict[str, Any]: Dictionary containing status information.
        """
        with self.lock:
            if self.running:
                uptime = time.time() - self.start_time
            else:
                uptime = 0

            modules_status = {}
            for name, info in self.modules.items():
                modules_status[name] = {
                    "status": info["status"],
                    "dependencies": info["dependencies"],
                    "config": info.get("config", {})  # Include configuration
                }

                # Add health information from error recovery manager
                if hasattr(self, "error_recovery"):
                    health_info = self.error_recovery.get_module_health_status(name)
                    modules_status[name]["health"] = health_info

                # If the module has a get_status() method, include that info
                instance = info["instance"]
                get_status_method = getattr(instance, "get_status", None)
                if callable(get_status_method):
                    try:
                        module_status = get_status_method()
                        if isinstance(module_status, dict):
                            modules_status[name].update(module_status)
                    except Exception as e:
                        self.logger.error(f"Error retrieving status for '{name}': {e}")

            status = {
                "instance_id": self.instance_id,
                "running": self.running,
                "uptime": uptime,
                "modules": modules_status,
                "queue_size": len(self.message_queue),
                "async_messaging_enabled": self.enable_async_messaging
            }

            # Add recovery enabled flag if error recovery is available
            if hasattr(self, "error_recovery"):
                status["recovery_enabled"] = True

            return status



    def update_module_status(self, module_name: str, new_status: str):
        """Updates the status of a module and persists it."""
        with self.lock:
            if module_name in self.modules:
                self.modules[module_name]['status'] = new_status
                self._persist_module(module_name)
                self.logger.info(f"Module '{module_name}' status updated to '{new_status}'")
            else:
                self.logger.error(f"Cannot update status: Module '{module_name}' not found")

# --- fix 6.5: publish/subscribe bus (see core/event_bus.py)
from grandnexus.core.event_bus import install_event_bus as _install_event_bus
_install_event_bus(NexusCore)
