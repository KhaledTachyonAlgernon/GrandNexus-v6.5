"""
registry.py - Advanced Module Registry for GrandNexus

This module implements:
- Enhanced module registration and dependency management
- Bidirectional tracking of dependencies and dependents
- Module lifecycle state tracking and logging of state changes
- Interface verification to ensure basic compatibility
- Dynamic discovery of modules from a Python package or path
- Automatic sync with NexusCore upon registration (if desired)
- Extended metadata handling (version, author, config, etc.)
"""

import logging
import importlib
import pkgutil
import inspect
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set, Union

class ModuleRegistry:
    """
    Advanced registry for GrandNexus modules.

    The ModuleRegistry is responsible for:
        - Keeping a centralized record of modules (by name).
        - Storing module dependencies (which other modules a given module depends on).
        - Storing dependents (which modules depend on a given module).
        - Tracking module metadata, including lifecycle states, timestamps, version, etc.
        - Managing the registration and unregistration process, ensuring no orphan dependencies.
        - Optionally synchronizing module info with NexusCore (ex: register_module calls).
        - Providing a method to discover and automatically register modules from a Python package.

    This registry can be seen as a companion to NexusCore, focusing on structural and
    informational aspects, while NexusCore handles the operational lifecycle (start/stop).
    """

    VALID_STATES: Set[str] = {
        "REGISTERED",  # Initially registered but not yet started
        "STARTING",    # In the process of starting
        "RUNNING",     # Successfully started and operational
        "STOPPING",    # In the process of stopping
        "STOPPED",     # Successfully stopped
        "ERROR"        # Error state (start, stop, or runtime error)
    }

    def __init__(self, core: Any):
        """
        Initialize the ModuleRegistry.

        Args:
            core: A reference to the NexusCore instance (or similar) that orchestrates
                  the system's lifecycle. This registry can delegate calls to that
                  instance, such as 'register_module', 'update_module_status', etc.
        """
        self.logger: logging.Logger = logging.getLogger("ModuleRegistry")
        self.core = core

        # The following dicts store key info about modules:
        #   self.modules       : { module_name: module_instance }
        #   self.dependencies  : { module_name: [dependencies...] }
        #   self.dependents    : { module_name: set(modules_that_depend_on_it) }
        #   self.metadata      : { module_name: { arbitrary_metadata... } }

        self.modules: Dict[str, Any] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Lock to ensure thread safety
        self.lock: threading.RLock = threading.RLock()

        self.logger.info("ModuleRegistry initialized successfully.")

    def register(
        self,
        name: str,
        module_instance: Any,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync_with_core: bool = True,
        auto_verify: bool = True
    ) -> bool:
        """
        Registers a module along with its dependencies and any custom metadata.

        By default, this method also registers the module with the NexusCore (sync_with_core=True).

        Args:
            name (str):
                The unique identifier for the module. Typically a short string or camelCase name.
            module_instance (Any):
                The Python object (class instance) representing the module. Should have
                at least a 'start()' or 'handle_message()' method to be recognized.
            dependencies (List[str], optional):
                Names of other modules upon which this module depends. If any of those
                modules are not yet registered, a warning is logged. Defaults to None.
            metadata (Dict[str, Any], optional):
                Additional information about the module (version, config, author, etc.).
                This dictionary is stored internally; it may also be partially sent to
                the NexusCore if relevant. Defaults to None.
            sync_with_core (bool):
                If True, automatically invokes 'core.register_module' to keep the
                NexusCore's records aligned. If False, no call is made. Defaults to True.
            auto_verify (bool):
                If True, automatically calls _verify_module_interface to check for
                required methods. Set to False to skip verification. Defaults to True.

        Returns:
            bool: True if registration was successful, False otherwise.

        Raises:
            ValueError:
                If the module fails interface verification (no start() or handle_message()).
        """
        deps = dependencies or []
        meta = metadata or {}

        try:
            with self.lock:
                # If this module is already registered, remove its old dependency links
                if name in self.modules:
                    self.logger.warning(f"Module '{name}' was already registered. Overwriting old entry.")
                    self._remove_dependency_links(name)

                # Validate minimal interface
                if auto_verify:
                    self._verify_module_interface(name, module_instance)

                # Check if dependencies exist in the registry
                for dep in deps:
                    if dep not in self.modules and dep != name:
                        # We do not forcibly raise an error, but we do log it
                        self.logger.warning(f"Dependency '{dep}' for module '{name}' is not yet registered. "
                                            "This may cause startup order issues.")

                # Store the module and its dependencies
                self.modules[name] = module_instance
                self.dependencies[name] = deps.copy()  # store a shallow copy
                self.metadata.setdefault(name, {})
                self.metadata[name].update(meta)       # merge new metadata
                # Set up bidirectional links
                for dep in deps:
                    self.dependents.setdefault(dep, set()).add(name)
                if name not in self.dependents:
                    self.dependents[name] = set()

                # Mark the module's initial state
                self._set_module_state(name, "REGISTERED")

                # Optionally inform the NexusCore
                if sync_with_core and hasattr(self.core, "register_module") and callable(self.core.register_module):
                    # If there's a 'config' key in the metadata, pass it as config param
                    if 'config' in meta:
                        self.core.register_module(name, module_instance, deps, meta['config'])
                    else:
                        self.core.register_module(name, module_instance, deps)

                self.logger.info(f"Module '{name}' registered with dependencies={deps}.")
                return True

        except Exception as e:
            self.logger.error(f"Failed to register module '{name}': {e}", exc_info=True)
            return False

    def unregister(self, name: str) -> bool:
        """
        Unregister a module from the registry, removing any references to it.

        This method checks if any other modules still depend on the one being removed.
        If so, it raises ValueError to prevent accidentally removing a module that is
        still in use.

        Args:
            name (str): The name of the module to unregister.

        Returns:
            bool: True if the unregistration succeeded, False if it could not be done
                  (e.g., module wasn't found).

        Raises:
            ValueError:
                If other modules still depend on the one being removed.
        """
        with self.lock:
            if name not in self.modules:
                self.logger.warning(f"Cannot unregister '{name}': module not found in registry.")
                return False

            # If something depends on this module, we can't remove it
            if name in self.dependents and self.dependents[name]:
                blocking_deps = ", ".join(sorted(self.dependents[name]))
                raise ValueError(
                    f"Cannot unregister module '{name}': the following modules depend on it: {blocking_deps}"
                )

            # Remove references
            self._remove_dependency_links(name)
            del self.modules[name]
            del self.dependencies[name]
            if name in self.dependents:
                del self.dependents[name]
            if name in self.metadata:
                del self.metadata[name]

            # Optionally, one could also call `self.core.unregister_module` if that were desired,
            # but the code from Gemini doesn't do that by default.

            self.logger.info(f"Module '{name}' successfully unregistered.")
            return True

    def get(self, name: str) -> Optional[Any]:
        """
        Retrieve the module instance by name.

        Args:
            name (str): The name of the module to retrieve.

        Returns:
            Optional[Any]: The module instance, or None if it doesn't exist.
        """
        with self.lock:
            return self.modules.get(name)

    def get_all_modules(self) -> Dict[str, Any]:
        """
        Return a dictionary of all registered modules.

        Returns:
            Dict[str, Any]: Copy of the internal registry map (name -> module_instance).
        """
        with self.lock:
            return dict(self.modules)

    def list_module_names(self) -> List[str]:
        """
        Get a list of all module names currently in the registry.

        Returns:
            List[str]: Sorted list of module names.
        """
        with self.lock:
            return sorted(self.modules.keys())

    def get_dependencies(self, name: str) -> List[str]:
        """
        Return the list of modules that the specified module depends on.

        Args:
            name (str): The module whose dependencies we want to retrieve.

        Returns:
            List[str]: Names of the modules that 'name' depends on.

        Raises:
            KeyError: If 'name' is not in the registry.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            return list(self.dependencies[name])

    def get_dependents(self, name: str) -> List[str]:
        """
        Return the list of modules that depend on the specified module.

        Args:
            name (str): The module name for which we want to retrieve dependents.

        Returns:
            List[str]: Names of the modules that depend on 'name'.

        Raises:
            KeyError: If 'name' is not in the registry.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            return sorted(self.dependents.get(name, set()))

    def get_module_metadata(self, name: str) -> Dict[str, Any]:
        """
        Retrieve the metadata dictionary for a given module.

        Args:
            name (str): Name of the module.

        Returns:
            Dict[str, Any]: A (copied) dictionary of the metadata for that module,
                            or an empty dict if none is stored.

        Raises:
            KeyError: If the module is not in the registry.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            meta = self.metadata.get(name, {})
            return dict(meta)  # return a shallow copy

    def set_module_metadata(self, name: str, new_meta: Dict[str, Any]) -> None:
        """
        Set or update the metadata for a given module.

        This merges the existing metadata with 'new_meta', overwriting any
        keys that appear in both.

        Args:
            name (str): The module name to update.
            new_meta (Dict[str, Any]): The new metadata to store.

        Raises:
            KeyError: If the module is not in the registry.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            self.metadata.setdefault(name, {})
            self.metadata[name].update(new_meta)
            self.logger.debug(f"Module '{name}' metadata updated with {new_meta}.")

    def get_module_state(self, name: str) -> str:
        """
        Get the current lifecycle state of a module.

        Args:
            name (str): Name of the module.

        Returns:
            str: The module's current state (REGISTERED, RUNNING, etc.), or "UNKNOWN"
                 if no state is recorded.

        Raises:
            KeyError: If 'name' is not registered.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            return self.metadata.get(name, {}).get("state", "UNKNOWN")

    def set_module_state(self, name: str, state: str, error_reason: Optional[str] = None) -> None:
        """
        Public method to forcibly set a module's state. Typically, internal calls use
        the protected _set_module_state, but you may invoke this method if you need
        to set or override the state externally.

        Args:
            name (str): The module name whose state is changing.
            state (str): The new state (must be in VALID_STATES).
            error_reason (str, optional): Explanation if we are transitioning to ERROR.

        Raises:
            KeyError: If the module is not in the registry.
            ValueError: If the provided state is invalid.
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry.")
            self._set_module_state(name, state, error_reason=error_reason)

    def resolve_dependency_order(self) -> List[str]:
        """
        Determine a valid startup (or shutdown) order based on module dependencies,
        typically by delegating to the NexusCore's method, if implemented.

        Returns:
            List[str]: A list of module names in an order that respects dependencies.

        Raises:
            ValueError: If circular dependencies are detected or if the resolution fails.
        """
        if hasattr(self.core, "_resolve_dependencies") and callable(self.core._resolve_dependencies):
            self.logger.debug("Resolving dependency order by delegating to NexusCore.")
            return self.core._resolve_dependencies()
        else:
            # If for any reason we cannot delegate to NexusCore, we can do a local topological sort
            self.logger.debug("NexusCore has no _resolve_dependencies method. Performing local topological sort.")
            return self._local_topo_sort()

    def get_module_dependencies(self, name: str, recursive: bool = False, max_depth: int = 10) -> List[str]:
        """
        Get the dependencies of a module, with option for recursive dependency resolution.

        Args:
            name (str): Module name to get dependencies for
            recursive (bool): If True, include all indirect dependencies
            max_depth (int): Maximum recursion depth for recursive dependencies

        Returns:
            List[str]: List of module names that this module depends on

        Raises:
            KeyError: If module name not found
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry")

            # Direct dependencies
            direct_deps = list(self.dependencies[name])

            # If not recursive, return direct dependencies
            if not recursive:
                return direct_deps

            # Recursive dependency resolution
            all_deps = set(direct_deps)
            to_process = deque([(dep, 1) for dep in direct_deps])  # (dep, depth)

            while to_process:
                dep, depth = to_process.popleft()

                # Stop if we reach max depth
                if depth >= max_depth:
                    continue

                if dep in self.dependencies:
                    for indirect_dep in self.dependencies[dep]:
                        if indirect_dep not in all_deps:
                            all_deps.add(indirect_dep)
                            to_process.append((indirect_dep, depth + 1))

            return list(all_deps)

    def get_module_dependents(self, name: str, recursive: bool = False, max_depth: int = 10) -> List[str]:
        """
        Get modules that depend on the specified module, with option for recursive resolution.

        Args:
            name (str): Module name to get dependents for
            recursive (bool): If True, include all indirect dependents
            max_depth (int): Maximum recursion depth for recursive dependents

        Returns:
            List[str]: List of module names that depend on this module

        Raises:
            KeyError: If module name not found
        """
        with self.lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' not found in the registry")

            # Direct dependents
            direct_deps = list(self.dependents.get(name, set()))

            # If not recursive, return direct dependents
            if not recursive:
                return direct_deps

            # Recursive dependent resolution
            all_deps = set(direct_deps)
            to_process = deque([(dep, 1) for dep in direct_deps])  # (dep, depth)

            while to_process:
                dep, depth = to_process.popleft()

                # Stop if we reach max depth
                if depth >= max_depth:
                    continue

                if dep in self.dependents:
                    for indirect_dep in self.dependents[dep]:
                        if indirect_dep not in all_deps:
                            all_deps.add(indirect_dep)
                            to_process.append((indirect_dep, depth + 1))

            return list(all_deps)

    def check_dependency_cycles(self) -> List[List[str]]:
        """
        Check for dependency cycles in the registry.

        Returns:
            List[List[str]]: List of cycles found, where each cycle is a list of module names
            forming a circular dependency chain. Empty list if no cycles found.
        """
        with self.lock:
            # Convert dependencies to a directed graph
            graph = {name: set(deps) for name, deps in self.dependencies.items()}

            # Build reverse graph
            reverse_graph = {name: set() for name in self.modules}
            for node, deps in graph.items():
                for dep in deps:
                    if dep in reverse_graph:
                        reverse_graph[dep].add(node)

            # Try to do a topological sort
            in_degree = {node: 0 for node in graph}
            for node in graph:
                for dep in graph.get(node, []):
                    if dep in in_degree:
                        in_degree[dep] += 1

            queue = [node for node, degree in in_degree.items() if degree == 0]
            visited = set()

            while queue:
                node = queue.pop(0)
                visited.add(node)

                for neighbor in list(graph.get(node, [])):
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)

            # If topological sort visited all nodes, there are no cycles
            if len(visited) == len(self.modules):
                return []

            # Find cycles using DFS
            remaining = set(self.modules.keys()) - visited

            # Use _find_dependency_cycles from NexusCore if available
            if hasattr(self.core, "_find_dependency_cycles") and callable(self.core._find_dependency_cycles):
                return self.core._find_dependency_cycles(remaining, graph, reverse_graph)

            # Fallback implementation
            cycles = []
            cycle_visited = set()

            def find_cycle(node, path, visited_in_path):
                if node in visited_in_path:
                    # Found a cycle
                    cycle_start = path.index(node)
                    cycles.append(path[cycle_start:] + [node])
                    return

                cycle_visited.add(node)
                visited_in_path.add(node)
                path.append(node)

                for neighbor in graph.get(node, []):
                    if neighbor in remaining and neighbor not in cycle_visited:
                        find_cycle(neighbor, path, visited_in_path.copy())

                path.pop()
                visited_in_path.remove(node)

            # Try to find cycles starting from each remaining node
            for node in remaining:
                if node not in cycle_visited:
                    find_cycle(node, [], set())

            return cycles

    def get_dependency_graph_summary(self) -> Dict[str, Any]:
        """
        Creates a summary of the dependency graph structure for visualization or debugging.

        Returns:
            Dict with:
            - 'modules': list of all module names
            - 'dependencies': dict mapping module names to their dependencies
            - 'dependents': dict mapping module names to their dependents
            - 'orphaned': list of modules with no dependencies or dependents
            - 'cycles': list of dependency cycles if any
            - 'stats': general statistics about the graph
        """
        with self.lock:
            modules = list(self.modules.keys())
            dependencies = {name: list(deps) for name, deps in self.dependencies.items()}
            dependents = {name: list(deps) for name, deps in self.dependents.items()}

            # Find orphaned modules (no dependencies or dependents)
            orphaned = [name for name in modules
                        if not dependencies.get(name) and not dependents.get(name)]

            # Check for cycles
            cycles = self.check_dependency_cycles()

            # Calculate some stats
            stats = {
                "total_modules": len(modules),
                "total_dependency_links": sum(len(deps) for deps in dependencies.values()),
                "max_dependencies": max((len(deps) for deps in dependencies.values()), default=0),
                "max_dependents": max((len(deps) for deps in dependents.values()), default=0),
                "orphaned_count": len(orphaned),
                "cycles_count": len(cycles)
            }

            return {
                "modules": modules,
                "dependencies": dependencies,
                "dependents": dependents,
                "orphaned": orphaned,
                "cycles": cycles,
                "stats": stats
            }

    def discover_modules(self, package_path: str) -> List[str]:
        """
        Dynamically discover and auto-register modules from the specified Python package.

        This scans for submodules, imports them, and looks for a class whose name matches
        the submodule's last part (case-insensitive). If found, an instance is created
        and automatically registered under that name.

        Args:
            package_path (str): The dotted path to the Python package to scan.

        Returns:
            List[str]: A list of module names that were discovered and registered.
        """
        discovered = []
        try:
            package = importlib.import_module(package_path)
            if not hasattr(package, "__path__"):
                self.logger.warning(f"Package '{package_path}' has no __path__. Cannot discover submodules.")
                return discovered

            for _, submodule_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if is_pkg:
                    # We skip sub-packages in this example
                    continue

                try:
                    submod = importlib.import_module(submodule_name)
                    # The last element of submodule_name might be something like "some_module"
                    short_name = submodule_name.split(".")[-1].lower()

                    # Search for a class matching short_name
                    for attr_name, attr_value in vars(submod).items():
                        if inspect.isclass(attr_value):
                            if attr_name.lower() == short_name:
                                instance = attr_value()
                                # The registry name is the class name, but lowercased
                                module_name = attr_name.lower()
                                self.register(module_name, instance)
                                discovered.append(module_name)
                                break

                except Exception as e:
                    self.logger.error(f"Error loading submodule '{submodule_name}': {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error discovering modules in '{package_path}': {e}", exc_info=True)

        if discovered:
            self.logger.info(f"Discovered and registered modules: {discovered}")
        else:
            self.logger.info(f"No modules discovered in '{package_path}'.")

        return discovered

    ############################################################
    #            INTERNAL (PROTECTED) METHODS                  #
    ############################################################

    def _set_module_state(self, name: str, state: str, error_reason: Optional[str] = None) -> None:
        """
        Protected method to change a module's state internally, recording the time
        of the change and also notifying the NexusCore if it has a suitable method.

        Args:
            name (str): The module name.
            state (str): A string in VALID_STATES.
            error_reason (str, optional): If transitioning to ERROR, explain the cause.

        Raises:
            ValueError: If the new state is not recognized.
        """
        if state not in self.VALID_STATES:
            raise ValueError(f"Invalid state '{state}'. Must be one of: {sorted(self.VALID_STATES)}")

        with self.lock:
            old_state = self.metadata.get(name, {}).get("state", "UNKNOWN")
            self.metadata.setdefault(name, {})
            self.metadata[name]["state"] = state
            self.metadata[name]["state_changed"] = time.time()
            if state == "ERROR" and error_reason:
                self.metadata[name]["error_reason"] = error_reason

            self.logger.info(f"Module '{name}' changed state: {old_state} -> {state} "
                             f"{'(reason: ' + error_reason + ')' if error_reason else ''}")

            # Attempt to update the core's view as well
            if hasattr(self.core, "update_module_status") and callable(self.core.update_module_status):
                try:
                    self.core.update_module_status(name, state)
                except Exception as e:
                    self.logger.error(f"Error updating NexusCore module status for '{name}': {e}", exc_info=True)

    def _verify_module_interface(self, name: str, module: Any) -> None:
        """
        Ensure that a module has at least a minimal set of expected methods.

        By default, we require that a module has either a 'start' method or a 'handle_message'
        method. You may expand this logic if your project demands more strict interface checks.

        Args:
            name (str): The module name for logging.
            module (Any): The module instance.

        Raises:
            ValueError: If the module doesn't have either start() or handle_message() method.
        """
        has_start = hasattr(module, "start") and callable(getattr(module, "start"))
        has_handle_message = hasattr(module, "handle_message") and callable(getattr(module, "handle_message"))
        has_stop = hasattr(module, "stop") and callable(getattr(module, "stop"))

        if not (has_start or has_handle_message):
            msg = (f"Module '{name}' provides neither start() nor handle_message(). "
                  f"It might not function properly within GrandNexus.")
            self.logger.warning(msg)
            raise ValueError(msg)

        if has_start and not has_stop:
            self.logger.warning(f"Module '{name}' has start() but no stop() method. "
                              "This may cause issues during shutdown.")

        # Check for additional recommended interfaces
        recommended_methods = {
            "get_status": "report module status",
            "set_configuration": "receive configuration updates"
        }

        for method, purpose in recommended_methods.items():
            if not (hasattr(module, method) and callable(getattr(module, method))):
                self.logger.debug(f"Module '{name}' does not implement optional '{method}()' to {purpose}")

    def _remove_dependency_links(self, name: str) -> None:
        """
        Remove any references to 'name' in the dependencies or dependents structures.

        This is typically used when a module is unregistered or re-registered.
        """
        # For each of 'name's dependencies, remove 'name' from the dep's dependents set
        deps = self.dependencies.get(name, [])
        for dep in deps:
            if dep in self.dependents and name in self.dependents[dep]:
                self.dependents[dep].remove(name)

    def _local_topo_sort(self) -> List[str]:
        """
        Perform a local topological sort of the modules using DFS, to handle dependencies
        if we can't rely on NexusCore.

        Returns:
            List[str]: A valid topological ordering (modules in the correct sequence).

        Raises:
            ValueError: If a cycle is detected.
        """
        visited = set()
        temp_visited = set()
        result = []

        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{node}'.")
            if node not in visited:
                temp_visited.add(node)
                for neighbor in self.dependencies.get(node, []):
                    if neighbor in self.modules:  # only consider known modules
                        visit(neighbor)
                temp_visited.remove(node)
                visited.add(node)
                result.append(node)

        with self.lock:
            for node in self.modules.keys():
                if node not in visited:
                    visit(node)

        # The result is reversed so that dependencies appear first
        return list(reversed(result))
