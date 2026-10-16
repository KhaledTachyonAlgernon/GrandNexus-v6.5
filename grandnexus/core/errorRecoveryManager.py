"""
error_recovery.py - Module de gestion de récupération d'erreurs pour GrandNexus

Ce module implémente un système de récupération d'erreurs qui collabore avec NexusCore
sans modifier son paradigme d'orchestration fondamental. Il fournit des stratégies
de récupération adaptatives, une surveillance de l'état des modules, et des mécanismes
d'isolation pour les modules problématiques.
"""

import threading
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple, Set

class ErrorRecoveryManager:
    """
    Gestionnaire de récupération d'erreurs pour GrandNexus.

    Cette classe implémente diverses stratégies pour récupérer des erreurs,
    y compris les tentatives automatiques avec backoff exponentiel, la surveillance
    de l'état des modules, et l'isolation des modules problématiques.

    Elle est conçue pour collaborer avec NexusCore sans modifier son paradigme
    d'orchestration fondamental.
    """

    # Error severity levels
    SEVERITY_LOW = 1     # Minor issues, can be ignored
    SEVERITY_MEDIUM = 2  # Significant issues, must be addressed
    SEVERITY_HIGH = 3    # Critical issues, require immediate attention

    # Recovery strategies
    STRATEGY_IGNORE = "ignore"           # Journaliser et continuer
    STRATEGY_RETRY = "retry"             # Retry the operation
    STRATEGY_RESTART = "restart"         # Restart the module
    STRATEGY_ISOLATE = "isolate"         # Isolate the module
    STRATEGY_NOTIFY = "notify"           # Notify administrators

    def __init__(self, core_reference):
        """
        Initialise le gestionnaire de récupération d'erreurs.

        Args:
            core_reference: Référence à l'instance NexusCore (utilisée uniquement pour les callbacks)
        """
        self.core = core_reference
        self.logger = logging.getLogger("ErrorRecoveryManager")

        # Monitoring module status and recovery attempts
        self.module_health: Dict[str, Dict[str, Any]] = {}
        self.recovery_attempts: Dict[str, Dict[str, Any]] = {}

        # Recovery strategy configuration
        self.max_retry_attempts = 3
        self.max_restart_attempts = 2
        self.backoff_base = 2  # Base for exponential backoff
        self.health_check_interval = 60  # secondes

        # Start the state monitoring thread
        self.stop_monitoring = threading.Event()
        self.monitoring_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("ErrorRecoveryManager initialisé")

    def handle_error(self, module_name: str, error: Exception, operation: str,
                    severity: int = SEVERITY_MEDIUM) -> Dict[str, Any]:
        """
        Gère une erreur survenue dans un module.

        Args:
            module_name: Nom du module où l'erreur s'est produite
            error: L'exception qui a été levée
            operation: L'opération qui était en cours (ex: "start", "stop", "message")
            severity: Niveau de gravité de l'erreur

        Returns:
            Dict contenant l'action de récupération prise et le résultat
        """
        self.logger.error(f"Erreur dans le module '{module_name}' pendant {operation}: {error}")

        # Initialize module state tracking if it doesn't exist
        if module_name not in self.module_health:
            self.module_health[module_name] = {
                "error_count": 0,
                "last_error_time": 0,
                "consecutive_errors": 0,
                "status": "healthy"
            }

        # Update the module state
        health = self.module_health[module_name]
        health["error_count"] += 1
        health["last_error_time"] = time.time()
        health["consecutive_errors"] += 1

        # Determine the recovery strategy based on severity and context
        strategy = self._determine_strategy(module_name, error, operation, severity)

        # Execute the recovery strategy
        result = self._execute_strategy(module_name, strategy, error, operation)

        # Update the module state based on the recovery outcome
        if result.get("success", False):
            health["consecutive_errors"] = 0
            if health["status"] == "degraded":
                health["status"] = "recovering"
        else:
            if health["consecutive_errors"] > 3:
                health["status"] = "degraded"
            if health["consecutive_errors"] > 5:
                health["status"] = "critical"

        return result

    def _determine_strategy(self, module_name: str, error: Exception,
                           operation: str, severity: int) -> str:
        """
        Détermine la stratégie de récupération appropriée en fonction du contexte.

        Args:
            module_name: Nom du module
            error: L'exception qui s'est produite
            operation: L'opération en cours
            severity: Niveau de gravité de l'erreur

        Returns:
            La stratégie de récupération sélectionnée
        """
        # Get module state information
        health = self.module_health.get(module_name, {})
        consecutive_errors = health.get("consecutive_errors", 0)

        # Get information about recovery attempts
        if module_name not in self.recovery_attempts:
            self.recovery_attempts[module_name] = {
                "retry_count": 0,
                "restart_count": 0,
                "last_attempt_time": 0
            }
        attempts = self.recovery_attempts[module_name]

        # Logic to determine the strategy
        if severity == self.SEVERITY_LOW:
            return self.STRATEGY_IGNORE

        elif severity == self.SEVERITY_MEDIUM:
            # For start/stop operations, try to restart
            if operation in ["start", "stop"]:
                if attempts["restart_count"] < self.max_restart_attempts:
                    return self.STRATEGY_RESTART
                else:
                    return self.STRATEGY_ISOLATE

            # For message processing, attempt to retry
            elif operation == "message":
                if attempts["retry_count"] < self.max_retry_attempts:
                    return self.STRATEGY_RETRY
                else:
                    return self.STRATEGY_IGNORE

            # Default for medium severity
            return self.STRATEGY_RETRY

        elif severity == self.SEVERITY_HIGH:
            # For critical errors, try restarting once, then isolate
            if attempts["restart_count"] < 1:
                return self.STRATEGY_RESTART
            else:
                return self.STRATEGY_ISOLATE

        #! Default strategy
        return self.STRATEGY_RETRY

    def _execute_strategy(self, module_name: str, strategy: str,
                         error: Exception, operation: str) -> Dict[str, Any]:
        """
        Exécute la stratégie de récupération sélectionnée.

        Args:
            module_name: Nom du module
            strategy: La stratégie de récupération à exécuter
            error: L'exception qui s'est produite
            operation: L'opération en cours

        Returns:
            Dict contenant le résultat de l'action de récupération
        """
        result = {
            "module": module_name,
            "strategy": strategy,
            "operation": operation,
            "success": False,
            "message": ""
        }

        attempts = self.recovery_attempts.get(module_name, {
            "retry_count": 0,
            "restart_count": 0,
            "last_attempt_time": 0
        })

        # Calculate backoff time if necessary
        current_time = time.time()
        time_since_last_attempt = current_time - attempts.get("last_attempt_time", 0)

        if strategy == self.STRATEGY_IGNORE:
            result["success"] = True
            result["message"] = "Erreur journalisée et ignorée"

        elif strategy == self.STRATEGY_RETRY:
            # Update the attempt counter
            attempts["retry_count"] += 1
            attempts["last_attempt_time"] = current_time

            # Calculate the backoff time
            backoff_time = self._calculate_backoff(attempts["retry_count"])

            # If we need to wait, do so
            if time_since_last_attempt < backoff_time:
                time.sleep(backoff_time - time_since_last_attempt)

            # Retry the operation via a callback to NexusCore
            try:
                # Use callbacks to maintain separation of concerns
                if operation == "start":
                    success = self._callback_start_module(module_name)
                    if success:
                        result["success"] = True
                        result["message"] = f"Module '{module_name}' démarré avec succès après nouvelle tentative"
                elif operation == "stop":
                    success = self._callback_stop_module(module_name)
                    if success:
                        result["success"] = True
                        result["message"] = f"Module '{module_name}' arrêté avec succès après nouvelle tentative"
                elif operation == "message":
                    # We can't retry message delivery directly here
                    result["success"] = False
                    result["message"] = "Nouvelle tentative de message non implémentée dans ce contexte"
            except Exception as e:
                result["success"] = False
                result["message"] = f"Nouvelle tentative échouée: {str(e)}"

        elif strategy == self.STRATEGY_RESTART:
            # Update the restart counter
            attempts["restart_count"] += 1
            attempts["last_attempt_time"] = current_time

            # Calculate the backoff time
            backoff_time = self._calculate_backoff(attempts["restart_count"])

            # If we need to wait, do so
            if time_since_last_attempt < backoff_time:
                time.sleep(backoff_time - time_since_last_attempt)

            # Try to stop, then start the module via callbacks
            try:
                # First, try to stop it if it's running
                module_status = self._callback_get_module_status(module_name)
                if module_status == "running":
                    self._callback_stop_module(module_name)

                # Then try to start it
                success = self._callback_start_module(module_name)
                if success:
                    result["success"] = True
                    result["message"] = f"Module '{module_name}' redémarré avec succès"
                else:
                    result["success"] = False
                    result["message"] = f"Échec du redémarrage du module '{module_name}'"
            except Exception as e:
                result["success"] = False
                result["message"] = f"Redémarrage échoué: {str(e)}"

        elif strategy == self.STRATEGY_ISOLATE:
            try:
                # Mark the module as isolated via a callback
                success = self._callback_update_module_status(module_name, "isolated")
                if success:
                    # Journaliser l'isolation
                    self.logger.warning(f"Module '{module_name}' a été isolé en raison d'erreurs persistantes")

                    result["success"] = True
                    result["message"] = f"Module '{module_name}' a été isolé"
                else:
                    result["success"] = False
                    result["message"] = f"Impossible d'isoler: Module '{module_name}' non trouvé"
            except Exception as e:
                result["success"] = False
                result["message"] = f"Isolation échouée: {str(e)}"

        elif strategy == self.STRATEGY_NOTIFY:
            # This would typically integrate with an external notification system
            # For now, we just log it to make it visible
            self.logger.critical(f"ERREUR CRITIQUE dans le module '{module_name}': {error}")
            result["success"] = True
            result["message"] = "Notification d'erreur journalisée"

        # Update retry attempts
        self.recovery_attempts[module_name] = attempts

        return result

    def _calculate_backoff(self, attempt_count: int) -> float:
        """
        Calcule le temps de backoff exponentiel basé sur le nombre de tentatives.

        Args:
            attempt_count: Nombre de tentatives jusqu'à présent

        Returns:
            Temps de backoff en secondes
        """
        # Exponential backoff with jitter
        backoff = (self.backoff_base ** attempt_count) * (0.5 + 0.5 * (uuid.uuid4().int % 100) / 100)
        return min(backoff, 60)  # Cap at 60 seconds

    def _health_monitoring_loop(self):
        """
        Thread d'arrière-plan qui vérifie périodiquement l'état des modules.
        """
        self.logger.info("Thread de surveillance de l'état démarré")

        while not self.stop_monitoring.is_set():
            try:
                self._check_module_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Erreur dans la surveillance de l'état: {e}")
                time.sleep(5)  # Short sleep on error

        self.logger.info("Thread de surveillance de l'état arrêté")

    def _check_module_health(self):
        """
        Vérifie l'état de tous les modules et tente une récupération si nécessaire.
        """
        # Get the list of modules and their status via a callback
        modules_info = self._callback_get_all_modules_info()

        for module_name, module_info in modules_info.items():
            # Ignore modules that are not running
            if module_info.get("status") != "running":
                continue

            # Check whether the module has a health_check method
            has_health_check = self._callback_has_health_check(module_name)

            if has_health_check:
                try:
                     # Call the module's health_check method via a callback
                    health_status = self._callback_check_module_health(module_name)

                     # If the health check fails, attempt a recovery
                    if not health_status.get("healthy", True):
                        self.logger.warning(f"Vérification de l'état échouée pour le module '{module_name}': {health_status.get('message', 'Aucun détail')}")

                         # Create a synthetic error for the recovery
                        error = Exception(f"Vérification de l'état échouée: {health_status.get('message', 'Aucun détail')}")
                        self.handle_error(module_name, error, "health_check", self.SEVERITY_MEDIUM)
                except Exception as e:
                    self.logger.error(f"Erreur pendant la vérification de l'état du module '{module_name}': {e}")

    def reset_recovery_attempts(self, module_name: str):
        """
        Réinitialise les tentatives de récupération pour un module.

        Args:
            module_name: Nom du module
        """
        if module_name in self.recovery_attempts:
            self.recovery_attempts[module_name] = {
                "retry_count": 0,
                "restart_count": 0,
                "last_attempt_time": 0
            }
            self.logger.info(f"Tentatives de récupération réinitialisées pour le module '{module_name}'")

    def get_module_health_status(self, module_name: str = None) -> Dict[str, Any]:
        """
        Obtient l'état de santé d'un module spécifique ou de tous les modules.

        Args:
            module_name: Nom du module, ou None pour tous les modules

        Returns:
            Dict contenant les informations sur l'état de santé
        """
        if module_name:
            return self.module_health.get(module_name, {
                "error_count": 0,
                "last_error_time": 0,
                "consecutive_errors": 0,
                "status": "unknown"
            })
        else:
            return self.module_health

    def shutdown(self):
        """
        Arrête le gestionnaire de récupération d'erreurs.
        """
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Arrêt d'ErrorRecoveryManager terminé")

     # Callback methods to maintain separation of responsibilities
     # These methods call NexusCore without altering its orchestration paradigm

    def _callback_start_module(self, module_name: str) -> bool:
        """
        Callback pour démarrer un module via NexusCore.

        Args:
            module_name: Nom du module à démarrer

        Returns:
            bool: True si le démarrage a réussi, False sinon
        """
        try:
             # Call the _start_module method of NexusCore
             # This approach maintains separation of responsibilities
            self.core._start_module(module_name)
            return True
        except Exception as e:
            self.logger.error(f"Callback de démarrage échoué pour '{module_name}': {e}")
            return False

    def _callback_stop_module(self, module_name: str) -> bool:
        """
        Callback pour arrêter un module via NexusCore.

        Args:
            module_name: Nom du module à arrêter

        Returns:
            bool: True si l'arrêt a réussi, False sinon
        """
        try:
             # Call the _stop_module method of NexusCore
            self.core._stop_module(module_name)
            return True
        except Exception as e:
            self.logger.error(f"Callback d'arrêt échoué pour '{module_name}': {e}")
            return False

    def _callback_get_module_status(self, module_name: str) -> str:
        """
        Callback pour obtenir l'état d'un module via NexusCore.

        Args:
            module_name: Nom du module

        Returns:
            str: État du module
        """
        try:
             # Access the module's state via NexusCore
            with self.core.lock:
                if module_name in self.core.modules:
                    return self.core.modules[module_name]["status"]
                return "unknown"
        except Exception as e:
            self.logger.error(f"Callback d'obtention d'état échoué pour '{module_name}': {e}")
            return "unknown"

    def _callback_update_module_status(self, module_name: str, status: str) -> bool:
        """
        Callback pour mettre à jour l'état d'un module via NexusCore.

        Args:
            module_name: Nom du module
            status: Nouvel état

        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        try:
             # Use the update_module_status method of NexusCore if it exists
            if hasattr(self.core, "update_module_status") and callable(getattr(self.core, "update_module_status")):
                self.core.update_module_status(module_name, status)
                return True

             # Otherwise, update directly (less ideal)
            with self.core.lock:
                if module_name in self.core.modules:
                    self.core.modules[module_name]["status"] = status
                    self.core._persist_module(module_name)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Callback de mise à jour d'état échoué pour '{module_name}': {e}")
            return False

    def _callback_get_all_modules_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Callback pour obtenir les informations sur tous les modules via NexusCore.

        Returns:
            Dict: Informations sur tous les modules
        """
        try:
            # Access module information via NexusCore
            with self.core.lock:
                return {name: info.copy() for name, info in self.core.modules.items()}
        except Exception as e:
            self.logger.error(f"Callback d'obtention des informations sur les modules échoué: {e}")
            return {}

    def _callback_has_health_check(self, module_name: str) -> bool:
        """
        Callback pour vérifier si un module a une méthode health_check.

        Args:
            module_name: Nom du module

        Returns:
            bool: True si le module a une méthode health_check, False sinon
        """
        try:
            # Check if the module has a health_check method
            with self.core.lock:
                if module_name in self.core.modules:
                    instance = self.core.modules[module_name]["instance"]
                    return hasattr(instance, "health_check") and callable(getattr(instance, "health_check"))
                return False
        except Exception as e:
            self.logger.error(f"Callback de vérification de health_check échoué pour '{module_name}': {e}")
            return False

    def _callback_check_module_health(self, module_name: str) -> Dict[str, Any]:
        """
        Callback pour vérifier l'état de santé d'un module.

        Args:
            module_name: Nom du module

        Returns:
            Dict: Résultat de la vérification de l'état de santé
        """
        try:
            # Call the module's health_check method
            with self.core.lock:
                if module_name in self.core.modules:
                    instance = self.core.modules[module_name]["instance"]
                    health_check = getattr(instance, "health_check", None)
                    if callable(health_check):
                        return health_check()
                return {"healthy": True}
        except Exception as e:
            self.logger.error(f"Callback de vérification de l'état de santé échoué pour '{module_name}': {e}")
            return {"healthy": False, "message": f"Erreur lors de la vérification: {str(e)}"}


# Extension class for NexusCore that preserves its orchestration paradigm
class ErrorRecoveryExtension:
    """
    Extension pour NexusCore qui ajoute des capacités de récupération d'erreurs
    sans modifier le paradigme d'orchestration fondamental.

    Cette classe est conçue pour être utilisée comme un mixin ou une extension
    de NexusCore, en ajoutant des points d'intégration minimaux pour la gestion
    d'erreurs.
    """

    def initialize_error_recovery(self):
        """
        Initialise le gestionnaire de récupération d'erreurs.
        Cette méthode doit être appelée après l'initialisation de NexusCore.
        """
        self.error_recovery = ErrorRecoveryManager(self)
        self.logger.info("Gestionnaire de récupération d'erreurs initialisé")

    def _start_module_with_recovery(self, name: str) -> None:
        """
        Version améliorée de _start_module avec récupération d'erreurs.

        Args:
            name (str): Nom du module à démarrer.
        """
        with self.lock:
            if name not in self.modules:
                self.logger.error(f"Impossible de démarrer '{name}': Module non trouvé.")
                return

            module_info = self.modules[name]
            if module_info["status"] == "running":
                self.logger.debug(f"Module '{name}' est déjà en cours d'exécution.")
                return

            # Ignore isolated modules
            if module_info["status"] == "isolated":
                self.logger.warning(f"Module '{name}' est isolé en raison d'erreurs précédentes. Démarrage ignoré.")
                return

            instance = module_info["instance"]
            try:
                start_method = getattr(instance, "start", None)
                if callable(start_method):
                    start_method()
                    self.logger.info(f"Module '{name}' démarré avec succès.")
                else:
                    self.logger.debug(f"Module '{name}' n'a pas de méthode start(). Marqué comme 'running' quand même.")
                module_info["status"] = "running"
                self._persist_module(name)  # Persist the new state

                # Reset recovery attempts on successful startup
                self.error_recovery.reset_recovery_attempts(name)

            except Exception as e:
                # Use the error recovery handler
                recovery_result = self.error_recovery.handle_error(name, e, "start")

                if not recovery_result["success"]:
                    self.logger.error(f"Erreur lors du démarrage du module '{name}': {e}")
                    module_info["status"] = "error"
                    self._persist_module(name)  # Persist the error state

    def _stop_module_with_recovery(self, name: str) -> None:
        """
        Version améliorée de _stop_module avec récupération d'erreurs.

        Args:
            name (str): Nom du module à arrêter.
        """
        with self.lock:
            if name not in self.modules:
                self.logger.error(f"Impossible d'arrêter '{name}': Module non trouvé.")
                return

            module_info = self.modules[name]
            if module_info["status"] != "running":
                self.logger.debug(f"Module '{name}' n'est pas en cours d'exécution. État actuel: '{module_info['status']}'.")
                return

            instance = module_info["instance"]
            try:
                stop_method = getattr(instance, "stop", None)
                if callable(stop_method):
                    stop_method()
                    self.logger.info(f"Module '{name}' arrêté avec succès.")
                else:
                    self.logger.debug(f"Module '{name}' n'a pas de méthode stop(). Marqué comme 'stopped' quand même.")
                module_info["status"] = "stopped"
                self._persist_module(name)  # Persister

                # Reset recovery attempts on successful shutdown
                self.error_recovery.reset_recovery_attempts(name)

            except Exception as e:
                # Use the error recovery handler
                recovery_result = self.error_recovery.handle_error(name, e, "stop")

                if not recovery_result["success"]:
                    self.logger.error(f"Erreur lors de l'arrêt du module '{name}': {e}")
                    module_info["status"] = "error"
                    self._persist_module(name)  # Persister

    def _deliver_message_with_recovery(self, message: Dict[str, Any]) -> None:
        """
        Version améliorée de _deliver_message avec récupération d'erreurs.

        Args:
            message (dict): Le message à livrer.
        """
        target = message["target"]
        with self.lock:
            if target in self.modules:
                module_info = self.modules[target]

                # Ignore message delivery to isolated modules
                if module_info["status"] == "isolated":
                    self.logger.warning(f"Livraison de message ignorée: Module '{target}' est isolé.")
                    return

                target_module = module_info["instance"]
                handle_method = getattr(target_module, "handle_message", None)
                if callable(handle_method):
                    try:
                        handle_method(message)
                    except Exception as e:
                        # Use the error recovery manager
                        recovery_result = self.error_recovery.handle_error(target, e, "message")

                        if not recovery_result["success"]:
                            self.logger.error(f"Erreur lors du traitement du message par '{target}': {e}")

    def reset_module(self, name: str) -> bool:
        """
        Réinitialise un module qui est en état d'erreur ou isolé, permettant de le démarrer à nouveau.

        Args:
            name (str): Nom du module à réinitialiser

        Returns:
            bool: True si la réinitialisation a réussi, False sinon
        """
        with self.lock:
            if name not in self.modules:
                self.logger.error(f"Impossible de réinitialiser '{name}': Module non trouvé.")
                return False

            module_info = self.modules[name]
            current_status = module_info["status"]

            if current_status in ["error", "isolated"]:
                # Reset the module state
                module_info["status"] = "registered"
                self._persist_module(name)

                # Reset recovery attempts
                self.error_recovery.reset_recovery_attempts(name)

                self.logger.info(f"Module '{name}' a été réinitialisé de '{current_status}' à 'registered'")
                return True
            else:
                self.logger.warning(f"Module '{name}' est en état '{current_status}', pas 'error' ou 'isolated'. Aucune réinitialisation nécessaire.")
                return False

    def get_enhanced_status(self) -> Dict[str, Any]:
        """
        Version améliorée de get_status qui inclut les informations de santé.

        Returns:
            Dict[str, Any]: Dictionnaire contenant les informations d'état.
        """
        # Call the original get_status method first
        status = self.get_status()

        # Add health information for each module
        for name in status["modules"]:
            health_info = self.error_recovery.get_module_health_status(name)
            status["modules"][name]["health"] = health_info

        # Add error recovery information
        status["recovery_enabled"] = True

        return status

    def shutdown_with_recovery(self) -> None:
        """
        Version améliorée de stop qui arrête également le gestionnaire de récupération d'erreurs.
        """
        # First stop NexusCore normally
        self.stop()

        # Then stop the error recovery manager
        if hasattr(self, "error_recovery"):
            self.error_recovery.shutdown()


