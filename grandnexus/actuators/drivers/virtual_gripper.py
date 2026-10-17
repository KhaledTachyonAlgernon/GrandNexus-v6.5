# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 59543-59590
# grandnexus/actuators/drivers/virtual_gripper.py
# ──────────────────────────────────────────────────────────────────────────
"""
Exemple : driver simple simulant un gripper (ouvrir / fermer).
"""
import time  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook

class VirtualGripperDriver:  # implémente ActuatorDriver
    def __init__(self):
        self.driver_id = "virt_gripper"
        self.name = "Virtual Gripper"
        self.running = False
        self.state = "open"
        self.latency_budget_ms = 10.0

    def start(self): self.running = True; return True
    def stop(self): self.running = False; return True

    def execute(self, cmd):
        action = cmd.get("action")
        if action not in ("open", "close"):
            return {"success": False, "msg": "unknown action"}
        self.state = action
        time.sleep(0.002)    # simulate latency
        return {"success": True, "state": self.state}

    def health(self):
        return {"state": self.state}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.actuators.actuator_hub import ActuatorHub

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    hub = ActuatorHub(nexus).start()

    # Envoi d’une commande
    nexus.send_message(
        source="planner",
        target="actuator_hub",
        message_type="action_command",
        content={
            "driver_id": "virt_gripper",
            "command": {"action": "close"},
            "priority": 2,
        },
    )
