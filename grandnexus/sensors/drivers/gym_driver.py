# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 59139-59200
# grandnexus/sensors/drivers/gym_driver.py
# ──────────────────────────────────────────────────────────────────────────
"""
Exemple Driver : OpenAI Gym observation collector.
"""

import uuid
import gymnasium as gym

class GymObservationDriver:  # Conforme à SensorDriver Protocol
    def __init__(self):
        self.driver_id = "gym_env"
        self.name = "OpenAI-Gym CartPole"
        self.frequency_hz = 20.0
        self.running = False
        self._env = gym.make("CartPole-v1", render_mode=None)
        self._obs, _ = self._env.reset(seed=42)

    # Obligatoire
    def start(self) -> bool:
        self.running = True
        return True

    def stop(self) -> bool:
        self.running = False
        self._env.close()
        return True

    def read(self):
        if not self.running:
            return None
        action = self._env.action_space.sample()
        self._obs, reward, terminated, truncated, info = self._env.step(action)
        return {
            "observation": self._obs.tolist(),
            "reward": reward,
            "done": terminated or truncated,
        }

    def health(self):
        return {"env_alive": self.running}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    """Élément	Détail	Bénéfice
    Cadence externe	Souscription au tick fast pour maintenir la phase avec CognitiveClock.	Alignement exact des frames sensoriels et du pipeline perception ➜ raisonnement.
    Drivers sandboxés (opt.)	sandbox_drivers=True ➜ pilote lancé dans un multiprocessing.Process, communication ZMQ.	Confinement de code tiers et tolérance aux crashs.
    Observabilité fine	health_check() renvoie jitter moyen par driver ; ErrorRecoveryManager peut isoler un driver défaillant.	Stabilise la stack face à des capteurs intermittents.
    Hot-swap	Ajout d’un nouveau fichier dans grandnexus/sensors/drivers/ → reload dynamique via RPC (hub.reload_plugins(), non montré ici).
    """

    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.sensors.sensor_hub import SensorHub

    nexus = NexusCore()
    clock = CognitiveClock(nexus).start()
    hub   = SensorHub(nexus).start()

    # Consommer depuis perception
    def debug_printer(msg):
        if msg["type"] == "sensor_data": print(msg["content"]["driver_id"], msg["content"]["data"][:2])
    nexus.send_message = debug_printer  # monkey-patch pour test rapide
