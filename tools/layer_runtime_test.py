"""
Test de conflit EN FONCTIONNEMENT : cœur 29/05 + orchestrateur + couche vivante du 15/05
sur un même NexusCore démarré, pendant quelques secondes.
Mesures : collisions de noms, messages envoyés / livrés / sans destinataire,
erreurs journalisées par module, statut des modules, fuite de threads.
Usage : python tools/layer_runtime_test.py [secondes]
"""
import collections, importlib, logging, sys, threading, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

LAYER = [  # (module, classe)
    ("grandnexus.clock.cognitive_clock", "CognitiveClock"),
    ("grandnexus.sensors.sensor_hub", "SensorHub"),
    ("grandnexus.actuators.actuator_hub", "ActuatorHub"),
    ("grandnexus.homeostasis.homeostasis_manager", "HomeostasisManager"),
    ("grandnexus.planning.executive_planner", "ExecutivePlanner"),
    ("grandnexus.cognition.perception.filters.attention_filter_v2", "AttentionFilter"),
    ("grandnexus.memory.working_v2", "WorkingMemory"),
    ("grandnexus.security.safety_guard", "SafetyGuard"),
    ("grandnexus.cognition.metacognition.self_model", "SelfModelService"),
    ("grandnexus.cognition.metacognition.affect_regulator", "AffectRegulator"),
    ("grandnexus.ops.cost_optimizer", "CostOptimizer"),
    ("grandnexus.kernel.kernel_bridge", "KernelBridge"),
]

class Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = collections.Counter(); self.samples = {}
    def emit(self, r):
        key = r.name
        self.errors[key] += 1
        self.samples.setdefault(key, r.getMessage()[:140])

def main(seconds=3.0):
    threads_before = threading.active_count()
    col = Collector(); logging.getLogger().addHandler(col)
    from grandnexus.boot import boot
    gn = boot(log_level=logging.CRITICAL)
    for name in list(logging.root.manager.loggerDict):        # certains modules ont leur propre handler
        lg = logging.getLogger(name); lg.propagate = True
    logging.getLogger().setLevel(logging.ERROR)
    core = gn.core
    print(f"démarrage du cœur : {sum(s.ok for s in gn.report)}/{len(gn.report)} étapes")

    sent, undelivered = collections.Counter(), collections.Counter()
    orig_send = core.send_message
    def send(source, target, message_type, content, priority=1):
        sent[(source, message_type)] += 1
        if target not in core.modules:
            undelivered[(source, target)] += 1
        return orig_send(source, target, message_type, content, priority)
    core.send_message = send

    # 1) collisions de noms avant tout démarrage
    instances, report = {}, []
    for mod, cls in LAYER:
        try:
            C = getattr(importlib.import_module(mod), cls)
            name = getattr(C, "MODULE_NAME", cls.lower())
            if name in core.modules:
                report.append((cls, "CONFLIT", f"le nom '{name}' est déjà pris par "
                               f"{type(core.modules[name]['instance']).__name__} : non démarré"))
                continue
            instances[cls] = C(core)
        except Exception as e:
            report.append((cls, "ÉCHEC création", f"{type(e).__name__}: {e}"))

    # 2) démarrage du cœur puis de la couche
    t0 = time.time()
    try:
        core.start()
    except Exception as e:
        report.append(("NexusCore", "ÉCHEC start", f"{type(e).__name__}: {e}"))
    for cls, inst in instances.items():
        try:
            ok = inst.start()
            report.append((cls, "démarré" if ok is not False else "start()=False", ""))
        except Exception as e:
            report.append((cls, "ÉCHEC start", f"{type(e).__name__}: {e}"))
    time.sleep(seconds)

    # 3) santé
    for cls, inst in instances.items():
        hc = getattr(inst, "health_check", None)
        if callable(hc):
            try:
                h = hc(); report.append((cls, "santé", str(h)[:90]))
            except Exception as e:
                report.append((cls, "ÉCHEC santé", f"{type(e).__name__}: {e}"))
    # 4) une question pendant que tout tourne
    try:
        r = gn.orchestrator.process_input("All men are mortal. Socrates is a man. Is Socrates mortal?")
        ans = [o.get("answer") for o in (r["output"] or []) if isinstance(o, dict) and "answer" in o]
        report.append(("Orchestrateur", "question en charge", f"réponses={ans[:2]} succès={r['success']}"))
    except Exception as e:
        report.append(("Orchestrateur", "ÉCHEC question", f"{type(e).__name__}: {e}"))

    # 5) arrêt
    for cls, inst in instances.items():
        try: inst.stop()
        except Exception as e: report.append((cls, "ÉCHEC stop", str(e)[:80]))
    gn.shutdown()
    time.sleep(1.0)
    elapsed = time.time() - t0

    print("\n— Modules de la couche 15/05")
    for cls, status, detail in report:
        print(f"  {cls:22} {status:20} {detail}")
    print(f"\n— Messages en {elapsed:.1f}s : {sum(sent.values())} envoyés")
    for (src, typ), n in sent.most_common(8):
        print(f"  {src:22} {typ:22} {n}")
    print(f"— Sans destinataire enregistré : {sum(undelivered.values())}")
    for (src, tgt), n in undelivered.most_common(8):
        print(f"  {src:22} -> {tgt:22} {n}")
    print(f"— Erreurs journalisées : {sum(col.errors.values())}")
    for name, n in col.errors.most_common(10):
        print(f"  {name:28} {n:5}  ex: {col.samples[name]}")
    print(f"— Threads : avant={threads_before} après arrêt={threading.active_count()}")
    leftover = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    if leftover:
        print("  encore vivants :", collections.Counter(n.split('-')[0] for n in leftover).most_common(8))

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 3.0)
