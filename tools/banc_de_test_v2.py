# ═══ Banc de test GrandNexus v2 — autonome, sans Gemini. N'affiche JAMAIS le texte des items. ═══
import os, sys, json, glob, time, hashlib, subprocess, collections, logging
def _sous_content(motif):
    """glob récursif dans /content SANS descendre dans le Drive monté (peut être immense)."""
    out = []
    for racine, dossiers, _ in os.walk('/content'):
        if racine == '/content':
            dossiers[:] = [d for d in dossiers if d not in ('drive', 'sample_data')]
        out += glob.glob(os.path.join(racine, motif))
    return out
WORK = globals().get('WORK') or sorted({os.path.dirname(os.path.dirname(p)) for p in _sous_content('grandnexus/boot.py')}, key=len)[0]
if not os.environ.get('ANTHROPIC_API_KEY'):
    from google.colab import userdata
    os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
try:
    import anthropic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'anthropic'])
sys.path.insert(0, WORK)
for m_ in [m_ for m_ in list(sys.modules) if m_.startswith('grandnexus')]:
    del sys.modules[m_]
from grandnexus.boot import boot
gn = boot(log_level=logging.ERROR, llm='on', llm_model='claude-sonnet-5-5')
tr = gn.modules()['text_reasoner']
for name in ('GrandNexus.Cognition.Manager', 'GrandNexus.Metacognition.Monitor', 'GrandNexus.Metacognition.Adaptation'):
    logging.getLogger(name).setLevel(logging.CRITICAL)
print('GrandNexus :', WORK, '| version Git :', subprocess.run('git log --oneline | head -1', shell=True, cwd=WORK, capture_output=True, text=True).stdout.strip())

def empreinte_items(items):
    """Empreinte du CONTENU (textes + réponses attendues), indépendante de la date de figement."""
    return hashlib.sha256(json.dumps([list(x) for x in items], ensure_ascii=False).encode()).hexdigest()[:16]

def temoins():
    """Tous les témoins figés présents dans /content : fichier, empreinte du fichier, empreinte du contenu."""
    out = []
    for p in sorted(_sous_content('tests_lab/*.json')):
        blob = open(p, encoding='utf-8').read()
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if 'items' in d:
            out.append({'fichier': p, 'sha_fichier': hashlib.sha256(blob.encode()).hexdigest()[:16],
                        'sha_items': empreinte_items(d['items']), 'n': len(d['items']), 'items': [tuple(x) for x in d['items']]})
    return out

def charge(empreinte):
    """Charge un témoin par empreinte de fichier OU de contenu."""
    for t in temoins():
        if empreinte in (t['sha_fichier'], t['sha_items']):
            return t['items']
    raise FileNotFoundError(empreinte)

def repondre(text, mode):
    o = tr.answer(text, mode=mode, write_back=False)
    defaut = (o['answer'] == 'inconnu' and '?' in text and not o['extraction']['goals']
              and mode not in ('llm_direct',) and o.get('parser') != 'llm-equation')
    return str(o['answer']), defaut

def mesure_temoin(items, nom, runs=3, modes=('grammar', 'llm', 'hybrid', 'llm_direct')):
    """Fige (sans jamais écraser), mesure, sauvegarde. '∅' = « inconnu » sans question formalisée (bon par défaut, pas compris)."""
    labels = ['oui', 'non', 'inconnu', 'contradictoire']
    os.makedirs(os.path.join(WORK, 'tests_lab', 'resultats'), exist_ok=True)
    chemin = os.path.join(WORK, 'tests_lab', nom + '.json')
    deja = [t for t in temoins() if t['sha_items'] == empreinte_items(items)]
    if os.path.exists(chemin) or deja:
        print('déjà figé :', [t['fichier'] for t in deja] or chemin, '→ pas de nouveau figement')
    else:
        blob = json.dumps({'auteur': 'Khaled', 'fige_le': time.strftime('%Y-%m-%d %H:%M'),
                           'items': [list(x) for x in items]}, ensure_ascii=False, indent=1)
        open(chemin, 'w', encoding='utf-8').write(blob)
        subprocess.run(f'git add -A && git -c user.name=Khaled -c user.email=khaled@grandnexus.local commit -qm "freeze {nom} before any run"', shell=True, cwd=WORK)
        print('figé :', chemin)
    print('empreinte du contenu :', empreinte_items(items))
    s0 = dict(gn.translator.stats)
    ans = {m: [[] for _ in items] for m in modes}
    for r in range(runs):
        gn.translator.cache.clear()
        for m in modes:
            if m == 'grammar' and r > 0:
                for i in range(len(items)): ans[m][i].append(ans[m][i][0])
                continue
            for i, (text, gold) in enumerate(items):
                ans[m][i].append(repondre(text, m))
    print(f"\n{'#':>2}  {'attendu':15}" + ''.join(f'{m:19}' for m in modes))
    resume = {}
    for i, (_, gold) in enumerate(items):
        cells = ''
        for m in modes:
            top, n = collections.Counter(a for a, _ in ans[m][i]).most_common(1)[0]
            d = any(df for a, df in ans[m][i] if a == top)
            cells += f"{('✓' if top == gold else '✗') + top + ('∅' if d else '') + (f' {n}/{runs}' if n < runs else ''):19}"
        print(f'{i+1:>2}  {gold:15}' + cells)
    print('\n∅ = « inconnu » donné faute de question formalisée (bonne réponse par défaut, pas une compréhension)')
    for m in modes:
        maj = [collections.Counter(a for a, _ in ans[m][i]).most_common(1)[0][0] for i in range(len(items))]
        ok = sum(a == g for a, (_, g) in zip(maj, items))
        par_defaut = sum(1 for i, (a, (_, g)) in enumerate(zip(maj, items)) if a == g and any(df for x, df in ans[m][i] if x == a))
        fy = sum(a == 'oui' and g != 'oui' for a, (_, g) in zip(maj, items))
        inv = sum(a in ('oui', 'non') and g == 'inconnu' for a, (_, g) in zip(maj, items))
        resume[m] = {'vote': ok, 'par_defaut': par_defaut, 'oui_a_tort': fy, 'invente_sur_inconnu': inv}
        print(f'{m:11} vote {ok}/{len(items)} (dont par défaut : {par_defaut}) | oui à tort : {fy} | oui/non inventé sur un « inconnu » : {inv}')
    st = gn.translator.stats
    print('meilleur témoin bête :', max(sum(g == l for _, g in items) for l in labels), '/', len(items),
          '| appels API :', st['calls'] - s0['calls'], '| rejets :', st['rejected_items'] - s0['rejected_items'])
    json.dump({'nom': nom, 'sha_items': empreinte_items(items), 'resume': resume, 'reponses': {m: [[a for a, _ in x] for x in v] for m, v in ans.items()}},
              open(os.path.join(WORK, 'tests_lab', 'resultats', nom + f'_{time.strftime("%Y%m%d_%H%M")}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    subprocess.run(f'git add -A && git -c user.name=Khaled -c user.email=khaled@grandnexus.local commit -qm "results {nom}"', shell=True, cwd=WORK)
    return resume

print('\nTémoins présents (fichier | empreinte fichier | empreinte contenu) :')
for t in temoins():
    print(f"  {t['sha_fichier']} | {t['sha_items']} | {t['n']:>2} items | {t['fichier']}")
