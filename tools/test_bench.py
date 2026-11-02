# === GrandNexus test bench v2 - standalone (exec in Colab or a script). NEVER prints the items' texts. ===
import os, sys, json, glob, time, hashlib, subprocess, collections, logging
def _sous_content(motif):
    """Recursive glob under /content WITHOUT entering the mounted Drive (can be huge)."""
    out = []
    for racine, folders, _ in os.walk('/content'):
        if racine == '/content':
            folders[:] = [d for d in folders if d not in ('drive', 'sample_data')]
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
from grandnexus.nexus_logic import normalise_label
gn = boot(log_level=logging.ERROR, llm='on', llm_model='claude-sonnet-5-5')
tr = gn.modules()['text_reasoner']
for name in ('GrandNexus.Cognition.Manager', 'GrandNexus.Metacognition.Monitor', 'GrandNexus.Metacognition.Adaptation'):
    logging.getLogger(name).setLevel(logging.CRITICAL)
print('GrandNexus :', WORK, '| version Git :', subprocess.run('git log --oneline | head -1', shell=True, cwd=WORK, capture_output=True, text=True).stdout.strip())

def fingerprint_items(items):
    """CONTENT fingerprint (texts + expected answers), independent of the freeze date."""
    return hashlib.sha256(json.dumps([list(x) for x in items], ensure_ascii=False).encode()).hexdigest()[:16]

def temoins():
    """Every frozen control set under /content: file, file fingerprint, content fingerprint."""
    out = []
    for p in sorted(_sous_content('tests_lab/*.json')):
        blob = open(p, encoding='utf-8').read()
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if 'items' in d:
            out.append({'file': p, 'sha_file': hashlib.sha256(blob.encode()).hexdigest()[:16],
                        'sha_items': fingerprint_items(d['items']), 'n': len(d['items']), 'items': [tuple(x) for x in d['items']]})
    return out

def charge(fingerprint):
    """Load a control set by file OR content fingerprint."""
    for t in temoins():
        if fingerprint in (t['sha_file'], t['sha_items']):
            return t['items']
    raise FileNotFoundError(fingerprint)

def repondre(text, mode):
    o = tr.answer(text, mode=mode, write_back=False)
    defaut = (o['answer'] == 'unknown' and '?' in text and not o['extraction']['goals'] and mode != 'nx'
              and mode not in ('llm_direct',) and o.get('parser') != 'llm-equation')
    return str(o['answer']), defaut

def measure_temoin(items, nom, runs=3, modes=('grammar', 'nx', 'llm', 'llm_direct')):
    """Freeze (never overwrite), measure, save. '∅' = "unknown" with no formalised question (right by default, not understood)."""
    items = [(t, normalise_label(g)) for t, g in items]
    labels = ['yes', 'no', 'unknown', 'contradictory']
    os.makedirs(os.path.join(WORK, 'tests_lab', 'resultats'), exist_ok=True)
    chemin = os.path.join(WORK, 'tests_lab', nom + '.json')
    deja = [t for t in temoins() if t['sha_items'] == fingerprint_items(items)]
    if os.path.exists(chemin) or deja:
        print('already frozen:', [t['file'] for t in deja] or chemin, '-> not frozen again')
    else:
        blob = json.dumps({'auteur': 'Khaled', 'fige_le': time.strftime('%Y-%m-%d %H:%M'),
                           'items': [list(x) for x in items]}, ensure_ascii=False, indent=1)
        open(chemin, 'w', encoding='utf-8').write(blob)
        subprocess.run(f'git add -A && git -c user.name=Khaled -c user.email=khaled@grandnexus.local commit -qm "freeze {nom} before any run"', shell=True, cwd=WORK)
        print('frozen:', chemin)
    print('content fingerprint:', fingerprint_items(items))
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
    print(f"\n{'#':>2}  {'expected':15}" + ''.join(f'{m:19}' for m in modes))
    resume = {}
    for i, (_, gold) in enumerate(items):
        cells = ''
        for m in modes:
            top, n = collections.Counter(a for a, _ in ans[m][i]).most_common(1)[0]
            d = any(df for a, df in ans[m][i] if a == top)
            cells += f"{('✓' if top == gold else '✗') + top + ('∅' if d else '') + (f' {n}/{runs}' if n < runs else ''):19}"
        print(f'{i+1:>2}  {gold:15}' + cells)
    print('\n∅ = "unknown" given for lack of a formalised question (right by default, not an understanding)')
    for m in modes:
        maj = [collections.Counter(a for a, _ in ans[m][i]).most_common(1)[0][0] for i in range(len(items))]
        ok = sum(a == g for a, (_, g) in zip(maj, items))
        par_defaut = sum(1 for i, (a, (_, g)) in enumerate(zip(maj, items)) if a == g and any(df for x, df in ans[m][i] if x == a))
        fy = sum(a == 'yes' and g != 'yes' for a, (_, g) in zip(maj, items))
        inv = sum(a in ('yes', 'no') and g == 'unknown' for a, (_, g) in zip(maj, items))
        resume[m] = {'vote': ok, 'par_defaut': par_defaut, 'oui_a_tort': fy, 'invente_sur_inconnu': inv}
        print(f'{m:11} vote {ok}/{len(items)} (by default: {par_defaut}) | false yes : {fy} | yes/no invented on an « inconnu » : {inv}')
    st = gn.translator.stats
    print('best dumb baseline :', max(sum(g == l for _, g in items) for l in labels), '/', len(items),
          '| API calls :', st['calls'] - s0['calls'], '| rejections :', st['rejected_items'] - s0['rejected_items'])
    json.dump({'nom': nom, 'sha_items': fingerprint_items(items), 'resume': resume, 'reponses': {m: [[a for a, _ in x] for x in v] for m, v in ans.items()}},
              open(os.path.join(WORK, 'tests_lab', 'resultats', nom + f'_{time.strftime("%Y%m%d_%H%M")}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    subprocess.run(f'git add -A && git -c user.name=Khaled -c user.email=khaled@grandnexus.local commit -qm "results {nom}"', shell=True, cwd=WORK)
    return resume

print('\nControl sets present (file | file fingerprint | content fingerprint):')
for t in temoins():
    print(f"  {t['sha_file']} | {t['sha_items']} | {t['n']:>2} items | {t['file']}")
