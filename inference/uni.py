import sys, os, numpy as np
from safetensors.torch import load_file
from collections import defaultdict
from scipy.stats import spearmanr
FA, FB = (sys.argv[1], sys.argv[2])
LA = sys.argv[3] if len(sys.argv) > 3 else 'A'
LB = sys.argv[4] if len(sys.argv) > 4 else 'B'
print(f'Loading {FA} ...')
ta = load_file(FA)
print(f'Loading {FB} ...')
tb = load_file(FB)

def parse(tensors):
    out = defaultdict(dict)
    for k, v in tensors.items():
        p = k.split('|')
        if len(p) == 4 and p[0] == 'dom':
            out[int(p[2]), p[3]][p[1]] = v.numpy().astype(np.float64)
    return out
va, vb = (parse(ta), parse(tb))
keys = sorted(set(va) & set(vb))
behs = sorted(set.intersection(*[set().union(*[set(va[k]) for k in keys]), set().union(*[set(vb[k]) for k in keys])]))
layers = sorted(set((k[0] for k in keys)))
comps = sorted(set((k[1] for k in keys)))
print(f'Shared: {len(behs)} behaviors, {len(keys)} (layer,comp) pairs, layers {min(layers)}..{max(layers)}')

def linear_cka(X, Y):
    K = X @ X.T
    L = Y @ Y.T
    num = (K * L).sum()
    den = np.sqrt((K * K).sum() * (L * L).sum())
    return float(num / den) if den > 1e-10 else 0.0

def rsa(X, Y):
    from scipy.spatial.distance import squareform, pdist
    dX = squareform(pdist(X, 'cosine'))
    dY = squareform(pdist(Y, 'cosine'))
    iu = np.triu_indices(len(dX), k=1)
    r, _ = spearmanr(dX[iu], dY[iu])
    return float(r)
results = []
for key in keys:
    bs = [b for b in behs if b in va[key] and b in vb[key]]
    if len(bs) < 3:
        continue
    X = np.stack([va[key][b] for b in bs])
    Y = np.stack([vb[key][b] for b in bs])
    cka = linear_cka(X, Y)
    rsa_val = rsa(X, Y) if len(bs) >= 4 else 0.0
    results.append({'layer': key[0], 'comp': key[1], 'cka': cka, 'rsa': rsa_val, 'n_behs': len(bs)})
print(f'\nComputed {len(results)} comparisons')
print(f"\n{'Comp':<6s} {'CKA':>8} {'RSA':>8}")
print('-' * 25)
for comp in comps:
    cr = [r for r in results if r['comp'] == comp]
    print(f"  {comp:<6s} {np.mean([r['cka'] for r in cr]):.4f}  {np.mean([r['rsa'] for r in cr]):.4f}")
with open('geometry_results.txt', 'w') as f:
    f.write(f"DoM Geometry: {LA} vs {LB}\nBehaviors: {len(behs)}\n{'=' * 60}\n\n")
    f.write(f"{'Layer':>5} {'Comp':<6} {'CKA':>8} {'RSA':>8}\n{'-' * 30}\n")
    for r in sorted(results, key=lambda x: (x['layer'], x['comp'])):
        f.write(f"L{r['layer']:<4d} {r['comp']:<6s} {r['cka']:>8.4f} {r['rsa']:>8.4f}\n")
print('Saved geometry_results.txt')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = 'geometry_plots'
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], 'cka', 'CKA'), (axes[1], 'rsa', 'RSA')]:
        for comp, color in [('attn', '#5BA4CF'), ('mlp', '#5BBD72')]:
            cr = sorted([r for r in results if r['comp'] == comp], key=lambda x: x['layer'])
            ax.plot([r['layer'] for r in cr], [r[metric] for r in cr], color=color, label=comp, linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f'DoM Geometry: {LA} vs {LB}')
    fig.tight_layout()
    fig.savefig(f'{pdir}/cka_rsa.png', dpi=150)
    plt.close()
    print(f'Plots: {pdir}/')
except:
    pass
print('Done.')
