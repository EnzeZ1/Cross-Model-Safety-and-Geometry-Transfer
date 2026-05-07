import sys, os, json, numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
FA, FB = (sys.argv[1], sys.argv[2])

LA = sys.argv[3] if len(sys.argv) > 3 else 'A'
LB = sys.argv[4] if len(sys.argv) > 4 else 'B'

COMPS = ['attn', 'mlp']
ja = json.load(open(FA))
jb = json.load(open(FB))
behs = sorted(set(ja) & set(jb))
num_layers = min(len(ja[behs[0]]), len(jb[behs[0]]))
print(f'Shared: {len(behs)} behaviors, {num_layers} layers')

def get(d, b, l, c):
    v = d[b][str(l)][c]
    return v.get('train', 0.0) if isinstance(v, dict) else float(v)
    
print(f"\n{'Behavior':<35s} {'Peak A':>7} {'Peak B':>7} {'|d|':>5} {'CurveCorr':>10}")
print('-' * 65)

corrs = {}
for b in behs:
    for c in COMPS:
        ca = [get(ja, b, l, c) for l in range(num_layers)]
        cb = [get(jb, b, l, c) for l in range(num_layers)]
        r, _ = pearsonr(ca, cb)
        corrs[b, c] = r
        
    ca_m = [get(ja, b, l, 'mlp') for l in range(num_layers)]
    cb_m = [get(jb, b, l, 'mlp') for l in range(num_layers)]
    
    pa, pb = (int(np.argmax(ca_m)), int(np.argmax(cb_m)))
    print(f"{b:<35s} L{pa:<5d} L{pb:<5d} {abs(pa - pb):>5d} {corrs[b, 'mlp']:>10.3f}")
    
mlp_corrs = [corrs[b, 'mlp'] for b in behs]
print(f'\nMLP curve correlation: mean={np.mean(mlp_corrs):.3f}')
with open('cross_jsd_results.txt', 'w') as f:
    f.write(f"Cross JSD: {LA} vs {LB}\n{'=' * 60}\n")
    for b in behs:
        ca = [get(ja, b, l, 'mlp') for l in range(num_layers)]
        cb = [get(jb, b, l, 'mlp') for l in range(num_layers)]
        f.write(f"{b:<35s} peak_A=L{int(np.argmax(ca))} peak_B=L{int(np.argmax(cb))} r={corrs[b, 'mlp']:.3f}\n")
    f.write(f'\nMean r: {np.mean(mlp_corrs):.3f}\n')
print('Saved cross_jsd_results.txt')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    pdir = 'cross_jsd_plots'
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for ax, c in [(axes[0], 'attn'), (axes[1], 'mlp')]:
        for d, lb, color, st in [(ja, LA, '#5BA4CF', '-'), (jb, LB, '#E8736C', '--')]:
            ys = [np.mean([get(d, b, l, c) for b in behs]) for l in range(num_layers)]
            ax.plot(range(num_layers), ys, color=color, linestyle=st, label=lb, linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('JSD')
        ax.set_title(c.upper())
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        
    fig.suptitle(f'JSD: {LA} vs {LB}')
    fig.tight_layout()
    fig.savefig(f'{pdir}/overlay.png', dpi=150)
    plt.close()
    print(f'Plots: {pdir}/')
except:
    pass
print('Done.')
