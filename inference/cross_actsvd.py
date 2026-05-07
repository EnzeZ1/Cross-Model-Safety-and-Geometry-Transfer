import sys, os, numpy as np
from safetensors.torch import load_file
from collections import defaultdict
from scipy.stats import pearsonr

FA, FB = (sys.argv[1], sys.argv[2])
LA = sys.argv[3] if len(sys.argv) > 3 else 'A'
LB = sys.argv[4] if len(sys.argv) > 4 else 'B'

print(f'Loading {FA} ...')
ta = load_file(FA)
print(f'Loading {FB} ...')
tb = load_file(FB)

def parse(t):
    out = {}
    for k, v in t.items():
        p = k.split('|')
        if len(p) == 2 and p[0] in ('U_s', 'U_u'):
            out.setdefault(p[1], {})[p[0]] = v.float().numpy()
    return out
    
va, vb = (parse(ta), parse(tb))
shared = sorted(set(va) & set(vb))
print(f'Shared: {len(shared)} weight matrices')

def phi(U1, U2):
    r1, r2 = (U1.shape[1], U2.shape[1])
    return float(((U1.T @ U2) ** 2).sum() / min(r1, r2))
results = []

for name in shared:
    parts = name.split('.')
    li = int(parts[1])
    lt = 'attn' if 'self_attn' in name else 'mlp'
    results.append({'name': name, 'layer': li, 'type': lt, 'proj': parts[-1], 'phi_su_a': phi(va[name]['U_s'], va[name]['U_u']), 'phi_su_b': phi(vb[name]['U_s'], vb[name]['U_u']), 'phi_ss': phi(va[name]['U_s'], vb[name]['U_s']), 'phi_uu': phi(va[name]['U_u'], vb[name]['U_u']), 'phi_su_x': phi(va[name]['U_s'], vb[name]['U_u'])})

def m(l):
    return sum(l) / len(l) if l else 0

def s(l):
    u = m(l)
    return (sum(((x - u) ** 2 for x in l)) / len(l)) ** 0.5 if l else 0
    
print(f"\n{'':45s} {'Attn':>12s} {'MLP':>12s}")
print('-' * 72)
for metric, label in [('phi_su_a', f'phi(Us,Uu) {LA}'), ('phi_su_b', f'phi(Us,Uu) {LB}'), ('phi_ss', f'phi(Us^A,Us^B) safety'), ('phi_uu', f'phi(Uu^A,Uu^B) utility'), ('phi_su_x', 'phi cross-type')]:
    a = [r[metric] for r in results if r['type'] == 'attn']
    ml = [r[metric] for r in results if r['type'] == 'mlp']
    print(f'  {label:<45s} {m(a):.4f}+/-{s(a):.3f} {m(ml):.4f}+/-{s(ml):.3f}')
    
print(f"\nKEY: cross-model safety phi={m([r['phi_ss'] for r in results]):.4f} vs within phi={m([r['phi_su_a'] for r in results]):.4f}")
with open('cross_actsvd_results.txt', 'w') as f:
    f.write(f"Cross ActSVD: {LA} vs {LB}\n{'=' * 80}\n\n")
    for r in sorted(results, key=lambda x: (x['layer'], x['name'])):
        f.write(f"{r['name']:<45s} su_a={r['phi_su_a']:.4f} su_b={r['phi_su_b']:.4f} ss={r['phi_ss']:.4f} uu={r['phi_uu']:.4f} su_x={r['phi_su_x']:.4f} {r['type']}\n")
print('Saved cross_actsvd_results.txt')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = 'cross_actsvd_plots'
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for ax, comp in [(axes[0], 'attn'), (axes[1], 'mlp')]:
        cr = [r for r in results if r['type'] == comp]
        ld = defaultdict(lambda: defaultdict(list))
        for r in cr:
            for mt in ['phi_ss', 'phi_uu', 'phi_su_a', 'phi_su_x']:
                ld[r['layer']][mt].append(r[mt])
        ls = sorted(ld)
        for mt, lb, c, st in [('phi_ss', 'safety cross', '#5BBD72', '-'), ('phi_uu', 'utility cross', '#5BA4CF', '-'), ('phi_su_a', f'within {LA}', '#E8736C', '--'), ('phi_su_x', 'cross-type', '#999', ':')]:
            ax.plot(ls, [m(ld[l][mt]) for l in ls], color=c, linestyle=st, label=lb, linewidth=1.5)
            
        ax.set_xlabel('Layer')
        ax.set_ylabel('phi')
        ax.set_title(comp.upper())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        
    fig.suptitle(f'Cross ActSVD: {LA} vs {LB}')
    fig.tight_layout()
    fig.savefig(f'{pdir}/layer_curves.png', dpi=150)
    plt.close()
    print(f'Plots: {pdir}/')
except:
    pass
print('Done.')
