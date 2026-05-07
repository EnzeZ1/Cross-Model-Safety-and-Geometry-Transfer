import os, sys
gpu_list = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
MODEL_B_NAME = sys.argv[2]
DATA_MODEL_B = sys.argv[3]
DOM_A_FILE = sys.argv[4]
LABEL_A = sys.argv[5] if len(sys.argv) > 5 else 'A'
LABEL_B = sys.argv[6] if len(sys.argv) > 6 else 'B'
import json, warnings
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
COMPONENTS = ['attn', 'mlp']
MODEL_COL = 'model_name'
SYSTEM_OT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

def build_input(block):
    if DATA_MODEL_B == 'ot-7b':
        return f"<|im_start|>system\n{SYSTEM_OT}<|im_end|>\n<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n{block['sentence']}\n<|end_of_thought|>\n\n<|im_end|>\n"
    elif DATA_MODEL_B in ('r1-8b', 'r1-32b', 'r1-8b-0528'):
        return f"<｜begin▁of▁sentence｜><｜User｜>{block['query']}<｜Assistant｜><think>\n{block['sentence']}\n</think>\n\n<｜end▁of▁sentence｜>"
    elif DATA_MODEL_B in ('QwQ', 'qwen3-8b'):
        return f"<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<think>\n{block['sentence']}\n</think>\n\n<|im_end|>\n"
    else:
        raise ValueError(f"Unknown '{DATA_MODEL_B}'")
print(f"Loading {LABEL_A}'s vectors from {DOM_A_FILE} ...")
tensors_a = load_file(DOM_A_FILE)
vecs_a = defaultdict(dict)
for k, v in tensors_a.items():
    p = k.split('|')
    if len(p) == 4 and p[0] == 'dom':
        vecs_a[int(p[2]), p[3]][p[1]] = v.numpy().astype(np.float64)
a_behaviors = sorted(set().union(*[set(vecs_a[k]) for k in vecs_a]))
num_layers_a = max((k[0] for k in vecs_a)) + 1
print(f'  {len(a_behaviors)} behaviors, {num_layers_a} layers')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
print(f'Loading model B: {MODEL_B_NAME} on GPUs [{gpu_list}] ...')
model = AutoModelForCausalLM.from_pretrained(MODEL_B_NAME, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_B_NAME)
num_layers = AutoConfig.from_pretrained(MODEL_B_NAME).num_hidden_layers
input_device = next(model.parameters()).device
acts = {}

def make_hook(name):

    def hook(mod, inp, out):
        acts[name] = out[0] if isinstance(out, (tuple, list)) else out
    return hook
for i, layer in enumerate(model.model.layers):
    layer.self_attn.o_proj.register_forward_hook(make_hook(f'L{i}_attn'))
    layer.mlp.register_forward_hook(make_hook(f'L{i}_mlp'))
store = json.load(open('data.json'))
data = store['data']
valid_ids = set((i for i, e in enumerate(data) if e[MODEL_COL] == DATA_MODEL_B))
test = {b: [p for p in ps if p['pos'] in valid_ids and p['neg'] in valid_ids] for b, ps in store['test'].items() if any((p['pos'] in valid_ids and p['neg'] in valid_ids for p in ps))}
train = {b: [p for p in ps if p['pos'] in valid_ids and p['neg'] in valid_ids] for b, ps in store['train'].items() if any((p['pos'] in valid_ids and p['neg'] in valid_ids for p in ps))}
shared = sorted(set(a_behaviors) & set(test.keys()) & set(train.keys()))
print(f'Shared behaviors: {len(shared)}')

def extract(block):
    full_str = build_input(block)
    target = block['sentence']
    char_st = full_str.find(target)
    if char_st == -1:
        raise ValueError('sentence not found')
    char_ed = char_st + len(target)
    enc = tokenizer(full_str, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc['offset_mapping'][0]
    indices = [i for i, (s, e) in enumerate(offsets.tolist()) if s < char_ed and e > char_st]
    if not indices:
        st, ed = (len(enc['input_ids'][0]) - 1, len(enc['input_ids'][0]))
    else:
        st, ed = (indices[0], indices[-1] + 1)
    acts.clear()
    with torch.no_grad():
        model(enc['input_ids'].to(input_device))
    h = {}
    for l in range(num_layers):
        h[l] = {}
        for c in COMPONENTS:
            t_gpu = acts[f'L{l}_{c}'][0, st:ed]
            v_gpu = t_gpu.float().mean(0)
            h[l][c] = v_gpu.detach().cpu().numpy().astype(np.float64)
    acts.clear()
    return h
cache = {}
needed = set()
for b in shared:
    for pair in train[b] + test[b]:
        needed.add(pair['pos'])
        needed.add(pair['neg'])
print(f'\nExtracting {len(needed)} samples ...')
for idx in tqdm(sorted(needed), desc=f'{LABEL_B} extract'):
    if idx not in cache:
        cache[idx] = extract(data[idx])
try:
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f'Warning: GPU cleanup failed ({e}), continuing...')
vecs_b = defaultdict(dict)
for b in shared:
    for l in range(num_layers):
        for comp in COMPONENTS:
            pos = np.stack([cache[p['pos']][l][comp] for p in train[b]])
            neg = np.stack([cache[p['neg']][l][comp] for p in train[b]])
            vecs_b[l, comp][b] = pos.mean(0) - neg.mean(0)

def eval_auroc(v, pairs, layer, comp):
    vn = np.linalg.norm(v)
    if vn < 1e-10:
        return None
    v = v / vn
    scores, labels = ([], [])
    for pair in pairs:
        scores.append(float(cache[pair['pos']][layer][comp] @ v))
        labels.append(1)
        scores.append(float(cache[pair['neg']][layer][comp] @ v))
        labels.append(0)
    if len(set(labels)) < 2:
        return None
    try:
        return roc_auc_score(labels, scores)
    except:
        return None
results = []
for b in tqdm(shared, desc='transfer'):
    if not test.get(b):
        continue
    for l in range(min(num_layers, num_layers_a)):
        for comp in COMPONENTS:
            key = (l, comp)
            bb = eval_auroc(vecs_b[key][b], test[b], l, comp) if b in vecs_b.get(key, {}) else None
            ab = eval_auroc(vecs_a[key][b], test[b], l, comp) if b in vecs_a.get(key, {}) else None
            results.append({'behavior': b, 'layer': l, 'comp': comp, 'baseline_B': bb, 'transfer_A2B': ab})

def jsd_1d(p, n, bins=50):
    a = np.concatenate([p, n])
    if np.std(a) < 1e-10:
        return 0.0
    lo, hi = (a.min(), a.max())
    if lo == hi:
        return 0.0
    b = np.linspace(lo - 1e-08, hi + 1e-08, bins + 1)
    ph = np.histogram(p, bins=b)[0].astype(np.float64) + 1e-10
    nh = np.histogram(n, bins=b)[0].astype(np.float64) + 1e-10
    ph /= ph.sum()
    nh /= nh.sum()
    m = 0.5 * (ph + nh)
    return float(0.5 * (ph * np.log2(ph / m)).sum() + 0.5 * (nh * np.log2(nh / m)).sum())
jsd_data = {}
for b in shared:
    jsd_data[b] = {}
    for l in range(num_layers):
        jsd_data[b][l] = {}
        for comp in COMPONENTS:
            v = vecs_b.get((l, comp), {}).get(b)
            if v is None:
                jsd_data[b][l][comp] = 0.0
                continue
            vn = np.linalg.norm(v)
            if vn < 1e-10:
                jsd_data[b][l][comp] = 0.0
                continue
            v = v / vn
            pp = np.array([cache[p['pos']][l][comp] @ v for p in train[b]])
            np_ = np.array([cache[p['neg']][l][comp] @ v for p in train[b]])
            jsd_data[b][l][comp] = jsd_1d(pp, np_)
with open(f'jsd_{DATA_MODEL_B}.json', 'w') as f:
    json.dump(jsd_data, f, indent=2)

def sm(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else 0
print(f"\n{'=' * 70}\nREAL TRANSFER: {LABEL_A} -> {LABEL_B}\n{'=' * 70}")
print(f"{'Behavior':<35s} {'Base':>7} {'Trans':>7} {'Gap':>7}")
beh_sum = {}
for b in shared:
    br = [r for r in results if r['behavior'] == b]
    base = sm([r['baseline_B'] for r in br])
    trans = sm([r['transfer_A2B'] for r in br])
    beh_sum[b] = {'base': base, 'trans': trans}
    print(f'{b:<35s} {base:>7.3f} {trans:>7.3f} {trans - base:>+7.3f}')
ob = sm([s['base'] for s in beh_sum.values()])
ot_ = sm([s['trans'] for s in beh_sum.values()])
print(f"\n{'OVERALL':<35s} {ob:>7.3f} {ot_:>7.3f} {ot_ - ob:>+7.3f}")
with open(f'real_transfer_{LABEL_A}_to_{LABEL_B}.txt', 'w') as f:
    f.write(f'Real Transfer: {LABEL_A} -> {LABEL_B}\n')
    for b, s in sorted(beh_sum.items()):
        f.write(f"{b:<35s} base={s['base']:.3f} trans={s['trans']:.3f} gap={s['trans'] - s['base']:+.3f}\n")
    f.write(f'\nOverall: base={ob:.3f} trans={ot_:.3f} gap={ot_ - ob:+.3f}\n')
print(f'\nDone. {len(results)} evals, {len(cache)} activations.')
