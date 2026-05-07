import os, sys
gpu_list = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
MODEL_NAME = sys.argv[2]
DATA_MODEL = sys.argv[3]

SAFETY_RANK = int(sys.argv[4]) if len(sys.argv) > 4 else 100
UTILITY_RANK = int(sys.argv[5]) if len(sys.argv) > 5 else 100

MAX_SAMPLES = 128
import json, random, warnings, numpy as np, torch
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import Counter

warnings.filterwarnings('ignore')
random.seed(42)

SAFE_LABELS = {'I_REPHRASE_PROMPT', 'I_FLAG_USER_TESTING', 'I_SPECULATE_USER_MOTIVE', 'II_FLAG_PROMPT_AS_HARMFUL', 'II_STATE_ETHICAL_MORAL_CONCERN', 'II_STATE_LEGAL_CONCERN', 'II_STATE_SAFETY_CONCERN', 'III_EXPRESS_UNCERTAINTY_CONFUSION', 'III_PLAN_IMMEDIATE_REASONING_STEP', 'III_STATE_FACT_OR_KNOWLEDGE', 'III_SUMMARIZE_INTERNAL_REASONING', 'III_SELF_CORRECT_INFO_OR_DETAIL', 'IV_CONSIDER_BENIGN_REINTERPRETATION', 'IV_INTEND_REFUSAL_OR_SAFE_ACTION', 'IV_SUGGEST_SAFE_CONSTRUCTIVE_ALTERNATIVE', 'VI_NEUTRAL_FILLER_TRANSITION', 'VI_OTHER'}
UNSAFE_LABELS = {'V_DETAIL_HARMFUL_METHOD_OR_INFO', 'V_INTEND_HARMFUL_COMPLIANCE', 'V_NOTE_RISK_WHILE_DETAILING_HARM'}
SYSTEM_OT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

def build_input(block):
    if DATA_MODEL == 'ot-7b':
        return f"<|im_start|>system\n{SYSTEM_OT}<|im_end|>\n<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n{block['sentence']}\n<|end_of_thought|>\n\n<|im_end|>\n"
    elif DATA_MODEL in ('r1-8b', 'r1-32b', 'r1-8b-0528'):
        return f"<｜begin▁of▁sentence｜><｜User｜>{block['query']}<｜Assistant｜><think>\n{block['sentence']}\n</think>\n\n<｜end▁of▁sentence｜>"
    elif DATA_MODEL in ('QwQ', 'qwen3-8b'):
        return f"<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<think>\n{block['sentence']}\n</think>\n\n<|im_end|>\n"
    else:
        raise ValueError(f"Unknown '{DATA_MODEL}'")
        
print(f'GPUs [{gpu_list}]: loading {MODEL_NAME} ...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_layers = AutoConfig.from_pretrained(MODEL_NAME).num_hidden_layers
input_device = next(model.parameters()).device
store = json.load(open('data.json'))
data = store['data']
meta = store.get('meta', {})
LC = meta.get('label_col', meta.get('label_field', 'behavior'))

safe_ids = [i for i, d in enumerate(data) if d.get('model_name') == DATA_MODEL and d.get(LC, d.get('behavior', '')) in SAFE_LABELS]
util_ids = [i for i, d in enumerate(data) if d.get('model_name') == DATA_MODEL and d.get(LC, d.get('behavior', '')) in UNSAFE_LABELS]

print(f'  safe={len(safe_ids)}, unsafe={len(util_ids)}')
random.shuffle(safe_ids)
random.shuffle(util_ids)
safe_ids = safe_ids[:MAX_SAMPLES]
util_ids = util_ids[:MAX_SAMPLES]

def collect(indices, desc):
    la = {}
    hooks = []
    buf = {}

    def mh(n):
        def h(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, torch.Tensor):
                buf[n] = x.detach().float().mean(1).squeeze(0).cpu()
        return h
        
    tgts = {}
    for li, layer in enumerate(model.model.layers):
        for pn in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            p = getattr(layer.self_attn, pn, None)
            if p:
                tgts[f'layers.{li}.self_attn.{pn}'] = p
        for pn in ['gate_proj', 'up_proj', 'down_proj']:
            p = getattr(layer.mlp, pn, None)
            if p:
                tgts[f'layers.{li}.mlp.{pn}'] = p
                
    for n, mod in tgts.items():
        hooks.append(mod.register_forward_hook(mh(n)))
    for idx in tqdm(indices, desc=desc):
        enc = tokenizer(build_input(data[idx]), return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)
        buf.clear()
        with torch.no_grad():
            model(enc['input_ids'].to(input_device))
        for n, a in buf.items():
            la.setdefault(n, []).append(a)
    for h in hooks:
        h.remove()
    return {n: torch.stack(v, dim=1) for n, v in la.items()}
    
X_s = collect(safe_ids, 'safe')
X_u = collect(util_ids, 'unsafe')

names = sorted(set(X_s) & set(X_u))
flat = {}
info = {}
for name in tqdm(names, desc='ActSVD'):
    parts = name.split('.')
    li = int(parts[1])
    layer = model.model.layers[li]
    proj = getattr(layer.self_attn, parts[3]) if 'self_attn' in name else getattr(layer.mlp, parts[3])
    w_dev = proj.weight.device
    W = proj.weight.data.float()
    
    Us, Ss, _ = torch.linalg.svd(W @ X_s[name].to(w_dev).float(), full_matrices=False)
    Uu, Su, _ = torch.linalg.svd(W @ X_u[name].to(w_dev).float(), full_matrices=False)
    
    rs = min(SAFETY_RANK, Us.shape[1])
    ru = min(UTILITY_RANK, Uu.shape[1])
    
    Us, Uu = (Us[:, :rs], Uu[:, :ru])
    phi = (Us.T @ Uu).pow(2).sum().item() / min(rs, ru)
    lt = 'attn' if 'self_attn' in name else 'mlp'
    info[name] = {'phi': phi, 'type': lt}
    
    flat[f'U_s|{name}'] = Us.cpu().half().contiguous()
    flat[f'U_u|{name}'] = Uu.cpu().half().contiguous()
    del Us, Uu
    torch.cuda.empty_cache()
    
save_file(flat, f'actsvd_{DATA_MODEL}.safetensors')
attn = [v['phi'] for v in info.values() if v['type'] == 'attn']
mlp = [v['phi'] for v in info.values() if v['type'] == 'mlp']

print(f'\nAttn: {np.mean(attn):.4f}±{np.std(attn):.4f}  MLP: {np.mean(mlp):.4f}±{np.std(mlp):.4f}')
with open(f'actsvd_results_{DATA_MODEL}.txt', 'w') as f:
    for n in sorted(info):
        f.write(f"{n:<45s} {info[n]['phi']:.4f} {info[n]['type']}\n")
print('Done.')
