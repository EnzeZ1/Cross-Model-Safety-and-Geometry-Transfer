import os, sys
gpu_list = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
MODEL_NAME = sys.argv[2]
DATA_MODEL = sys.argv[3]

import json, warnings
import torch
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings('ignore')
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
COMPONENTS = ['attn', 'mlp', 'residual']
MODEL_COL = 'model_name'
SYSTEM_OT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

def build_input(block):
    if DATA_MODEL == 'ot-7b':
        return f"<|im_start|>system\n{SYSTEM_OT}<|im_end|>\n<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n{block['sentence']}\n<|end_of_thought|>\n\n<|im_end|>\n"
    elif DATA_MODEL in ('r1-8b', 'r1-32b', 'r1-llama-8b', 'r1-8b-0528', 'r1-qwen-32b'):
        return f"<｜begin▁of▁sentence｜><｜User｜>{block['query']}<｜Assistant｜><think>\n{block['sentence']}\n</think>\n\n<｜end▁of▁sentence｜>"
    elif DATA_MODEL in ('QwQ', 'qwen3-8b'):
        return f"<|im_start|>user\n{block['query']}<|im_end|>\n<|im_start|>assistant\n<think>\n{block['sentence']}\n</think>\n\n<|im_end|>\n"
    else:
        raise ValueError(f"Unknown DATA_MODEL '{DATA_MODEL}'")
        
print(f'GPUs [{gpu_list}]: loading {MODEL_NAME} ({DATA_MODEL}) ...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
input_device = next(model.parameters()).device
print(f'  Input device: {input_device}')
acts = {}

def make_hook(name):
    def hook(mod, inp, out):
        acts[name] = out[0] if isinstance(out, (tuple, list)) else out
    return hook
    
for i, layer in enumerate(model.model.layers):
    layer.self_attn.o_proj.register_forward_hook(make_hook(f'L{i}_attn'))
    layer.mlp.register_forward_hook(make_hook(f'L{i}_mlp'))
    layer.register_forward_hook(make_hook(f'L{i}_residual'))
    
store = json.load(open('data.json'))
data = store['data']
valid_ids = set((i for i, e in enumerate(data) if e[MODEL_COL] == DATA_MODEL))
print(f"  {len(valid_ids)}/{len(data)} entries for '{DATA_MODEL}'")
train = {}

for b, pairs in store['train'].items():
    filtered = [p for p in pairs if p['pos'] in valid_ids and p['neg'] in valid_ids]
    if filtered:
        train[b] = filtered
print(f'  {len(train)} behaviors')

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
            h[l][c] = v_gpu.detach().cpu().clone()
    acts.clear()
    return h
    
all_behaviors = sorted(train.keys())
steering = {}
cache = {}
for behavior in tqdm(all_behaviors, desc=f'DoM ({DATA_MODEL})'):
    for pair in tqdm(train[behavior], desc=behavior, leave=False):
        for idx in [pair['pos'], pair['neg']]:
            if idx not in cache:
                cache[idx] = extract(data[idx])
                
    steering[behavior] = {}
    for l in range(num_layers):
        steering[behavior][l] = {}
        for comp in COMPONENTS:
            pos = torch.stack([cache[p['pos']][l][comp] for p in train[behavior]]).float()
            neg = torch.stack([cache[p['neg']][l][comp] for p in train[behavior]]).float()
            steering[behavior][l][comp] = torch.nan_to_num(pos).mean(0) - torch.nan_to_num(neg).mean(0)
    torch.cuda.empty_cache()
    
flat = {}
for b, ld in steering.items():
    for l, cd in ld.items():
        for c, t in cd.items():
            flat[f'dom|{b}|{l}|{c}'] = t.cpu()
            
out = f'dom_{DATA_MODEL}.safetensors'
save_file(flat, out)
print(f'Saved {out} -- done')
