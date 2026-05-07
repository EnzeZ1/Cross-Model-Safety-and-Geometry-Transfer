# Disentangling Safety and Harm: Do Behavioral Signals Share Geometry Across Models?

This repository contains the code and result files for a CS639 final project studying whether safety-relevant behavioral representations are preserved across post-trained large reasoning models.

The project compares cross-model geometry at four levels:

1. **ActSVD subspaces**: safety-vs.-harm subspace overlap induced by model projection weights.
2. **DoM steering vectors**: Difference-of-Means behavior directions in activation space.
3. **CKA/RSA geometry**: layer-wise similarity of behavior-level DoM directions across models.
4. **Functional transfer and JSD dynamics**: whether a direction learned in one model transfers to another, and whether behavior separation emerges at similar layers.

## Model pairs and datasets

| Dataset | Model pair | Purpose |
|---|---|---|
| `ishitakakkar-10/HarmThoughts` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` vs. `Qwen/QwQ-32B` | 32B shared-family reasoning-safety analysis |
| `AISafety-Student/reasoning-safety-behaviours` | `Qwen/Qwen3-8B` vs. `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B shared-family reasoning-safety analysis |

The data loaders create prompt-level 80/20 train/test splits so that sentences from the same reasoning trace do not appear in both splits.

## Repository structure

```text
.
├── harmthoughts_data.py          # Prepare HARMTHOUGHTS data.json
├── safety_student_data.py        # Prepare AISafety-Student data.json
├── work_actsvd.py                # ActSVD extraction for HARMTHOUGHTS-style labels
├── work_actsvd2.py               # ActSVD extraction for AISafety-Student-style labels
├── work_dom.py                   # DoM steering-vector extraction
├── uni.py                        # Cross-model DoM geometry: CKA + RSA
├── cross_actsvd.py               # Cross-model ActSVD subspace overlap
├── cross_jsd.py                  # Cross-model JSD layer-dynamics comparison
├── real_transfer.py              # Cross-model functional transfer AUROC
├── run.sh                        # Full 32B HARMTHOUGHTS pipeline
├── run2.sh                       # Full 8B AISafety-Student pipeline
```

> Note: `safety_ai_studnets/` follows the folder name in the current archive. You may want to rename it to `safety_ai_students/` before publishing.

## Installation

This code assumes a CUDA environment with enough GPU memory to load the target models. The original experiments used multi-GPU setups.

```bash
conda create -n safety-geometry python=3.10 -y
conda activate safety-geometry

pip install torch transformers datasets safetensors tqdm numpy scipy scikit-learn pandas matplotlib
```

If you use a shared cluster or offline Hugging Face cache, edit the cache paths in `run.sh` and `run2.sh`:

```bash
export HF_HOME="/path/to/hf_cache"
export TRANSFORMERS_CACHE="/path/to/hf_cache"
```

## Quick start

### 1. Prepare data

For HARMTHOUGHTS:

```bash
python harmthoughts_data.py r1-32b
python harmthoughts_data.py QwQ
```

For AISafety-Student:

```bash
python safety_student_data.py qwen3-8b
python safety_student_data.py r1-8b-0528
```

Each command writes a `data.json` file containing the filtered examples, prompt-level split, and contrastive train/test pairs.

### 2. Extract ActSVD subspaces

Example for HARMTHOUGHTS:

```bash
python work_actsvd.py "0,1,2,3" deepseek-ai/DeepSeek-R1-Distill-Qwen-32B r1-32b 100 100
python work_actsvd.py "0,1,2,3" Qwen/QwQ-32B QwQ 100 100
```

Example for AISafety-Student:

```bash
python work_actsvd2.py "0,1,2,3" Qwen/Qwen3-8B qwen3-8b 100 100
python work_actsvd2.py "0,1,2,3" deepseek-ai/DeepSeek-R1-0528-Qwen3-8B r1-8b-0528 100 100
```

Outputs:

```text
actsvd_<model-tag>.safetensors
actsvd_results_<model-tag>.txt
```

### 3. Extract DoM steering vectors

```bash
python work_dom.py "0,1,2,3" Qwen/Qwen3-8B qwen3-8b
python work_dom.py "0,1,2,3" deepseek-ai/DeepSeek-R1-0528-Qwen3-8B r1-8b-0528
```

Output:

```text
dom_<model-tag>.safetensors
```

### 4. Compare cross-model geometry

DoM CKA/RSA:

```bash
python uni.py dom_qwen3-8b.safetensors dom_r1-8b-0528.safetensors qwen3-8b r1-8b-0528
```

ActSVD overlap:

```bash
python cross_actsvd.py actsvd_qwen3-8b.safetensors actsvd_r1-8b-0528.safetensors qwen3-8b r1-8b-0528
```

JSD layer dynamics:

```bash
python cross_jsd.py jsd_qwen3-8b.json jsd_r1-8b-0528.json qwen3-8b r1-8b-0528
```

### 5. Run transfer evaluation

```bash
python real_transfer.py "0,1,2,3" deepseek-ai/DeepSeek-R1-0528-Qwen3-8B r1-8b-0528 dom_qwen3-8b.safetensors qwen3-8b r1-8b-0528
python real_transfer.py "0,1,2,3" Qwen/Qwen3-8B qwen3-8b dom_r1-8b-0528.safetensors r1-8b-0528 qwen3-8b
```

Outputs include transfer AUROC summaries and `jsd_<model-tag>.json` files.

## Full pipelines

For the 32B HARMTHOUGHTS experiment:

```bash
bash run.sh
```

For the 8B AISafety-Student experiment:

```bash
bash run2.sh
```

Before running, check the GPU IDs and Hugging Face cache paths inside each script.

## Main outputs

The repository includes text summaries from the reported experiments:

```text
harmthoughts/
├── geometry_results_r1-32b_vs_QwQ.txt
├── cross_actsvd_results_r1-32b_vs_QwQ.txt
├── cross_jsd_results_r1-32b_vs_QwQ.txt
├── real_transfer_r1-32b_to_QwQ.txt
└── real_transfer_QwQ_to_r1-32b.txt

safety_ai_studnets/
├── geometry_results_qwen3-8b_vs_r1-8b-0528.txt
├── cross_actsvd_results_qwen3-8b_vs_r1-8b-0528.txt
├── cross_jsd_results_qwen3-8b_vs_r1-8b-0528.txt
├── real_transfer_qwen3-8b_to_r1-8b-0528.txt
└── real_transfer_r1-8b-0528_to_qwen3-8b.txt
```

## Summary of findings

Across both model pairs, DoM directions preserve substantial activation-level cross-model structure: CKA/RSA remain high across many layers, transfer AUROC drops are relatively small, and JSD layer dynamics are correlated across paired models.

In contrast, ActSVD subspaces are less stable across post-training: cross-model safety, harm, and cross-type overlaps become much closer to one another than the corresponding within-model safety/harm overlaps. This suggests that activation-level behavioral geometry transfers more reliably than absolute ActSVD subspaces.

A fine-grained behavior analysis shows that harmful-compliance intent is especially asymmetric across the 8B model pair, suggesting that decision-style safety behaviors may be more strongly reshaped by post-training than recognition-style behaviors such as legal, ethical, or safety-concern statements.

## Reproducibility notes

- The scripts download datasets through Hugging Face `datasets`.
- Model loading uses `transformers.AutoModelForCausalLM` with `device_map="auto"`.
- Large models require substantial GPU memory; adjust `GPU_LIST` and cache paths for your environment.
- Raw datasets, model weights, and generated `.safetensors` activation files are not included in this repository.
- Random splitting uses seed `42` in the data preparation scripts.

## Authors

- Enze Zhang
- Youran Wang
- Sungjin Cheong
- Yuliang Wu

Department of Computer Sciences, University of Wisconsin--Madison.
