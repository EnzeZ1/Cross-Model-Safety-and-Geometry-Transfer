#!/bin/bash
set -e

export HF_HOME="/nobackup/enzez/hf_cache"
export TRANSFORMERS_CACHE="/nobackup/enzez/hf_cache"

GPUS="0,1,2,3,4,5,6,7"
QWEN_HF="Qwen/Qwen3-8B"
R1_HF="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

echo "╔════════════════════════════════════════════╗"
echo "║  AISafety-Student: qwen3-8b + r1-8b-0528   ║"
echo "║  GPUs: [$GPUS]                             ║"
echo "╚════════════════════════════════════════════╝"

echo ""; echo "════ qwen3-8b ════"
python -u safety_student_data.py qwen3-8b
python -u work_actsvd2.py "$GPUS" "$QWEN_HF" qwen3-8b 100 100 2>&1 | tee log_qwen3-8b_actsvd.txt
python -u work_dom.py "$GPUS" "$QWEN_HF" qwen3-8b 2>&1 | tee log_qwen3-8b_dom.txt

echo ""; echo "════ r1-8b-0528 ════"
python -u safety_student_data.py r1-8b-0528
python -u work_actsvd2.py "$GPUS" "$R1_HF" r1-8b-0528 100 100 2>&1 | tee log_r1-8b-0528_actsvd.txt
python -u work_dom.py "$GPUS" "$R1_HF" r1-8b-0528 2>&1 | tee log_r1-8b-0528_dom.txt

echo ""; echo "════ Cross-model ════"
python -u uni.py dom_qwen3-8b.safetensors dom_r1-8b-0528.safetensors qwen3-8b r1-8b-0528 2>&1 | tee log_uni.txt
python -u cross_actsvd.py actsvd_qwen3-8b.safetensors actsvd_r1-8b-0528.safetensors qwen3-8b r1-8b-0528 2>&1 | tee log_cross_actsvd.txt
python -u cross_jsd.py jsd_qwen3-8b.json jsd_r1-8b-0528.json qwen3-8b r1-8b-0528 2>&1 | tee log_cross_jsd.txt

echo ""; echo "════ Transfer: qwen3-8b → r1-8b-0528 ════"
python -u safety_student_data.py r1-8b-0528
python -u real_transfer.py "$GPUS" "$R1_HF" r1-8b-0528 dom_qwen3-8b.safetensors qwen3-8b r1-8b-0528 2>&1 | tee log_transfer_qwen3-8b_to_r1-8b-0528.txt

echo ""; echo "════ Transfer: r1-8b-0528 → qwen3-8b ════"
python -u safety_student_data.py qwen3-8b
python -u real_transfer.py "$GPUS" "$QWEN_HF" qwen3-8b dom_r1-8b-0528.safetensors r1-8b-0528 qwen3-8b 2>&1 | tee log_transfer_r1-8b-0528_to_qwen3-8b.txt

echo ""
echo "╔══════════════════════════╗"
echo "║  All done!               ║"
echo "╚══════════════════════════╝"
ls -lh actsvd_*.safetensors dom_*.safetensors jsd_*.json 2>/dev/null
ls -lh *results*.txt real_transfer_*.txt 2>/dev/null
