CUDA_VISIBLE_DEVICES=0 python main.py \
    --cuda  \
    --seed 2024 \
    --exp-name moe+logits+cl_arxiv-lora_5e-4 \
    --train-path /feature/arxiv_new/lora/train.jsonl \
    --val-path /feature//arxiv_new/lora/val.jsonl \
    --test-path /feature/arxiv_new/lora/test_ood.jsonl \
    --batch-size 64 \
    --lr 5e-4 \
    --train
    # --test
    # --inference \
    # --pred-name arxiv_lora 