import os
import sys
import torch
import random
import argparse
import numpy as np

from dataloader import get_dataloader
from model import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--n-family', type=int, default=5, help='number of families including human and generated')
    parser.add_argument('--exp-name', type=str, default='family_moe_logits', help='model experiment name')
    parser.add_argument('--pred-name', type=str, default='arxiv', help='prediction file name, just for inference')
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--val-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--pretrain-model', default='/data/shiyuhui/pretrained/roberta-base')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--model-save-dir', default='./params')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference',action='store_true') 
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--is-binary',  action='store_true', help='True indicate binary classification,False indicate multi-classification')
    parser.add_argument('--is-cl',  action='store_true', help='if use contrastive learning')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(args.seed)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    model_save_path = os.path.join(args.model_save_dir, f'params_{args.exp_name}.pt')

    label2id = {
        'human': 0,
        'generated': 1,
        'llama': 1,
        'mistral': 2,
        'gemma': 3,
        'Qwen2.5': 4,
    }


    train_dataloader = get_dataloader(args.train_path, args.pretrain_model, args.batch_size, args.max_len, label2id, shuffle=True) if not args.test else None
    val_dataloader = get_dataloader(args.val_path, args.pretrain_model, args.batch_size, args.max_len, label2id, shuffle=False) if not args.test else None
    test_dataloader = get_dataloader(args.test_path, args.pretrain_model, args.batch_size, args.max_len, label2id, shuffle=False)
    trainer = Trainer(device, args.pretrain_model, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, model_save_path, args.n_family, args.is_cl, args.is_binary)

    if not args.test:
        trainer.train()
    else:
        trainer.model.load_state_dict(torch.load(model_save_path))
        results = trainer.test(test_dataloader)
        print(results)

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
