import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm


def access_api(text, api_url, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_features(type, input_file, output_file):
    """
    get [losses, begin_idx_list, ll_tokens_list, label_int, label] based on raw lines
    """

    en_model_names = ['llama2', 'gemma', 'mistral','Qwen2.5-7B']
    llama_api = 'http://127.0.0.1:6009/inference'
    gemma_api = 'http://127.0.0.1:6010/inference'
    mistral_api = 'http://127.0.0.1:6011/inference'
    qwen2_api = 'http://127.0.0.1:6012/inference'

    en_model_apis = [llama_api, gemma_api, mistral_api, qwen2_api]

    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]


    print('input file:{}, length:{}'.format(input_file, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label_binary = data['label_binary']
            label_family = data['label_family']
            label_model = data['label_model']

            losses = []
            begin_idx_list = []
            ll_tokens_list = []
            model_apis = en_model_apis

            error_flag = False
            for api in model_apis:
                try:
                    loss, begin_word_idx, ll_tokens = access_api(line, api)
                except TypeError:
                    print("return NoneType, probably gpu OOM, discard this sample")
                    error_flag = True
                    break
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)
            if error_flag:
                continue

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_binary': label_binary,
                'label_family': label_family,
                'label_model': label_model,
                'text': line
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def process_features(input_file, output_file, do_normalize=False):
    """
    Process features from raw features.

        raw_features: {losses, begin_idx_list, ll_tokens_list, label_int, label, text}
        ==>
        processed_features: {values, label_int, label}

        values = {losses, lt_zero_percents, std_deviations, pearson_list, spearmann_list}
    """

    with open(input_file, 'r') as f:
        raw_features = [json.loads(line) for line in f.readlines()]
    print('input file:{}, length:{}'.format(input_file, len(raw_features)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for raw_feature in tqdm(raw_features):
            losses = raw_feature['losses']
            begin_idx_list = raw_feature['begin_idx_list']
            ll_tokens_list = raw_feature['ll_tokens_list']
            label = raw_feature['label']
            text = raw_feature['text']

            try:
                begin_idx_list = np.array(begin_idx_list)
                max_begin_idx = np.max(begin_idx_list)
                for idx, ll_tokens in enumerate(ll_tokens_list):
                    ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
                min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
                for idx, ll_tokens in enumerate(ll_tokens_list):
                    ll_tokens_list[idx] = ll_tokens[:min_len]


                if do_normalize:

                    ll_tokens_list_normalized = normalize(ll_tokens_list, norm='l1')

                    lls = ll_tokens_list_normalized.tolist()
                else:

                    lls = ll_tokens_list


            except Exception as e:
                """
                [0, 0, 0, 0], too short, discard this sample
                """
                print(e)
                print("fail to process this sample, discard it, text:{}".format(text))
                print()
                continue

            try:
                lt_zero_percents = []
                std_deviations = []
                deviations = []
                pearson_list = []
                spearmann_list = []
                
                for i in range((len(lls))):
                    for j in range(i + 1, len(lls)):
                        # lls[i], ll[j]
                        deviation_ij = [li - lj for li, lj in zip(lls[i], lls[j])]
                        # `lt` means `less than`
                        deviation_lt_zero_ij = [d < 0 for d in deviation_ij]
                        lt_zero_pct_ij = sum(deviation_lt_zero_ij) / len(
                            deviation_lt_zero_ij)
                        std_ij = np.std(deviation_ij)
                        lt_zero_percents.append(lt_zero_pct_ij)
                        std_deviations.append(std_ij)
                        deviations.append(deviation_ij)
                        pearson = scipy.stats.pearsonr(lls[i], lls[j])[0]
                        spearmann = scipy.stats.spearmanr(lls[i], lls[j])[0]

                        pearson_list.append(pearson)
                        spearmann_list.append(spearmann)

                values = {'losses': losses,
                        'lt_zero_percents': lt_zero_percents,
                        'std_deviations': std_deviations,
                        'pearson_list': pearson_list,
                        'spearmann_list': spearmann_list}

                processed_feature = {'values': values,
                                    # 'label_int': label_int,
                                    'label': label,
                                    'text': text}

                f.write(json.dumps(processed_feature, ensure_ascii=False) + '\n')
            except:
                print("fail may due to speraman or pearson")
                print(text)
                print(lls[i], lls[j])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--get_en_features", action="store_true", help="generate en logits and losses")
    parser.add_argument("--get_cn_features", action="store_true", help="generate cn logits and losses")
    parser.add_argument("--get_en_features_multithreading", action="store_true", help="multithreading generate en logits and losses")
    parser.add_argument("--get_cn_features_multithreading", action="store_true", help="multithreading generate cn logits and losses")
    parser.add_argument("--process_features", action="store_true", help="process the raw features")

    parser.add_argument("--do_normalize", action="store_true", help="normalize the features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.get_en_features:
        """
        retrieve english features in a single file 
        python gen_features.py --get_en_features --input_file raw_data/en_alpaca_lines.jsonl --output_file ../features/raw_features/en_alpaca_features.jsonl
        python gen_features.py --get_en_features --input_file raw_data/en_dolly_lines.jsonl --output_file ../features/raw_features/en_dolly_features.jsonl
        
        python gen_features.py --get_en_features --input_file gpt3_ablation_data/gpt3_ablation_train_lines.jsonl --output_file ../features/gpt3_ablation_features/gpt3_ablation_train_features.jsonl
        python gen_features.py --get_en_features --input_file gpt3_ablation_data/gpt3_ablation_test_lines.jsonl --output_file ../features/gpt3_ablation_features/gpt3_ablation_test_features.jsonl
        """
        get_features(type='en', input_file=args.input_file, output_file=args.output_file)

    elif args.get_cn_features:
        """
        retrieve chinese features in a single file 
        python gen_features.py --get_cn_features --input_file aligned_data/cn_wenzhong_aligned_lines.jsonl --output_file ../features/aligned_features/cn_wenzhong_aligned_features.jsonl

        python gen_features.py --get_cn_features --input_file aligned_data/cn_moss_aligned_lines.jsonl --output_file ../features/aligned_features/cn_moss_aligned_features.jsonl
        """
        get_features(type='cn', input_file=args.input_file, output_file=args.output_file)

    elif args.get_en_features_multithreading:
        """
        retrieve english features in multiple files, use multithreading for faster speed
        python gen_features.py --get_en_features_multithreading
        """

        en_input_files = [
        'data/QA/lora/train.jsonl',
        'data/QA/lora/val.jsonl',
        'data/QA/lora/test.jsonl',
        ]
        en_output_files = [
        'features/QA/lora/train.jsonl',
        'features/QA/lora/val.jsonl',
        'features/QA/lora/test.jsonl',
        ]

        threads = []
        for i in range(len(en_input_files)):
            t = threading.Thread(target=get_features, args=('en', en_input_files[i], en_output_files[i]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    elif args.process_features:
        
        print(args.do_normalize)
        process_features(args.input_file, args.output_file, args.do_normalize)

    else:
        print("please select an action")
