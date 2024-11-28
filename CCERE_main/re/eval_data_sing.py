import os
import sys
import torch
import logging
import hydra
from hydra import utils
from deepke.relation_extraction.standard.tools import Serializer
from deepke.relation_extraction.standard.tools import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_relation_data , _lm_serialize
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from deepke.relation_extraction.standard.utils import load_pkl, load_csv
import deepke.relation_extraction.standard.models as models
from itertools import combinations
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _preprocess_data(data, cfg):
    
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)

    if cfg.model_name != 'lm':
        vocab = load_pkl(os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl'), verbose=False)
        cfg.vocab_size = vocab.count
        serializer = Serializer(do_chinese_split=cfg.chinese_split)
        serial = serializer.serialize

        _serialize_sentence(data, serial, cfg)
        _convert_tokens_into_index(data, vocab)
        _add_pos_seq(data, cfg)
        logger.info('start sentence preprocess...')
        formats = '\nsentence: {}\nchinese_split: {}\nreplace_entity_with_type:  {}\nreplace_entity_with_scope: {}\n' \
                'tokens:    {}\ntoken2idx: {}\nlength:    {}\nhead_idx:  {}\ntail_idx:  {}'
        logger.info(
            formats.format(data[0]['sentence'], cfg.chinese_split, cfg.replace_entity_with_type,
                        cfg.replace_entity_with_scope, data[0]['tokens'], data[0]['token2idx'], data[0]['seq_len'],
                        data[0]['head_idx'], data[0]['tail_idx']))
    else:
        _lm_serialize(data,cfg)

    return data, rels


def _get_predict_instance(cfg):
    flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
    flag = flag.strip().lower()
    if flag == 'y' or flag == 'yes':
        sentence = '﻿以2个超级杂交稻两优培九和两优E32为材料,汕优63作对照,比较分析了与光合生态生理有关的株型性状'
        head = '汕优63'
        tail = '光合生态'
        head_type = 'BREED'
        tail_type = 'PHENOTYPE'
    elif flag == 'n' or flag == 'no':
        sentence = input('请输入句子：')
        head = input('请输入句中需要预测关系的头实体：')
        head_type = input('请输入头实体类型：')
        tail = input('请输入句中需要预测关系的尾实体：')
        tail_type = input('请输入尾实体类型：')
    elif flag == 'exit':
        sys.exit(0)
    else:
        print('please input yes or no, or exit!')
        _get_predict_instance()

    instance = dict()
    instance['sentence'] = sentence.strip()
    instance['head'] = head.strip()
    instance['tail'] = tail.strip()
    if head_type.strip() == '' or tail_type.strip() == '':
        cfg.replace_entity_with_type = False
        instance['head_type'] = 'None'
        instance['tail_type'] = 'None'
    else:
        instance['head_type'] = head_type.strip()
        instance['tail_type'] = tail_type.strip()

    return instance


def read_and_output_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def combine_elements(lst, k):
    # 使用 combinations() 生成给定长度的所有组合
    result = list(combinations(lst, k))
    return result


def process_data(data):
    entity = []
    sentence = ''.join(data['sentence'])
    if len(sentence) > 510: 
        return None
    else:
        for ner_item in data['ner']:
            start_index, end_index = ner_item['index'][0], ner_item['index'][-1]
            ner_value = sentence[start_index:end_index+1]
            entity.append((ner_value, ner_item['type']))
    entity_group = list(combinations(entity, 2))
    rows = []
    for group in entity_group:
        head, head_type = group[0]
        tail, tail_type = group[1]
        if head != tail:
            data = {'sentence': sentence, 'head':head, 'head_type':head_type,
                    'tail':tail,'tail_type':tail_type}
            rows.append(data)
    return rows

def process_concat_parallel(data_all):
    count = 0
    df = pd.DataFrame(columns=['sentence', 'head', 'head_type', 'tail', 'tail_type'])

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, data) for data in data_all]

        # 使用 tqdm 打印进度条
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                rows = future.result()
                if rows is not None:
                    df = df.append(rows, ignore_index=True)
                    count += len(rows)
                pbar.update(1)

    return df, rows


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    # cwd = cwd[0:-5]
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2
    print(cfg.pretty())

    # get predict instance
    # instance = _get_predict_instance(cfg)
    # data = [instance]

    df = pd.read_csv(cfg.read_relation)
    df = df.astype(str)
    # df = df.head(1000000)
    # df = df[1000000:4000000]
    df = df[cfg.starts:cfg.ends]
    # df = process_concat(datas)
    print('当前共有：' + str(len(df)) + ' 个关系待检测')
    data = df.to_dict(orient='records')

    # preprocess data
    data, rels = _preprocess_data(data, cfg)
    # print(data)
    # model
    __Model__ = {
        'cnn': models.PCNN,
        'rnn': models.BiLSTM,
        'transformer': models.Transformer,
        'gcn': models.GCN,
        'capsule': models.Capsule,
        'lm': models.LM,
    }

    # 最好在 cpu 上预测
    cfg.use_gpu = True
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')

    model = __Model__[cfg.model_name](cfg)
    logger.info(f'model name: {cfg.model_name}')
    logger.info(f'\n {model}')
    model.load(cfg.fp, device=device)
    model.to(device)
    model.eval()

    x = dict()
    dfs = pd.DataFrame(columns=['head', 'head_type', 'tail', 'tail_type', 'Sure'])  # 替换列名为你实际的列名

    for num in tqdm(range(len(data))):
        x['word'], x['lens'] = torch.tensor([data[num]['token2idx'] + [0] * (512 - len(data[num]['token2idx']))]), torch.tensor([data[num]['seq_len']])
        
        if cfg.model_name != 'lm':
            x['head_pos'], x['tail_pos'] = torch.tensor(data[0]['head_pos'] + [0] * (512 - len(data[0]['token2idx']))), torch.tensor(data[0]['tail_pos'] + [0] * (512 - len(data[0]['token2idx'])))
            if cfg.model_name == 'cnn':
                if cfg.use_pcnn:
                    x['pcnn_mask'] = torch.tensor([data[0]['entities_pos']])
            if cfg.model_name == 'gcn':
                # 没找到合适的做 parsing tree 的工具，暂时随机初始化
                adj = torch.empty(1,512,512).random_(2)
                x['adj'] = adj


        for key in x.keys():
            x[key] = x[key].to(device)

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]

            prob = y_pred.max().item()
            print(len(rels))
            print(y_pred.argmax())
            prob_rel = list(rels.keys())[y_pred.argmax().item()]
            if prob > 0.1:
                da = {'head': data[num]['head'], 'head_type': data[num]['head_type'],
                    'tail': data[num]['tail'], 'tail_type': data[num]['tail_type'], 'relation': prob_rel,'sure': round(prob, 3)}
                dfs = dfs._append(da, ignore_index=True)
                print(str(data[num]['head']) + ' 和 ' +  str(data[num]['tail']) + '的关系为' + str(prob_rel) + ' 置信度为' + str(round(prob,3)))
                logger.info(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{prob_rel}\"，置信度为{prob:.2f}。")
        # except:
            # dad = 1
    dfs = dfs.drop_duplicates()
    print('总共有 ' + str(len(dfs)) + ' 个三元组')
    # dfs.to_csv(cfg.sa_path, index=False)


if __name__ == '__main__':
    main()
