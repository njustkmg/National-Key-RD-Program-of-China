from deepke.name_entity_re.standard import *
import hydra
from hydra import utils
import pickle
import os

import warnings
warnings.filterwarnings("ignore")


def constract_data(class_all, data_all):
    index_list = []
    words_dict = {}
    for i in range(len(class_all)):
        if class_all[i].startswith('B'):
            index_list.append(i)
    for i in range(len(index_list)):
        start = index_list[i]
        end = index_list[i + 1] if i + 1 < len(index_list) else None
        words_list = data_all[start:end]
        classes = str(words_list[0][1]).split('-')[1]
        words = ''
        for word in words_list:
            words += word[0] 
        # print(classes)
        # print(words)
        # print('-'*40)
        words_dict[words] = classes
    # print('construct success')
    return words_dict


@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    if cfg.model_name == 'lstmcrf':
        with open(os.path.join(utils.get_original_cwd(), cfg.data_dir, cfg.model_vocab_path), 'rb') as inp:
            word2id = pickle.load(inp)
            label2id = pickle.load(inp)
            id2label = pickle.load(inp)

        model = InferNer(utils.get_original_cwd() + '/' + "checkpoints/", cfg, len(word2id), len(label2id), word2id, id2label)
    elif cfg.model_name == 'bert':
        model = InferNer(os.path.join(utils.get_original_cwd(), cfg.output_dirs), cfg)
    else:
        raise NotImplementedError(f"model type {cfg.model_name} not supported")
    
    text = str(cfg.text).replace(' ', '').replace('。', '.').replace('/', '').replace('、', ',').replace(':', '')

    print("NER句子:")
    print(text)
    print('NER结果:')

    result = model.predict(text)
    print(result)
    classes_list = [item[1] for item in result]
    # result = {item[0]: item[1] for item in result}
    # print(result)
    # class_list = list(result.values())
    # result = [[key, value] for key, value in result.items()]
    if len(result) != 0:
        words_dict = constract_data(classes_list, result)
    print(words_dict)
    # for k,v in result.items():
    #     if v:
    #         print(v,end=': ')
    #         if k=='PER':
    #             print('Person')
    #         elif k=='LOC':
    #             print('Location')
    #         elif k=='ORG':
    #             print('Organization')
   
    
if __name__ == "__main__":
    main()
