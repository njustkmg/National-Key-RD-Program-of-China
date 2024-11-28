import os
import json
import torch
import random
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import AdamW, BertConfig, BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import hydra
from hydra import utils
from deepke.name_entity_re.standard import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainNer(BertForTokenClassification):

    def forward(
        self, 
        input_ids, 
        attention_mask=None,
        token_type_ids=None,  
        labels=None,
        valid_ids=None,
        attention_mask_label=None,
        device=None
    ):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    # Use gpu or not
    USE_MULTI_GPU = cfg.use_multi_gpu
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        MULTI_GPU = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        MULTI_GPU = False
    if not MULTI_GPU:
        n_gpu = 1
        if cfg.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda', cfg.gpu_ids)
        else:
            device = torch.device('cpu')
        
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(cfg.gradient_accumulation_steps))
    
    # Preprocess the input dataset
    processor = NerProcessor()
    label_list = processor.get_labels(cfg)
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(r'E:\PyCharmProject\FoundationModel-Agriculture\ner\\'+cfg.output_dirs, do_lower_case=cfg.do_lower_case)

    config = BertConfig.from_pretrained(r'E:\PyCharmProject\FoundationModel-Agriculture\ner\\'+cfg.output_dirs, num_labels=num_labels, finetuning_task=cfg.task_name)

    model = TrainNer.from_pretrained(r'E:\PyCharmProject\FoundationModel-Agriculture\ner\\'+cfg.output_dirs, from_tf=False,config=config)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    if cfg.do_eval:
        if cfg.eval_ons == "dev":
            eval_examples = processor.get_dev_examples(os.path.join(utils.get_original_cwd(), cfg.data_dir))
        elif cfg.eval_ons == "test":
            eval_examples = processor.get_test_examples(os.path.join(utils.get_original_cwd(), cfg.data_dir))
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = convert_examples_to_features(eval_examples, label_list, cfg.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=cfg.eval_batch_size * n_gpu)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,valid_ids=valid_ids,attention_mask_label=l_mask,device=device)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        
                        if logits[i][j] != 0:
                            temp_2.append(label_map[logits[i][j]])
                        else:
                            temp_2.append(None)

        report = classification_report(y_true, y_pred)
        logger.info("\n%s", report)
        # output_eval_file = os.path.join(os.path.join(utils.get_original_cwd(), (cfg.output_dir+'_eval')), "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
            # logger.info("***** Eval results *****")
            # logger.info("\n%s", report)
            # writer.write(report)


if __name__ == '__main__':
    main()
