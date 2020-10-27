import os
import time
from tqdm import tqdm, trange
import argparse
import numpy as np

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel

from read_text_from_data import readDUCorTACText
from utils import Logger, create_dir, save_features


def predict_sim(model, tokenizer, text_docs, max_seq_length, batch_size, cuda_dev, name, seg_pos=None):
    for di, text_doc in tqdm(enumerate(text_docs), desc='[Checking...]'):
        num_sent = len(text_doc)
        print(name[di])
        if seg_pos is not None:
            assert len(seg_pos[di]) == num_sent, '{} != {}'.format(len(seg_pos[di]), num_sent)
    print('checked same length of seg_pos and num_sent')

    model.train(False)
    pred_sim_list = []
    for di, text_doc in tqdm(enumerate(text_docs), desc='[Document]'):
        num_sent = len(text_doc)
        if seg_pos is not None:
            assert len(seg_pos[di]) == num_sent, '{} != {}'.format(len(seg_pos[di]), num_sent)

        input_ids_list, segment_ids_list, input_mask_list = [], [], []

        sent_same_list = []
        for i in range(num_sent):
            for j in range(i + 1, num_sent):
                sent1, sent2 = text_doc[i], text_doc[j]

                if sent1 == sent2:
                    sent_same_list.append(True)
                    continue

                sent_same_list.append(False)

                # longer sentence first
                if len(sent2) > len(sent1):
                    temp = sent1
                    sent1 = sent2
                    sent2 = temp
                input_ids = tokenizer.encode(
                    sent1, sent2, add_special_tokens=True, max_length=max_seq_length)

                if len(input_ids) > max_seq_length:
                    input_ids = input_ids[:max_seq_length - 1]
                    input_ids[-1] = tokenizer.sep_token_id

                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                segment_b = False
                for k, ids in enumerate(input_ids):
                    if segment_b:
                        segment_ids[k] = 1
                    if ids == tokenizer.sep_token_id:
                        segment_b = True

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                segment_ids += padding
                input_mask += padding

                input_ids_list.append(np.array(input_ids, dtype=int))
                segment_ids_list.append(np.array(segment_ids, dtype=int))
                input_mask_list.append(np.array(input_mask, dtype=int))

        # tensorize
        input_ids = torch.from_numpy(
            np.stack(input_ids_list, axis=0)).type(torch.LongTensor)
        input_mask = torch.from_numpy(
            np.stack(input_mask_list, axis=0)).type(torch.LongTensor)
        segment_ids = torch.from_numpy(
            np.stack(segment_ids_list, axis=0)).type(torch.LongTensor)

        # predict
        total_ins = input_ids.size(0)
        prob_out = np.zeros(total_ins)
        num_iter = total_ins // batch_size + \
            1 if total_ins % batch_size != 0 else total_ins // batch_size
        for niter in trange(num_iter, desc='Sentence'):
            input_ids_ = input_ids[batch_size *
                                   niter:batch_size * (niter + 1), :].cuda(cuda_dev)
            input_mask_ = input_mask[batch_size *
                                     niter:batch_size * (niter + 1), :].cuda(cuda_dev)
            segment_ids_ = segment_ids[batch_size *
                                       niter:batch_size * (niter + 1), :].cuda(cuda_dev)

            outputs = model(input_ids_, input_mask_, segment_ids_)
            pred = outputs[0].detach().cpu().numpy()
            prob_out[batch_size * niter:batch_size * (niter + 1)] = pred.flatten()

        pred_mat = np.zeros((num_sent, num_sent))
        pidx, sidx = 0, 0
        for i in range(num_sent):
            for j in range(i + 1, num_sent):
                if sent_same_list[sidx]:
                    pred_mat[i, j] = 1.0
                else:
                    pred_mat[i, j] = prob_out[pidx]
                    pidx += 1
                sidx += 1

        pred_mat = pred_mat + np.transpose(pred_mat) + np.eye(num_sent)  # square matrix
        pred_sim_list.append(pred_mat)

    return pred_sim_list


def predict_imp(model, tokenizer, text_docs, pos, max_seq_length, batch_size, cuda_dev):
    model.train(False)
    pred_imp_list, pool_imp_list = [], []
    for di, text_doc in tqdm(enumerate(text_docs), desc='Document'):
        pos_ = pos[di]
        sent_id_1st = [pi for pi, ids in enumerate(pos_) if ids == 0]
        num_words_leadn = int(max_seq_length * 0.8 / len(sent_id_1st))

        input_ids_list, segment_ids_list, input_mask_list = [], [], []
        for si, sent in enumerate(text_doc):
            sent_leadn = [text_doc[ii][:num_words_leadn] for ii in sent_id_1st]
            sent_concat = ' '.join(sent_leadn)
            input_ids = tokenizer.encode(
                sent, sent_concat, add_special_tokens=True, max_length=max_seq_length)

            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length - 1]
                input_ids[-1] = tokenizer.sep_token_id

            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)

            segment_b = False
            for i, ids in enumerate(input_ids):
                if segment_b:
                    segment_ids[i] = 1
                if ids == tokenizer.sep_token_id:
                    segment_b = True

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            segment_ids += padding
            input_mask += padding

            input_ids_list.append(np.array(input_ids))
            segment_ids_list.append(np.array(segment_ids))
            input_mask_list.append(np.array(input_mask))

        # tensorize
        input_ids = torch.from_numpy(
            np.stack(input_ids_list, axis=0)).type(torch.LongTensor)
        input_mask = torch.from_numpy(
            np.stack(input_mask_list, axis=0)).type(torch.LongTensor)
        segment_ids = torch.from_numpy(
            np.stack(segment_ids_list, axis=0)).type(torch.LongTensor)

        # predict
        total_ins = input_ids.size(0)
        prob_out = np.zeros(total_ins)
        pool_out = np.zeros((total_ins, 768))

        num_iter = total_ins // batch_size + 1 if total_ins % batch_size != 0 else total_ins // batch_size
        for niter in range(num_iter):
            input_ids_ = input_ids[batch_size *
                                   niter:batch_size * (niter + 1), :].cuda(cuda_dev)
            input_mask_ = input_mask[batch_size *
                                     niter:batch_size * (niter + 1), :].cuda(cuda_dev)
            segment_ids_ = segment_ids[batch_size *
                                       niter:batch_size * (niter + 1), :].cuda(cuda_dev)

            outputs = model(input_ids_, input_mask_, segment_ids_)

            pred = outputs[0].detach().cpu().numpy()    # logits
            prob_out[batch_size * niter:batch_size * (niter + 1)] = pred.flatten()

            pred = outputs[1].detach().cpu().numpy()    # dense_output
            pool_out[batch_size * niter:batch_size * (niter + 1), :] = pred

        pred_imp_list.append(prob_out)
        pool_imp_list.append(pool_out)

    return pred_imp_list, pool_imp_list


def run_predict(args, model, tokenizer, logger, batch_size, cuda_dev, is_sim_running=False, is_run_force=False):
    if args.dataset == 0:
        # DUC
        duc_base = os.path.dirname(args.DUC_data_path[0])
        if args.split == 'train':
            data_path = os.path.join(args.base_path, duc_base, args.data_type, 'train')
            sum_path = data_path
        else:
            data_path = os.path.join(args.base_path, duc_base, args.data_type, 'test')
            sum_path = data_path

        text_cls = readDUCorTACText(data_path, sum_path=sum_path, is_duc=True,
                                    data_st=args.data_start, data_en=args.data_end)
    elif args.dataset == 1:
        # TAC
        tac_base = os.path.dirname(args.TAC_data_path[0])
        if args.split == 'train':
            data_path = os.path.join(args.base_path, tac_base, args.data_type, 'train')
            sum_path = data_path
        else:
            data_path = os.path.join(args.base_path, tac_base, args.data_type, 'test')
            sum_path = data_path

        text_cls = readDUCorTACText(data_path, sum_path=sum_path, is_duc=False,
                                    data_st=args.data_start, data_en=args.data_end)

    BERT_base_dir = os.path.join(data_path, 'BERT_features', 'extractions')
    if not os.path.exists(BERT_base_dir):
        os.makedirs(BERT_base_dir)

    # retrieve text data
    text_docs = text_cls.text
    Y = text_cls.Y
    name = text_cls.name
    pos = text_cls.pos
    seg_pos = text_cls.seg

    y_name_pos_file = os.path.join(BERT_base_dir,
                                   '{}_y_name_pos_{}-{}.pkl'.format(args.split, args.data_start, args.data_end))
    save_features(y_name_pos_file, {'Y': Y, 'name': name, 'pos': pos})
    logger.write('docs files are saved in {}.'.format(y_name_pos_file))

    pred_fn = 'sim' if is_sim_running else 'imp'
    pred_fn = '{}_{}_{}-{}'.format(args.split, pred_fn, args.data_start, args.data_end)
    pred_file = os.path.join(BERT_base_dir, pred_fn)

    pool_fn = 'imp_vector'
    pool_fn = '{}_{}_{}-{}'.format(args.split, pool_fn, args.data_start, args.data_end)
    if not is_sim_running:
        pool_file = os.path.join(BERT_base_dir, pool_fn)

    if not os.path.exists(pred_file) or is_run_force:
        st_ext = time.time()

        if is_sim_running:
            pred_list = predict_sim(
                model, tokenizer, text_docs, 128, batch_size, cuda_dev, name, seg_pos)
        else:
            pred_list, pool_list = predict_imp(
                model, tokenizer, text_docs, pos, 512, batch_size, cuda_dev)

        elpased_time = time.time() - st_ext
        logger.write('prediction time for {} is {}sec: avg. {}sec/doc.'.format(data_path,
                                                                               elpased_time, elpased_time / len(text_docs)))

        save_features(pred_file, pred_list)
        logger.write('{} file stored!'.format(pred_file))
        if not is_sim_running:
            save_features(pool_file, pool_list)
            logger.write('{} file stored!'.format(pool_file))
    else:
        logger.write('{} file exists... skip prediction!'.format(pred_file))
        if not is_sim_running:
            logger.write('{} file exists... skip prediction!'.format(pool_file))


class BertBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBinaryClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        dense_output = self.dense(pooled_output)

        dp_output = self.dropout(dense_output)
        logits = self.classifier(dp_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            outputs = (loss,) + outputs
        else:
            sm = nn.Sigmoid()
            outputs = (sm(logits),) + (dense_output,) + outputs[2:]

        return outputs


def load_model(config, input_weights, gpu_id=None):
    model = BertBinaryClassification(config)

    # load trained model
    model_dict = model.state_dict()
    if gpu_id is None:
        trained_model = torch.load(input_weights)
    else:
        trained_model = torch.load(input_weights, map_location=torch.device('cuda:{}'.format(gpu_id)))
        trained_model_dict = trained_model['model_state']
    trained_dict = {k.replace('module.', ''): v for k, v in trained_model_dict.items() if k.replace('module.', '') in model_dict}
    model_dict.update(trained_dict)
    model.load_state_dict(model_dict)
    logger_log.write('trained model [{}] is loaded... dict size={}'.format(input_weights, len(trained_dict)))

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    logger_log.write('total num. parameters to be trained: {}\n'.format(pytorch_total_params))
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=int, default=0, help='0:DUC, 1:TAC')
    parser.add_argument('--data_type', default='sent', choices=['sent', 'xlnet', 'tree'])
    parser.add_argument('--split', default='train', choices=['train', 'test'])

    parser.add_argument('--base_path', type=str, default='../data')
    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'])
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'])

    parser.add_argument('--data_start', type=float, default=0)
    parser.add_argument('--data_end', type=float, default=1)

    # I/0
    parser.add_argument('--output', type=str, default='finetune_out')
    parser.add_argument('--input_weights', type=str, default=None, help='not used. hard coded')

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_parallel', action='store_true')

    # model
    parser.add_argument('--model_type', type=str,
                        default='base', choices=['base', 'large'])
    # parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size_imp', type=int, default=15)
    parser.add_argument('--batch_size_sim', type=int, default=95)

    parser.add_argument('--seed', default=777)
    parser.add_argument('--is_force', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {}, n_gpu: {}".format(device, n_gpu))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.model_type == 'base':
        bert_weights = 'bert-base-uncased'
    else:
        bert_weights = 'bert-large-uncased'

    # output folder
    output_dir = os.path.join(args.output, 'feat_ext')
    create_dir(output_dir)

    # logger
    logger_arg = Logger(os.path.join(output_dir, 'args.txt'))
    logger_arg.write(args.__repr__())
    logger_log = Logger(os.path.join(output_dir, 'log.txt'))

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_weights)
    sep_token = tokenizer.sep_token_id

    # config
    config = BertConfig.from_pretrained(bert_weights)

    # GPU, model
    if args.gpu_parallel:
        model = load_model(config, './finetune_out/pair_leadn/model_epoch1.pth', gpu_id=None)
        model = nn.DataParallel(model).cuda()
    else:
        model = load_model(config, './finetune_out/pair_leadn/model_epoch1.pth', args.gpu_id)
        cuda_dev = torch.device('cuda:{}'.format(args.gpu_id))
        model = model.cuda(cuda_dev)

    # save classifier weights and bias
    model_dict = model.state_dict()
    cls_weights = model_dict['classifier.weight'].cpu().numpy().flatten()
    cls_bias = model_dict['classifier.bias'].cpu().numpy().flatten()
    cls_w = np.concatenate([cls_weights, cls_bias])
    print(cls_bias)
    assert (len(cls_w) == 769), '{} != {}'.format(len(cls_w), 768 + 1)

    imp_cls_w_fn = os.path.join('./DPP_scripts', 'weights', 'imp_cls_w.npy')
    dir_name = os.path.dirname(imp_cls_w_fn)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.save(imp_cls_w_fn, cls_w)
    print('weights are saved in {}'.format(imp_cls_w_fn))

    # importance
    run_predict(args, model, tokenizer, logger_log,
                args.batch_size_imp, cuda_dev,
                is_sim_running=False, is_run_force=args.is_force)
    torch.cuda.empty_cache()

    # GPU, model
    if args.gpu_parallel:
        model = load_model(config, './finetune_out/pair/model_epoch1.pth', gpu_id=None)
        model = nn.DataParallel(model).cuda()
    else:
        model = load_model(config, './finetune_out/pair/model_epoch1.pth', args.gpu_id)
        cuda_dev = torch.device('cuda:{}'.format(args.gpu_id))
        model = model.cuda(cuda_dev)

    # similarity
    run_predict(args, model, tokenizer, logger_log,
                args.batch_size_sim, cuda_dev,
                is_sim_running=True, is_run_force=args.is_force)
    torch.cuda.empty_cache()
