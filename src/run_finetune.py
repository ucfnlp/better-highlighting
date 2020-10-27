import argparse
import os

import torch
import torch.nn as nn
from dataset_cnndm import CNNDMDataset
from torch.utils.data import DataLoader
from train_finetune import train
from transformers import AdamW, BertModel, BertPreTrainedModel, BertTokenizer
from utils import Logger, create_dir


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

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def parse_args():
    parser = argparse.ArgumentParser()

    # I/0
    parser.add_argument('--output', type=str, default='finetune_out')
    parser.add_argument('--data_type', type=str, default='pair_leadn',
                        choices=['pair', 'pair_leadn', 'single'])
    parser.add_argument('--data_path', type=str,
                        default='../../data/cnn_dm_sum_pair')
    parser.add_argument('--input_weights', type=str, default=None, help='trained weights')

    # model
    parser.add_argument('--max_seq_len', type=int,
                        default=512, choices=[512, 128], help='input seq. length of data')

    # train
    parser.add_argument('--model_type', type=str,
                        default='base', choices=['base', 'large'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--lr_init', type=float, default=1e-3)

    # dataset
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=777)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.model_type == 'base':
        bert_weights = 'bert-base-uncased'
    else:
        bert_weights = 'bert-large-uncased'

    # output folder
    output_dir = os.path.join(args.output, args.data_type)
    create_dir(output_dir)

    # logger
    logger_arg = Logger(os.path.join(output_dir, 'args.txt'))
    logger_arg.write(args.__repr__())
    logger_log = Logger(os.path.join(output_dir, 'log.txt'))

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_weights)
    sep_token = tokenizer.sep_token_id

    # dataset
    train_dset = CNNDMDataset(data_path=os.path.join(args.data_path, 'train'),
                              data_type=args.data_type,
                              sep_token=sep_token,
                              logger=logger_log,
                              max_seq_len=args.max_seq_len,
                              debug=args.debug)
    trainval_loader = DataLoader(train_dset, args.batch_size,
                                 shuffle=True, num_workers=1)

    test_dset = CNNDMDataset(data_path=os.path.join(args.data_path, 'test'),
                             data_type=args.data_type,
                             sep_token=sep_token,
                             logger=logger_log,
                             max_seq_len=args.max_seq_len,
                             debug=args.debug)
    test_loader = DataLoader(test_dset, args.batch_size,
                             shuffle=False, num_workers=1)

    # model
    model = BertBinaryClassification.from_pretrained(bert_weights)
    model = nn.DataParallel(model).cuda()
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    logger_log.write(
        'total num. parameters to be trained: {}'.format(pytorch_total_params))

    # load trained model
    if args.input_weights is not None:
        model_dict = model.state_dict()
        trained_model = torch.load(
            os.path.join(output_dir, args.input_weights))
        trained_dict = {k: v for k, v in trained_model.items()
                        if k in model_dict}
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
        logger_log.write(
            'trained model [{}] is loaded...'.format(args.input_weights))

        optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(trained_model.get(
            'optimizer_state', trained_model))
        epoch = trained_model['epoch'] + 1
    else:
        optim = None
        epoch = 0

    train(model, trainval_loader, test_loader, args.epochs, output_dir,
          logger_log, optim, epoch, args.batch_size, device, n_gpu, args.lr_init)
