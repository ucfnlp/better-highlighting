#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np


class readDUCorTACText():
    def __init__(self, data_path, topn_sent=1000, sum_path=None, is_duc=True, reg_exp=None, data_st=0, data_en=1):
        # data_path: path to one of split (train, test)
        self.data_path = data_path
        self.topn_sent = topn_sent
        self.sum_path = sum_path
        self.is_duc = is_duc
        self.data_st = data_st
        self.data_en = data_en

        self.text = []
        self.Y = []
        self.name = []
        self.pos = []
        self.ref = []
        self.seg = None

        if reg_exp is None:
            if self.is_duc:
                self.re_exp = '^d[0-9]+'
            else:
                self.re_exp = '^D[0-9]+\-[A-B]'
        else:
            self.re_exp = reg_exp

        self.read_split()

    def read_split(self):
        # access data from here
        # text: list of list
        # Y: list of list
        # name: list (doc name)
        # pos: list of list
        self.read_text(self.data_path, self.re_exp, self.sum_path, self.data_st, self.data_en)

    def get_files(self, root_path, ext=None):
        if ext is not None:
            return [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f)) and f.endswith(ext)]
        return [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f)) and not f.endswith('.py')]

    def load_txt_data(self, file_path):
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        sents = []
        for line in lines:
            if len(line.strip()) > 0:
                words = line.split()
                words = [word.strip() for word in words]
                line_clean = ' '.join(words)
                sents.append(line_clean)
        return sents

    def load_data(self, file_path):
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return lines

    def read_text(self, data_path, re_exp, sum_path=None, data_st=0, data_en=1):
        folders = sorted([os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and re.search(re_exp, d)])
        num_folders = len(folders)
        st_id = int(num_folders * data_st)
        en_id = int(num_folders * data_en)
        folders = folders[st_id:en_id]
        print(data_path, '[{}-{}] {} folders found'.format(data_st, data_en, len(folders)))

        summary_path = sum_path if sum_path is not None else data_path
        sum_folders = [os.path.join(summary_path, os.path.basename(folder)) for folder in folders]
        text, Y, name, pos, seg = [], [], [], [], []

        y_len = []      # for stats
        text_len = []   # for stats
        for folder in folders:
            file_name = os.path.basename(folder)

            file_path = os.path.join(folder, file_name + '.pos')
            pos_sent = self.load_data(file_path)
            position = [int(p) - 1 for p in pos_sent]
            pos_top = [p for p in position if p < self.topn_sent]

            pos_dict = {i: 1 if p < self.topn_sent else 0 for i, p in enumerate(position)}
            pos_new_id = list(np.zeros(len(position), dtype=int))
            for i in range(1, len(pos_new_id)):
                pos_new_id[i] = pos_new_id[i - 1] + pos_dict[i]
            pos.append(pos_top)

            file_path = os.path.join(folder, file_name + '.txt')
            sents = self.load_txt_data(file_path)
            sents_top = [sent for i, sent in enumerate(sents) if pos_dict[i]]
            text.append(sents_top)

            file_path = os.path.join(folder, file_name + '.Y')
            y_label = self.load_data(file_path)
            y_label_int = [int(y) - 1 for y in y_label]
            y_label_top = [pos_new_id[y] for y in y_label_int if pos_dict[y]]
            Y.append(y_label_top)

            file_path = os.path.join(folder, file_name + '.seg')
            if os.path.exists(file_path):
                pos_seg = self.load_data(file_path)
                pos_seg_int = [int(ps) - 1 for ps in pos_seg]
                seg.append(pos_seg_int)

            name.append(file_name)

            y_len.append(len(y_label_top))
            text_len.append(len(sents_top))

        print('y_label - num doc: {}, max_y_len: {}, min_y_len: {}, mean_y_len: {:.3f}, std_y_len: {:.3f}'.format(len(y_len), np.max(y_len), np.min(y_len), np.mean(y_len), np.std(y_len)))
        print('text - num doc: {}, max_num_sents: {}, min_num_sents: {}, mean_num_sents: {:.3f}, std_num_sents: {:.3f}'.format(len(text_len), np.max(text_len), np.min(text_len), np.mean(text_len), np.std(text_len)))

        ref_list = []
        for folder in sum_folders:
            # summaries
            refs = self.get_files(folder, ext='.sum')
            ref_txt = []
            for ref in refs:
                ref_sents = self.load_txt_data(os.path.join(folder, ref))
                for rsent in ref_sents:
                    ref_txt.append(rsent)
            ref_list.append(ref_txt)

        seg = seg if len(seg) > 0 else None

        self.text, self.Y, self.name, self.pos, self.ref, self.seg = text, Y, name, pos, ref_list, seg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=int, default=0, help='0:DUC, 1:TAC')
    parser.add_argument('--base_path', type=str, default='../data')
    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'])
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'])
    parser.add_argument('--topn_sent', type=int, default=500, help='500 for all sentence on DUC,TAC')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # DUC or TAC
    if args.dataset == 0:
        # DUC
        train_path = os.path.join(args.base_path, args.DUC_data_path[0])
        test_path = os.path.join(args.base_path, args.DUC_data_path[1])

        text_train = readDUCorTACText(train_path, is_duc=True)
        text_test = readDUCorTACText(test_path, is_duc=True)

    elif args.dataset == 1:
        # TAC
        train_path = os.path.join(args.base_path, args.TAC_data_path[0])
        test_path = os.path.join(args.base_path, args.TAC_data_path[1])
        train_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[0])
        test_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[1])

        text_train = readDUCorTACText(train_path, sum_path=train_sum_path, is_duc=False)
        text_test = readDUCorTACText(test_path, sum_path=test_sum_path, is_duc=False)

    # DUC & TAC
    docs_train = text_train.text    # list of list : This is the texts for train set
    docs_test = text_test.text      # This is the texts for test set
