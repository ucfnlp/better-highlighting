#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from read_text_from_data import readDUCorTACText
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
from utils import load_features, print_stats, save_features


def draw_stats(data_list, splits=['train', 'test'], dataset='DUC', cut_thres=15):
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False, sharex=True, dpi=100)

    for i, data in enumerate(data_list):
        data = np.array(data)
        data_ = data[data <= cut_thres]
        sns.countplot(data_, ax=axes[i])

    plt.xlim(0, cut_thres)
    plt.tight_layout()

    fig.savefig('full_sent_pos_{}.png'.format(dataset))


def scaling(X, mode=0):
    if isinstance(X, list):
        X = np.array(X).reshape((len(X), 1))
    if mode == 0:
        X = X
    elif mode == 1:
        X = StandardScaler().fit_transform(X)
    elif mode == 2:
        X = MinMaxScaler().fit_transform(X)
    elif mode == 3:
        X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    else:
        raise NotImplementedError

    return X.ravel()


def decode_index(seg_index):
    if len(seg_index) < 6:
        st = int(seg_index[:2])
        ed = int(seg_index[2:])
    else:
        st = int(seg_index[:3])
        ed = int(seg_index[3:])
    return (st, ed)


def gen_segment_text(segment, parsed_words):
    st, ed = decode_index(segment[0])
    seg_text = ' '.join(parsed_words[st:ed])
    return seg_text


def data_processing(files, seg_per_sent=3, debug=False):
    data_all = []

    # stats
    full_sent_pos, seg_num_doc = [], []
    total_instance = 0
    for fi, file in tqdm(enumerate(files)):
        data = load_features(file)
        # [(name, [(time_elapsed, parsed_words, segments, full_sent_pos), ...])]

        doc_data = []
        for doc in data:
            # doc
            doc_name = doc[0]
            if debug:
                print(doc_name, fi)

            num_sent = len(doc[1])
            seg_nums = 0
            sent_data = []

            for i in range(num_sent):
                # sentence
                sent = doc[1][i]

                segments = []
                for ssent in sent:
                    parsed_words, segments_all = ssent[1], ssent[2]
                    num_seg = len(segments_all)

                    sub_segments = []
                    is_full_sent = False
                    sum_list = [0, 1]

                    if isinstance(segments_all, str):
                        full_sent_key = '000{:03}'.format(len(parsed_words))
                        sub_segments.append([full_sent_key, 0, 0, True, True, [0 for _ in sum_list]])
                        full_sent_pos.append(0)
                    else:
                        segments_all = list(segments_all.items())

                        probs = [[] for _ in sum_list]
                        max_ed = -1
                        for j in range(num_seg):
                            key, prob = segments_all[j][0], segments_all[j][1]

                            for p in range(len(probs)):
                                probs[p].append(prob[p][1] + prob[p][2])    # P(comma) + P(period)
                            _, ed = decode_index(key)
                            max_ed = max(max_ed, ed)

                        full_sent_key = '000{:03d}'.format(max_ed)

                        mode = 0    # if > 0, scaling
                        if mode > 0:
                            probs = [scaling(pr, mode) for pr in probs]

                        # ! filtering RULE !
                        quant_ratio = 0.75  # 0.5 for median
                        thres = [np.quantile(prs, quant_ratio) for prs in probs]

                        num_sel_seg = 0
                        for j in range(num_seg):
                            key, prob = segments_all[j][0], segments_all[j][1]
                            st, ed = decode_index(key)
                            rouge_score = 0
                            sel_type = 0

                            # segment selection by threshold
                            if probs[0][j] < thres[0] or probs[1][j] < thres[1]:
                                sel_type = 1

                            # optional RULE
                            if parsed_words[st] == 'and' or parsed_words[ed - 1] == 'and':
                                sel_type = 2

                            if sel_type == 0:
                                num_sel_seg += 1

                            if key == full_sent_key:
                                full_sent_pos.append(j)
                                is_full_sent = True

                            # segment: [key, psum, rouge, is_full_sent, is_sel, probs.]
                            sub_segments.append([key, np.sum([probs[p][j] for p in sum_list]), rouge_score, is_full_sent, sel_type, [probs[p][j] for p in sum_list]])
                            is_full_sent = False

                        if debug:
                            # raw segments based on XLNet prob. dist.
                            segment_sorted = sorted(sub_segments, key=lambda x: x[1], reverse=True)
                            print('\n[Full sentence]: ', ' '.join(parsed_words))
                            print('\n[XLNet segments - sorted by prob. sum]')
                            topn_seg = 10
                            for si, seg in enumerate(segment_sorted):
                                if si < topn_seg or si >= len(segment_sorted) - topn_seg:
                                    seg_text = gen_segment_text(seg, parsed_words)
                                    app_txt = 'top-{}'.format(topn_seg) if si < topn_seg else 'bot-{}'.format(topn_seg)
                                    fi = '-f' if seg[0] == full_sent_key else ''
                                    prob_txt = 'sel:[{}] Sum:{:.3e}, L-C + L-P:{:.3e}, R-C: + R-P:{:.3e}'.format(seg[4], seg[1], seg[5][0], seg[5][1])
                                    print('{:03}/{:03} [{}{}]'.format(si + 1, len(segment_sorted), app_txt, fi), prob_txt, seg_text)

                        sub_segments = [sseg for sseg in sub_segments if (sseg[4] == 0) and (not sseg[3])]
                        sub_segments = sorted(sub_segments, key=lambda x: x[1], reverse=True)

                        if debug:
                            # selected segments
                            print('\n[candidate segments filtered by median - sorted]')
                            for si, seg in enumerate(sub_segments):
                                seg_text = gen_segment_text(seg, parsed_words)
                                fi = '-f' if seg[3] else ''
                                prob_txt = 'sel:[{}] Sum:{:.3e}, L-C + L-P:{:.3e}, R-C: + R-P:{:.3e}'.format(seg[4], seg[1], seg[5][0], seg[5][1])
                                print('{:03}/{:03}{}'.format(si + 1, num_sel_seg, fi), prob_txt, seg_text)

                        # final sub-segments
                        if len(sub_segments) == 0:
                            sub_segments.append([full_sent_key, 0, 0, True, True, [0 for _ in sum_list]])
                        else:
                            sub_segments = sub_segments[:seg_per_sent]

                        if debug:
                            # selected segments
                            print('\n[final candidate segments]')
                            for si, seg in enumerate(sub_segments):
                                seg_text = gen_segment_text(seg, parsed_words)
                                fi = '-f' if seg[3] else ''
                                prob_txt = 'sel:[{}] Sum:{:.3e}, L-C + L-P:{:.3e}, R-C: + R-P:{:.3e}'.format(seg[4], seg[1], seg[5][0], seg[5][1])
                                print('{:03}/{:03}{}'.format(si + 1, num_sel_seg, fi), prob_txt, seg_text)
                            pdb.set_trace()

                        seg_nums += len(sub_segments)

                    # exclude less than 5 words (not chunks)
                    final_sub_segments = []
                    for sseg in sub_segments:
                        seg_text = gen_segment_text(sseg, parsed_words)
                        if len(seg_text.split()) >= 5:
                            final_sub_segments.append(sseg)
                    segments.append((parsed_words, final_sub_segments))

                sent_data.append(segments)

            doc_data.append((doc_name, sent_data))
            seg_num_doc.append(seg_nums)

        data_all = data_all + doc_data
        total_instance += len(data)

    print('data num.: {} {}'.format(total_instance, len(data_all)))

    return data_all, full_sent_pos, seg_num_doc


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    # data
    parser.add_argument('--seg_per_sent', type=int, default=2, help='num. of segments per sub-sentence to be generated')
    parser.add_argument('--dataset', type=int, default=0, help='0:DUC, 1:TAC')
    parser.add_argument('--base_path', type=str, default='../data')

    parser.add_argument('--topn_sent', type=int, default=20, help='top-n sentences per each article on DUC,TAC')

    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'], help='TAC text data path')
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'], help='TAC summary data path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 0:
        # DUC
        train_path = os.path.join(args.base_path, args.DUC_data_path[0])
        test_path = os.path.join(args.base_path, args.DUC_data_path[1])
        text_train = readDUCorTACText(train_path, is_duc=True, topn_sent=args.topn_sent)
        text_test = readDUCorTACText(test_path, is_duc=True, topn_sent=args.topn_sent)

        output_dir = os.path.dirname(args.DUC_data_path[0])
        data_name = 'DUC'
    elif args.dataset == 1:
        # TAC
        train_path = os.path.join(args.base_path, args.TAC_data_path[0])
        test_path = os.path.join(args.base_path, args.TAC_data_path[1])
        train_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[0])
        test_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[1])
        text_train = readDUCorTACText(train_path, sum_path=train_sum_path, is_duc=False, topn_sent=args.topn_sent)
        text_test = readDUCorTACText(test_path, sum_path=test_sum_path, is_duc=False, topn_sent=args.topn_sent)

        output_dir = os.path.dirname(args.TAC_data_path[0])
        data_name = 'TAC'

    # output
    search_dir = 'seg_prob'
    out_dir_name = 'xlnet'
    input_path = os.path.join(args.base_path, output_dir, search_dir)
    output_path = os.path.join(args.base_path, output_dir, out_dir_name)

    ori_text = [text_test, text_train]
    full_sent_pos_list = []
    splits = ['test', 'train']
    for i, split in enumerate(splits):
        print(input_path + '/{}*'.format(split))
        files = sorted(glob.glob(input_path + '/{}*'.format(split)))
        print('{} files are found'.format(len(files)))

        filename = os.path.join(output_path, split + '.pkl')
        filename_stats = os.path.join(output_path, split + '_stats.pkl')

        if not os.path.exists(filename):
            st_time = time.time()

            # merge data based on filtering rule
            data_all, full_sent_pos, seg_num_doc = data_processing(files, args.seg_per_sent, args.debug)

            save_features(filename, data_all)
            save_features(filename_stats, [full_sent_pos, seg_num_doc])

            print('total num. sentences', len(full_sent_pos))
            print('elapsed time: {:.3f}s'.format(time.time() - st_time))
        else:
            data_all = load_features(filename)
            full_sent_pos, seg_num_doc = load_features(filename_stats)
            print('data is loaded from {} and {}'.format(filename, filename_stats))

        full_sent_pos_list.append(full_sent_pos)
        print_stats(seg_num_doc, '{}-seg_num_doc'.format(split))

    # draw data stats
    draw_stats(full_sent_pos_list, splits, data_name)
