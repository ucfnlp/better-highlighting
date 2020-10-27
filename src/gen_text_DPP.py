import argparse
import os
import time

import numpy as np
from nltk.tokenize import word_tokenize
from read_text_from_data import readDUCorTACText
# from rouge.rougescore import rouge_2
from rouge_new.rouge import Rouge
from tqdm import tqdm
from utils import create_dir, load_features, loadJson, print_stats

shared_dir = 'shared'
prp_list = ['he', 'him', 'himself', 'she', 'her', 'herself', 'them', 'their', 'it', 'itself']


def generate_text(data, data_ext, text_dir, type, topn_sent=20, is_stat=False):
    start_time = time.time()

    if type == 'xlnet':
        (words_train, doc_seg_train, doc_name_list_train) = generate_text_from_xlnet(data[0],
                                                                                     data_ext[0], text_dir[0], topn_sent, is_stat=is_stat)
        (words_test, doc_seg_test, doc_name_list_test) = generate_text_from_xlnet(data[1],
                                                                                  data_ext[1], text_dir[1], topn_sent, is_stat=is_stat)
    elif type == 'tree':
        (words_train, doc_seg_train, doc_name_list_train) = generate_text_from_tree(data[0],
                                                                                    data_ext[0], text_dir[0], topn_sent, is_stat=is_stat)
        (words_test, doc_seg_test, doc_name_list_test) = generate_text_from_tree(data[1],
                                                                                 data_ext[1], text_dir[1], topn_sent, is_stat=is_stat)
    elif type == 'sent':
        (words_train, doc_seg_train, doc_name_list_train) = generate_text_from_sent(data_ext[0], text_dir[0])
        (words_test, doc_seg_test, doc_name_list_test) = generate_text_from_sent(data_ext[1], text_dir[1])

    if is_stat:
        exit(0)

    words = words_train.union(words_test)
    print('total {} unique words found'.format(len(words)))

    doc_counts_train, total_sents_train = compute_idf(words, doc_seg_train, doc_name_list_train, text_dir[0])
    doc_counts_test, total_sents_test = compute_idf(words, doc_seg_test, doc_name_list_test, text_dir[1])

    shared_path = os.path.join(os.path.dirname(text_dir[0]), shared_dir)
    create_dir(shared_path)

    # dict
    dict_path = os.path.join(shared_path, 'dict')
    words_dict = {word: i + 1 for i, word in enumerate(words)}
    with open(dict_path, 'w') as f:
        for word in words:
            f.write('{}\t{}\n'.format(word, words_dict[word]))
    print('dict file is generated in {}.'.format(dict_path))

    # prp
    prp_path = os.path.join(shared_path, 'prp')
    with open(prp_path, 'w') as f:
        for prp in prp_list:
            prp = prp.lower()
            if prp in words:
                f.write('{}\n'.format(words_dict[prp]))
    print('prp file is generated in {}.'.format(prp_path))

    # idf global
    doc_counts_all = doc_counts_train + doc_counts_test
    total_docs_all = total_sents_train + total_sents_test
    idf = [np.log2(total_docs_all / doc_cnt) if doc_cnt > 0 else -1 for doc_cnt in doc_counts_all]

    idf_path = os.path.join(shared_path, 'idf')
    with open(idf_path, 'w') as f:
        for w_id, word in enumerate(words):
            if idf[w_id] >= 0:
                f.write('{0}\t{1:.17f}\n'.format(w_id + 1, idf[w_id]))

    print('elpased time: {}sec'.format(time.time() - start_time))


def tokenize(text):
    tokenized_txt = []
    for doc in text:
        tokenized_doc = []
        for sent in doc:
            tokens = word_tokenize(sent)
            tokens_processed = [token.lower().strip() for token in tokens]
            tokenized_doc.append(tokens_processed)
        tokenized_txt.append(tokenized_doc)
    return tokenized_txt


def decode_index(seg_index):
    if len(seg_index) < 6:
        st = int(seg_index[:2])
        ed = int(seg_index[2:])
    else:
        st = int(seg_index[:3])
        ed = int(seg_index[3:])
    return (st, ed)


def generate_text_from_xlnet(data, data_ext, text_dir, topn_sent, is_stat=False):
    rouge = Rouge(metrics=['rouge-2'], stats=['f'])

    create_dir(text_dir)
    summary, Y, name, pos = data_ext
    name2id = {nm: i for i, nm in enumerate(name)}

    # 0. generate txt, pos, cost, Y
    doc_seg, doc_name_list = [], []
    y_skip_case = []
    seg_num_doc, seg_num_sent, seg_len, y_doc, seg_num_word = [], [], [], [], []
    for doc in tqdm(data):
        doc_name = doc[0]

        doc_name_list.append(doc_name)
        doc_dir = os.path.join(text_dir, doc_name)
        create_dir(doc_dir)

        if doc_name not in name2id:
            continue
        idx = name2id[doc_name]
        pos_ = pos[idx]
        y_ = Y[idx]
        summary_ = summary[idx]

        # get top-n position
        pos_dict = {i: 1 if p < topn_sent else 0 for i, p in enumerate(pos_)}

        if not is_stat:
            cost_file = os.path.join(doc_dir, doc_name + '.cost')
            f_cost = open(cost_file, 'w')
            pos_file = os.path.join(doc_dir, doc_name + '.pos')
            f_pos = open(pos_file, 'w')
            seg_file = os.path.join(doc_dir, doc_name + '.seg')
            f_seg = open(seg_file, 'w')
            txt_file = os.path.join(doc_dir, doc_name + '.txt')
            f_txt = open(txt_file, 'w')
            y_file = os.path.join(doc_dir, doc_name + '.Y')
            f_y = open(y_file, 'w')
            sum_file = os.path.join(doc_dir, doc_name + '.sum')
            f_sum = open(sum_file, 'w')

        num_sent = len(doc[1])
        assert num_sent == len(pos_), '{} != {}'.format(num_sent, len(pos_))

        text_all = ''
        line_num, pos_num, seg_num = 0, 0, 0
        seg_nums_doc = 0
        sent_seg, pos_seg, cost_seg, y_seg, seg_seg = [], [], [], [], []
        for si, sent in enumerate(doc[1]):
            if not pos_dict[si]:
                continue

            refresh_line_num = True if pos_[si] == 0 else False
            get_rouge = True if si in y_ else False
            y_dict_text = {}

            seg_nums_sent = 0
            for ssent in sent:
                parsed_words, segments = ssent

                num_seg = len(segments)
                seg_nums_doc += num_seg
                seg_nums_sent += num_seg

                for segi, seg in enumerate(segments):
                    key = seg[0]

                    # txt
                    st, ed = decode_index(key)
                    seg_text = ' '.join(parsed_words[st:ed]).strip()
                    text_all = text_all + seg_text + '\n'
                    sent_seg.append(seg_text)
                    seg_len.append(len(seg_text))
                    seg_num_word.append(len(seg_text.split()))

                    # pos
                    if refresh_line_num:
                        pos_num = 0
                        refresh_line_num = False
                    pos_seg.append(pos_num)
                    pos_num += 1

                    # cost
                    cost_seg.append(len(seg_text))

                    # Y
                    if get_rouge:
                        y_dict_text[line_num] = seg_text

                    # seg
                    seg_seg.append(seg_num)

                    line_num += 1

            if seg_nums_sent > 0:
                seg_num += 1
                if get_rouge:
                    # get segments with the highest rouge score
                    y_seg_rouge = []
                    for k, txt in y_dict_text.items():
                        rouge_score = rouge.get_scores(txt, ' '.join(summary_))[0]['rouge-2']['f']
                        y_seg_rouge.append((k, rouge_score))

                    y_list = sorted(y_seg_rouge, key=lambda x: x[1], reverse=True)
                    y_list_sel = [yl[0] for yl in y_list[:2]]
                    y_seg += y_list_sel

            else:
                if refresh_line_num:
                    pos_num = 0
                if get_rouge:
                    y_skip_case.append((doc_name, si))

            if seg_nums_sent > 0:
                seg_num_sent.append(seg_nums_sent)

        doc_seg.append(sent_seg)
        seg_num_doc.append(seg_nums_doc)
        y_doc.append(len(y_seg))

        # write to files: txt, pos, cost, Y
        if not is_stat:
            f_txt.write('{}'.format(text_all))
            for pi, ps in enumerate(pos_seg):
                f_pos.write('{}\n'.format(ps + 1))
                f_cost.write('{}\n'.format(cost_seg[pi]))
                f_seg.write('{}\n'.format(seg_seg[pi] + 1))
            for y in y_seg:
                f_y.write('{}\n'.format(y + 1))
            for summ in summary_:
                f_sum.write('{}\n'.format(summ.strip()))
            f_txt.close()
            f_pos.close()
            f_cost.close()
            f_y.close()
            f_seg.close()
            f_sum.close()

    print('y_skip_case', len(y_skip_case), y_skip_case)

    print_stats(seg_num_sent, '{}-seg_per_sent'.format(os.path.basename(text_dir)))
    print_stats(seg_num_doc, '{}-seg_per_doc'.format(os.path.basename(text_dir)))
    print_stats(y_doc, '{}-summary_per_doc'.format(os.path.basename(text_dir)))
    print_stats(seg_num_word, '{}-word_per_seg'.format(os.path.basename(text_dir)))
    print_stats(seg_len, '{}-seg_len'.format(os.path.basename(text_dir)))

    # 1. get all vocabulary: words
    if not is_stat:
        doc_seg_tokenized = tokenize(doc_seg)

        words = set()
        for doc in doc_seg_tokenized:
            for sent in doc:
                words = words.union(set(sent))

        return (words, doc_seg_tokenized, doc_name_list)
    return (set(), [], doc_name_list)


def generate_text_from_sent(data_ext, text_dir, is_stat=False):
    create_dir(text_dir)
    summary, Y, name, pos, text = data_ext

    # 0. generate txt, pos, cost, Y
    doc_name_list = []
    for di, doc in enumerate(text):
        doc_name = name[di]
        doc_name_list.append(doc_name)
        doc_dir = os.path.join(text_dir, doc_name)
        create_dir(doc_dir)

        pos_ = pos[di]
        y_ = Y[di]
        summary_ = summary[di]

        assert len(doc) == len(pos_), '{} != {}'.format(len(doc), len(pos_))

        # find a specific word in dataset
        for si, sent in enumerate(doc):
            words = [w.lower().strip() for w in sent.split()]
            if '737-400' in words:
                print(doc_name, si)

        if not is_stat:
            cost_file = os.path.join(doc_dir, doc_name + '.cost')
            f_cost = open(cost_file, 'w')
            pos_file = os.path.join(doc_dir, doc_name + '.pos')
            f_pos = open(pos_file, 'w')
            txt_file = os.path.join(doc_dir, doc_name + '.txt')
            f_txt = open(txt_file, 'w')
            y_file = os.path.join(doc_dir, doc_name + '.Y')
            f_y = open(y_file, 'w')
            sum_file = os.path.join(doc_dir, doc_name + '.sum')
            f_sum = open(sum_file, 'w')

            # write to files: txt, pos, cost, Y
            for si, sent in enumerate(doc):
                f_txt.write('{}\n'.format(sent.strip()))
                f_pos.write('{}\n'.format(pos_[si] + 1))
                f_cost.write('{}\n'.format(len(sent)))
            for y in y_:
                f_y.write('{}\n'.format(y + 1))
            for summ in summary_:
                f_sum.write('{}\n'.format(summ.strip()))
            f_txt.close()
            f_pos.close()
            f_cost.close()
            f_y.close()
            f_sum.close()

    if not is_stat:
        # 1. get all vocabulary: words
        text_tokenized = tokenize(text)

        words = set()
        for doc in text_tokenized:
            for sent in doc:
                words = words.union(set(sent))

        return (words, text_tokenized, doc_name_list)
    return (set(), [], doc_name_list)


def generate_text_from_tree(data, data_ext, text_dir, topn_sent=20, min_word=4, is_stat=False):
    # assume tree segments are based on "top-20" sentences on DUC, TAC
    # data: {['n_article'], ['n_abstract'], ['article'], ['abstract']}
    # ['abstract'] - [ (['index'], ['originalText']), ... ]
    # ['article']  - [ (['index'], ['originalText'], ['tokens'], ['parse'], ['segments']), ...]
    # ['tokens']   - [ {['index'], ['word'], ['origianlText'], ['characterOffsetBegin'], ['characterOffsetEnd'], ['pos'], ['before'], ['after']}, ...]
    # data[i]['article'][j]['segments'][k] - ith document, jth sentence, kth segment

    rouge = Rouge(metrics=['rouge-2'], stats=['f'])
    create_dir(text_dir)
    summary, Y, name, pos = data_ext

    doc_seg, doc_name_list = [], []
    y_skip_case = []
    seg_num_doc, seg_num_sent, seg_len, y_doc = [], [], [], []
    for di, doc in enumerate(data):
        article = doc['article']
        num_sent = len(article)

        doc_name = name[di]
        doc_name_list.append(doc_name)
        doc_dir = os.path.join(text_dir, doc_name)
        create_dir(doc_dir)

        pos_ = pos[di]
        y_ = Y[di]
        summary_ = summary[di]

        assert num_sent == len(pos_), '{} != {}'.format(num_sent, len(pos_))

        # get top-n position
        pos_dict = {i: 1 if p < topn_sent else 0 for i, p in enumerate(pos_)}
        pos_new_id = list(np.zeros(len(pos_), dtype=int))
        for i in range(1, len(pos_new_id)):
            pos_new_id[i] = pos_new_id[i - 1] + pos_dict[i]
        y_ = [pos_new_id[y] for y in y_ if pos_dict[y]]

        if not is_stat:
            cost_file = os.path.join(doc_dir, doc_name + '.cost')
            f_cost = open(cost_file, 'w')
            pos_file = os.path.join(doc_dir, doc_name + '.pos')
            f_pos = open(pos_file, 'w')
            seg_file = os.path.join(doc_dir, doc_name + '.seg')
            f_seg = open(seg_file, 'w')
            txt_file = os.path.join(doc_dir, doc_name + '.txt')
            f_txt = open(txt_file, 'w')
            y_file = os.path.join(doc_dir, doc_name + '.Y')
            f_y = open(y_file, 'w')
            sum_file = os.path.join(doc_dir, doc_name + '.sum')
            f_sum = open(sum_file, 'w')

        text_all = ''
        line_num, pos_num, seg_num = 0, 0, 0
        seg_cnt_all, y_nums_doc = 0, 0
        sent_seg, pos_seg, cost_seg, y_seg, seg_seg, seg_num_word = [], [], [], [], [], []
        for si, sent in enumerate(article):
            if not pos_dict[si]:
                continue

            tokens = sent['tokens']
            if len(tokens) == 0:
                continue

            segments = sent['segments']
            segments = sorted(segments, key=lambda x: x[-1], reverse=False)
            len_seg = len(segments)
            # ----------------------------------------------------------------------
            # heuristic RULE - min. of 3 & max. of 7 based on number of segments !!!
            if len_seg <= 6:
                segments = segments[:3]
            elif len_seg <= 10:
                idx = int(len_seg / 2.)
                segments = segments[:idx]
            else:
                idx = min(int(len_seg / 2.), 7)
                segments = segments[:idx]
            # ----------------------------------------------------------------------
            segments = [seg for seg in segments if seg[1] - seg[0] + 1 >= min_word]

            refresh_line_num = True if pos_[si] == 0 else False
            get_rouge = True if si in y_ else False
            y_dict = {}

            seg_cnt = 0
            for segi, seg in enumerate(segments):
                st, ed, stc, edc, d = seg

                # check if a segment is a full sentence
                if st == 0 and ed == len(tokens) - 1:
                    continue

                seg_cnt += 1
                seg_cnt_all += 1

                seg_text = ''.join([tokens[s]['word'] + tokens[s]['after'] for s in range(st, ed + 1)]).strip()

                # txt
                text_all = text_all + seg_text + '\n'
                sent_seg.append(seg_text)
                seg_len.append(len(seg_text))
                seg_num_word.append(len(seg_text.split()))

                # pos
                if refresh_line_num:
                    pos_num = 0
                    refresh_line_num = False
                pos_seg.append(pos_num)
                pos_num += 1

                # cost
                cost_seg.append(len(seg_text))

                # Y - avoid to compute R-2 for a full sentence
                if get_rouge:
                    rouge_score = rouge.get_scores(seg_text, summary_)[0]['rouge-2']['f']
                    # rouge_score = rouge_2(seg_text, summary_, alpha=0.5)
                    y_dict[line_num] = rouge_score

                # seg
                seg_seg.append(seg_num)
                line_num += 1

            seg_num_sent.append(seg_cnt)

            if get_rouge:
                if len(y_dict) > 0:
                    # set all segments as oracle segments
                    for line, rouge in y_dict.items():
                        y_seg.append(line)
                        y_nums_doc += 1
                else:
                    y_skip_case.append((doc_name, si))

            if seg_cnt > 0:
                seg_num += 1
            else:
                if refresh_line_num:
                    pos_num = 0

        doc_seg.append(sent_seg)
        seg_num_doc.append(seg_cnt_all)
        y_doc.append(y_nums_doc)

        # write to files: txt, pos, cost, Y
        if not is_stat:
            f_txt.write('{}'.format(text_all))
            for pi, ps in enumerate(pos_seg):
                f_pos.write('{}\n'.format(ps + 1))
                f_cost.write('{}\n'.format(cost_seg[pi]))
                f_seg.write('{}\n'.format(seg_seg[pi] + 1))
            for y in y_seg:
                f_y.write('{}\n'.format(y + 1))
            for summ in summary_:
                f_sum.write('{}\n'.format(summ.strip()))
            f_txt.close()
            f_pos.close()
            f_cost.close()
            f_y.close()
            f_seg.close()
            f_sum.close()

    print('y_skip_case', len(y_skip_case), y_skip_case)

    print_stats(seg_num_doc, '{}-seg_num_doc'.format(os.path.basename(text_dir)))
    print_stats(seg_len, '{}-seg_len'.format(os.path.basename(text_dir)))
    print_stats(seg_num_sent, '{}-seg_num_sent'.format(os.path.basename(text_dir)))
    print_stats(y_doc, '{}-y_doc'.format(os.path.basename(text_dir)))
    print_stats(seg_num_word, '{}-seg_num_word'.format(os.path.basename(text_dir)))

    # 1. get all vocabulary: words
    if not is_stat:
        doc_seg_tokenized = tokenize(doc_seg)

        words = set()
        for doc in doc_seg_tokenized:
            for sent in doc:
                words = words.union(set(sent))

        return (words, doc_seg_tokenized, doc_name_list)

    return (set(), [], doc_name_list)


def compute_idf(words, doc_seg, doc_name_list, text_dir):
    # doc_seg: tokenized text
    # 2. compute IDF
    words_dict = {word: i + 1 for i, word in enumerate(words)}

    print('begin computing idf...')
    num_words = len(words)
    doc_counts = np.zeros(num_words)
    total_sents = 0
    sent_ind = 1

    # summary text is not included for computing idf
    for di, doc in tqdm(enumerate(doc_seg)):
        doc_name = doc_name_list[di]
        words_file = os.path.join(text_dir, doc_name, doc_name + '.words')
        f_words = open(words_file, 'w')

        for si, sent in enumerate(doc):
            for t_num, token in enumerate(sent):
                # token = token.strip()
                f_words.write('{} {} {}\n'.format(si + 1, t_num + 1, words_dict[token]))  # sent_ind
            sent_ind += 1
        f_words.close()

        # idf
        for w_id, word in enumerate(words):
            for sent in doc:
                count = sent.count(str(word))
                if count > 0:
                    doc_counts[w_id] += 1
        total_sents += len(doc)

    print('total doc counts: {}'.format(total_sents))

    idf = [np.log2(total_sents / doc_cnt) if doc_cnt > 0 else -1 for doc_cnt in doc_counts]

    # idf
    idf_filename = os.path.join(text_dir, 'idf')
    with open(idf_filename, 'w') as f:
        for w_id, word in enumerate(words):
            if idf[w_id] >= 0:
                f.write('{0}\t{1:.17f}\n'.format(w_id + 1, idf[w_id]))

    return (doc_counts, total_sents)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=int, default=0, help='0:DUC, 1:TAC')
    parser.add_argument('--data_type', default='sent', choices=['sent', 'xlnet', 'tree'], help='sent is only for CNN')

    parser.add_argument('--base_path', type=str, default='../data')
    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'])
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'])

    parser.add_argument('--topn_sent', type=int, default=20, help='top-n sentences from each article on DUC,TAC')

    parser.add_argument('--is_stat', action='store_true', help='enable this to get stat results')
    parser.add_argument('--is_force', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 0:
        # DUC
        train_path = os.path.join(args.base_path, args.DUC_data_path[0])
        test_path = os.path.join(args.base_path, args.DUC_data_path[1])

        if args.data_type == 'tree':
            text_train = readDUCorTACText(train_path, is_duc=True, topn_sent=args.topn_sent)
            text_test = readDUCorTACText(test_path, is_duc=True, topn_sent=args.topn_sent)
        else:
            text_train = readDUCorTACText(train_path, is_duc=True, topn_sent=args.topn_sent)
            text_test = readDUCorTACText(test_path, is_duc=True, topn_sent=args.topn_sent)

        dest_dir = os.path.join(args.base_path, os.path.dirname(args.DUC_data_path[0]), args.data_type)
        data_name = 'DUC'

    elif args.dataset == 1:
        # TAC
        train_path = os.path.join(args.base_path, args.TAC_data_path[0])
        test_path = os.path.join(args.base_path, args.TAC_data_path[1])
        train_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[0])
        test_sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[1])

        if args.data_type == 'tree':
            text_train = readDUCorTACText(train_path, sum_path=train_sum_path, is_duc=False, topn_sent=args.topn_sent)
            text_test = readDUCorTACText(test_path, sum_path=test_sum_path, is_duc=False, topn_sent=args.topn_sent)
        else:
            text_train = readDUCorTACText(train_path, sum_path=train_sum_path, is_duc=False, topn_sent=args.topn_sent)
            text_test = readDUCorTACText(test_path, sum_path=test_sum_path, is_duc=False, topn_sent=args.topn_sent)

        dest_dir = os.path.join(args.base_path, os.path.dirname(args.TAC_data_path[0]), args.data_type)
        data_name = 'TAC'

    train_ids = None
    test_ids = None

    if args.data_type == 'xlnet':

        train_file = os.path.join(dest_dir, 'train.pkl')
        train_data = load_features(train_file)
        test_file = os.path.join(dest_dir, 'test.pkl')
        test_data = load_features(test_file)

        data_ext = []
        summary, Y = text_train.ref[:train_ids], text_train.Y[:train_ids]
        name, pos = text_train.name[:train_ids], text_train.pos[:train_ids]
        data_ext.append([summary, Y, name, pos])
        summary, Y = text_test.ref[:test_ids], text_test.Y[:test_ids]
        name, pos = text_test.name[:test_ids], text_test.pos[:test_ids]
        data_ext.append([summary, Y, name, pos])
        text_dir = [os.path.join(dest_dir, 'train'), os.path.join(dest_dir, 'test')]

        generate_text([train_data, test_data], data_ext, text_dir, args.data_type, topn_sent=args.topn_sent,
                      is_stat=args.is_stat)

    elif args.data_type == 'tree':

        test_file = os.path.join(dest_dir, '{}_test_seg'.format(data_name))
        test_data = loadJson(test_file)
        train_file = os.path.join(dest_dir, '{}_train_seg'.format(data_name))
        train_data = loadJson(train_file)

        data_ext = []
        summary, Y = text_train.ref[:train_ids], text_train.Y[:train_ids]
        name, pos = text_train.name[:train_ids], text_train.pos[:train_ids]
        # text = text_train.text[:train_ids]
        data_ext.append([summary, Y, name, pos])
        summary, Y = text_test.ref[:test_ids], text_test.Y[:test_ids]
        name, pos = text_test.name[:test_ids], text_test.pos[:test_ids]
        # text = text_test.text[:train_ids]
        data_ext.append([summary, Y, name, pos])
        text_dir = [os.path.join(dest_dir, 'train'), os.path.join(dest_dir, 'test')]

        generate_text([train_data, test_data], data_ext, text_dir, args.data_type, topn_sent=args.topn_sent,
                      is_stat=args.is_stat)

    elif args.data_type == 'sent':
        # CNN
        data_ext = []
        summary, Y = text_train.ref[:train_ids], text_train.Y[:train_ids]
        name, pos, text = text_train.name[:train_ids], text_train.pos[:train_ids], text_train.text[:train_ids]
        data_ext.append([summary, Y, name, pos, text])
        summary, Y = text_test.ref[:test_ids], text_test.Y[:test_ids]
        name, pos, text = text_test.name[:test_ids], text_test.pos[:test_ids], text_test.text[:test_ids]
        data_ext.append([summary, Y, name, pos, text])
        text_dir = [os.path.join(dest_dir, 'train'), os.path.join(dest_dir, 'test')]

        generate_text([], data_ext, text_dir, args.data_type, topn_sent=args.topn_sent, is_stat=args.is_stat)
