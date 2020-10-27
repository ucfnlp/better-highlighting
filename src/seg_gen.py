import argparse
import os
import time
from collections import defaultdict

import numpy as np
import spacy
import torch
import torch.nn as nn
from read_text_from_data import readDUCorTACText
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm, trange
from transformers import XLNetLMHeadModel, XLNetTokenizer
from utils import print_stats, save_features


class compSegProb():
    # given tokens, predict prob. dist. of <MASK> located "before" or "after" tokens
    def __init__(self, model, tokenizer, nlp_parser, cuda_device, batch_size=15):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp_parser = nlp_parser
        self.cuda_device = cuda_device
        self.sep_id = 4
        self.comma_id = 19
        self.period_id = 9
        self.cls_id = 3
        self.mask_id = 6
        self.batch_size = batch_size

        self.model.train(False)

    def predict_batch_prob(self, sent, ref, max_token=350, min_word=4, scaling=0, printout=False):
        '''
        compute prob. dist. of <mask> for all segments given a sentence
        '''
        st_time = time.time()
        parsed_sent = self.nlp_parser(sent)

        parsed_words = [token.text for token in parsed_sent]
        n_w = len(parsed_words)

        if n_w <= min_word:
            return (0, parsed_words, sent)

        words_ids = [tokenizer.encode(word) for word in parsed_words]
        # +1 for <mask> token
        total_tks = int(np.sum([len(ids) for ids in words_ids]) + 1)
        # total_segs = (n_w-1) * n_w # x2 for pre, post <mask>

        # limit num. of tokens
        cut_id = None
        if total_tks > max_token:
            total_len, wid = 0, 0
            for wi, ids in enumerate(words_ids):
                if total_len + len(ids) >= max_token:
                    wid = wi
                    break
                total_len += len(ids)

            cut_id = wid
            words_ids = words_ids[:wid]
            parsed_words = parsed_words[:wid]
            total_tks = int(np.sum([len(ids) for ids in words_ids]) + 1)
            n_w = len(parsed_words)
        assert total_tks <= max_token, 'error {} <= {}'.format(total_tks, max_token)

        cnt = 0
        for i in range(n_w):
            for j in range(i, n_w):
                if j + 1 - i < min_word:
                    continue
                cnt += 1
        total_segs = int(cnt * 2)
        prob_out = np.zeros((total_segs, 3))

        # generate all segments
        input_ids = np.zeros((total_segs, total_tks))  # (B, segments)
        perm_mask = np.zeros((total_segs, total_tks, total_tks))    # (B, segments, segments)
        target_mapping = np.zeros((total_segs, 1, total_tks))   # (B, 1, segments)
        attention_mask = np.zeros((total_segs, total_tks))  # (B, segments)

        batch_idx = 0
        for i in range(n_w):
            for j in range(i, n_w):
                if j + 1 - i < min_word:
                    continue
                # post <mask>
                token_ids = [ids for word_ids in words_ids[i:j + 1] for ids in word_ids] + [self.mask_id]
                len_tk_ids = len(token_ids)
                input_ids[batch_idx, :len_tk_ids] = np.array(token_ids)
                attention_mask[batch_idx, :len_tk_ids] = np.ones_like(token_ids)
                perm_mask[batch_idx, :, len_tk_ids - 1] = 1.0
                target_mapping[batch_idx, 0, len_tk_ids - 1] = 1.0
                batch_idx += 1
                # pre <mask>
                token_ids.pop()
                token_ids.insert(0, self.mask_id)
                input_ids[batch_idx, :len_tk_ids] = np.array(token_ids)
                attention_mask[batch_idx, :len_tk_ids] = np.ones_like(token_ids)
                perm_mask[batch_idx, :, 0] = 1.0
                target_mapping[batch_idx, 0, 0] = 1.0
                batch_idx += 1

        assert (batch_idx == total_segs), '{} != {}'.format(batch_idx, total_segs)

        input_ids = torch.from_numpy(input_ids).type(torch.LongTensor)
        attention_mask = torch.from_numpy(attention_mask).type(torch.FloatTensor)
        perm_mask = torch.from_numpy(perm_mask).type(torch.FloatTensor)
        target_mapping = torch.from_numpy(target_mapping).type(torch.FloatTensor)

        # Model prediction
        num_iter = total_segs // self.batch_size + \
            1 if total_segs % self.batch_size != 0 else total_segs // self.batch_size
        for niter in range(num_iter):
            input_ids_ = input_ids[self.batch_size * niter:self.batch_size * (niter + 1), :].cuda(self.cuda_device)
            perm_mask_ = perm_mask[self.batch_size * niter:self.batch_size * (niter + 1), :, :].cuda(self.cuda_device)
            target_mapping_ = target_mapping[self.batch_size * niter:self.batch_size * (niter + 1), :, :].cuda(self.cuda_device)
            attention_mask_ = attention_mask[self.batch_size * niter:self.batch_size * (niter + 1), :].cuda(self.cuda_device)

            outputs = self.model(input_ids_, perm_mask=perm_mask_,
                                 target_mapping=target_mapping_, attention_mask=attention_mask_)
            cur_batch_size = outputs[0].size()[0]
            prob = outputs[0].view(cur_batch_size, -1)
            prob_norm = torch.nn.functional.softmax(prob, dim=-1)
            prob_norm = prob_norm.cpu().detach().numpy()
            prob_out[self.batch_size * niter:self.batch_size * (niter + 1), :] = prob_norm[:, [self.sep_id, self.comma_id, self.period_id]]
        torch.cuda.empty_cache()
        self.prob_out = prob_out

        # segment
        # key: [pre-mask prob, post-mask prob, index]
        idx = 0
        seg_dict = {}
        segments = defaultdict(list)
        for i in range(n_w):
            for j in range(i, n_w):
                if j + 1 - i < min_word:
                    continue

                # segment id: [i,j+1)
                st_id = '{:03d}'.format(i)
                ed_id = '{:03d}'.format(j + 1)
                sid = st_id + ed_id
                segments[sid] = [self.prob_out[2 * idx + 1, :], self.prob_out[2 * idx, :], idx]
                idx += 1

        ed_time = time.time()
        # --- end of segment generation ---

        # Scaling
        if printout:
            n_sample = self.prob_out.shape[0]
            self.prob_out[:, 0] = self.scaling(
                self.prob_out[:, 0].reshape(n_sample, 1), scaling)
            self.prob_out[:, 1] = self.scaling(
                self.prob_out[:, 1].reshape(n_sample, 1), scaling)
            self.prob_out[:, 2] = self.scaling(
                self.prob_out[:, 2].reshape(n_sample, 1), scaling)

            # Method 1
            idx = 0
            seg_dict = {}
            for i in range(n_w):
                for j in range(i + 1, n_w):
                    seg_key = parsed_words[i:j + 1]
                    seg_key = ' '.join(seg_key)
                    seg_dict[seg_key] = np.sum(
                        self.prob_out[2 * idx, :]) + np.sum(self.prob_out[2 * idx + 1, :])
                    idx += 1
            sorted_seg = sorted(
                seg_dict.items(), key=lambda kv: kv[1], reverse=True)

            for i, seg in enumerate(sorted_seg):
                print('{:<2} {:.4} {}'.format(i + 1, seg[1], seg[0]))

            # Method 2
            idx = 0
            probs = np.zeros((n_w + 1, 3))  # 3 for <sep> , .
            prob_cnt = np.zeros((n_w + 1,))
            for i in range(n_w):
                for j in range(i + 1, n_w):
                    probs[i] += self.prob_out[2 * idx + 1, :]
                    probs[j + 1] += self.prob_out[2 * idx, :]
                    prob_cnt[i] += 1.
                    prob_cnt[j + 1] += 1.
                    idx += 1

            for i in range(n_w + 1):
                probs[i, :] /= prob_cnt[i]

            idx = 0
            seg_dict = {}
            for i in range(n_w):
                for j in range(i + 1, n_w):
                    seg_key = parsed_words[i:j + 1]
                    seg_key = ' '.join(seg_key)

                    seg_dict[seg_key] = np.sum(
                        probs[i, :]) + np.sum(probs[j + 1, :])
                    idx += 1
            sorted_seg = sorted(
                seg_dict.items(), key=lambda kv: kv[1], reverse=True)

            print('\n')
            for i, seg in enumerate(sorted_seg):
                print('{:<2} {:.4} {}'.format(i + 1, seg[1], seg[0]))

        return (ed_time - st_time, parsed_words[:cut_id], segments)

    def scaling(self, X, mode):
        # X: [[]]
        if mode == 0:
            X = X
        elif mode == 1:
            X = StandardScaler().fit_transform(X)
        elif mode == 2:
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
    st, ed = decode_index(segment)
    seg_text = ' '.join(parsed_words[st:ed])
    return seg_text


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


def print_seg(segments, parsed_words):

    if not isinstance(parsed_words[0], str):
        parsed_words = [ps.text for ps in parsed_words]

    segments = list(segments.items())

    print_mode = 1

    max_ed = -1
    sum_list = [0, 1]
    probs = [[] for _ in sum_list]
    for si, seg in enumerate(segments):
        key, prob = seg
        if print_mode == 0:
            probs[0].append(prob[0][0])
            probs[1].append(prob[0][1])
            probs[2].append(prob[0][2])
            probs[3].append(prob[1][0])
            probs[4].append(prob[1][1])
            probs[5].append(prob[1][2])
        elif print_mode == 1:
            for i in range(len(probs)):
                probs[i].append(prob[i][1] + prob[i][2])
        _, ed = decode_index(key)
        max_ed = max(max_ed, ed)

    full_sent_key = '000{:03d}'.format(max_ed)

    mode = 0
    if mode > 0:
        probs = [scaling(pr, mode) for pr in probs]

    quant_ratio = 0.75  # 0.5 for median
    thres = [np.quantile(prs, quant_ratio) for prs in probs]

    segment_sorted = []
    for k in range(len(segments)):
        key, prob = segments[k][0], segments[k][1]
        segment_sorted.append([key, np.sum([probs[p][k] for p in sum_list]), [probs[p][k] for p in sum_list]])
    segment_sorted = sorted(segment_sorted, key=lambda x: x[1], reverse=True)
    print('\n[Full sentence]: ', ' '.join(parsed_words))
    print('\n[XLNet segments - sorted by prob. sum]')
    topn_seg = 20
    for si, seg in enumerate(segment_sorted):
        if si < topn_seg or si >= len(segment_sorted) - topn_seg:
            seg_text = gen_segment_text(seg[0], parsed_words)
            app_txt = 'top-20' if si < topn_seg else 'bot-20'
            fi = 'f' if seg[0] == full_sent_key else ''
            if print_mode == 0:
                prob_txt = 'Sum:{:.3e}, L-S:{:.3e}, L-C:{:.3e}, L-P:{:.3e}, R-S:{:.3e}, R-C:{:.3e}, R-P:{:.3e}'.format(seg[1], seg[2][0], seg[2][1], seg[2][2], seg[2][3], seg[2][4], seg[2][5])
            elif print_mode == 1:
                prob_txt = 'Sum:{:.3e}, L-C + L-P:{:.3e}, R-C: + R-P:{:.3e}'.format(seg[1], seg[2][0], seg[2][1])
            print('{:03}/{:03} [{}] {}'.format(si + 1, len(segment_sorted), app_txt, fi), prob_txt, seg_text)

    num_sel_seg = 0
    is_full_sent = False
    segments_new = []
    for j in range(len(segments)):
        key, prob = segments[j][0], segments[j][1]
        rouge_score = 0

        # segment selection RULE: > upper quantile
        sel = True
        if probs[0][j] < thres[0] or probs[1][j] < thres[1]:
            sel = False

        if sel:
            num_sel_seg += 1

        if key == full_sent_key:
            is_full_sent = True

        segments_new.append([key, np.sum([probs[p][j] for p in sum_list]), rouge_score, is_full_sent, sel, [probs[p][j] for p in sum_list]])
        is_full_sent = False

    # segments: [key, prob_sum, Rouge2, is_full_sent, is_selection]
    id_order = 0
    # qualified candidate segments
    print('\n[candidate segments filtered by median - sorted]')
    segments_new = sorted(segments_new, key=lambda x: x[1], reverse=True)
    for si, seg in enumerate(segments_new):
        if seg[4]:
            seg_text = gen_segment_text(seg[0], parsed_words)
            fi = 'f' if seg[3] else ''
            if print_mode == 0:
                prob_txt = 'Sum:{:.3e}, L-S:{:.3e}, L-C:{:.3e}, L-P:{:.3e}, R-S:{:.3e}, R-C:{:.3e}, R-P:{:.3e}'.format(seg[1], seg[5][0], seg[5][1], seg[5][2], seg[5][3], seg[5][4], seg[5][5])
            elif print_mode == 1:
                prob_txt = 'Sum:{:.3e}, L-C + L-P:{:.3e}, R-C: + R-P:{:.3e}'.format(seg[1], seg[5][0], seg[5][1])
            print('{:03}/{:03}-{}'.format(id_order + 1, num_sel_seg, fi), prob_txt, seg_text)
            id_order += 1


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--xlnet_model', type=str, default='xlnet-large-cased')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_lg')

    # segment params
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--min_word', type=int, default=4)
    parser.add_argument('--max_token', type=int, default=200)
    parser.add_argument('--topn_sent', type=int, default=20, help='read top N sentences')

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_parallel', action='store_true')

    # data
    parser.add_argument('--dataset', type=int, default=0,
                        help='0:DUC, 1:TAC')
    parser.add_argument('--split', default='train', choices=['train', 'test'])

    parser.add_argument('--base_path', type=str, default='../data')
    parser.add_argument('--output', type=str, default='seg_prob')
    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'], help='TAC text data path')
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'], help='TAC summary data path')

    parser.add_argument('--data_start', type=float, default=0, help='start point of data in 0-1 for DUC or TAC')
    parser.add_argument('--data_end', type=float, default=1, help='end point of data in 0-1 for DUC or TAC')

    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # XLNet models
    tokenizer = XLNetTokenizer.from_pretrained(args.xlnet_model)
    model = XLNetLMHeadModel.from_pretrained(args.xlnet_model)
    if args.gpu_parallel:
        model = nn.DataParallel(model).cuda()
    else:
        cuda_dev = torch.device('cuda:{}'.format(args.gpu_id))
        model = model.cuda(cuda_dev)
    model.train(False)

    # spaCy: used for merge noun chunks & name entities
    spacy.prefer_gpu()
    nlp = spacy.load(args.spacy_model)

    def merge_entities_and_nouns(doc, ret=True):
        assert doc.is_parsed
        with doc.retokenize() as retokenizer:
            seen_words = set()
            for ent in doc.ents:
                attrs = {"tag": ent.root.tag, "dep": ent.root.dep, "ent_type": ent.label}
                retokenizer.merge(ent, attrs=attrs)
                seen_words.update(w.i for w in ent)
            for npc in doc.noun_chunks:
                if any(w.i in seen_words for w in npc):
                    continue
                attrs = {"tag": npc.root.tag, "dep": npc.root.dep}
                retokenizer.merge(npc, attrs=attrs)
                seen_words.update(w.i for w in npc)
            if ret:
                return doc

    nlp.add_pipe(merge_entities_and_nouns, name='merge_entities_and_nouns')

    def merge_punc(doc, ret=True):
        assert doc.is_parsed
        spans = []
        for word in doc[:-1]:
            if word.is_punct or not word.nbor(1).is_punct:
                continue
            start = word.i
            end = word.i + 1
            while end < len(doc) and doc[end].is_punct:
                end += 1
            span = doc[start:end]
            spans.append((span, word.tag_, word.lemma_, word.ent_type_))

        with doc.retokenize() as retokenizer:
            for span, tag, lemma, ent_type in spans:
                attrs = {"tag": tag, "lemma": lemma, "ent_type": ent_type}
                retokenizer.merge(span, attrs=attrs)
            if ret:
                return doc

    nlp.add_pipe(merge_punc, name='merge_punc')

    # segment generation
    compSP = compSegProb(model, tokenizer, nlp, cuda_dev, batch_size=args.batch_size)

    if args.debug:
        # one sentence test
        sent = "drumming up of advance orders and dreams of knocking Boeing's 747 off its perch as the top bird in passenger transport"

        time_elapsed, parsed_sents, segments = compSP.predict_batch_prob(sent, '', args.max_token, args.min_word, scaling=0, printout=False)
        print_seg(segments, parsed_sents)
        exit(0)

    if args.dataset == 0:
        # DUC
        if args.split == 'train':
            data_path = os.path.join(args.base_path, args.DUC_data_path[0])
        else:
            data_path = os.path.join(args.base_path, args.DUC_data_path[1])

        text_cls = readDUCorTACText(data_path, is_duc=True, topn_sent=args.topn_sent,
                                    data_st=args.data_start, data_en=args.data_end)

        output_dir = os.path.dirname(args.DUC_data_path[0])
        offset = 0

    elif args.dataset == 1:
        # TAC
        if args.split == 'train':
            data_path = os.path.join(args.base_path, args.TAC_data_path[0])
            sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[0])
        else:
            data_path = os.path.join(args.base_path, args.TAC_data_path[1])
            sum_path = os.path.join(args.base_path, args.TAC_sum_data_path[1])

        text_cls = readDUCorTACText(data_path, sum_path=sum_path, is_duc=False, topn_sent=args.topn_sent,
                                    data_st=args.data_start, data_en=args.data_end)

        output_dir = os.path.dirname(args.TAC_data_path[0])
        offset = 0

    # output
    output_dir = os.path.join(args.base_path, output_dir, args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids = None
    texts = text_cls.text[:ids]
    refs = text_cls.ref[:ids]
    names = text_cls.name[:ids]

    if False:
        # dataset stats
        num_sent = []
        num_word = []
        for di in tqdm(range(0, len(texts)), desc='Document'):
            doc = texts[di]
            num_sent.append(len(doc))
            for sent in doc:
                words = sent.split()
                num_word.append(len(words))

        print_stats(num_sent, desc='num_sents - {}'.format(args.split))
        print_stats(num_word, desc='num_words - {}'.format(args.split))
        exit(0)

    begin_id = 0
    save_id = 0
    sum_elapsed_time, sum_sent = 0, 0
    doc_data = []
    for di in tqdm(range(begin_id, len(texts)), desc='Document'):
        sent_data = []
        doc = texts[di]
        ref = refs[di]
        name = names[di]
        print(name)

        tbar = trange(len(doc))
        for si in tbar:
            sent = doc[si]

            # split sentence
            parsed_sent = nlp(sent)
            parsed_words = [ps.text.strip() for ps in parsed_sent]
            split_tag = [-1]
            for i in range(len(parsed_words)):
                if parsed_words[i].endswith((',', '.', ':', ';')):
                    parsed_words[i] = parsed_words[i][:-1].strip()
                    split_tag.append(i)
            if len(parsed_words) - 1 not in split_tag:
                split_tag.append(len(parsed_words) - 1)
            sent_sub = []
            for i in range(1, len(split_tag)):
                st, ed = split_tag[i - 1], split_tag[i]
                sent_sub.append(' '.join(parsed_words[st + 1:ed + 1]).strip())

            ss_data = []
            te = 0
            for ss in sent_sub:
                time_elapsed, parsed_sents, segments = compSP.predict_batch_prob(ss, ref, args.max_token, args.min_word, scaling=0, printout=False)
                ss_data.append((time_elapsed, parsed_sents, segments))
                te += time_elapsed
            te /= len(sent_sub)

            sent_data.append(ss_data)

            sum_elapsed_time += te
            sum_sent += 1
            tbar.set_description('[{}/{} sentence] avg. time/sent: {:.3f}'.format(si, len(doc), sum_elapsed_time / sum_sent))

        doc_data.append((name, sent_data))

        if (di + 1) % args.save_freq == 0 or (di + 1) == len(texts):
            text_format = '{}{:05d}.pkl'.format(args.split, save_id)
            output_path = os.path.join(output_dir, text_format)
            save_features(output_path, doc_data)
            doc_data = []
            save_id += 1
