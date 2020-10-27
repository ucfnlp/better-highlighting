import argparse
import errno
import glob
import json
import os
import pickle
import random
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.core.example import example_pb2
from tqdm import tqdm
from transformers import BertTokenizer

names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'),
                  ('summary_text', 'string'), ('corefs', 'json')]


def save_features(file_path, to_file):
    with open(file_path, 'wb') as f:
        pickle.dump(to_file, f)


def load_features(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class cnn_dm_data(object):
    def __init__(self, max_len=512, tokenizer=None):
        self.max_len = max_len

        self.unique_words = set()
        self.unique_words_freq = {}

        self.x_train_sent_pairs = []
        self.y_train = []
        self.x_test_sent_pairs = []
        self.y_test = []
        self.x_dev_sent_pairs = []
        self.y_dev = []

        self.x_train_sent_single = []
        self.y_train_single = []
        self.x_test_sent_single = []
        self.y_test_single = []
        self.x_dev_sent_single = []
        self.y_dev_single = []

        self.unique_words = set()
        self.unique_words_freq = {}

        self.tokenizer = tokenizer

    def gen_all_pairs(self, data_split, file_path, to_save=[1, 1, 0], max_length=[128, 512, 128],
                      most_similar=False, split_num=1, lead_N=5, dest_dir=None):
        '''
        to_save: [pair, pair_leadN, single] to save it or not
        max_length: max. length for each input
        most_similar: use the most similar sentence or a list of most similar sentences
        split_num: save data as multiples of chunks based on this number
        lead_N: lead sentence number

        input: pairs - [[sent1, sent2], ...] / singles = [[sent], ...]
        output (tokenized): pairs - [[sent1_tokens, sent2_tokens, label], ...] / singles = [[sent_tokens, label], ...]
        saved in .pkl
        '''
        for split in data_split:
            x_pair, y_pair_leadn, x_single, labels, src_idx_list = self.generate_sentence_pairs(file_path + '/' + split + '*',
                                                                                                lead_N=lead_N, most_similar=most_similar)

            assert (len(x_pair) == len(x_single))
            assert (len(x_pair) == len(y_pair_leadn))
            assert (len(x_pair) == len(src_idx_list))

            # split data
            x_pair_split, x_pair_leadn_split, x_single_split = [], [], []
            label_split = []
            src_idx_split = []
            chunk_num = len(x_pair) // split_num

            for i in np.arange(split_num):
                if to_save[0]:
                    x_pair_split.append(x_pair[chunk_num * i:chunk_num * (i + 1)])
                if to_save[1]:
                    x_pair_leadn_split.append(x_pair[chunk_num * i:chunk_num * (i + 1)])
                if to_save[2]:
                    x_single_split.append(x_single[chunk_num * i:chunk_num * (i + 1)])
                label_split.append(labels[chunk_num * i:chunk_num * (i + 1)])
                src_idx_split.append(src_idx_list[chunk_num * i:chunk_num * (i + 1)])
            del x_pair, y_pair_leadn, x_single

            save_data_path = os.path.join(dest_dir, split)
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)

            # tokenize & save
            if to_save[0]:
                for i, split_pairs in tqdm(enumerate(x_pair_split)):
                    x_pair_tokenized = []
                    y_pair_labels = label_split[i]
                    src_idx = src_idx_split[i]
                    assert (len(split_pairs) == len(y_pair_labels))
                    assert (len(split_pairs) == len(src_idx))

                    for j, pair in enumerate(split_pairs):
                        sent_most, sent_sum = pair
                        input_ids = tokenizer.encode(sent_most, sent_sum, add_special_tokens=True, max_length=max_length[0])
                        ########################
                        # DATA to use !!!
                        tokenized = [input_ids, y_pair_labels[j], src_idx[j]]
                        ########################
                        x_pair_tokenized.append(tokenized)

                    filepath = os.path.join(save_data_path, str(i) + '_pair.pkl')
                    save_features(filepath, x_pair_tokenized)
                print('[{}] pair tokenization & saving pairs finished '.format(split))

            if to_save[1]:
                for i, split_pairs_leadn in tqdm(enumerate(x_pair_leadn_split)):
                    x_pair_leadn_tokenized = []
                    y_pair_labels = label_split[i]
                    src_idx = src_idx_split[i]
                    assert (len(split_pairs_leadn) == len(y_pair_labels))
                    assert (len(split_pairs_leadn) == len(src_idx))

                    for j, pair in enumerate(split_pairs_leadn):
                        sent_most, sent_leadn = pair
                        sent_concat = ''.join(sent_leadn)
                        input_ids = tokenizer.encode(sent_most, sent_concat, add_special_tokens=True, max_length=max_length[1])
                        ########################
                        # DATA to use !!!
                        tokenized = [input_ids, y_pair_labels[j], src_idx[j]]
                        ########################
                        x_pair_leadn_tokenized.append(tokenized)

                    filepath = os.path.join(save_data_path, str(i) + '_pair_leadn.pkl')
                    save_features(filepath, x_pair_leadn_tokenized)
                print('[{}] pair leadN tokenization & saving pairs finished '.format(split))

            if to_save[2]:
                for i, split_singles in tqdm(enumerate(x_single_split)):
                    x_single_tokenized = []
                    y_single_labels = label_split[i]
                    src_idx = src_idx_split[i]
                    assert (len(split_singles) == len(y_single_labels))
                    assert (len(split_singles) == len(src_idx))

                    for j, sent in enumerate(split_singles):
                        input_ids = tokenizer.encode(sent, add_special_tokens=True, max_length=max_length[2])
                        ########################
                        # DATA to use !!!
                        tokenized = [input_ids, y_pair_labels[j], src_idx[j]]
                        ########################
                        x_single_tokenized.append(tokenized)

                    filepath = os.path.join(save_data_path, str(i) + '_single.pkl')
                    save_features(filepath, x_single_tokenized)
                print('[{}] single tokenization & saving singles finished '.format(split))

            if split == 'train':
                print('[cnn_dm] generated sentence pairs / single - train')

            elif split == 'test':
                print('[cnn_dm] generated sentence pairs / single - test')

            elif split == 'val':
                print('[cnn_dm] generated sentence pairs / single - val')

    def generate_sentence_pairs(self, file_path, lead_N=5, most_similar=False):
        print('processing {}'.format(file_path))
        source_files = sorted(glob.glob(file_path))
        print('{} files are found'.format(len(source_files)))

        if most_similar:
            end_index = 1
        else:
            end_index = -1

        sent_pairs, sent_single, sent_leadn = [], [], []
        labels = []
        src_idx_list = []
        src_len_stat_list = []
        num_skipped_doc = 0

        total = len(source_files) * 1000
        example_generator = self.example_generator(file_path, True)
        for example in tqdm(example_generator, total=total):
            raw_article_sents, similar_source_indices_list, summary_text = self.unpack_tf_example(example, names_to_types[:-1])

            summary_text_list = [sent.strip() for sent in summary_text.split('\n') if len(sent.strip()) > 0]

            # length of document constraint
            if len(raw_article_sents) < lead_N * 2:
                num_skipped_doc += 1
                continue
            src_len_stat_list.append(len(raw_article_sents))

            # sentences in same document except summary sentences
            sum_index = set([idx for si in similar_source_indices_list for idx in si])
            # add first N sentence indices not to choose them as negative labels
            for i in range(lead_N):
                sum_index.add(i)
            all_index = np.arange(len(raw_article_sents))
            np.random.shuffle(all_index)
            neg_cand_index = [ind for ind in all_index if ind not in sum_index]
            neg_cand_index_sorted = sorted(neg_cand_index)

            for si, sum_sent in enumerate(summary_text_list):
                neg_ind = 0
                for sisi in similar_source_indices_list[si][:end_index]:
                    if neg_ind < len(neg_cand_index):
                        # pair
                        # positive sentence pair
                        pos_index = sisi
                        sent_pairs.append([raw_article_sents[pos_index], sum_sent])
                        labels.append(1)
                        src_idx_list.append(pos_index)

                        # negative sentence pair
                        neg_index = neg_cand_index[neg_ind]
                        sent_pairs.append([raw_article_sents[neg_index], sum_sent])
                        labels.append(0)
                        src_idx_list.append(neg_index)

                        # single
                        # positive sentence
                        sent_single.append(raw_article_sents[pos_index])
                        # label_single.append(1)

                        # negative sentence
                        sent_single.append(raw_article_sents[neg_index])
                        # label_single.append(0)

                        # lead-n
                        # positive
                        pos_leadn = [raw_article_sents[sent_id] for sent_id in neg_cand_index_sorted[:lead_N]]
                        sent_leadn.append([raw_article_sents[pos_index], pos_leadn])

                        # negative
                        neg_leadn = [raw_article_sents[sent_id] for sent_id in neg_cand_index_sorted[-lead_N:]]
                        sent_leadn.append([raw_article_sents[neg_index], neg_leadn])

                        neg_ind += 1

        print('[cnn_dm-{}] # of generated pairs/singles: {} {}'.format(os.path.basename(file_path), len(sent_pairs), len(sent_single)))
        print('num. skipped doc: {} (num. sents < {})'.format(num_skipped_doc, lead_N * 2))
        print('source doc len stats {}: max:{}, min:{}, mean:{}, median:{}'.format(len(src_len_stat_list), np.max(src_len_stat_list), np.min(src_len_stat_list),
                                                                                   np.mean(src_len_stat_list), np.median(src_len_stat_list)))
        pos_src_index_list = src_idx_list[::2]
        print('positive source index stats: max:{}, min:{}, mean:{}, median:{}'.format(np.max(pos_src_index_list), np.min(pos_src_index_list),
                                                                                       np.mean(pos_src_index_list), np.median(pos_src_index_list)))
        neg_src_index_list = src_idx_list[1::2]
        print('negative source index stats: max:{}, min:{}, mean:{}, median:{}'.format(np.max(neg_src_index_list), np.min(neg_src_index_list),
                                                                                       np.mean(neg_src_index_list), np.median(neg_src_index_list)))
        return sent_pairs, sent_leadn, sent_single, labels, src_idx_list

    '''
    [cnn_dm-test*] # of generated pairs/singles: 85296 85296
    num. skipped doc: 248 (num. sents < 10) 0.29%
    source doc len stats 11242: max:231, min:10, mean:34.612880270414514, median:30.0
    positive source index stats: max:140, min:0, mean:9.946351528793848, median:5.0
    negative source index stats: max:225, min:5, mean:20.074821797036204, median:16.0
    
    [cnn_dm-val*] # of generated pairs/singles: 105194 105194
    num. skipped doc 13104: 264 (num. sents < 10) 0.25%
    source doc len stats: max:319, min:10, mean:34.005570818070815, median:29.0
    positive source index stats: max:194, min:0, mean:9.81215658687758, median:5.0
    negative source index stats: max:200, min:5, mean:20.1385820484058, median:16.0
    
    [cnn_dm-train*] # of generated pairs/singles: 2077132 2077132
    num. skipped doc 281966: 5147 (num. sents < 10) 0.24%
    source doc len stats: max:404, min:10, mean:40.24281650979196, median:34.0
    positive source index stats: max:341, min:0, mean:13.170475444025705, median:9.0
    negative source index stats: max:299, min:5, mean:23.31311250320153, median:18.0
    '''

    def example_generator(self, data_path, single_pass):
        """Generates tf.Examples from data files.
        Args:
          data_path:
            Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
          single_pass:
            Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

        Yields:
          Deserialized tf.Example.
        """
        while True:
            filelist = glob.glob(data_path)  # get the list of datafiles
            assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
            if single_pass:
                filelist = sorted(filelist)
            else:
                random.shuffle(filelist)
            for f in filelist:
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes:
                        break  # finished reading this file
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                    yield example_pb2.Example.FromString(example_str)
            if single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break

    def decode_text(self, text):
        try:
            text = text.decode('utf-8')
        except:
            try:
                text = text.decode('latin-1')
            except:
                raise
        return text

    def unpack_tf_example(self, example, names_to_types):
        def get_string(name):
            return self.decode_text(example.features.feature[name].bytes_list.value[0])

        def get_string_list(name):
            texts = get_list(name)
            texts = [self.decode_text(text) for text in texts]
            return texts

        def get_list(name):
            return example.features.feature[name].bytes_list.value

        def get_delimited_list(name):
            text = get_string(name)
            return text.split(' ')

        def get_delimited_list_of_lists(name):
            text = get_string(name)
            return [[int(i) for i in (l.split(' ') if l != '' else [])] for l in text.split(';')]

        def get_delimited_list_of_tuples(name):
            list_of_lists = get_delimited_list_of_lists(name)
            return [tuple(l) for l in list_of_lists]

        def get_json(name):
            text = get_string(name)
            return json.loads(text)
        func = {'string': get_string,
                'list': get_list,
                'string_list': get_string_list,
                'delimited_list': get_delimited_list,
                'delimited_list_of_lists': get_delimited_list_of_lists,
                'delimited_list_of_tuples': get_delimited_list_of_tuples,
                'json': get_json}

        res = []
        for name, type in names_to_types:
            if name not in example.features.feature:
                raise Exception('%s is not a feature of TF Example' % name)
            res.append(func[type](name))
        return res


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stat_run', action='store_true', help='stat. running or processing')

    # data
    parser.add_argument('--src_data', type=str, default='../../data/cnn_dm_sum', help='source data path')
    parser.add_argument('--dest_data', type=str, default='../../data/cnn_dm_sum_pair/', help='dest data path')

    # param
    parser.add_argument('--most_similar', action='store_true', help='use only the most similar sentence')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # source data directory
    source_dir = args.src_data

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # dest. data directory
    dest_dir = args.dest_data
    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # data processing
    cnn_dm = cnn_dm_data(tokenizer=tokenizer)

    if args.stat_run:
        file_name = 'src_idx_list.pkl'

        # data
        if not os.path.exists(file_name):
            src_idx_split_list = []
            splits = ['train', 'val', 'test']
            for split in splits:
                _, _, _, _, src_idx_list = cnn_dm.generate_sentence_pairs(source_dir + '/' + split + '*',
                                                                          lead_N=5, most_similar=True)
                src_idx_split_list.append(src_idx_list)

            save_features(file_name, src_idx_split_list)
        else:
            src_idx_split_list = load_features(file_name)

        # Plot stats
        kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
        color_list = ["dodgerblue", "gold", "deeppink", "green", "red", "olive"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False, sharex=True, dpi=100)
        for i, src_idx_list in enumerate(src_idx_split_list):
            sns.distplot(src_idx_list[::2], color=color_list[2 * i], ax=axes[i], label='Pos', axlabel=splits[i], **kwargs)
            sns.distplot(src_idx_list[1::2], color=color_list[2 * i + 1], ax=axes[i], label='Neg', axlabel=splits[i], **kwargs)
            axes[i].legend()
        plt.xlim(0, 100)
        plt.tight_layout()

        fig.savefig('cnn_dm_sum-sent_stat.png')
        fig.savefig('cnn_dm_sum-sent_stat.pdf')

    else:
        start_time = time.time()
        paths = ['test', 'val']
        cnn_dm.gen_all_pairs(paths, source_dir, to_save=[1, 1, 0], max_length=[128, 512, 128],
                             most_similar=args.most_similar, split_num=1, lead_N=args.leadn, dest_dir=dest_dir)
        paths = ['train']
        cnn_dm.gen_all_pairs(paths, source_dir, to_save=[1, 1, 0], max_length=[128, 512, 128],
                             most_similar=args.most_similar, split_num=10, lead_N=args.leadn, dest_dir=dest_dir)
        print('elapsed training time: {:3.3f} hrs'.format((time.time() - start_time) / 3600))
