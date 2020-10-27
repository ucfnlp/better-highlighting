from __future__ import print_function

import errno
import os
import re
import glob
import numpy as np
import operator
import functools
import time
import json
import h5py
import pickle
import scipy.io as scio
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


EPS = 1e-7


def loadJson(fileName):
    f = open(fileName + ".json", 'r')
    instances = []
    for lid, line in enumerate(f):
        # print(lid)
        data = json.loads(line)
        instances.append(data)
    print('{} instances are loaded'.format(len(instances)))
    return instances


def print_stats(data, desc=''):
    print('{} - min:{}, max:{}, mean:{:.3f}, median:{}, std:{:.3f}, sum:{}'.format(desc, np.min(data), np.max(data), np.mean(data), np.median(data), np.std(data), np.sum(data)))


def save_features_h5(file_path, to_file, data_name='data'):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    hf = h5py.File(file_path, 'w')
    for i, data in enumerate(to_file):
        hf.create_dataset(str(i), data=to_file[i])
    hf.close()


def save_features(file_path, to_file):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, 'wb') as f:
        pickle.dump(to_file, f)


def load_features(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_best_model(output_dir):
    models = []
    for file in os.listdir(output_dir):
        if file.startswith('model_epoch'):
            dot_index = file.find('.pth')
            models.append(int(file[11:dot_index]))
    max_model = np.max(models)
    best_model = os.path.join(
        output_dir, 'model_epoch{}.pth'.format(max_model))
    return best_model


def convert2mat(base_dir, is_force=False):
    print(base_dir)
    path_pair = os.path.join(base_dir, 'sim.pkl')
    if os.path.exists(path_pair):
        data_pair = load_features(path_pair)
    path_single = os.path.join(base_dir, 'imp.pkl')
    if os.path.exists(path_single):
        data_single = load_features(path_single)
    path_single_vector = os.path.join(base_dir, 'imp_vector.pkl')
    if os.path.exists(path_single_vector):
        data_single_vector = load_features(path_single_vector)
    path_name = os.path.join(base_dir, 'y_name_pos.pkl')
    if os.path.exists(path_name):
        data_name = load_features(path_name)

    if (not os.path.exists(os.path.join(base_dir, 'pair.mat')) or is_force) and os.path.exists(path_pair):
        scio.savemat(os.path.join(base_dir, 'pair.mat'), mdict={'pair': data_pair})
        print('pair.mat is generated in {}'.format(os.path.join(base_dir, 'pair.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'single.mat')) or is_force) and os.path.exists(path_single):
        scio.savemat(os.path.join(base_dir, 'single.mat'), mdict={'single': data_single})
        print('single.mat is generated in {}'.format(os.path.join(base_dir, 'single.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'single_vector.mat')) or is_force) and os.path.exists(path_single_vector):
        scio.savemat(os.path.join(base_dir, 'single_vector.mat'), mdict={'single_vector': data_single_vector})
        print('single_vector.mat is generated in {}'.format(os.path.join(base_dir, 'single_vector.mat')))
    if (not os.path.exists(os.path.join(base_dir, 'name.mat')) or is_force) and os.path.exists(path_name):
        scio.savemat(os.path.join(base_dir, 'name.mat'), mdict={'name': data_name['name']})
        print('name.mat is generated in {}'.format(os.path.join(base_dir, 'name.mat')))


def convert2mat_npy(base_dir, is_force=False):
    path_w = os.path.join(base_dir, 'imp_cls_w.npy')
    if os.path.exists(path_w):
        data_weight = np.load(path_w)
    if (not os.path.exists(os.path.join(base_dir, 'imp_cls_w.mat')) and os.path.exists(path_w)) or is_force:
        scio.savemat(os.path.join(base_dir, 'imp_cls_w.mat'), mdict={'W_Extract': data_weight})
        print('imp_cls_w.mat is generated in {}'.format(os.path.join(base_dir, 'imp_cls_w.mat')))

    path_w = os.path.join(base_dir, 'sim_cls_w.npy')
    if os.path.exists(path_w):
        data_weight = np.load(path_w)
    if (not os.path.exists(os.path.join(base_dir, 'sim_cls_w.mat')) and os.path.exists(path_w)) or is_force:
        scio.savemat(os.path.join(base_dir, 'sim_cls_w.mat'), mdict={'W_Extract': data_weight})
        print('sim_cls_w.mat is generated in {}'.format(os.path.join(base_dir, 'sim_cls_w.mat')))


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real - expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def assert_tensor_eq(real, expected, eps=EPS):
    assert (torch.abs(real - expected) < eps).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write('nParams=\t' + str(nParams))


def save_model(path, model, epoch, optimizer=None):
    model_dict = {
        'epoch': epoch,
        'model_state': model.state_dict()
    }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}
        self.writer = SummaryWriter(
            log_dir=f'{dirname}/tb_logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}')

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

    def tensorboard_log(self, info, global_step):
        """
        Log at tensorboard
        :param info: info to log
        :param global_step: step in training
        """
        for k, v in info.items():
            self.writer.add_scalar(k, v, global_step)

    def tensorboard_confusion_matrix(self, ground_truth, prediction, classes, global_step):
        """
        Log a confusion matrix in tensorboard
        :param ground_truth: ground truth labels
        :param prediction: predicted labels
        :param classes: class names corresponding to label indexes
        :param global_step: step in training
        """
        fig = self.plot_confusion_matrix(ground_truth, prediction, classes)
        self.writer.add_figure('matplotlib', fig, global_step=global_step)

    @staticmethod
    def plot_confusion_matrix(ground_truth, prediction, classes):
        """
        Create a plot of a confusion matrix
        :param ground_truth: ground truth labels
        :param prediction: predicted labels
        :param classes: class names corresponding to label indexes
        :return: plot of confusion matrix
        """
        num_classes = len(classes)

        # Normalized confusion matrix
        cnf_matrix = confusion_matrix(ground_truth, prediction)
        cnf_matrix = cnf_matrix.astype(
            'float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        cnf_plot = ax.imshow(cnf_matrix, interpolation='nearest',
                             cmap=plt.cm.Blues, vmin=0, vmax=1)
        fig.colorbar(cnf_plot)

        # Ticks - class names
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=-70, fontsize=4)
        plt.yticks(tick_marks, fontsize=4)
        return fig


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def remove_annotations(s):
    return re.sub(r'\[[^ ]+ ', '', s).replace(']', '')


def get_sent_data(file_path):
    phrases = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for sent in f:
            str = remove_annotations(sent.strip())
            phrases.append(str)

    return phrases


# Find position of a given sublist
# return the index of the last token
def find_sublist(arr, sub):
    sublen = len(sub)
    first = sub[0]
    indx = -1
    while True:
        try:
            indx = arr.index(first, indx + 1)
        except ValueError:
            break
        if sub == arr[indx: indx + sublen]:
            return indx + sublen - 1
    return -1


def calculate_iou(obj1, obj2):
    area1 = calculate_area(obj1)
    area2 = calculate_area(obj2)
    intersection = get_intersection(obj1, obj2)
    area_int = calculate_area(intersection)
    return area_int / (area1 + area2 - area_int)


def calculate_area(obj):
    return (obj[2] - obj[0]) * (obj[3] - obj[1])


def get_intersection(obj1, obj2):
    left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
    top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
    right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
    bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
    if left > right or top > bottom:
        return [0, 0, 0, 0]
    return [left, top, right, bottom]


def get_match_index(src_bboxes, dst_bboxes):
    indices = set()
    for src_bbox in src_bboxes:
        for i, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            if iou >= 0.5:
                indices.add(i)
    return list(indices)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


class ChunkDataManager_finetune(object):
    '''
    data is list type & saved as pickle
    '''

    def __init__(self, load_data_path=None, save_data_path=None):
        self.load_data_path = load_data_path
        self.save_data_path = save_data_path

    def load(self, search_key=None):
        data = []
        num_loaded = 0
        find_format = os.path.join(
            self.load_data_path, '*' + search_key + '.pkl')
        file_list = sorted(glob.glob(find_format))

        for pkl_file in file_list:
            data += load_features(pkl_file)
            num_loaded += 1

        if False:
            file_list = sorted(os.listdir(self.load_data_path))
            print(file_list, len(file_list))
            for pfile in tqdm(list(file_list)):
                if not pfile.endswith('.pkl'):
                    continue
                # merge all lists
                if search_key in os.path.splitext(pfile)[0]:
                    filepath = os.path.join(self.load_data_path, pfile)
                    data += load_features(filepath)
                    num_loaded += 1
        print('{}/{} files are loaded'.format(num_loaded, len(file_list)))
        return data

    def save(self, data):
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        print('Saving data of shapes:', [len(item) for item in data])
        for i, item in tqdm(enumerate(data)):
            filepath = os.path.join(self.save_data_path, str(i) + '.pkl')
            save_features(filepath, item)
