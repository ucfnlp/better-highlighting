import numpy as np
import torch
from utils import ChunkDataManager_finetune


class CNNDMDataset(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: data path
        data_type: 'pair_leadn' for BERT-imp, 'pair' for BERT-sim, 'single'
        sep_token: [SEP] token id
        logger: logger instance
        max_seq_len: max. sequence length for input
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 data_type,
                 sep_token,
                 logger,
                 max_seq_len=512,
                 debug=False
                 ):
        self.debug = debug
        self.data_path = data_path
        self.data_type = data_type
        self.sep_token = sep_token
        self.max_seq_len = max_seq_len
        self.logger = logger

        if 'train' in self.data_path:
            self.mode = 'train'
        elif 'val' in self.data_path:
            self.mode = 'val'
        elif 'test' in self.data_path:
            self.mode = 'test'

        print('sep token id', self.sep_token)
        self.load_data()

    def load_data(self):
        # load data
        cdm_data = ChunkDataManager_finetune(load_data_path=self.data_path)
        self.data = cdm_data.load(search_key=self.data_type)

        if self.debug:
            self.data = self.data[:100]

        self.logger.write('{}-{} {} data is loaded from {}'.format(self.mode,
                                                                   self.data_type, len(self.data), self.data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, label, src_idx = self.data[index]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len - 1]
            input_ids[-1] = self.sep_token

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        segment_b = False
        for i, ids in enumerate(input_ids):
            if segment_b:
                segment_ids[i] = 1
            if ids == self.sep_token:
                segment_b = True

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        # tensorize
        input_ids = torch.from_numpy(
            np.array(input_ids)).type(torch.LongTensor)
        input_mask = torch.from_numpy(
            np.array(input_mask)).type(torch.LongTensor)
        segment_ids = torch.from_numpy(
            np.array(segment_ids)).type(torch.LongTensor)
        label = torch.tensor(label, dtype=torch.float)

        return input_ids, input_mask, segment_ids, label
