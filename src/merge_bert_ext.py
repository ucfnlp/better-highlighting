import os
import glob
import argparse

from utils import save_features, load_features, save_features_h5
from utils import convert2mat, convert2mat_npy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=int, default=0, help='0:DUC, 1:TAC, 2:CNN')
    parser.add_argument('--data_type', default='sent', choices=['sent', 'xlnet', 'tree'])
    parser.add_argument('--split', default='train', choices=['train', 'test'])

    parser.add_argument('--base_path', type=str, default='../data')
    parser.add_argument('--DUC_data_path', default=['DUC/2003', 'DUC/2004'])
    parser.add_argument('--TAC_data_path', default=['TAC/s080910_gen_proc', 'TAC/s11_gen_proc'])
    parser.add_argument('--TAC_sum_data_path', default=['TAC/s080910_gen', 'TAC/s11_gen'])

    parser.add_argument('--is_force', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 0:
        # DUC
        base_name = os.path.dirname(args.DUC_data_path[0])
    elif args.dataset == 1:
        # TAC
        base_name = os.path.dirname(args.TAC_data_path[0])

    if args.split == 'train':
        data_path = os.path.join(args.base_path, base_name, args.data_type, 'train')
    else:
        data_path = os.path.join(args.base_path, base_name, args.data_type, 'test')

    BERT_base_dir = os.path.join(data_path, 'BERT_features', 'extractions')
    BERT_output_dir = os.path.dirname(BERT_base_dir)

    # files to find
    y_name_pos_file = '{}_y_name_pos*'.format(args.split)
    imp_file = '{}_imp_[0-9]*'.format(args.split)
    imp_vector_file = '{}_imp_vector_[0-9]*'.format(args.split)
    sim_file = '{}_sim*'.format(args.split)
    file_pattern = [y_name_pos_file, imp_file, sim_file, imp_vector_file]
    file_names = ['y_name_pos.pkl', 'imp.pkl', 'sim.pkl', 'imp_vector.pkl']

    for i, pf in enumerate(zip(file_pattern, file_names)):
        pattern, fn = pf
        pattern_ = os.path.join(BERT_base_dir, pattern)
        file_n = 'imp_vector.h5' if args.dataset == 2 and i == 3 else fn
        file_name = os.path.join(BERT_output_dir, file_n)

        files = sorted(glob.glob(pattern_))
        print('found {} files for {}'.format(len(files), pattern_))
        if i == 0:
            Y_data, name_data, pos_data = [], [], []
            for file in files:
                data = load_features(file)
                # 'Y': Y, 'name': name, 'pos': pos
                Y_data = Y_data + data['Y']
                name_data = name_data + data['name']
                pos_data = pos_data + data['pos']
            save_features(file_name, {'Y': Y_data, 'name': name_data, 'pos': pos_data})
        else:
            data_all = []
            for file in files:
                data = load_features(file)
                data_all = data_all + data
            if args.dataset == 2 and i == 3:
                save_features_h5(file_name, data_all)
            else:
                save_features(file_name, data_all)
            print('saved in {}'.format(file_name))

    # convert to mat file
    convert2mat(BERT_output_dir, is_force=args.is_force)
    print('converted files in {}'.format(BERT_output_dir))

    base_dir = './DPP_scripts/weights'
    convert2mat_npy(base_dir, is_force=args.is_force)
    print('converted files in {}'.format(base_dir))
