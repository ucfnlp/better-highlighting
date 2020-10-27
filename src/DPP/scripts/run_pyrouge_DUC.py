# -*- coding: utf-8 -*-

from pyrouge import Rouge155
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--system_dir', type=str)
    parser.add_argument('--ref_dir', type=str)
    parser.add_argument('--score_dir', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    r = Rouge155()

    r.system_dir = args.system_dir
    r.model_dir = args.ref_dir
    r.system_filename_pattern = 'd3(\d+).sum'
    r.model_filename_pattern = 'd3#ID#.sum'

    output = r.convert_and_evaluate()
    print(output)
    data = r.output_to_dict(output)

    score_file = args.score_dir
    with open(score_file, 'w') as out_file:
        out_file.write('%s ' % data['rouge_1_precision'])
        out_file.write('%s ' % data['rouge_1_recall'])
        out_file.write('%s ' % data['rouge_1_f_score'])
        out_file.write('%s ' % data['rouge_2_precision'])
        out_file.write('%s ' % data['rouge_2_recall'])
        out_file.write('%s ' % data['rouge_2_f_score'])
        out_file.write('%s ' % data['rouge_su4_precision'])
        out_file.write('%s ' % data['rouge_su4_recall'])
        out_file.write('%s ' % data['rouge_su4_f_score'])
        out_file.write('%s ' % data['rouge_l_precision'])
        out_file.write('%s ' % data['rouge_l_recall'])
        out_file.write('%s ' % data['rouge_l_f_score'])
