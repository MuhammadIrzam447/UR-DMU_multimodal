import argparse
from random import seed
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--output_path', type = str, default = 'outputs/')
    parser.add_argument('--root_dir', type = str, default = 'outputs/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--modal', type = str, default = 'multimodal',choices = ["rgb,flow,both"])
    parser.add_argument('--model_path', type = str, default = 'models/')
    parser.add_argument('--lr', type = str, default = '[0.0001]*3000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--num_segments', type = int, default = 32)
    parser.add_argument('--seed', type = int, default = 7777, help = 'random seed (-1 for no manual seed) 2022 default')
    parser.add_argument('--model_file', type = str, default = "trans_{}.pkl".format(seed), help = 'the path of pre-trained model file')
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--missing_modality', action='store_true', help='Set this flag for multimodal missing modality evaluation')
    # parser.add_argument('--single_branch', action='store_true', help='Set this flag if single_branch training and evaluation is required')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
