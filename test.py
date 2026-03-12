from ignite.distributed import model_name

from envsocial import EnvSocial
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict
import numpy as np
import random, torch
import setproctitle

torch.set_num_threads(8)
torch.cuda.set_device(6)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/ucy.yaml')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--data_dict_path', default='/mnt/d/EnvSocial/dataset_built/ucy.pkl',help='Path to the training data file (PKL format) that contains all data required for model training.')
    parser.add_argument('--model_name', default='/mnt/d/EnvSocial/experiments/UCY.pkl',help="PKL file path of pre-trained model for testing.")
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for k, v in vars(args).items():
        config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset
    config['data_dict_path'] = args.data_dict_path
    #
    print(f"Current data path: {config['data_dict_path']}")
    config = EasyDict(config)
    agent = EnvSocial(config)


    model_name = args.model_name



    agent.load_current_model(model_name)

    keyattr = ["lr", "ft_lr", "data_dict_path", "data_dict_path", "epochs", "total_epochs", "dataset", "batch_size",
               "diffnet", "seed"]

    keys = {}
    for k, v in config.items():
        if k in keyattr:
            keys[k] = v

    pprint(keys)

    agent._test()



if __name__ == '__main__':
    main()
