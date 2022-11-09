import argparse

import easyfl
from client import FedSSLClient
from dataset import get_semi_supervised_dataset
from easyfl.datasets.data import CIFAR100
from easyfl.distributed import slurm
from model import get_model, BYOLNoEMA, BYOL, BYOLNoSG, BYOLNoEMA_NoSG
from server import FedSSLServer
import wandb
import torch
import numpy as np
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic =True
    torch.backends.cudnn.benchmark = False

def run():
    set_random_seed(0)
    dataset = 'cifar10'
    user_num = 4
    fed_ema = False
    personalized = True
    heterogeneous_network = {
        'f0000000': 'resnet18',
        'f0000001': 'resnet34',
        'f0000002': 'vgg9',
        'f0000003': 'alexnet',
        # 'f0000004': 'vgg9',
    }
    lamda = 1
    MD = False
    # whether you use individual model without aggregation
    semantic_align = True
    fed_para = False
    track_loss = True
    semantic_method = 'QR'
    aggregation_method = 'semantic'

    if fed_ema:
        personalized = False
        model = 'byol'
        name0 = 'fedema'
        update_encoder = 'dynamic_ema_online'
        update_predictor = 'dynamic_dapu'
    else:
        model = 'byol'
        update_encoder = 'online'
        update_predictor = 'global'

        name0 = model

    if personalized:
        name1 = '_local_'
    else:
        name1 = '_weights_agg_'

    if semantic_align:
        name3 = semantic_method + '_' + aggregation_method
    else:
        name3 = ''
    if fed_para:
        name3 = 'fed_para'

    name = name0+name1+name3+'_'+str(lamda)
    if MD:
        name += 'MD'
    # name += '_Non_IID'
    task_id = name
    wandb.init(project='11.9.Swift_EasyFL_{}'.format(dataset), name=name, entity='peilab')
    parser = argparse.ArgumentParser(description='FedSSL')
    parser.add_argument("--task_id", type=str, default=task_id)
    parser.add_argument("--dataset", type=str, default=dataset, help='options: cifar10, cifar100')
    parser.add_argument("--data_partition", type=str, default='dir', help='options: class, iid, dir')
    parser.add_argument("--dir_alpha", type=float, default=100000, help='alpha for dirichlet sampling')
    parser.add_argument('--model', default=model, type=str, help='options: byol, simsiam, simclr, moco, moco_v2')
    parser.add_argument('--encoder_network', default='resnet18', type=str,
                        help='network architecture of encoder, options: resnet18, resnet50')
    parser.add_argument('--predictor_network', default='2_layer', type=str,
                        help='network of predictor, options: 1_layer, 2_layer')
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--rounds', default=200, type=int)
    parser.add_argument('--num_of_clients', default=user_num, type=int)
    parser.add_argument('--clients_per_round', default=user_num, type=int)
    parser.add_argument('--class_per_client', default=10, type=int,
                        help='for non-IID setting, number of classes each client, based on CIFAR10')
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.032, type=float)
    parser.add_argument('--lr_type', default='cosine', type=str, help='cosine decay learning rate')
    parser.add_argument('--random_selection', action='store_true', help='whether randomly select clients')
    parser.add_argument('--aggregate_encoder', default='online', type=str, help='options: online, target')
    parser.add_argument('--update_encoder', default=update_encoder, type=str, help='options: online, target, both, none')
    parser.add_argument('--update_predictor', default=update_predictor, type=str, help='options: global, local, dapu')
    parser.add_argument('--dapu_threshold', default=0.4, type=float, help='DAPU threshold value')
    parser.add_argument('--weight_scaler', default=1.0, type=float, help='weight scaler for different class per client')
    parser.add_argument('--auto_scaler', default='y', type=str, help='use value to compute auto scaler')
    parser.add_argument('--auto_scaler_target', default=0.7, type=float,
                        help='target weight for the first time scaling')
    parser.add_argument('--encoder_weight', type=float, default=0,
                        help='for ema encoder update, apply on local encoder')
    parser.add_argument('--predictor_weight', type=float, default=0,
                        help='for ema predictor update, apply on local predictor')

    parser.add_argument('--test_every', default=10, type=int, help='test every x rounds')
    parser.add_argument('--save_model_every', default=100, type=int, help='save model every x rounds')
    parser.add_argument('--save_predictor', action='store_true', help='whether save predictor')

    parser.add_argument('--semi_supervised', default=0, help='whether to train with semi-supervised data')
    parser.add_argument('--label_ratio', default=0.0, type=float, help='percentage of labeled data')

    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--run_count', default=0, type=int)

    args = parser.parse_args()
    print("arguments: ", args)

    class_per_client = args.class_per_client
    if args.dataset == CIFAR100:
        class_per_client *= 10

    task_id = args.task_id
    if task_id == "":
        task_id = f"{args.dataset}_{args.model}_{args.encoder_network}_{args.data_partition}_" \
                  f"aggregate_{args.aggregate_encoder}_update_{args.update_encoder}_predictor_{args.update_predictor}_" \
                  f"run{args.run_count}"

    momentum_update = True
    if args.model == BYOLNoEMA:
        args.model = BYOL
        momentum_update = False
    elif args.model == BYOLNoEMA_NoSG:
        args.model = BYOLNoSG
        momentum_update = False

    image_size = 32

    config = {
        "task_id": task_id,
        "data": {
            "dataset": args.dataset,
            "num_of_clients": args.num_of_clients,
            "split_type": args.data_partition,
            "class_per_client": class_per_client,
            "data_amount": 1,
            "iid_fraction": 1,
            "min_size": 10,
            "alpha": args.dir_alpha,
        },
        "model": args.model,
        "test_mode": "test_in_server",
        "server": {
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "test_every": args.test_every,
            "save_model_every": args.save_model_every,
            "clients_per_round": args.clients_per_round,
            "random_selection": args.random_selection,
            "save_predictor": args.save_predictor,
            "test_all": True,
        },
        "client": {
            "drop_last": True,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            # application specific
            "model": args.model,
            "rounds": args.rounds,
            "gaussian": False,
            "image_size": image_size,

            "aggregate_encoder": args.aggregate_encoder,
            "update_encoder": args.update_encoder,
            "update_predictor": args.update_predictor,
            "dapu_threshold": args.dapu_threshold,
            "weight_scaler": args.weight_scaler,
            "auto_scaler": args.auto_scaler,
            "auto_scaler_target": args.auto_scaler_target,
            "random_selection": args.random_selection,
            "encoder_weight": args.encoder_weight,
            "predictor_weight": args.predictor_weight,
            "momentum_update": momentum_update,
            'personalized': personalized,
            'semantic_align': semantic_align,
            'semantic_method': semantic_method,
            'aggregation_method': aggregation_method,
            'lamda': lamda,
            'track_loss': track_loss,
            'MD': MD
        },
        'device': 'cuda',
        'resource_heterogeneous': {"grouping_strategy": ""},
        'personalized': personalized,
        'semantic_align': semantic_align,
        'semantic_method': semantic_method,
        'aggregation_method': aggregation_method,
        'encoder_network': args.encoder_network,
        'predictor_network': args.predictor_network,
        'fed_para': fed_para,
        'heterogeneous_network': heterogeneous_network,
        'MD': MD,
        'track_loss': track_loss,
        'test_dis': False
    }

    if args.gpu > 1:
        rank, local_rank, world_size, host_addr = slurm.setup()
        distribute_config = {
            "gpu": world_size,
            "distributed": {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "init_method": host_addr
            },
        }
        config.update(distribute_config)

    if args.semi_supervised:
        train_data, test_data, _ = get_semi_supervised_dataset(args.dataset,
                                                               args.num_of_clients,
                                                               args.data_partition,
                                                               class_per_client,
                                                               args.label_ratio)
        easyfl.register_dataset(train_data, test_data)

    model = get_model(args.model, args.encoder_network, args.predictor_network, fed_para)
    model.to('cuda')
    easyfl.register_model(model)
    easyfl.register_client(FedSSLClient)
    easyfl.register_server(FedSSLServer)
    easyfl.init(config, init_all=True)
    easyfl.run()


if __name__ == '__main__':
    run()
