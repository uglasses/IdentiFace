import argparse
import sys
import os
from dotenv import load_dotenv
load_dotenv()

def get_args():

    parser = argparse.ArgumentParser(add_help=False)
    ## Model ##
    parser.add_argument('--expt_name', type=str, default="segface")
    parser.add_argument('--dataset', type=str, default="lapa")
    parser.add_argument('--backbone', type=str, default="segface")
    parser.add_argument('--model', type=str, default="swin_base")
    ## Optimizer & Scheduler ##
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_schedule', type=str, default='10,15')
    parser.add_argument('--lr_schedule_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    ## Loss Functions ##
    parser.add_argument('--input_resolution', type=int, default=512)
    ## Dataloader ## 
    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    ## Logging & Saving ##
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--save_all_states', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()

    args.lr_schedule = [int(x) for x in args.lr_schedule.split(',')]
    args.ckpt_path = os.path.join(os.getenv('LOG_PATH'), args.ckpt_path)

    return args