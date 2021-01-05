import argparse
from args import init_parser, post_processing
import numpy as np
from envs import make_env

# find the carla module 
import os
import math 
import random
import time
import torch
import shutil

parser = argparse.ArgumentParser(description='SPC')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()
args = post_processing(args)


CARLA8_TIMEOUT = 100000
CARLA9_TIMEOUT = 20.0

def init_dirs(dir_list):
    for path in dir_list:
        if not os.path.isdir(path):
            os.makedirs(path)


def setup_dirs(args):
    save_path = args.save_path
    model_path = os.path.join(save_path, 'model')
    optim_path = os.path.join(save_path, 'optimizer')
    init_dirs([model_path, optim_path])


def create_carla9_env(args):
    from envs.CARLA.carla9 import World
    import carla # here the carla is installed by pip/conda
    try:
        import glob
        import sys
        sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

    client = carla.Client("localhost", args.port)
    client.set_timeout(CARLA9_TIMEOUT)
    carla_world = client.get_world()
    settings = carla_world.get_settings()
    settings.synchronous_mode = True
    client.get_world().apply_settings(settings)
    env = World(args, carla_world)
    return env


def main():
    if not args.resume:
        if args.debug:
            print("run spc in debug mode")
            if os.path.isdir(args.save_path):
                print(args.save_path)
                shutil.rmtree(args.save_path)
            setup_dirs(args)
            shutil.copytree('scripts', os.path.join(args.save_path, 'scripts'))
        elif os.path.isdir(args.save_path):
            print("the save path has already existed!")
            exit(0)
        else:
            setup_dirs(args)
            shutil.copytree('scripts', os.path.join(args.save_path, 'scripts'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = None # placeholder
    if 'carla9' in args.env:
        # select CARLA v0.9.x as the platform
        env = create_carla9_env(args)
    elif 'carla8' in args.env:
        # select CARLA v0.8.x as the platform
        from envs.CARLA.carla.client import make_carla_client
        from envs.CARLA.carla_env import CarlaEnv
        client = make_carla_client('localhost', args.port, CARLA8_TIMEOUT)
        env = CarlaEnv(client, args)
    else:
        # select PyTorcs or GTAV as the platform
        env = make_env(args)

    if args.eval:
        from evaluate import evaluate_policy
        evaluate_policy(args, env)
    else:
        from train import train_policy
        train_policy(args, env)


if __name__ == '__main__':
    main()

