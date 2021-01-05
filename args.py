def set_env_params(parser):
    '''
    set the simulator environment parameters
    ----- preset weather index in CARLA8 -----
    0 - Default; 1 - ClearNoon; 2 - CloudyNoon; 3 - WetNoon; 4 - WetCloudyNoon; 5 - MidRainyNoon; 6 - HardRainNoon; 7 - SoftRainNoon
    8 - ClearSunset; 9 - CloudySunset; 10 - WetSunset; 11 - WetCloudySunset; 12 - MidRainSunset; 13 - HardRainSunset; 14 - SoftRainSunset
    '''
    parser.add_argument('--vehicle-num', type=int, default=120)
    parser.add_argument('--weather-id', type=int, default=1)
    parser.add_argument('--ped-num', type=int, default=0)
    parser.add_argument('--notify', type=bool, default=False)
    parser.add_argument('--autopilot', action='store_true')
    parser.add_argument('--monitor-video-dir', type=str, default="monitor_record")
    parser.add_argument('--imitation', action='store_true', help="whether use expert demonstration for imitation learning")
    parser.add_argument("--steer-clip", type=float, default=0.0, help="threshold to clip small steering values")
    parser.add_argument('--frame-height', type=int, default=256)
    parser.add_argument('--frame-width', type=int, default=256)

def set_train_params(parser):
    '''
    set the parameters for training model
    '''
    # part1: basic params
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path', type=str, default='demo', help="output path to save evaluation results")
    parser.add_argument('--port', type=int, default=6666)
    parser.add_argument('--num-train-steps', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=4000000000)
    parser.add_argument('--max-eval-step', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', help='to use debug mode')
    parser.add_argument('--thr', type=int, default=2)

    # part2: supervision signals combat
    parser.add_argument('--no-supervision', action='store_true') # deprecated in IPC
    parser.add_argument('--use-depth', action='store_true')
    parser.add_argument('--use-guidance', action='store_true')
    parser.add_argument('--use-collision', action='store_true')
    parser.add_argument('--use-offroad', action='store_true')
    parser.add_argument('--use-speed', action='store_true')
    parser.add_argument('--use-offlane', action="store_true")
    parser.add_argument('--use-detection', action='store_true')
    parser.add_argument('--use-3d-detection', action='store_true')
    parser.add_argument('--use-orientation', action='store_true')
    parser.add_argument('--use-collision-other', action='store_true')
    parser.add_argument('--use-colls-with', action='store_true')  

    # part3: parameters for action selection strategy
    parser.add_argument('--safe-length-collision', type=int, default=5)
    parser.add_argument('--safe-length-offroad', type=int, default=5)
    parser.add_argument('--safe-length-offlane', type=int, default=5)
    parser.add_argument('--sample-type', type=str, default='binary')
    parser.add_argument('--sample-with-offroad', action='store_true')
    parser.add_argument('--sample-with-collision', action='store_true')
    parser.add_argument('--sample-with-offlane', action="store_true")
    parser.add_argument('--speed-threshold', type=float, default=15)
    parser.add_argument('--time-decay', type=float, default=0.97)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--CEM', action='store_true', help='using CEM for action sample')
    parser.add_argument('--SAS', action='store_true', help="whether to enable sequential action sampling")
    parser.add_argument('--SAS_thred', type=int, default=5, help="number of action candidates remaining after the first stage of SAS")

    # part4: training params
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR', help='learning rate')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--frame-history-len', type=int, default=3)
    parser.add_argument('--pred-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--save-path', type=str, default='spc')
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--num-total-act', type=int, default=2)
    parser.add_argument('--epsilon-frames', type=int, default=50000)
    parser.add_argument('--learning-freq', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=40000000, help="maximum step in training")
    parser.add_argument('--braking', action='store_true', help="whether use braking signal")
    # return parser


def set_model_params(parser):
    # set model configs
    # parser.add_argument('--detach-seg', action='store_true', help='detach the feature map for segmentation prediction')
    parser.add_argument('--pretrain-model', type=str, default='pretrain/spc_p10.pth', help='the base pretrain model for initializing model')
    parser.add_argument('--expert-bar', type=int, default=50)
    parser.add_argument('--expert-ratio', type=float, default=0.05)
    parser.add_argument('--bin-divide', type=list, default=[5, 5])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--drn-model', type=str, default='dla46x_c')
    parser.add_argument('--classes', type=int, default=6)
    # return parser


def set_common_params(parser):
    # set common parameters not related to model/env/training configs
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--save-record', action='store_true', help="whether to save visulization of real-time observations")
    parser.add_argument('--logger_path', type=str, default="wandb_log.txt")
    parser.add_argument('--env', type=str, default='carla')
    parser.add_argument('--server', type=bool, default=False)
    parser.add_argument('--wandb', action="store_true", help="whether to use wandb for train logging")


def init_parser(parser):
    set_env_params(parser)
    set_train_params(parser)
    set_model_params(parser)
    set_common_params(parser)

def post_processing(args):
    import os
    import torchvision.transforms as transforms
    args.env = args.env.lower()
    args.save_path = '{}vehicle/{}'.format(args.vehicle_num, args.id)
    if args.debug:
        args.save_path = args.save_path + '_debug'

    if 'carla' in args.env:
        args.save_path = os.path.join('exps', args.save_path)
    else:
        args.save_path = os.path.join('gta', args.save_path)

    args.sync = 'torcs' in args.env or 'carla' in args.env

    # transform on the original image / 255
    args.trans = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # to recover the image feature map
    args.invtrans = transforms.Compose([ 
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
    ])
    return args
