import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='acer',
                        help='algorithm to use: acer(only supported)')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--rms-alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--clip', type=float, default=10.0,
                        help='clip value in importance weight truncation')
    parser.add_argument('--no-trust-region', action='store_true', default=False,
                        help='disable TRPO')
    parser.add_argument('--delta', type=float, default=1.0,
                        help='delta value in TRPO')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='alpha value in average model for TRPO')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=10.0,
                        help='max norm of gradients (default: 10.0)')
    parser.add_argument('--seed', type=int, default=1122,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=4,
                        help='how many training CPU processes to use (default: 4) '
                             'Note that ACER requires a LOT of RAM!')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in ACER (default: 20)')
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='size of replay buffer')
    parser.add_argument('--replay-ratio', type=int, default=4,
                        help='number of replay ratio in ACER (default: 4)')
    parser.add_argument('--replay-start', type=int, default=10000,
                        help='number of saved memories to start off-policy learning in ACER (default: 10000)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
