import argparse
import numpy as np
import os

cwd = os.getcwd()
parser = argparse.ArgumentParser()

# env parameters
parser.add_argument('--game', type=str, default='CartPole-v0',
                    help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
parser.add_argument('--use_gpu', type=bool, default=False)

# input parameters
parser.add_argument('--asset_num', type=int, default=5)
parser.add_argument('--info_num', type=int, default=12)
parser.add_argument('--input_size', type=int, default=60,
                    help='input size = asset num * info num')

# model parameters
parser.add_argument('--lstm1_unit', type=int, default=128,
                    help='the output size of lstm1')
parser.add_argument('--lstm_unit', type=int, default=128,
                    help='the output size of lstm')
parser.add_argument('--state_feature_num', type=int, default=64,
                    help='the num of feature extracted from both state and allocation')
parser.add_argument('--entropy_beta', type=float, default=0.01)

# finance parameters
parser.add_argument('--gamma', type=float, default=0.9999,
                    help='daily discount rate, 0.9999 equals to capital return rate 103.7% per year')
parser.add_argument('--risk_beta', type=float, default=0.5,
                    help='the multiplier for gauss mean l1 loss, represents the risk preference, greater risk_beta means safer')
# train parameters
parser.add_argument('--local_t_max', type=int, default=32,
                    help='async interval of a single thread. In fact it is the same as batch size')
parser.add_argument('--max_time_step', type=int, default=10*10**7)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--thread_num', type=int, default=6)

# log parameters
parser.add_argument('--log_interval', type=int, default=2000,
                    help='log interval')
parser.add_argument('--performance_log_interval', type=int, default=1000,
                    help='performance log interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--log_file', type=str, default='tmp/a3c_log')
parser.add_argument('--use_chechpoint', type=bool, default=False)

# gradient applier parameters
parser.add_argument('--rmsp_alpha', type=float, default=0.99)
parser.add_argument('--rmsp_epsilon', type=float, default=0.1)
parser.add_argument('--grad_norm_clip', type=float, default=40.0)

args = parser.parse_args()

# additional parameters
args.action_size = 7
args.gauss_sigma = 0.5*np.eye(args.action_size-1)

