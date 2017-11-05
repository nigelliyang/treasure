import argparse
import numpy as np
import os

cwd = os.getcwd()
parser = argparse.ArgumentParser()

# env parameters
parser.add_argument('--game', type=str, default='CartPole-v0',
                    help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')

# input parameters
parser.add_argument('--asset_num', type=int, default=11)
parser.add_argument('--info_num', type=int, default=3)
parser.add_argument('--input_size', type=int, default=33,
                    help='input size = asset num * info num')

# model parameters
parser.add_argument('--share_variable', type=bool, default=True,
                    help='wether use the independent lstm')
parser.add_argument('--dropout', type=bool, default=True,
                    help='use droup_out')
parser.add_argument('--short_sight', type=bool, default=False,
                    help='lookfoward only a few steps')
parser.add_argument('--lstm1_unit', type=int, default=15,
                    help='the output size of indepent_lstm1')
parser.add_argument('--lstm_unit', type=int, default=64,
                    help='the output size of lstm')
parser.add_argument('--state_feature_num', type=int, default=64,
                    help='the num of feature extracted from both state and allocation')
parser.add_argument('--alloRNN_fc0_output_dim', type=int, default=64,
                    help='the dimention of fc0 of the direct allocation RNN')
parser.add_argument('--keep_prob', type=float, default=0.5,
                    help='keep probability in droup out')
parser.add_argument('--short_sight_step', type=int, default=30,
                    help='lookfoward steps')
parser.add_argument('--entropy_beta', type=float, default=0.01)

# finance parameters
parser.add_argument('--gamma', type=float, default=0.9999,
                    help='daily discount rate, 0.9999 equals to capital return rate 103.7% per year')
parser.add_argument('--risk_beta', type=float, default=0.1,
                    help='the multiplier for gauss mean l1 loss, represents the risk preference, greater risk_beta means safer')
# train parameters
parser.add_argument('--local_t_max', type=int, default=32,
                    help='async interval of a single thread. In fact it is the same as batch size')
parser.add_argument('--max_time_step', type=int, default=10 * 10 ** 5)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--thread_num', type=int, default=4)
parser.add_argument('--sigma', type=float, default=1.0,
                    help='init sigma as sigme*I')

# log parameters
# the directory saved ckpt is joined by two parts
# the parent dir "checkpoint" and the identical dir "test_name"
# this design is for the convenient of various test
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--use_checkpoint', type=bool, default='False')
parser.add_argument('--test_steps', type=int, default=5000)
parser.add_argument('--test_name', type=str, default='latest_test')

# gradient applier parameters
parser.add_argument('--rmsp_alpha', type=float, default=0.99)
parser.add_argument('--rmsp_epsilon', type=float, default=0.1)
parser.add_argument('--grad_norm_clip', type=float, default=40.0)

args = parser.parse_args()

# additional parameters
args.action_size = args.asset_num + 1
args.use_checkpoint = False
args.gauss_sigma = args.sigma * np.eye(args.action_size - 1)
args.only_train_positive = True

args.mean = 0
args.std = 0
