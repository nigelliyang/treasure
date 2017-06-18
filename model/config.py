#!/usr/bin/python



import argparse

parser = argparse.ArgumentParser()

#network parameter
parser.add_argument('--lstm_num_units',type=int, default=32,help='the num of lstm units')
parser.add_argument('--input_size',type=int,default=70,help='the input size of the network')

parser.add_argument('--batch_size',type=int,default=100,help='the batch size or the time step of the network input')
parser.add_argument('--batch_num',type=int,default=100000,help='the num of batch trainning turn')


#data parameter
parser.add_argument('--futures_num',type=int,default=7,help='the num of futures')
parser.add_argument('--infofield_num',type=int,default=10,help='one future info field num')




args = parser.parse_args()
args.input_size = args.futures_num * args.infofield_num





