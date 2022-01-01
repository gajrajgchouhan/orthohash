import argparse
import json

from scripts import test_hashing

parser = argparse.ArgumentParser()
parser.add_argument("-l", required=True, help="training logdir")
parser.add_argument("-m", type=float, default=0, help="threshold value for ternary")
parser.add_argument("-device", type=str, default="cuda", help="device")
parser.add_argument('-l', required=True, help='training logdir')
parser.add_argument('-m', type=float, default=0, help='threshold value for ternary')

args = parser.parse_args()

logdir = args.l
config = json.load(open(logdir + "/config.json"))

config.update({"map_threshold": args.m})
config.update({"device": args.device})
config.update({
    'map_threshold': args.m
})

test_hashing.main(config)
