import argparse
import json
import neptune.new as neptune

run = neptune.init(
    project="gajrajgchouhan/HashingProject",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1N2ViYzI5Yy0zY2U4LTQ1ZDItYmVmOS04OGE2Mjg3MjA1MWYifQ==",
)  # your credentials


from scripts import test_hashing

parser = argparse.ArgumentParser()
parser.add_argument("-l", required=True, help="training logdir")
parser.add_argument("-m", type=float, default=0, help="threshold value for ternary")
parser.add_argument("-device", type=str, default="cuda", help="device")
parser.add_argument("-one", type=bool, default=False, help="visualise one example")

args = parser.parse_args()

logdir = args.l
config = json.load(open(logdir + "/config.json"))

config.update({"map_threshold": args.m})
config.update({"device": args.device})
config.update({"one": args.one})
print(args.one)

test_hashing.main(config, run)
