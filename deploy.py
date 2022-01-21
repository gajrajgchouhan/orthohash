import argparse
import json
from pprint import pprint
from collections import defaultdict

import numpy as np

from extract_data import ExperimentDataset
from utils.misc import AverageMeter
from PIL import Image
from matplotlib import pyplot as plt

# python3 deploy.py -l logs/alexnet64_cifar10_1_100_0.0001_adam_1.0/orthohash_59209_000/ -device cpu

import torch
import configs
from scripts.train_hashing import prepare_model
from utils import io
from configs import compose_transform


parser = argparse.ArgumentParser()
parser.add_argument("-l", required=True, help="training logdir")
parser.add_argument("-m", type=float, default=0, help="threshold value for ternary")
parser.add_argument("-device", type=str, default="cuda", help="device")

args = parser.parse_args()

logdir = args.l
config = json.load(open(logdir + "/config.json"))

config.update({"map_threshold": args.m})
config.update({"device": args.device})

device = torch.device(config["device"])
io.init_save_queue()
configs.seeding(config["seed"])

logdir = config["logdir"]
pprint(config)

codebook = torch.load(f"{logdir}/outputs/codebook.pth", map_location=device).to(device)
model, extrabit = prepare_model(config, device, codebook)
model.load_state_dict(torch.load(f"{logdir}/models/best.pth", map_location=device))

model.eval()
meters = defaultdict(AverageMeter)
dataset = ExperimentDataset()

resize = config["dataset_kwargs"].get("resize", 0)
crop = config["dataset_kwargs"].get("crop", 0)
norm = config["dataset_kwargs"].get("norm", 2)

resizec = 0 if resize == 32 else resize
cropc = 0 if crop == 32 else crop

transform = compose_transform("test", resizec, cropc, norm)

i = 0

for cls in range(10):
    for idx in range(10):
        label, data = dataset.__getitem__(cls, idx)
        data = Image.fromarray(data)
        transformed_data = transform(data)
        transformed_data = transformed_data[None, ...]

        if not i:
            with torch.no_grad():
                logits, codes = model(transformed_data)
                if np.argmax(logits) != label:
                    print(cls, idx, np.argmax(logits), logits)
                # i += 1

print("Waiting for save queue to end")
io.join_save_queue()
print(f"Done: {logdir}")
