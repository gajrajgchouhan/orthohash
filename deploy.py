from typing import List
from extract_data import ExperimentDataset
from PIL import Image
from torchvision import transforms


# python3 deploy.py -l logs/alexnet64_cifar10_1_100_0.0001_adam_1.0/orthohash_59209_000/ -device cpu

import torch

# parser = argparse.ArgumentParser()
# parser.add_argument("-l", required=True, help="training logdir")
# parser.add_argument("-m", type=float, default=0, help="threshold value for ternary")
# parser.add_argument("-device", type=str, default="cuda", help="device")

# args = parser.parse_args()

# logdir = args.l
# config = json.load(open(logdir + "/config.json"))

# config.update({"map_threshold": args.m})
# config.update({"device": args.device})

# device = torch.device(config["device"])
# io.init_save_queue()
# configs.seeding(config["seed"])

# logdir = config["logdir"]
# pprint(config)

# codebook = torch.load(f"{logdir}/outputs/codebook.pth", map_location=device).to(device)
# model, extrabit = prepare_model(config, device, codebook)
# model.load_state_dict(torch.load(f"{logdir}/models/best.pth", map_location=device))

# model.eval()
# meters = defaultdict(AverageMeter)
# dataset = ExperimentDataset()

# resize = config["dataset_kwargs"].get("resize", 0)
# crop = config["dataset_kwargs"].get("crop", 0)
# norm = config["dataset_kwargs"].get("norm", 2)

# resizec = 0 if resize == 32 else resize
# cropc = 0 if crop == 32 else crop

# transform = compose_transform("test", resizec, cropc, norm)

# i = 0

# for cls in range(10):
#     for idx in range(10):
#         label, data = dataset.__getitem__(cls, idx)
#         data = Image.fromarray(data)
#         transformed_data = transform(data)
#         transformed_data = transformed_data[None, ...]

#         if not i:
#             with torch.no_grad():
#                 logits, codes = model(transformed_data)
#                 print(logits)
#                 print(codes)
#                 plt.imshow(data)
#                 plt.show()
#                 i += 1
#                 # if np.argmax(logits) != label:
#                 #     print(cls, idx, np.argmax(logits), logits)
#                 # i += 1

# print("Waiting for save queue to end")
# io.join_save_queue()
# print(f"Done: {logdir}")


class Deploy:
    def __init__(self):
        self.dataset = ExperimentDataset()

    def get(self, model, transform: transforms.Compose, device, cls, idx) -> List[torch.Tensor]:
        model.eval()
        label, data = self.dataset.__getitem__(cls, idx)
        data = Image.fromarray(data)
        transformed_data: torch.Tensor = transform(data)
        transformed_data = transformed_data[None, ...]
        transformed_data = transformed_data.to(device)

        with torch.no_grad():
            logits, codes = model(transformed_data)
            return logits, codes

    def get_all(self, model, transform):
        for cls in range(10):
            for idx in range(10):
                self.get(model, transform, cls, idx)
