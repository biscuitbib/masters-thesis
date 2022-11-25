import torch
from thesisproject.models import UNet
from collections import OrderedDict

checkpoint_path = "/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_3557/checkpoints/epoch=37-step=9500.ckpt"

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

label_keys = ["Lateral femoral cart.",
                "Lateral meniscus",
                "Lateral tibial cart.",
                "Medial femoral cartilage",
                "Medial meniscus",
                "Medial tibial cart.",
                "Patellar cart.",
                "Tibia"]
unet = UNet(1, 9, 384, class_names=label_keys)

#print(unet.state_dict().keys())
#print([(key.split(".", 1)[1], val) for key, val in checkpoint["state_dict"].items()])
modified_key_vals = [(key.split(".", 1)[1], val) for key, val in checkpoint["state_dict"].items()]
modified_state_dict = OrderedDict(modified_key_vals)
print(modified_state_dict.keys() == unet.state_dict().keys())
#unet.load_state_dict(checkpoint["state_dict"])