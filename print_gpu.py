import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if  torch.cuda.is_available():
    print(f"Running job with device: {torch.cuda.get_device_name(device=device)}")
else:
    print("Running job on cpu")