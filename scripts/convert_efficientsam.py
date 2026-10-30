# Converts an EfficientSAM .pt checkpoint to safetensors.  usage: python convert_efficientsam.py IN.pt OUT.safetensors
# Uses torch.load(weights_only=True), a restricted unpickler that only builds tensors and plain containers, so a
# malicious checkpoint cannot run code.
import sys, torch
from safetensors.torch import save_file
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
save_file({k: v.contiguous().float() for k, v in sd.items()}, sys.argv[2])
print(f"{len(sd)} tensors -> {sys.argv[2]}")
