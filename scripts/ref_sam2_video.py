# Dumps PyTorch (transformers) SAM 2 *video* activations on the synthetic clip (frames video_frame{i}.png, written by
# ref_sam3_video.py) for examples/sam2_video_parity.rs.   usage: python ref_sam2_video.py DIR [variant]
import os, sys, torch
import numpy as np
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor, Sam2ImageProcessorFast, Sam2VideoVideoProcessor
from safetensors.torch import save_file
torch.set_num_threads(16)
S = sys.argv[1]
V = sys.argv[2] if len(sys.argv) > 2 else "sam2.1-hiera-tiny"
D = os.path.expanduser(f"~/.cache/sam-rs/facebook--{V}")
N = 6
model = Sam2VideoModel.from_pretrained(D).eval()
proc = Sam2VideoProcessor(image_processor=Sam2ImageProcessorFast(), video_processor=Sam2VideoVideoProcessor())
cap = {}
oc, oe = model._prepare_memory_conditioned_features, model._encode_new_memory
def cond(*a, **k):
    r = oc(*a, **k); cap.setdefault("cond", []).append(r.detach().clone()); return r
def enc(*a, **k):
    r = oe(*a, **k); cap.setdefault("mem", []).append((r[0].detach().float().clone(), r[1].detach().float().clone())); return r
model._prepare_memory_conditioned_features, model._encode_new_memory = cond, enc
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(path):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).float()[None] / 255.0
    x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
    return (x - mean) / std
first = Image.open(f"{S}/video_frame0.png"); W, H = first.size
session = proc.init_video_session(video=None, inference_device="cpu", dtype=torch.float32)
out, pix = {}, []
with torch.no_grad():
    proc.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1, input_points=[[[[500.0, 375.0]]]], input_labels=[[[1]]], original_size=(H, W))
    for i in range(N):
        px = prep(f"{S}/video_frame{i}.png"); pix.append(px)
        o = model(inference_session=session, frame=px[0])
        out[f"f{i}_pred"] = o.pred_masks.float(); out[f"f{i}_obj"] = o.object_score_logits.float().reshape(1)
        st = session.output_dict_per_obj[0]; rec = st["cond_frame_outputs"].get(i) or st["non_cond_frame_outputs"].get(i)
        out[f"f{i}_ptr"] = rec["object_pointer"].float()
        print(i, "obj", float(o.object_score_logits.flatten()[0]), "mask px", int((o.pred_masks > 0).sum()), tuple(o.pred_masks.shape), flush=True)
for i, c in enumerate(cap["cond"]): out[f"cond{i + 1}"] = c.float()
for i, (m, p) in enumerate(cap["mem"]): out[f"mem{i}"], out[f"mempos{i}"] = m, p
out["pix"] = torch.cat(pix, 0)
sx, sy = 1024 / W, 1024 / H
out["point"] = torch.tensor([[500.0 * sx, 375.0 * sy]])
save_file({k: v.contiguous() for k, v in out.items()}, f"{S}/sam2_video_{V}_ref.safetensors")
print({k: tuple(v.shape) for k, v in out.items() if k.startswith(("cond1", "mem0", "f0_"))})
