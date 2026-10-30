# Dumps PyTorch (transformers) SAM 2 activations for a point and a box prompt, for examples/sam2_parity.rs.
# usage: python ref_sam2.py DIR [variant]   (DIR holds truck.jpg; variant e.g. sam2.1-hiera-tiny)
import os, sys, torch
import numpy as np
from PIL import Image
from transformers import Sam2Model
from safetensors.torch import save_file
torch.set_num_threads(16)
S = sys.argv[1]
V = sys.argv[2] if len(sys.argv) > 2 else "sam2.1-hiera-tiny"
D = os.path.expanduser(f"~/.cache/sam-rs/facebook--{V}")
model = Sam2Model.from_pretrained(D).eval()
img = Image.open(f"{S}/truck.jpg").convert("RGB")
W, H = img.size
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None] / 255.0
x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
px = (x - mean) / std
out = {"pixel_values": px}
with torch.no_grad():
    vo = model.vision_encoder.backbone(px)
    for i, h in enumerate(vo.intermediate_hidden_states): out[f"stage{i}"] = h            # NHWC
    vis = model.vision_encoder(px)
    for i, f in enumerate(vis.fpn_hidden_states): out[f"fpn{i}"] = f
    for i, f in enumerate(vis.fpn_position_encoding): out[f"fpn_pos{i}"] = f
    md = model.mask_decoder
    f0, f1 = md.conv_s0(vis.fpn_hidden_states[0]), md.conv_s1(vis.fpn_hidden_states[1])
    emb = vis.fpn_hidden_states[2] + model.no_memory_embedding.view(1, -1, 1, 1)
    out["s0"], out["s1"], out["emb"] = f0, f1, emb
    ipe = model.get_image_wide_positional_embeddings(); out["pe"] = ipe
    sx, sy = 1024 / W, 1024 / H
    def run(name, points=None, labels=None, boxes=None, multimask=True):
        sparse, dense = model.prompt_encoder(input_points=points, input_labels=labels, input_boxes=boxes, input_masks=None)
        masks, iou, tok, obj = md(image_embeddings=emb, image_positional_embeddings=ipe, sparse_prompt_embeddings=sparse,
                                  dense_prompt_embeddings=dense, multimask_output=multimask, high_resolution_features=[f0, f1])
        out[f"{name}_sparse"], out[f"{name}_masks"], out[f"{name}_iou"], out[f"{name}_obj"] = sparse, masks, iou, obj
        print(name, "iou", iou.flatten().tolist(), "obj", obj.flatten().tolist(), "masks", tuple(masks.shape))
    pt = torch.tensor([[[[500 * sx, 375 * sy]]]], dtype=torch.float32); lab = torch.tensor([[[1]]], dtype=torch.int32)
    out["pt"] = pt; run("pt", pt, lab, None, True); run("pt1", pt, lab, None, False)
    bx = torch.tensor([[[86 * sx, 282 * sy, 1710 * sx, 851 * sy]]], dtype=torch.float32); out["bx"] = bx
    run("bx", None, None, bx, True)
save_file({k: v.contiguous().float() if v.dtype.is_floating_point else v.contiguous() for k, v in out.items()}, f"{S}/sam2_{V}_ref.safetensors")
print({k: tuple(v.shape) for k, v in out.items() if k in ("stage0", "stage1", "stage2", "stage3", "fpn0", "fpn1", "fpn2", "emb", "pt_masks")})
