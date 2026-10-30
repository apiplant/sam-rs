# Dumps PyTorch (transformers) SAM 3 *tracker* activations for a point prompt and a box prompt, for
# examples/sam3_tracker_parity.rs.   usage: python ref_sam3_tracker.py DIR   (DIR holds truck.jpg)
import os, sys, torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3Processor
from safetensors.torch import save_file
torch.set_num_threads(16)
D = os.path.expanduser("~/.cache/sam-rs/facebook--sam3")
S = sys.argv[1]
vm = Sam3VideoModel.from_pretrained(D).eval()
proc = Sam3Processor.from_pretrained(D)
img = Image.open(f"{S}/truck.jpg").convert("RGB")
W, H = img.size
px = proc(images=img, text="truck", return_tensors="pt")["pixel_values"]
tr = vm.tracker_model
out = {}
with torch.no_grad():
    tok = vm.detector_model.vision_encoder.backbone(px).last_hidden_state  # [1, 5184, 1024]
    nchw = tok.view(1, 72, 72, -1).permute(0, 3, 1, 2)
    fpn, pos = vm.tracker_neck(nchw)
    for i, f in enumerate(fpn): out[f"t_fpn{i}"] = f
    f0 = tr.mask_decoder.conv_s0(fpn[0]); f1 = tr.mask_decoder.conv_s1(fpn[1])
    emb = fpn[2] + tr.no_memory_embedding.view(1, -1, 1, 1)
    out["t_s0"], out["t_s1"], out["t_emb"] = f0, f1, emb
    ipe = tr.get_image_wide_positional_embeddings()
    out["t_pe"] = ipe

    def run(name, points=None, labels=None, boxes=None, multimask=True):
        sparse, dense = tr.prompt_encoder(input_points=points, input_labels=labels, input_boxes=boxes, input_masks=None)
        masks, iou, tokens, obj = tr.mask_decoder(image_embeddings=emb, image_positional_embeddings=ipe, sparse_prompt_embeddings=sparse,
                                                  dense_prompt_embeddings=dense, multimask_output=multimask, high_resolution_features=[f0, f1])
        out[f"{name}_sparse"], out[f"{name}_masks"], out[f"{name}_iou"], out[f"{name}_obj"] = sparse, masks, iou, obj
        print(name, "iou", iou.flatten().tolist(), "obj", obj.flatten().tolist(), "masks", tuple(masks.shape))

    sx, sy = 1008 / W, 1008 / H
    pt = torch.tensor([[[[500 * sx, 375 * sy]]]], dtype=torch.float32)           # [B=1, P=1, N=1, 2], 1008-space
    lab = torch.tensor([[[1]]], dtype=torch.int32)
    out["pt"], out["pt_label"] = pt, lab.to(torch.int64)
    run("pt", pt, lab, None, True)
    run("pt1", pt, lab, None, False)                                              # single mask (dynamic stability)
    bx = torch.tensor([[[86 * sx, 282 * sy, 1710 * sx, 851 * sy]]], dtype=torch.float32)  # [1, 1, 4]
    out["bx"] = bx
    run("bx", None, None, bx, True)
save_file({k: v.contiguous().float() if v.dtype.is_floating_point else v.contiguous() for k, v in out.items()}, f"{S}/sam3_tracker_ref.safetensors")
print({k: tuple(v.shape) for k, v in out.items()})
