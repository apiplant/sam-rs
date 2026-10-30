# Dumps PyTorch (transformers) SAM 3 activations for tests/examples/sam3_parity.rs.
# usage: python ref_sam3.py DIR   (DIR holds truck.jpg; writes DIR/sam3_ref.safetensors)
# needs: pip install torch torchvision transformers safetensors pillow  (CPU torch is enough)
import sys, time, torch, numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from safetensors.torch import save_file
torch.set_num_threads(16)
import os
D = os.path.expanduser("~/.cache/sam-rs/facebook--sam3")
S = sys.argv[1]
model = Sam3Model.from_pretrained(D).eval()
proc = Sam3Processor.from_pretrained(D)
img = Image.open(f"{S}/truck.jpg").convert("RGB")
inputs = proc(images=img, text="truck", return_tensors="pt")
print({k: (tuple(v.shape), v.dtype) for k, v in inputs.items()}, flush=True)
out = {}
with torch.no_grad():
    t = time.time()
    v = model.get_vision_features(pixel_values=inputs["pixel_values"])
    print("vision", time.time() - t, flush=True)
    out["pixel_values"] = inputs["pixel_values"]
    out["input_ids"] = inputs["input_ids"].to(torch.int64)
    out["attention_mask"] = inputs["attention_mask"].to(torch.int64)
    out["backbone_last"] = v.last_hidden_state
    for i, f in enumerate(v.fpn_hidden_states): out[f"fpn{i}"] = f
    for i, f in enumerate(v.fpn_position_encoding): out[f"fpn_pos{i}"] = f
    tf = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    out["text_hidden"] = tf.last_hidden_state
    out["text_proj"] = tf.pooler_output
    t = time.time()
    r = model(vision_embeds=v, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    print("rest", time.time() - t, flush=True)
    out["pred_logits"] = r.pred_logits; out["pred_boxes"] = r.pred_boxes; out["presence_logits"] = r.presence_logits
    out["pred_masks"] = r.pred_masks
    if r.semantic_seg is not None: out["semantic_seg"] = r.semantic_seg
    out["enc_last"] = r.encoder_hidden_states[-1] if r.encoder_hidden_states else torch.zeros(1)
# --- geometry (box) prompt: text "visual" + one positive exemplar box, normalized (cx, cy, w, h)
W, H = img.size
x0, y0, x1, y1 = 86.0, 282.0, 1710.0, 851.0
boxes = torch.tensor([[[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]]], dtype=torch.float32).reshape(1, 1, 4)
labels = torch.tensor([[1]])
vis_in = proc(images=img, text="visual", return_tensors="pt")
with torch.no_grad():
    geo = model.geometry_encoder(box_embeddings=boxes, box_mask=torch.ones(1, 1, dtype=torch.bool), box_labels=labels,
                                 img_feats=v.fpn_hidden_states[:-1], img_pos_embeds=v.fpn_position_encoding[:-1])
    rg = model(vision_embeds=v, input_ids=vis_in["input_ids"], attention_mask=vis_in["attention_mask"], input_boxes=boxes, input_boxes_labels=labels)
out["geo_boxes"] = boxes; out["geo_labels"] = labels.to(torch.int64)
out["geo_ids"] = vis_in["input_ids"].to(torch.int64); out["geo_mask"] = vis_in["attention_mask"].to(torch.int64)
out["geo_feats"] = geo.last_hidden_state
out["geo_logits"] = rg.pred_logits; out["geo_boxes_out"] = rg.pred_boxes; out["geo_presence"] = rg.presence_logits; out["geo_masks"] = rg.pred_masks
print("geo prompt top scores:", (rg.pred_logits.sigmoid() * rg.presence_logits.sigmoid()).flatten().topk(3).values.tolist())
save_file({k: v.contiguous().float() if v.dtype.is_floating_point else v.contiguous() for k, v in out.items()}, f"{S}/sam3_ref.safetensors")
print({k: tuple(v.shape) for k, v in out.items()})
print("top scores:", r.pred_logits.sigmoid().flatten().topk(5).values.tolist(), "presence", r.presence_logits.sigmoid().flatten().tolist())
