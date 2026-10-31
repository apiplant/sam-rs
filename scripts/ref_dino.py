# Dumps PyTorch (transformers) Grounding DINO activations for examples/dino_parity.rs.
# usage: python ref_dino.py DIR [MODEL_DIR]   (DIR holds truck.jpg; writes DIR/dino_ref.safetensors)
# needs: pip install torch transformers safetensors pillow  (CPU torch is enough)
import os, sys, torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from safetensors.torch import save_file
# transformers >= 5.2x casts the sinusoidal embedding back to the dtype of its input, and the text position ids are
# integers, which truncates the embedding to 0s and 1s (a regression: 5.0 and the original GroundingDINO keep floats).
from transformers.models.grounding_dino import modeling_grounding_dino as _m
if hasattr(_m, "encode_sinusoidal_position_embedding"):
    _orig = _m.encode_sinusoidal_position_embedding
    _m.encode_sinusoidal_position_embedding = lambda t, *a, **k: _orig(t.float() if not t.is_floating_point() else t, *a, **k)
S = sys.argv[1]
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/sam-rs/IDEA-Research--grounding-dino-tiny")
PROMPT = "a wheel. a truck."
proc = AutoProcessor.from_pretrained(D)
model = GroundingDinoForObjectDetection.from_pretrained(D).eval()
img = Image.open(f"{S}/truck.jpg").convert("RGB")
inp = proc(images=img, text=PROMPT, return_tensors="pt")
out = {"pixel_values": inp["pixel_values"][0].contiguous(), "input_ids": inp["input_ids"][0].to(torch.int32)}
hooks = []
def grab(name, idx=None):
    def f(_m, _i, o):
        t = o if idx is None else o[idx]
        out[name] = t.detach().float().contiguous()
    return f
conv = model.model.backbone.conv_encoder
def swin_hook(m, i, o):
    for k, f in enumerate(o):
        out[f"swin{k+2}"] = f[0].detach().float().contiguous()
hooks.append(conv.register_forward_hook(swin_hook))
hooks.append(model.model.text_projection.register_forward_hook(grab("text_features")))
l0 = model.model.encoder.layers[0]
def pair(name_v, name_t):
    def f(_m, _i, o):
        out[name_v] = o[0][0][0].detach().float().contiguous()
        out[name_t] = o[1][0][0].detach().float().contiguous()
    return f
hooks.append(l0.fusion_layer.register_forward_hook(pair("l0_fusion_v", "l0_fusion_t")))
hooks.append(l0.text_enhancer_layer.register_forward_hook(lambda m, i, o: out.__setitem__("l0_text", o[0][0].detach().float().contiguous())))
hooks.append(l0.deformable_layer.register_forward_hook(lambda m, i, o: out.__setitem__("l0_deform", o[0][0].detach().float().contiguous())))
with torch.no_grad():
    r = model(**inp)
out["enc_vision"] = r.encoder_last_hidden_state_vision[0].float().contiguous()
out["enc_text"] = r.encoder_last_hidden_state_text[0].float().contiguous()
out["dec_hidden"] = r.intermediate_hidden_states[0, -1].float().contiguous()
out["topk_reference"] = r.init_reference_points[0].float().contiguous()
out["logits"] = r.logits[0, :, :inp["input_ids"].shape[1]].float().contiguous()
out["boxes"] = r.pred_boxes[0].float().contiguous()
res = proc.post_process_grounded_object_detection(r, inp["input_ids"], threshold=0.35, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
print("size", img.size, "pixel_values", tuple(inp["pixel_values"].shape), "ids", inp["input_ids"][0].tolist())
for s, l, b in zip(res["scores"], res.get("text_labels", res.get("labels")), res["boxes"]):
    print(f"{float(s):.3f} {l!r} {[round(float(x),1) for x in b]}")
save_file({k: v for k, v in out.items()}, f"{S}/dino_ref.safetensors")
print({k: tuple(v.shape) for k, v in out.items()})
