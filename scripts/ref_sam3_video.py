# Dumps PyTorch (transformers) SAM 3 *video tracker* activations on a synthetic clip (a sliding crop of truck.jpg)
# for examples/sam3_video_parity.rs.   usage: python ref_sam3_video.py DIR   (DIR holds truck.jpg)
import os, sys, torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3Processor, Sam3TrackerVideoProcessor
from safetensors.torch import save_file
torch.set_num_threads(16)
D = os.path.expanduser("~/.cache/sam-rs/facebook--sam3")
S = sys.argv[1]
N, STEP, CW = 6, 36, 1500
vm = Sam3VideoModel.from_pretrained(D).eval()
tr = vm.tracker_model
proc = Sam3Processor.from_pretrained(D)
vproc = Sam3TrackerVideoProcessor.from_pretrained(D)
base = Image.open(f"{S}/truck.jpg").convert("RGB")
W0, H0 = base.size
frames = [base.crop((i * STEP, 0, i * STEP + CW, H0)) for i in range(N)]   # the scene slides left
W, H = frames[0].size
out, pix = {}, []
cap = {}
orig_cond, orig_enc, orig_fwd = tr._prepare_memory_conditioned_features, tr._encode_new_memory, tr._single_frame_forward
def cond(*a, **k):
    r = orig_cond(*a, **k); cap.setdefault("cond", []).append(r.detach().clone()); return r
def enc(*a, **k):
    r = orig_enc(*a, **k); cap.setdefault("mem", []).append((r[0].detach().float().clone(), r[1].detach().float().clone())); return r
tr._prepare_memory_conditioned_features, tr._encode_new_memory = cond, enc

session = vproc.init_video_session(video=None, inference_device="cpu", dtype=torch.float32)
sx, sy = 1008 / W, 1008 / H
with torch.no_grad():
    vproc.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1, input_points=[[[[(500 - 0) , 375]]]], input_labels=[[[1]]], original_size=(H, W))
    for i, fr in enumerate(frames):
        px = proc(images=fr, text="x", return_tensors="pt")["pixel_values"]
        pix.append(px)
        tok = vm.detector_model.vision_encoder.backbone(px).last_hidden_state
        fpn, pos = vm.tracker_neck(tok.view(1, 72, 72, -1).permute(0, 3, 1, 2))
        f0 = tr.mask_decoder.conv_s0(fpn[0]); f1 = tr.mask_decoder.conv_s1(fpn[1])
        feats = [f.flatten(2).permute(2, 0, 1) for f in (f0, f1, fpn[2])]
        poss = [p.flatten(2).permute(2, 0, 1) for p in pos[:3]]
        session.cache.cache_vision_features(i, {"vision_feats": feats, "vision_pos_embeds": poss})
        o = tr(inference_session=session, frame=px[0])
        out[f"f{i}_pred"] = o.pred_masks.float()
        out[f"f{i}_obj"] = o.object_score_logits.float().reshape(1)
        store = session.output_dict_per_obj[0]
        rec = store["cond_frame_outputs"].get(i) or store["non_cond_frame_outputs"].get(i)
        out[f"f{i}_ptr"] = rec["object_pointer"].float()
        print(i, "obj", float(o.object_score_logits.flatten()[0]), "mask px", int((o.pred_masks > 0).sum()), "pred", tuple(o.pred_masks.shape), flush=True)
for i, c in enumerate(cap.get("cond", [])): out[f"cond{i + 1}"] = c.float()
for i, (m, p) in enumerate(cap["mem"]): out[f"mem{i}"], out[f"mempos{i}"] = m, p
out["pix"] = torch.cat(pix, 0)
out["point"] = torch.tensor([[500.0 * sx, 375.0 * sy]])
save_file({k: v.contiguous() for k, v in out.items()}, f"{S}/sam3_video_ref.safetensors")
for i, f in enumerate(frames): f.save(f"{S}/video_frame{i}.png")
print({k: tuple(v.shape) for k, v in out.items() if k != "pix"})
