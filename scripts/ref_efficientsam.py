# Runs the EfficientSAM ONNX exports (encoder + decoder) with onnxruntime and dumps their outputs as the numerical
# oracle for examples/efficientsam_parity.rs.   usage: python ref_efficientsam.py DIR [ti|s]   (DIR holds truck.jpg)
import sys, numpy as np, onnxruntime as ort
from PIL import Image
from safetensors.numpy import save_file
S = sys.argv[1]; V = sys.argv[2] if len(sys.argv) > 2 else "ti"
import os; D = os.path.expanduser("~/.cache/sam-rs/yunyangx--EfficientSAM")
enc = ort.InferenceSession(f"{D}/efficientsam_{V}_encoder.onnx", providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(f"{D}/efficientsam_{V}_decoder.onnx", providers=["CPUExecutionProvider"])
img = np.asarray(Image.open(f"{S}/truck.jpg").convert("RGB"))
H, W = img.shape[:2]
x = (img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
emb = enc.run(None, {"batched_images": x})[0]
out = {"image": x[0], "emb": emb[0]}
def run(name, pts, lab):
    p = np.array(pts, np.float32)[None, None]; l = np.array(lab, np.float32)[None, None]
    masks, iou, low = dec.run(None, {"image_embeddings": emb, "batched_point_coords": p, "batched_point_labels": l, "orig_im_size": np.array([H, W], np.int64)})
    out[f"{name}_coords"] = p[0, 0]; out[f"{name}_labels"] = l[0, 0]; out[f"{name}_masks"] = masks[0, 0].astype(np.float32); out[f"{name}_iou"] = iou[0, 0].astype(np.float32); out[f"{name}_low"] = low[0].astype(np.float32)
    print(name, "masks", masks.shape, "low", low.shape, "iou", iou[0, 0].tolist(), "areas", [(masks[0, 0, i] > 0).sum() for i in range(masks.shape[2])])
run("pt", [[500, 375]], [1])
run("pt2", [[500, 375], [1250, 600]], [1, 0])              # foreground + background
run("bx", [[86, 282], [1710, 851]], [2, 3])                # box corners
run("pt_inv", [[500, 375], [0, 0]], [1, -1])               # explicit invalid (padding) point
save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, f"{S}/efficientsam_{V}_ref.safetensors")
print("emb", emb.shape)
