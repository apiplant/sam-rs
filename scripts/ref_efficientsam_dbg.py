# Dumps intermediate tensors of the EfficientSAM decoder ONNX graph (prompt embedding, every LayerNorm output, the
# upscaled embedding, hypernetwork outputs) for the single point prompt, to locate where a port first diverges.
import sys, os, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto
from safetensors.numpy import save_file
S = sys.argv[1]; V = sys.argv[2] if len(sys.argv) > 2 else "ti"
D = os.path.expanduser("~/.cache/sam-rs/yunyangx--EfficientSAM")
m = onnx.load(f"{D}/efficientsam_{V}_decoder.onnx")
ln_nodes = [n for n in m.graph.node if n.op_type == "LayerNormalization"]
extra, names = [], []
for i, n in enumerate(ln_nodes):
    extra.append(n.output[0]); names.append(f"ln{i}")
# the sparse prompt embedding: the last Add after the four label-embedding Muls (first Add consuming a W(1,1,256) Mul chain end)
init = {i.name for i in m.graph.initializer}
for n in m.graph.node:
    if n.op_type == "Add" and n.output[0].endswith("Add_8_output_0") and "prompt_encoder" in n.output[0]:
        extra.append(n.output[0]); names.append("sparse"); break
for t in extra:
    m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
ref = {k: v for k, v in __import__("safetensors.numpy", fromlist=["load_file"]).load_file(f"{S}/efficientsam_{V}_ref.safetensors").items()}
emb = ref["emb"][None]
feeds = {"image_embeddings": emb, "batched_point_coords": ref["pt_coords"][None, None], "batched_point_labels": ref["pt_labels"][None, None], "orig_im_size": np.array([1200, 1800], np.int64)}
outs = sess.run(None, feeds)
n0 = len(sess.get_outputs()) - len(extra)
dump = {nm: np.ascontiguousarray(o.astype(np.float32)) for nm, o in zip(names, outs[n0:])}
for k, v in dump.items(): print(k, v.shape)
save_file(dump, f"{S}/efficientsam_{V}_dbg.safetensors")
