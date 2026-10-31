//! High-level API: load a model, embed an image once, then decode masks for many prompts.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;

use crate::image_proc::{self, Prepared};
use crate::sam1::{tinyvit::TinyVit, decoder::MaskDecoder, encoder::{EncoderConfig, ImageEncoder}, prompt::PromptEncoder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamVariant {
    VitB,
    VitL,
    VitH,
    /// MobileSAM: TinyViT-5M encoder with the SAM ViT-B prompt encoder and decoder
    Mobile,
}

impl SamVariant {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sam-vit-b" | "sam:100m" | "vit_b" => Some(Self::VitB),
            "sam-vit-l" | "sam:300m" | "vit_l" => Some(Self::VitL),
            "sam-vit-h" | "sam:600m" | "vit_h" => Some(Self::VitH),
            "mobile-sam" | "mobilesam" | "mobile_sam" => Some(Self::Mobile),
            _ => None,
        }
    }

    fn config(self) -> EncoderConfig {
        match self {
            Self::VitB => EncoderConfig::VIT_B,
            Self::VitL => EncoderConfig::VIT_L,
            Self::VitH => EncoderConfig::VIT_H,
            Self::Mobile => EncoderConfig::VIT_B, // unused: MobileSAM has its own encoder
        }
    }

    /// Hugging Face `(repo, file)` of the checkpoint.
    pub fn weights(self) -> (&'static str, &'static str) {
        match self {
            Self::VitB => ("lmz/candle-sam", "sam_vit_b_01ec64.safetensors"),
            Self::VitL => ("facebook/sam-vit-large", "model.safetensors"),
            Self::VitH => ("facebook/sam-vit-huge", "model.safetensors"),
            Self::Mobile => ("lmz/candle-sam", "mobile_sam-tiny-vitt.safetensors"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Prompt {
    /// `(x, y, foreground)` in original image pixels.
    Points(Vec<(f32, f32, bool)>),
    /// `(x0, y0, x1, y1)` in original image pixels.
    Box([f32; 4]),
}

/// A binary mask in original image resolution, row-major, with the predicted IoU.
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub iou: f32,
}

/// Image embedding plus the geometry needed to decode prompts against it.
pub struct Embedding {
    emb: Tensor,
    pre: Prepared,
}

enum Encoder {
    Vit(ImageEncoder),
    Tiny(TinyVit),
}

impl Encoder {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Vit(e) => e.forward(x),
            Self::Tiny(e) => e.forward(x),
        }
    }
}

pub struct Sam {
    encoder: Encoder,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    pub device: Device,
    pub dtype: DType,
    image_pe: Tensor,
    dense: Tensor,
}

impl Sam {
    pub fn from_file(path: &std::path::Path, variant: SamVariant, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        // Meta checkpoints use `image_encoder.*`; the Hugging Face (transformers) export names things differently.
        let vb = if vb.contains_tensor("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix") { vb } else { vb.rename_f(hf_to_meta) };
        let prompt = PromptEncoder::new(vb.pp("prompt_encoder"))?;
        Ok(Self {
            encoder: if variant == SamVariant::Mobile { Encoder::Tiny(TinyVit::new(vb.pp("image_encoder"), false)?) } else { Encoder::Vit(ImageEncoder::new(&variant.config(), vb.pp("image_encoder"))?) },
            decoder: MaskDecoder::new(vb.pp("mask_decoder"))?,
            image_pe: prompt.dense_pe(device, dtype)?,
            dense: prompt.dense_no_mask()?.to_dtype(dtype)?,
            prompt,
            device: device.clone(),
            dtype,
        })
    }

    /// Downloads the checkpoint on first use.
    pub fn from_pretrained(variant: SamVariant, device: &Device, dtype: DType) -> Result<Self> {
        let (repo, file) = variant.weights();
        let path = crate::download::fetch(repo, file)?;
        Self::from_file(&path, variant, device, dtype)
    }

    /// Blocks until queued device work is done (for timing).
    pub fn sync(&self) -> Result<()> {
        Ok(self.device.synchronize()?)
    }

    pub fn embed(&self, img: &RgbImage) -> Result<Embedding> {
        let pre = image_proc::prepare_sam1(img, &self.device, self.dtype)?;
        let emb = self.encoder.forward(&pre.tensor)?;
        Ok(Embedding { emb, pre })
    }

    /// Decodes masks for a prompt. With `multimask`, returns 3 candidates; otherwise 1.
    pub fn predict(&self, e: &Embedding, prompt: &Prompt, multimask: bool) -> Result<Vec<Mask>> {
        let s = e.pre.scale;
        let pts: Vec<(f32, f32, i32)> = match prompt {
            Prompt::Points(p) => p.iter().map(|&(x, y, fg)| (x * s, y * s, fg as i32)).collect(),
            Prompt::Box(b) => vec![(b[0] * s, b[1] * s, 2), (b[2] * s, b[3] * s, 3)],
        };
        let sparse = self.prompt.sparse(&pts, &self.device, self.dtype)?;
        let (masks, iou) = self.decoder.forward(&e.emb, &self.image_pe, &sparse, &self.dense, multimask)?;
        let (oh, ow) = e.pre.orig;
        let (rh, rw) = e.pre.resized;
        let iou = iou.to_dtype(DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;
        let bin: Vec<Vec<u8>> = if self.device.is_cpu() {
            let logits = masks.to_dtype(DType::F32)?.squeeze(0)?.to_vec3::<f32>()?;
            logits.iter().map(|m| resample_mask(m, rh, rw, oh, ow)).collect()
        } else {
            let up = masks.to_dtype(DType::F32)?.upsample_bilinear2d(1024, 1024, false)?.narrow(2, 0, rh)?.narrow(3, 0, rw)?;
            let up = up.upsample_bilinear2d(oh, ow, false)?;
            up.gt(0f32)?.to_dtype(DType::U8)?.squeeze(0)?.to_vec3::<u8>()?.into_iter().map(|m| m.into_iter().flatten().collect()).collect()
        };
        Ok(bin.into_iter().zip(iou).map(|(data, iou)| Mask { width: ow, height: oh, data, iou }).collect())
    }
}


/// 256x256 logits -> binary mask at the original size. Equivalent to bilinear-upsampling to 1024x1024,
/// cropping to the unpadded `rh x rw` region and bilinear-resizing to `oh x ow` (both align_corners=false),
/// but evaluated per output pixel (16 taps) in parallel, without the intermediate images.
fn resample_mask(logits: &[Vec<f32>], rh: usize, rw: usize, oh: usize, ow: usize) -> Vec<u8> {
    use rayon::prelude::*;
    // (index0, index1, frac) of an align_corners=false bilinear tap
    fn tap(dst: usize, scale: f32, max: usize) -> (usize, usize, f32) {
        let src = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src as usize).min(max);
        (i0, (i0 + 1).min(max), src - i0 as f32)
    }
    // stage 1: 256 -> 1024
    let low = |y: usize, x: usize| -> f32 {
        let (y0, y1, fy) = tap(y, 0.25, 255);
        let (x0, x1, fx) = tap(x, 0.25, 255);
        let top = logits[y0][x0] * (1.0 - fx) + logits[y0][x1] * fx;
        let bot = logits[y1][x0] * (1.0 - fx) + logits[y1][x1] * fx;
        top * (1.0 - fy) + bot * fy
    };
    let xt: Vec<_> = (0..ow).map(|x| tap(x, rw as f32 / ow as f32, rw - 1)).collect();
    let mut out = vec![0u8; oh * ow];
    out.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = tap(y, rh as f32 / oh as f32, rh - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1, fx) = xt[x];
            let top = low(y0, x0) * (1.0 - fx) + low(y0, x1) * fx;
            let bot = low(y1, x0) * (1.0 - fx) + low(y1, x1) * fx;
            *o = (top * (1.0 - fy) + bot * fy > 0.0) as u8;
        }
    });
    out
}

/// Meta tensor name -> the name used by the Hugging Face `transformers` SAM export.
fn hf_to_meta(name: &str) -> String {
    let n = name;
    let exact = [
        ("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", "shared_image_embedding.positional_embedding"),
        ("image_encoder.pos_embed", "vision_encoder.pos_embed"),
    ];
    for (a, b) in exact {
        if n == a {
            return b.to_string();
        }
    }
    let swap = |s: &str, from: &str, to: &str| s.replacen(from, to, 1);
    let mut r = n.to_string();
    if r.starts_with("image_encoder.") {
        r = swap(&r, "image_encoder.patch_embed.proj", "vision_encoder.patch_embed.projection");
        r = swap(&r, "image_encoder.blocks.", "vision_encoder.layers.");
        r = r.replace(".norm1.", ".layer_norm1.").replace(".norm2.", ".layer_norm2.");
        for (a, b) in [("neck.0", "neck.conv1"), ("neck.1", "neck.layer_norm1"), ("neck.2", "neck.conv2"), ("neck.3", "neck.layer_norm2")] {
            r = r.replace(a, b);
        }
        return swap(&r, "image_encoder.", "vision_encoder.");
    }
    r = r.replace("prompt_encoder.point_embeddings.", "prompt_encoder.point_embed.");
    r = r.replace("mask_decoder.output_upscaling.0", "mask_decoder.upscale_conv1");
    r = r.replace("mask_decoder.output_upscaling.1", "mask_decoder.upscale_layer_norm");
    r = r.replace("mask_decoder.output_upscaling.3", "mask_decoder.upscale_conv2");
    if r.starts_with("mask_decoder.transformer.") {
        r = r.replace(".norm_final_attn.", ".layer_norm_final_attn.");
        for i in 1..=4 {
            r = r.replace(&format!(".norm{i}."), &format!(".layer_norm{i}."));
        }
    }
    // three-layer MLP heads: Meta layers.0/1/2 = HF proj_in / layers.0 / proj_out
    if r.contains("output_hypernetworks_mlps.") || r.contains("iou_prediction_head.") {
        r = r.replace(".layers.2.", ".proj_out.").replace(".layers.1.", ".layers.TMP.").replace(".layers.0.", ".proj_in.").replace(".layers.TMP.", ".layers.0.");
    }
    r
}
