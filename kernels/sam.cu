// Fused CUDA kernels for sam-rs. Built to PTX by build.rs and loaded through candle's custom-module API.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// Block-wide reduction; `sh` holds 32 floats.
template <bool MAX>
__device__ __forceinline__ float block_reduce(float v, float* sh) {
  v = MAX ? warp_max(v) : warp_sum(v);
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (lane == 0) sh[w] = v;
  __syncthreads();
  v = lane < nw ? sh[lane] : (MAX ? -INFINITY : 0.f);
  v = MAX ? warp_max(v) : warp_sum(v);
  return v;
}

// ---------------------------------------------------------------------------------------------
// Attention with decomposed relative position bias, softmax part.
//   out[bh, r, :] = softmax_{kh,kw}( scale * S[bh, r, kh*W+kw] + RH[kh] + RW[kw] )   (r = i*W + j)
//   RH[kh] = q[bh, r, :] . table[i - kh + H-1, :]          RW[kw] = q[bh, r, :] . table[(2H-1) + j - kw + W-1, :]
// `table` is rel_pos_h (2H-1 rows) followed by rel_pos_w (2W-1 rows). One block per score row; the bias
// values are computed on the fly in shared memory, so the two rel-pos matmuls and their permutes vanish.
// ITEMS = ceil(H*W / blockDim.x), chosen on the host so every thread keeps its slice of the row in registers.
template <typename T, int ITEMS>
__device__ void relpos_softmax(const T* __restrict__ s, const T* __restrict__ q, const T* __restrict__ tab,
                               T* __restrict__ out, int H, int W, int HD, float scale) {
  extern __shared__ float sm[];
  float* qs = sm;
  float* rh = sm + HD;
  float* rw = rh + H;
  __shared__ float red[32];
  const int N = H * W;
  const long row = blockIdx.x;
  const int rem = (int)(row % N), i = rem / W, j = rem % W;
  for (int c = threadIdx.x; c < HD; c += blockDim.x) qs[c] = to_f(q[row * HD + c]);
  __syncthreads();
  for (int idx = threadIdx.x; idx < H + W; idx += blockDim.x) {
    const bool is_h = idx < H;
    const int kk = is_h ? idx : idx - H;
    const T* tr = tab + (is_h ? (long)(i - kk + H - 1) : (long)(2 * H - 1) + (j - kk + W - 1)) * HD;
    float a = 0.f;
    for (int c = 0; c < HD; ++c) a += qs[c] * to_f(tr[c]);
    (is_h ? rh : rw)[kk] = a;
  }
  __syncthreads();
  const T* sr = s + row * N;
  float v[ITEMS];
  float m = -INFINITY;
#pragma unroll
  for (int t = 0; t < ITEMS; ++t) {
    int k = threadIdx.x + t * blockDim.x;
    float x = -INFINITY;
    if (k < N) {
      int kh = k / W, kw = k - kh * W;
      x = to_f(sr[k]) * scale + rh[kh] + rw[kw];
    }
    v[t] = x;
    m = fmaxf(m, x);
  }
  m = block_reduce<true>(m, red);
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < ITEMS; ++t) {
    float e = __expf(v[t] - m);
    v[t] = e;
    sum += e;
  }
  sum = block_reduce<false>(sum, red);
  const float inv = 1.f / sum;
  T* o = out + row * N;
#pragma unroll
  for (int t = 0; t < ITEMS; ++t) {
    int k = threadIdx.x + t * blockDim.x;
    if (k < N) o[k] = from_f<T>(v[t] * inv);
  }
}

#define RELPOS(SUF, T, ITEMS)                                                                                  \
  extern "C" __global__ void relpos_softmax_i##ITEMS##_##SUF(const T* s, const T* q, const T* tab, T* out,     \
                                                             int H, int W, int HD, float scale) {              \
    relpos_softmax<T, ITEMS>(s, q, tab, out, H, W, HD, scale);                                                \
  }
#define RELPOS_ALL(SUF, T) RELPOS(SUF, T, 1) RELPOS(SUF, T, 2) RELPOS(SUF, T, 4) RELPOS(SUF, T, 8) RELPOS(SUF, T, 16) RELPOS(SUF, T, 32)
RELPOS_ALL(f32, float)
RELPOS_ALL(f16, __half)
RELPOS_ALL(bf16, __nv_bfloat16)

// ---------------------------------------------------------------------------------------------
// y[T, 3C] (+ bias[3C]) -> q [BH,N,HD], kT [BH,HD,N], v [BH,N,HD], packed back to back in `out`.
// Input rows are tokens (windows * N of them), columns are (qkv, head, d).
template <typename T>
__device__ void qkv_split(const T* __restrict__ y, const T* __restrict__ bias, T* __restrict__ out,
                          long total, int N, int NH, int HD) {
  const int C = NH * HD;
  const long plane = total / 3;  // elements per q / k / v
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const long t = idx / (3 * C);
    const int r = (int)(idx - t * 3 * C);
    const int which = r / C, cc = r - which * C;
    const int head = cc / HD, d = cc - head * HD;
    const long win = t / N;
    const int n = (int)(t - win * N);
    const long bh = win * NH + head;
    const T val = from_f<T>(to_f(y[idx]) + to_f(bias[r]));
    long dst = which == 1 ? (bh * HD + d) * N + n : (bh * N + n) * HD + d;
    out[which * plane + dst] = val;
  }
}
// out = res + y + bias[col]
template <typename T>
__device__ void bias_residual(const T* __restrict__ res, const T* __restrict__ y, const T* __restrict__ bias,
                              T* __restrict__ out, long total, int C) {
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x)
    out[idx] = from_f<T>(to_f(res[idx]) + to_f(y[idx]) + to_f(bias[idx % C]));
}
// out = gelu_erf(y + bias[col])
template <typename T>
__device__ void bias_gelu(const T* __restrict__ y, const T* __restrict__ bias, T* __restrict__ out, long total, int C) {
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    float x = to_f(y[idx]) + to_f(bias[idx % C]);
    out[idx] = from_f<T>(0.5f * x * (1.f + erff(x * 0.70710678118654752f)));
  }
}

#define ELEMENTWISE(SUF, T)                                                                                      \
  extern "C" __global__ void qkv_split_##SUF(const T* y, const T* b, T* out, long total, int N, int NH, int HD) { \
    qkv_split<T>(y, b, out, total, N, NH, HD);                                                                  \
  }                                                                                                              \
  extern "C" __global__ void bias_residual_##SUF(const T* r, const T* y, const T* b, T* out, long total, int C) { \
    bias_residual<T>(r, y, b, out, total, C);                                                                   \
  }                                                                                                              \
  extern "C" __global__ void bias_gelu_##SUF(const T* y, const T* b, T* out, long total, int C) {                \
    bias_gelu<T>(y, b, out, total, C);                                                                          \
  }
ELEMENTWISE(f32, float)
ELEMENTWISE(f16, __half)
ELEMENTWISE(bf16, __nv_bfloat16)

// ---------------------------------------------------------------------------------------------
// Resize (antialiased triangle filter, as PIL / the `image` crate), normalize and zero-pad in one pass.
//   src: u8 [h, w, 3]   out: [3, S, S]  (rows >= nh or cols >= nw are zero, i.e. bottom/right padding)
__device__ __forceinline__ float tri(float x) {
  x = fabsf(x);
  return x < 1.f ? 1.f - x : 0.f;
}
template <typename T>
__device__ void preprocess(const unsigned char* __restrict__ src, T* __restrict__ out, int h, int w, int nh, int nw,
                           int S, float m0, float m1, float m2, float i0, float i1, float i2) {
  const long total = 3L * S * S;
  const float sx = (float)w / nw, sy = (float)h / nh;
  const float fx = fmaxf(sx, 1.f), fy = fmaxf(sy, 1.f);  // filter scale (antialias only when shrinking)
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int c = (int)(idx / ((long)S * S));
    const int r = (int)(idx - (long)c * S * S);
    const int y = r / S, x = r - y * S;
    float v = 0.f;
    if (y < nh && x < nw) {
      const float cx = (x + 0.5f) * sx, cy = (y + 0.5f) * sy;
      const int x0 = max((int)(cx - fx + 0.5f), 0), x1 = min((int)(cx + fx + 0.5f), w);
      const int y0 = max((int)(cy - fy + 0.5f), 0), y1 = min((int)(cy + fy + 0.5f), h);
      float wsx = 0.f, wsy = 0.f;
      for (int xx = x0; xx < x1; ++xx) wsx += tri((xx + 0.5f - cx) / fx);
      for (int yy = y0; yy < y1; ++yy) wsy += tri((yy + 0.5f - cy) / fy);
      float acc = 0.f;
      for (int yy = y0; yy < y1; ++yy) {
        const float wy = tri((yy + 0.5f - cy) / fy);
        float row = 0.f;
        for (int xx = x0; xx < x1; ++xx) row += tri((xx + 0.5f - cx) / fx) * (float)src[((long)yy * w + xx) * 3 + c];
        acc += wy * row;
      }
      const float mean = c == 0 ? m0 : (c == 1 ? m1 : m2), inv = c == 0 ? i0 : (c == 1 ? i1 : i2);
      v = (acc / (wsx * wsy) - mean) * inv;
    }
    out[idx] = from_f<T>(v);
  }
}
#define PREPROCESS(SUF, T)                                                                                       \
  extern "C" __global__ void preprocess_##SUF(const unsigned char* src, T* out, int h, int w, int nh, int nw,    \
                                              int S, float m0, float m1, float m2, float i0, float i1, float i2) { \
    preprocess<T>(src, out, h, w, nh, nw, S, m0, m1, m2, i0, i1, i2);                                          \
  }
PREPROCESS(f32, float)
PREPROCESS(f16, __half)
PREPROCESS(bf16, __nv_bfloat16)

// ---------------------------------------------------------------------------------------------
// SAM 3 ViT: y[T, 3C] (+ bias[3C]) -> packed [3, T, H, HD] with 2-D RoPE applied to q and k.
// `cos`/`sin` are f32 tables [N, HD] (N tokens per window; HD pairs share an angle); math is f32 like the
// reference. One thread per (token, head, pair); v is only biased and copied.
template <typename T>
__device__ void qkv_rope(const T* __restrict__ y, const T* __restrict__ bias, const float* __restrict__ cs,
                         const float* __restrict__ sn, T* __restrict__ out, long rows, int N, int H, int HD) {
  const int C = H * HD, P = HD / 2;
  const long total = rows * 3L * H * P;
  const long plane = rows * (long)C;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int j = (int)(idx % P);
    const int h = (int)((idx / P) % H);
    const int which = (int)((idx / ((long)P * H)) % 3);
    const long t = idx / ((long)P * H * 3);
    const long src = t * 3L * C + (long)which * C + (long)h * HD + 2 * j;
    float x0 = to_f(y[src]) + to_f(bias[which * C + h * HD + 2 * j]);
    float x1 = to_f(y[src + 1]) + to_f(bias[which * C + h * HD + 2 * j + 1]);
    if (which < 2) {
      const int n = (int)(t % N);
      const float c = cs[n * HD + 2 * j], s = sn[n * HD + 2 * j];
      const float r0 = x0 * c - x1 * s, r1 = x1 * c + x0 * s;
      x0 = r0;
      x1 = r1;
    }
    const long dst = which * plane + t * (long)C + (long)h * HD + 2 * j;
    out[dst] = from_f<T>(x0);
    out[dst + 1] = from_f<T>(x1);
  }
}

// y: [Cout*4, H*W] (rows (co,dy,dx)) or, when `tokens`, [H*W, Cout*4]; + bias[Cout*4]; optional GELU;
// -> NCHW [Cout, 2H, 2W]. The output of a kernel-2 stride-2 transposed convolution.
template <typename T>
__device__ void pixel_shuffle2(const T* __restrict__ y, const T* __restrict__ bias, T* __restrict__ out,
                               int cout, int h, int w, int gelu, int tokens) {
  const long oh = 2L * h, ow = 2L * w, total = (long)cout * oh * ow;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const long ox = idx % ow, oy = (idx / ow) % oh;
    const int co = (int)(idx / (ow * oh));
    const int col = co * 4 + (int)(oy & 1) * 2 + (int)(ox & 1);
    const long p = (oy >> 1) * w + (ox >> 1);
    float v = to_f(tokens ? y[p * cout * 4 + col] : y[(long)col * h * w + p]) + to_f(bias[col]);
    if (gelu) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    out[idx] = from_f<T>(v);
  }
}

#define SAM3_OPS(SUF, T)                                                                                         \
  extern "C" __global__ void qkv_rope_##SUF(const T* y, const T* b, const float* cs, const float* sn, T* out,    \
                                            long rows, int N, int H, int HD) {                                   \
    qkv_rope<T>(y, b, cs, sn, out, rows, N, H, HD);                                                              \
  }                                                                                                              \
  extern "C" __global__ void pixel_shuffle2_##SUF(const T* y, const T* b, T* out, int cout, int h, int w,        \
                                                  int gelu, int tokens) {                                        \
    pixel_shuffle2<T>(y, b, out, cout, h, w, gelu, tokens);                                                      \
  }
SAM3_OPS(f32, float)
SAM3_OPS(f16, __half)
SAM3_OPS(bf16, __nv_bfloat16)

// ---------------------------------------------------------------------------------------------
// Depthwise 3x3 convolution (padding 1, stride 1 or 2) + bias, optional exact GELU. x: [C, H, W], w: [C, 9].
template <typename T>
__device__ void dwconv3(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ b, T* __restrict__ out,
                        int C, int H, int W, int OH, int OW, int stride, int gelu) {
  const long total = (long)C * OH * OW;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int ox = (int)(idx % OW), oy = (int)((idx / OW) % OH), c = (int)(idx / ((long)OW * OH));
    float acc = to_f(b[c]);
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      const int iy = oy * stride + ky - 1;
      if (iy < 0 || iy >= H) continue;
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        const int ix = ox * stride + kx - 1;
        if (ix < 0 || ix >= W) continue;
        acc += to_f(x[((long)c * H + iy) * W + ix]) * to_f(w[c * 9 + ky * 3 + kx]);
      }
    }
    if (gelu) acc = 0.5f * acc * (1.f + erff(acc * 0.70710678118654752f));
    out[idx] = from_f<T>(acc);
  }
}
#define DWCONV(SUF, T)                                                                                              \
  extern "C" __global__ void dwconv3_##SUF(const T* x, const T* w, const T* b, T* out, int C, int H, int W, int OH, \
                                           int OW, int stride, int gelu) {                                          \
    dwconv3<T>(x, w, b, out, C, H, W, OH, OW, stride, gelu);                                                        \
  }
DWCONV(f32, float)
DWCONV(f16, __half)
DWCONV(bf16, __nv_bfloat16)
