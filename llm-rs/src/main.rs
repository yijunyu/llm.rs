#![allow(
    dead_code,
    mutable_transmutes,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]

use std::f32::EPSILON;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicPtr, Ordering};

#[allow(unused_imports)]
use rayon::prelude::*;
extern "C" {
    static mut stdout: *mut FILE;
    fn fclose(__stream: *mut FILE) -> i32;
    fn fflush(__stream: *mut FILE) -> i32;
    fn fopen(_: *const u8, _: *const u8) -> *mut FILE;
    fn printf(_: *const u8, _: ...) -> i32;
    fn ftell(__stream: *mut FILE) -> i64;
    fn fseek(__stream: *mut FILE, __off: i64, __whence: i32) -> i32;
    fn fread(_: *mut usize, _: u64, _: u64, _: *mut FILE) -> u64;
    fn malloc(_: u64) -> *mut usize;
    fn calloc(_: u64, _: u64) -> *mut usize;
    fn exit(_: i32) -> !;
    fn free(__ptr: *mut usize);
    fn __ctype_b_loc() -> *mut *const u16;
    fn __assert_fail(
        __assertion: *const u8,
        __file: *const u8,
        __line: u32,
        __function: *const u8,
    ) -> !;
    fn logf(_: f32) -> f32;
    fn expf(_: f32) -> f32;
    fn tanhf(_: f32) -> f32;
    fn coshf(_: f32) -> f32;
    fn powf(_: f32, _: f32) -> f32;
    fn sqrtf(_: f32) -> f32;
    fn clock_gettime(__clock_id: clockid_t, __tp: *mut timespec) -> i32;
    fn memcpy(_: *mut usize, _: *const usize, _: u64) -> *mut usize;
    fn memset(_: *mut usize, _: i32, _: u64) -> *mut usize;
    fn access(__name: *const u8, __type: i32) -> i32;
}
pub type size_t = u64;
pub type __uint32_t = u32;
pub type __off_t = i64;
pub type __off64_t = i64;
pub type __time_t = i64;
pub type __clockid_t = i32;
pub type __syscall_slong_t = i64;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: i32,
    pub _IO_read_ptr: *mut u8,
    pub _IO_read_end: *mut u8,
    pub _IO_read_base: *mut u8,
    pub _IO_write_base: *mut u8,
    pub _IO_write_ptr: *mut u8,
    pub _IO_write_end: *mut u8,
    pub _IO_buf_base: *mut u8,
    pub _IO_buf_end: *mut u8,
    pub _IO_save_base: *mut u8,
    pub _IO_backup_base: *mut u8,
    pub _IO_save_end: *mut u8,
    pub _chain: *mut _IO_FILE,
    pub _fileno: i32,
    pub _flags2: i32,
    pub _old_offset: __off_t,
    pub _cur_column: u16,
    pub _vtable_offset: u8,
    pub _shortbuf: [u8; 1],
    pub _lock: *mut usize,
    pub _offset: __off64_t,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut usize,
    pub __pad5: size_t,
    pub _mode: i32,
    pub _unused2: [u8; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub type clockid_t = __clockid_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
pub type C2RustUnnamed = u32;
pub const _ISalnum: C2RustUnnamed = 8;
pub const _ISpunct: C2RustUnnamed = 4;
pub const _IScntrl: C2RustUnnamed = 2;
pub const _ISblank: C2RustUnnamed = 1;
pub const _ISgraph: C2RustUnnamed = 32768;
pub const _ISprint: C2RustUnnamed = 16384;
pub const _ISspace: C2RustUnnamed = 8192;
pub const _ISxdigit: C2RustUnnamed = 4096;
pub const _ISdigit: C2RustUnnamed = 2048;
pub const _ISalpha: C2RustUnnamed = 1024;
pub const _ISlower: C2RustUnnamed = 512;
pub const _ISupper: C2RustUnnamed = 256;
pub type uint32_t = __uint32_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParameterTensors {
    pub wte: *mut f32,
    pub wpe: *mut f32,
    pub ln1w: *mut f32,
    pub ln1b: *mut f32,
    pub qkvw: *mut f32,
    pub qkvb: *mut f32,
    pub attprojw: *mut f32,
    pub attprojb: *mut f32,
    pub ln2w: *mut f32,
    pub ln2b: *mut f32,
    pub fcw: *mut f32,
    pub fcb: *mut f32,
    pub fcprojw: *mut f32,
    pub fcprojb: *mut f32,
    pub lnfw: *mut f32,
    pub lnfb: *mut f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ActivationTensors {
    pub encoded: *mut f32,
    pub ln1: *mut f32,
    pub ln1_mean: *mut f32,
    pub ln1_rstd: *mut f32,
    pub qkv: *mut f32,
    pub atty: *mut f32,
    pub preatt: *mut f32,
    pub att: *mut f32,
    pub attproj: *mut f32,
    pub residual2: *mut f32,
    pub ln2: *mut f32,
    pub ln2_mean: *mut f32,
    pub ln2_rstd: *mut f32,
    pub fch: *mut f32,
    pub fch_gelu: *mut f32,
    pub fcproj: *mut f32,
    pub residual3: *mut f32,
    pub lnf: *mut f32,
    pub lnf_mean: *mut f32,
    pub lnf_rstd: *mut f32,
    pub logits: *mut f32,
    pub probs: *mut f32,
    pub losses: *mut f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2Config {
    pub max_seq_len: i32,
    pub vocab_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub channels: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2 {
    pub config: GPT2Config,
    pub params: ParameterTensors,
    pub param_sizes: [size_t; 16],
    pub params_memory: *mut f32,
    pub num_parameters: size_t,
    pub grads: ParameterTensors,
    pub grads_memory: *mut f32,
    pub m_memory: *mut f32,
    pub v_memory: *mut f32,
    pub acts: ActivationTensors,
    pub act_sizes: [size_t; 23],
    pub acts_memory: *mut f32,
    pub num_activations: size_t,
    pub grads_acts: ActivationTensors,
    pub grads_acts_memory: *mut f32,
    pub batch_size: i32,
    pub seq_len: i32,
    pub inputs: *mut i32,
    pub targets: *mut i32,
    pub mean_loss: f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DataLoader {
    pub B: i32,
    pub T: i32,
    pub tokens_file: *mut FILE,
    pub file_size: i64,
    pub current_position: i64,
    pub batch: *mut i32,
    pub inputs: *mut i32,
    pub targets: *mut i32,
    pub num_batches: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Tokenizer {
    pub vocab_size: uint32_t,
    pub token_table: *mut *mut u8,
    pub init_ok: i32,
}

pub unsafe fn encoder_forward(
    out: *mut f32,
    inp: *const i32,
    wte: *const f32,
    wpe: *const f32,
    B: usize,
    T: usize,
    C: usize,
) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"

    // Iterate over the batch dimension
    for b in 0..B {
        // Iterate over the sequence length
        for t in 0..T {
            // Calculate the base address for out[b,t,:]
            let out_bt = out.add(b * T * C + t * C);
            // Get the token index at inp[b, t]
            let ix = *inp.add(b * T + t) as usize;
            // Calculate the base address for wte[ix,:]
            let wte_ix = wte.add(ix * C);
            // Calculate the base address for wpe[t,:]
            let wpe_t = wpe.add(t * C);
            // Sum the token and position embeddings and store the result in out[b,t,:]
            for i in 0..C {
                *out_bt.add(i) = *wte_ix.add(i) + *wpe_t.add(i);
            }
        }
    }
}

pub unsafe fn encoder_backward(
    dwte: *mut f32,
    dwpe: *mut f32,
    dout: *const f32,
    inp: *const i32,
    B: usize,
    T: usize,
    C: usize,
) {
    // Iterate over the batch dimension
    for b in 0..B {
        // Iterate over the sequence length
        for t in 0..T {
            // Calculate the base address for dout[b,t,:]
            let dout_bt = dout.add(b * T * C + t * C);
            // Get the token index at inp[b, t]
            let ix = *inp.add(b * T + t) as usize;
            // Calculate the base address for dwte[ix,:]
            let dwte_ix = dwte.add(ix * C);
            // Calculate the base address for dwpe[t,:]
            let dwpe_t = dwpe.add(t * C);
            // Backpropagate the gradients
            for i in 0..C {
                let d = *dout_bt.add(i);
                *dwte_ix.add(i) += d;
                *dwpe_t.add(i) += d;
            }
        }
    }
}

pub unsafe fn layernorm_forward(
    out: *mut f32,
    mean: *mut f32,
    rstd: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: usize,
    T: usize,
    C: usize,
) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    for b in 0..B {
        // Iterate over the sequence length
        for t in 0..T {
            // Calculate the base address for inp[b,t,:]
            let x = inp.add(b * T * C + t * C);
            // Initialize mean and variance
            let mut m = 0.0f32;
            let mut v = 0.0f32;

            // Calculate the mean
            for i in 0..C {
                m += *x.add(i);
            }
            m /= C as f32;

            // Calculate the variance (without bias correction)
            for i in 0..C {
                let xshift = *x.add(i) - m;
                v += xshift * xshift;
            }
            v /= C as f32;

            // Calculate the rstd (reciprocal standard deviation)
            let s = 1.0f32 / (v + EPSILON).sqrt();

            // Calculate the base address for out[b,t,:]
            let out_bt = out.add(b * T * C + t * C);
            // Apply normalization, scaling, and shifting
            for i in 0..C {
                let n = s * (*x.add(i) - m); // normalize
                let o = n * *weight.add(i) + *bias.add(i); // scale and shift
                *out_bt.add(i) = o; // write
            }

            // Cache the mean and rstd for the backward pass later
            *mean.add(b * T + t) = m;
            *rstd.add(b * T + t) = s;
        }
    }
}

pub unsafe fn layernorm_backward(
    dinp: *mut f32,
    dweight: *mut f32,
    dbias: *mut f32,
    dout: *const f32,
    inp: *const f32,
    weight: *const f32,
    mean: *const f32,
    rstd: *const f32,
    B: usize,
    T: usize,
    C: usize,
) {
    // Iterate over the batch dimension
    for b in 0..B {
        // Iterate over the sequence length
        for t in 0..T {
            // Calculate the base addresses for dout, inp, and dinp at (b,t)
            let dout_bt = dout.add(b * T * C + t * C);
            let inp_bt = inp.add(b * T * C + t * C);
            let dinp_bt = dinp.add(b * T * C + t * C);
            let mean_bt = *mean.add(b * T + t);
            let rstd_bt = *rstd.add(b * T + t);

            // First: two reduce operations
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.add(i) * *dout_bt.add(i);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Now iterate again and accumulate all the gradients
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.add(i) * *dout_bt.add(i);
                
                // Gradient contribution to bias
                *dbias.add(i) += *dout_bt.add(i);
                
                // Gradient contribution to weight
                *dweight.add(i) += norm_bti * *dout_bt.add(i);
                
                // Gradient contribution to input
                let mut dval = 0.0f32;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                *dinp_bt.add(i) += dval;
            }
        }
    }
}

pub unsafe fn matmul_forward_naive(
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    let out = AtomicPtr::new(out);
    let inp = AtomicPtr::new(inp as *mut f32);
    let weight = AtomicPtr::new(weight as *mut f32);
    let bias = AtomicPtr::new(bias as *mut f32);

    // Create a parallel iterator over the batch dimension
    (0..B).into_par_iter().for_each(|b| {
        // Create a parallel iterator over the sequence length
        (0..T).into_par_iter().for_each(|t| {
            let bt = b * T + t;
            // Iterate over the output channels
            for o in 0..OC {
                // Initialize the output value with the bias if provided, otherwise 0.0
                let mut val = if !bias.load(Ordering::SeqCst).is_null() {
                    *bias.load(Ordering::SeqCst).add(o)
                } else {
                    0.0f32
                };
                // Perform the dot product
                for i in 0..C {
                    val += *inp.load(Ordering::SeqCst).add(bt * C + i) * *weight.load(Ordering::SeqCst).add(o * C + i);
                }
                // Store the result
                *out.load(Ordering::SeqCst).add(bt * OC + o) = val;
            }
        });
    });
}

pub unsafe fn matmul_forward(
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    const LOOP_UNROLL: usize = 8;
    let out = AtomicPtr::new(out);
    let inp = AtomicPtr::new(inp as *mut f32);
    let weight = AtomicPtr::new(weight as *mut f32);
    let bias = AtomicPtr::new(bias as *mut f32);

    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        matmul_forward_naive(out.load(Ordering::SeqCst), inp.load(Ordering::SeqCst), weight.load(Ordering::SeqCst), bias.load(Ordering::SeqCst), B, T, C, OC);
        return;
    }

    // Parallelize the outer loop using Rayon
    (0..B * T).into_par_iter().step_by(LOOP_UNROLL).for_each(|obt| {
        for o in 0..OC {
            // Initialize the result array with bias if present
            let mut result = [0.0f32; LOOP_UNROLL];
            for ibt in 0..LOOP_UNROLL {
                result[ibt] = if !bias.load(Ordering::SeqCst).is_null() { *bias.load(Ordering::SeqCst).add(o) } else { 0.0f32 };
            }

            // Cache the weight value and compute dot products
            for i in 0..C {
                let w = *weight.load(Ordering::SeqCst).add(i + o * C);
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt + ibt;
                    result[ibt] += *inp.load(Ordering::SeqCst).add(bt * C + i) * w;
                }
            }

            // Write results back to the output matrix
            for ibt in 0..LOOP_UNROLL {
                let bt = obt + ibt;
                *out.load(Ordering::SeqCst).add(bt * OC + o) = result[ibt];
            }
        }
    });
}

pub unsafe fn matmul_backward(
    dinp: *mut f32,
    dweight: *mut f32,
    dbias: *mut f32,
    dout: *const f32,
    inp: *const f32,
    weight: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy
    let dinp = AtomicPtr::new(dinp);
    let dweight = AtomicPtr::new(dweight);
    let dbias = AtomicPtr::new(dbias);
    let dout = AtomicPtr::new(dout as *mut f32);
    let inp = AtomicPtr::new(inp as *mut f32);
    let weight = AtomicPtr::new(weight as *mut f32);

    // Parallelize over B and T for input gradient computation
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dout_bt = dout.load(Ordering::SeqCst).add(b * T * OC + t * OC);
            let dinp_bt = dinp.load(Ordering::SeqCst).add(b * T * C + t * C);
            for o in 0..OC {
                let wrow = weight.load(Ordering::SeqCst).add(o * C);
                let d = *dout_bt.add(o);
                for i in 0..C {
                    *dinp_bt.add(i) += *wrow.add(i) * d;
                }
            }
        });
    });

    // Parallelize over output channels for weight and bias gradient computation
    (0..OC).into_par_iter().for_each(|o| {
        let dwrow = dweight.load(Ordering::SeqCst).add(o * C);
        for b in 0..B {
            for t in 0..T {
                let dout_bt = dout.load(Ordering::SeqCst).add(b * T * OC + t * OC);
                let inp_bt = inp.load(Ordering::SeqCst).add(b * T * C + t * C);
                let d = *dout_bt.add(o);
                if !dbias.load(Ordering::SeqCst).is_null() {
                    *dbias.load(Ordering::SeqCst).add(o) += d;
                }
                for i in 0..C {
                    *dwrow.add(i) += *inp_bt.add(i) * d;
                }
            }
        }
    });
}

pub unsafe fn attention_forward(
    out: *mut f32,
    preatt: *mut f32,
    att: *mut f32,
    inp: *const f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    let mut out = AtomicPtr::new(out);
    let mut preatt = AtomicPtr::new(preatt);
    let mut att = AtomicPtr::new(att);
    let mut inp = AtomicPtr::new(inp as *mut f32);

    // Parallelize the outer loops using Rayon
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                // Access pointers for query, preatt, and att
                let query_t = inp.load(Ordering::SeqCst).add(b * T * C3 + t * C3 + h * hs) ;
                let preatt_bth = preatt.load(Ordering::SeqCst).add(b * NH * T * T + h * T * T + t * T);
                let att_bth = att.load(Ordering::SeqCst).add(b * NH * T * T + h * T * T + t * T);

                // Pass 1: calculate query dot key and maxval
                let mut maxval = f32::MIN; // Initialize to the smallest possible value for stability
                for t2 in 0..=t {
                    let key_t2 = inp.load(Ordering::SeqCst).add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key

                    // Calculate dot product (query_t) dot (key_t2)
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += *query_t.add(i) * *key_t2.add(i);
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    *preatt_bth.add(t2) = val;
                }

                // Pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.0f32;
                for t2 in 0..=t {
                    let expv = (*preatt_bth.add(t2) - maxval).exp();
                    expsum += expv;
                    *att_bth.add(t2) = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: normalize to get the softmax
                for t2 in 0..T {
                    if t2 <= t {
                        *att_bth.add(t2) *= expsum_inv;
                    } else {
                        *att_bth.add(t2) = 0.0;
                    }
                }

                // Pass 4: accumulate weighted values into the output of attention
                let out_bth = out.load(Ordering::SeqCst).add(b * T * C + t * C + h * hs);
                for i in 0..hs {
                    *out_bth.add(i) = 0.0;
                }
                for t2 in 0..=t {
                    let value_t2 = inp.load(Ordering::SeqCst).add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                    let att_btht2 = *att_bth.add(t2);
                    for i in 0..hs {
                        *out_bth.add(i) += att_btht2 * *value_t2.add(i);
                    }
                }
            });
        });
    });
}

pub unsafe fn attention_backward(
    dinp: *mut f32,
    dpreatt: *mut f32,
    datt: *mut f32,
    dout: *const f32,
    inp: *const f32,
    att: *const f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    // Iterate over batch size
    for b in 0..B {
        // Iterate over sequence length
        for t in 0..T {
            // Iterate over number of heads
            for h in 0..NH {
                // Access pointers for att, datt, and dpreatt
                let att_bth = att.add(b * NH * T * T + h * T * T + t * T);
                let datt_bth = datt.add(b * NH * T * T + h * T * T + t * T);
                let dpreatt_bth = dpreatt.add(b * NH * T * T + h * T * T + t * T);
                let dquery_t = dinp.add(b * T * C3 + t * C3 + h * hs);
                let query_t = inp.add(b * T * C3 + t * C3 + h * hs);

                // Backward pass 4: through the value accumulation
                let dout_bth = dout.add(b * T * C + t * C + h * hs);
                for t2 in 0..=t {
                    let value_t2 = inp.add(b * T * C3 + t2 * C3 + h * hs + C * 2); // +C*2 because it's value
                    let dvalue_t2 = dinp.add(b * T * C3 + t2 * C3 + h * hs + C * 2);
                    for i in 0..hs {
                        // Forward pass was: out_bth[i] += att_bth[t2] * value_t2[i];
                        // Backward pass:
                        *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                        *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                    }
                }

                // Backward pass 2 & 3: the softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0f32 } else { 0.0f32 };
                        let local_derivative = *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                        *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                    }
                }

                // Backward pass 1: the query @ key matmul
                for t2 in 0..=t {
                    let key_t2 = inp.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    let dkey_t2 = dinp.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    for i in 0..hs {
                        // Forward pass was: preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // Backward pass:
                        *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                        *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                    }
                }
            }
        }
    }
}

pub unsafe fn gelu_forward(
    out: *mut f32, 
    inp: *const f32, 
    N: usize
) {
    for i in 0..N {
        // Access the input value
        let x = *inp.add(i);

        // Compute the cube term
        let cube = 0.044715 * x * x * x;

        // Apply the GeLU activation function
        let result = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());

        // Store the result in the output array
        *out.add(i) = result;
    }
}

pub unsafe fn gelu_backward(
    dinp: *mut f32,
    inp: *const f32,
    dout: *const f32,
    N: usize,
) {
    for i in 0..N {
        let gelu_scaling_factor: f32 = (2.0 / PI).sqrt();

        // Access the input value and output gradient
        let x = *inp.add(i);
        let dout_i = *dout.add(i);

        // Compute the cube term
        let cube = 0.044715 * x * x * x;

        // Compute the tanh argument
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);

        // Compute the local gradient
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);

        // Update the input gradient
        *dinp.add(i) += local_grad * dout_i;
    }
}

pub unsafe extern "C" fn residual_forward(
    out: *mut f32,
    inp1: *const f32,
    inp2: *const f32,
    N: usize,
) {
    for i in 0..N {
        // Access the elements of inp1 and inp2, add them, and store the result in out
        *out.add(i) = *inp1.add(i) + *inp2.add(i);
    }
}

pub unsafe fn residual_backward(
    dinp1: *mut f32,
    dinp2: *mut f32,
    dout: *const f32,
    N: usize,
) {
    for i in 0..N {
        // Access the elements of dout and update dinp1 and dinp2
        let grad = *dout.add(i);
        *dinp1.add(i) += grad;
        *dinp2.add(i) += grad;
    }
}

pub unsafe extern "C" fn softmax_forward(
    mut probs: *mut f32,
    mut logits: *mut f32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    let probs = AtomicPtr::new(probs);
    let logits = AtomicPtr::new(logits);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let logits_bt: *mut f32 = logits
                .load(Ordering::SeqCst)
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let probs_bt: *mut f32 = probs
                .load(Ordering::SeqCst)
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);

            let mut maxval: f32 = -10000.0f32;
            (0..V).into_iter().for_each(|i| {
                if *logits_bt.offset(i as isize) > maxval {
                    maxval = *logits_bt.offset(i as isize);
                }
            });

            let mut sum: f32 = 0.0f32;
            (0..V).into_iter().for_each(|i_0| {
                *probs_bt.offset(i_0 as isize) =
                    expf(*logits_bt.offset(i_0 as isize) - maxval);
                sum += *probs_bt.offset(i_0 as isize);
            });

            (0..V).into_iter().for_each(|i_1| {
                *probs_bt.offset(i_1 as isize) /= sum;
            });
        });
    });
}

pub unsafe extern "C" fn crossentropy_forward(
    mut losses: *mut f32,
    mut probs: *mut f32,
    mut targets: *mut i32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    for b in 0..B {
        (0..T).into_iter().for_each(|t| {
            let mut probs_bt: *mut f32 =
                probs.offset((b * T * V) as isize).offset((t * V) as isize);
            let ix: i32 = *targets.offset((b * T + t) as isize);
            *losses.offset((b * T + t) as isize) = -logf(*probs_bt.offset(ix as isize));
        });
    }
}

pub unsafe extern "C" fn crossentropy_softmax_backward(
    mut dlogits: *mut f32,
    mut dlosses: *mut f32,
    mut probs: *mut f32,
    mut targets: *mut i32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    for b in 0..B {
        for t in 0..T {
            let mut dlogits_bt: *mut f32 = dlogits
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut probs_bt: *mut f32 =
                probs.offset((b * T * V) as isize).offset((t * V) as isize);
            let dloss: f32 = *dlosses.offset((b * T + t) as isize);
            let ix: i32 = *targets.offset((b * T + t) as isize);

            (0..V).into_iter().for_each(|i| {
                let p: f32 = *probs_bt.offset(i as isize);
                let indicator: f32 = if i == ix { 1.0f32 } else { 0.0f32 };
                *dlogits_bt.offset(i as isize) += (p - indicator) * dloss;
            });
        }
    }
}

pub unsafe extern "C" fn malloc_and_point_parameters(
    mut params: *mut ParameterTensors,
    mut param_sizes: *mut size_t,
) -> *mut f32 {
    let mut num_parameters: size_t = 0 as i32 as size_t;
    (0..16 as usize).into_iter().for_each(|i| {
        num_parameters = num_parameters.wrapping_add(*param_sizes.offset(i as isize) as size_t);
    });
    let mut params_memory: *mut f32 =
        malloc(num_parameters.wrapping_mul(::core::mem::size_of::<f32>() as u64)) as *mut f32;
    let mut ptrs: [*mut *mut f32; 16] = [
        &mut (*params).wte,
        &mut (*params).wpe,
        &mut (*params).ln1w,
        &mut (*params).ln1b,
        &mut (*params).qkvw,
        &mut (*params).qkvb,
        &mut (*params).attprojw,
        &mut (*params).attprojb,
        &mut (*params).ln2w,
        &mut (*params).ln2b,
        &mut (*params).fcw,
        &mut (*params).fcb,
        &mut (*params).fcprojw,
        &mut (*params).fcprojb,
        &mut (*params).lnfw,
        &mut (*params).lnfb,
    ];
    let mut params_memory_iterator: *mut f32 = params_memory;
    (0..16 as usize).into_iter().for_each(|i_0| {
        *ptrs[i_0] = params_memory_iterator;
        params_memory_iterator =
            params_memory_iterator.offset(*param_sizes.offset(i_0 as isize) as isize);
    });
    return params_memory;
}

pub unsafe extern "C" fn malloc_and_point_activations(
    mut acts: *mut ActivationTensors,
    mut act_sizes: *mut size_t,
) -> *mut f32 {
    let mut num_activations: size_t = 0 as i32 as size_t;
    (0..23 as usize).into_iter().for_each(|i| {
        num_activations = num_activations.wrapping_add(*act_sizes.offset(i as isize) as size_t);
    });
    let mut acts_memory: *mut f32 =
        malloc(num_activations.wrapping_mul(::core::mem::size_of::<f32>() as u64)) as *mut f32;
    let mut ptrs: [*mut *mut f32; 23] = [
        &mut (*acts).encoded,
        &mut (*acts).ln1,
        &mut (*acts).ln1_mean,
        &mut (*acts).ln1_rstd,
        &mut (*acts).qkv,
        &mut (*acts).atty,
        &mut (*acts).preatt,
        &mut (*acts).att,
        &mut (*acts).attproj,
        &mut (*acts).residual2,
        &mut (*acts).ln2,
        &mut (*acts).ln2_mean,
        &mut (*acts).ln2_rstd,
        &mut (*acts).fch,
        &mut (*acts).fch_gelu,
        &mut (*acts).fcproj,
        &mut (*acts).residual3,
        &mut (*acts).lnf,
        &mut (*acts).lnf_mean,
        &mut (*acts).lnf_rstd,
        &mut (*acts).logits,
        &mut (*acts).probs,
        &mut (*acts).losses,
    ];
    let mut acts_memory_iterator: *mut f32 = acts_memory;
    (0..23 as usize).into_iter().for_each(|i_0| {
        *ptrs[i_0] = acts_memory_iterator;
        acts_memory_iterator =
            acts_memory_iterator.offset(*act_sizes.offset(i_0 as isize) as isize);
    });
    return acts_memory;
}

pub unsafe extern "C" fn gpt2_build_from_checkpoint(
    mut model: *mut GPT2,
    mut checkpoint_path: *const u8,
) {
    let mut model_file: *mut FILE = fopen(checkpoint_path, b"rb\0" as *const u8 as *const u8);
    if model_file.is_null() {
        printf(b"Error opening model file\n\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    let mut model_header: [i32; 256] = [0; 256];
    fread(
        model_header.as_mut_ptr() as *mut usize,
        ::core::mem::size_of::<i32>() as u64,
        256 as i32 as u64,
        model_file,
    );
    if model_header[0 as i32 as usize] != 20240326 as i32 {
        printf(b"Bad magic model file\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    if model_header[1 as i32 as usize] != 1 as i32 {
        printf(b"Bad version in model file\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    let mut maxT: size_t = 0;
    let mut V: size_t = 0;
    let mut L: size_t = 0;
    let mut NH: size_t = 0;
    let mut C: size_t = 0;
    maxT = model_header[2 as i32 as usize] as size_t;
    (*model).config.max_seq_len = maxT as i32;
    V = model_header[3 as i32 as usize] as size_t;
    (*model).config.vocab_size = V as i32;
    L = model_header[4 as i32 as usize] as size_t;
    (*model).config.num_layers = L as i32;
    NH = model_header[5 as i32 as usize] as size_t;
    (*model).config.num_heads = NH as i32;
    C = model_header[6 as i32 as usize] as size_t;
    (*model).config.channels = C as i32;
    printf(b"[GPT-2]\n\0" as *const u8 as *const u8);
    printf(b"max_seq_len: %zu\n\0" as *const u8 as *const u8, maxT);
    printf(b"vocab_size: %zu\n\0" as *const u8 as *const u8, V);
    printf(b"num_layers: %zu\n\0" as *const u8 as *const u8, L);
    printf(b"num_heads: %zu\n\0" as *const u8 as *const u8, NH);
    printf(b"channels: %zu\n\0" as *const u8 as *const u8, C);
    (*model).param_sizes[0 as i32 as usize] = V.wrapping_mul(C);
    (*model).param_sizes[1 as i32 as usize] = maxT.wrapping_mul(C);
    (*model).param_sizes[2 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[3 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[4 as i32 as usize] = L
        .wrapping_mul((3 as i32 as u64).wrapping_mul(C))
        .wrapping_mul(C);
    (*model).param_sizes[5 as i32 as usize] = L.wrapping_mul((3 as i32 as u64).wrapping_mul(C));
    (*model).param_sizes[6 as i32 as usize] = L.wrapping_mul(C).wrapping_mul(C);
    (*model).param_sizes[7 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[8 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[9 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[10 as i32 as usize] = L
        .wrapping_mul((4 as i32 as u64).wrapping_mul(C))
        .wrapping_mul(C);
    (*model).param_sizes[11 as i32 as usize] = L.wrapping_mul((4 as i32 as u64).wrapping_mul(C));
    (*model).param_sizes[12 as i32 as usize] = L
        .wrapping_mul(C)
        .wrapping_mul((4 as i32 as u64).wrapping_mul(C));
    (*model).param_sizes[13 as i32 as usize] = L.wrapping_mul(C);
    (*model).param_sizes[14 as i32 as usize] = C;
    (*model).param_sizes[15 as i32 as usize] = C;
    let mut num_parameters: size_t = 0 as i32 as size_t;
    (0..16 as usize).into_iter().for_each(|i| {
        num_parameters = num_parameters.wrapping_add((*model).param_sizes[i] as size_t);
    });
    printf(
        b"num_parameters: %zu\n\0" as *const u8 as *const u8,
        num_parameters,
    );
    (*model).num_parameters = num_parameters;
    (*model).params_memory =
        malloc_and_point_parameters(&mut (*model).params, ((*model).param_sizes).as_mut_ptr());
    fread(
        (*model).params_memory as *mut usize,
        ::core::mem::size_of::<f32>() as u64,
        num_parameters,
        model_file,
    );
    fclose(model_file);
    (*model).acts_memory = 0 as *mut f32;
    (*model).grads_memory = 0 as *mut f32;
    (*model).m_memory = 0 as *mut f32;
    (*model).v_memory = 0 as *mut f32;
    (*model).grads_acts_memory = 0 as *mut f32;
    (*model).inputs = 0 as *mut i32;
    (*model).targets = 0 as *mut i32;
    (*model).batch_size = 0 as i32;
    (*model).seq_len = 0 as i32;
    (*model).mean_loss = -1.0f32;
}

pub unsafe extern "C" fn gpt2_forward(
    mut model: *mut GPT2,
    mut inputs: *mut i32,
    mut targets: *mut i32,
    mut B: size_t,
    mut T: size_t,
) {
    if ((*model).params_memory).is_null() {
        printf(b"Error: model was not initialized properly.\n\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    let mut V: size_t = (*model).config.vocab_size as size_t;
    let mut L: size_t = (*model).config.num_layers as size_t;
    let mut NH: size_t = (*model).config.num_heads as size_t;
    let mut C: size_t = (*model).config.channels as size_t;
    (0..(B.wrapping_mul(T) as usize)).into_iter().for_each(|i| {
        if !((0 <= *inputs.offset(i as isize)) && (*inputs.offset(i as isize) as u64) < V) {
            panic!("Assertion failed: 0 <= inputs[i] && inputs[i] < V, file /home/linuxohos/demo2/train_gpt2.c, line 677");
        }

        // Check targets if not null
        if !targets.is_null() {
            if !((0 <= *targets.offset(i as isize)) && (*targets.offset(i as isize) as u64) < V) {
                panic!("Assertion failed: 0 <= targets[i] && targets[i] < V, file /home/linuxohos/demo2/train_gpt2.c, line 679");
            }
        }
    });

    if ((*model).acts_memory).is_null() {
        (*model).batch_size = B as i32;
        (*model).seq_len = T as i32;
        (*model).act_sizes[0 as i32 as usize] = B.wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[1 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[2 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T);
        (*model).act_sizes[3 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T);
        (*model).act_sizes[4 as i32 as usize] = L
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(3 as i32 as u64)
            .wrapping_mul(C);
        (*model).act_sizes[5 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[6 as i32 as usize] = L
            .wrapping_mul(B)
            .wrapping_mul(NH)
            .wrapping_mul(T)
            .wrapping_mul(T);
        (*model).act_sizes[7 as i32 as usize] = L
            .wrapping_mul(B)
            .wrapping_mul(NH)
            .wrapping_mul(T)
            .wrapping_mul(T);
        (*model).act_sizes[8 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[9 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[10 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[11 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T);
        (*model).act_sizes[12 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T);
        (*model).act_sizes[13 as i32 as usize] = L
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(4 as i32 as u64)
            .wrapping_mul(C);
        (*model).act_sizes[14 as i32 as usize] = L
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(4 as i32 as u64)
            .wrapping_mul(C);
        (*model).act_sizes[15 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[16 as i32 as usize] = L.wrapping_mul(B).wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[17 as i32 as usize] = B.wrapping_mul(T).wrapping_mul(C);
        (*model).act_sizes[18 as i32 as usize] = B.wrapping_mul(T);
        (*model).act_sizes[19 as i32 as usize] = B.wrapping_mul(T);
        (*model).act_sizes[20 as i32 as usize] = B.wrapping_mul(T).wrapping_mul(V);
        (*model).act_sizes[21 as i32 as usize] = B.wrapping_mul(T).wrapping_mul(V);
        (*model).act_sizes[22 as i32 as usize] = B.wrapping_mul(T);
        let mut num_activations: size_t = 0 as i32 as size_t;
        (0..23 as usize).into_iter().for_each(|i_0| {
            num_activations = num_activations.wrapping_add((*model).act_sizes[i_0] as size_t);
        });
        printf(
            b"num_activations: %zu\n\0" as *const u8 as *const u8,
            num_activations,
        );
        (*model).num_activations = num_activations;
        (*model).acts_memory =
            malloc_and_point_activations(&mut (*model).acts, ((*model).act_sizes).as_mut_ptr());
        (*model).inputs = malloc(
            B.wrapping_mul(T)
                .wrapping_mul(::core::mem::size_of::<i32>() as u64),
        ) as *mut i32;
        (*model).targets = malloc(
            B.wrapping_mul(T)
                .wrapping_mul(::core::mem::size_of::<i32>() as u64),
        ) as *mut i32;
    } else if B != (*model).batch_size as u64 || T != (*model).seq_len as u64 {
        printf(
            b"Model: B=%d T=%d, Desired: B=%d T=%d\n\0" as *const u8 as *const u8,
            (*model).batch_size,
            (*model).seq_len,
            B as i32,
            T as i32,
        );
        exit(1 as i32);
    }
    memcpy(
        (*model).inputs as *mut usize,
        inputs as *const usize,
        B.wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<i32>() as u64),
    );
    if !targets.is_null() {
        memcpy(
            (*model).targets as *mut usize,
            targets as *const usize,
            B.wrapping_mul(T)
                .wrapping_mul(::core::mem::size_of::<i32>() as u64),
        );
    }
    let mut params: ParameterTensors = (*model).params;
    let mut acts: ActivationTensors = (*model).acts;
    let mut residual: *mut f32 = 0 as *mut f32;
    encoder_forward(
        acts.encoded,
        inputs,
        params.wte,
        params.wpe,
        B as usize,
        T as usize,
        C as usize,
    );
    (0..L as usize).into_iter().for_each(|l| {
        let residual = if l == 0 {
            acts.encoded
        } else {
            acts.residual3.offset(((l - 1) as u64 * B * T * C) as isize)
        };

        let l_ln1w = params.ln1w.offset((l as u64 * C) as isize);
        let l_ln1b = params.ln1b.offset((l as u64 * C) as isize);
        let l_qkvw = params.qkvw.offset(((l * 3) as u64 * C * C) as isize);
        let l_qkvb = params.qkvb.offset(((l * 3) as u64 * C) as isize);
        let l_attprojw = params.attprojw.offset((l as u64 * C * C) as isize);
        let l_attprojb = params.attprojb.offset((l as u64 * C) as isize);
        let l_ln2w = params.ln2w.offset((l as u64 * C) as isize);
        let l_ln2b = params.ln2b.offset((l as u64 * C) as isize);
        let l_fcw = params.fcw.offset(((l * 4) as u64 * C * C) as isize);
        let l_fcb = params.fcb.offset(((l * 4) as u64 * C) as isize);
        let l_fcprojw = params.fcprojw.offset((l as u64 * C * 4 * C) as isize);
        let l_fcprojb = params.fcprojb.offset((l as u64 * C) as isize);

        let l_ln1 = acts.ln1.offset((l as u64 * B * T * C) as isize);
        let l_ln1_mean = acts.ln1_mean.offset((l as u64 * B * T) as isize);
        let l_ln1_rstd = acts.ln1_rstd.offset((l as u64 * B * T) as isize);
        let l_qkv = acts.qkv.offset((l as u64 * B * T * 3 * C) as isize);
        let l_atty = acts.atty.offset((l as u64 * B * T * C) as isize);
        let l_preatt = acts.preatt.offset((l as u64 * B * NH * T * T) as isize);
        let l_att = acts.att.offset((l as u64 * B * NH * T * T) as isize);
        let l_attproj = acts.attproj.offset((l as u64 * B * T * C) as isize);
        let l_residual2 = acts.residual2.offset((l as u64 * B * T * C) as isize);
        let l_ln2 = acts.ln2.offset((l as u64 * B * T * C) as isize);
        let l_ln2_mean = acts.ln2_mean.offset((l as u64 * B * T) as isize);
        let l_ln2_rstd = acts.ln2_rstd.offset((l as u64 * B * T) as isize);
        let l_fch = acts.fch.offset((l as u64 * B * T * 4 * C) as isize);
        let l_fch_gelu = acts.fch_gelu.offset((l as u64 * B * T * 4 * C) as isize);
        let l_fcproj = acts.fcproj.offset((l as u64 * B * T * C) as isize);
        let l_residual3 = acts.residual3.offset((l as u64 * B * T * C) as isize);

        layernorm_forward(
            l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B as usize, T as usize, C as usize,
        );
        matmul_forward(
            l_qkv,
            l_ln1,
            l_qkvw,
            l_qkvb,
            B as usize,
            T as usize,
            C as usize,
            (3 as u64 * C) as usize,
        );
        attention_forward(
            l_atty, l_preatt, l_att, l_qkv, B as usize, T as usize, C as usize, NH as usize,
        );
        matmul_forward(
            l_attproj, l_atty, l_attprojw, l_attprojb, B as usize, T as usize, C as usize, C as usize,
        );
        residual_forward(l_residual2, residual, l_attproj, (B * T * C) as usize);
        layernorm_forward(
            l_ln2,
            l_ln2_mean,
            l_ln2_rstd,
            l_residual2,
            l_ln2w,
            l_ln2b,
            B as usize,
            T as usize,
            C as usize,
        );
        matmul_forward(
            l_fch,
            l_ln2,
            l_fcw,
            l_fcb,
            B as usize,
            T as usize,
            C as usize,
            (4 * C) as usize,
        );
        gelu_forward(l_fch_gelu, l_fch, (B * T * 4 * C) as usize);
        matmul_forward(
            l_fcproj,
            l_fch_gelu,
            l_fcprojw,
            l_fcprojb,
            B as usize,
            T as usize,
            (4 * C) as usize,
            C as usize,
        );
        residual_forward(l_residual3, l_residual2, l_fcproj, (B * T * C) as usize);
    });

    residual = (acts.residual3).offset(
        L.wrapping_sub(1 as i32 as u64)
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(C) as isize,
    );
    layernorm_forward(
        acts.lnf,
        acts.lnf_mean,
        acts.lnf_rstd,
        residual,
        params.lnfw,
        params.lnfb,
        B as usize,
        T as usize,
        C as usize,
    );
    matmul_forward(
        acts.logits,
        acts.lnf,
        params.wte,
        0 as *mut f32,
        B as usize,
        T as usize,
        C as usize,
        V as usize,
    );
    softmax_forward(acts.probs, acts.logits, B as i32, T as i32, V as i32);
    if !targets.is_null() {
        crossentropy_forward(
            (*model).acts.losses,
            (*model).acts.probs,
            targets,
            B as i32,
            T as i32,
            V as i32,
        );
        let mut mean_loss: f32 = 0.0f32;
        (0..(B.wrapping_mul(T) as usize)).into_iter().for_each(|i_1| {
            mean_loss += *((*model).acts.losses.offset(i_1 as isize));
        });
        mean_loss /= B.wrapping_mul(T) as f32;
        (*model).mean_loss = mean_loss;
    } else {
        (*model).mean_loss = -1.0f32;
    };
}

pub unsafe extern "C" fn gpt2_zero_grad(mut model: *mut GPT2) {
    if !((*model).grads_memory).is_null() {
        memset(
            (*model).grads_memory as *mut usize,
            0 as i32,
            ((*model).num_parameters).wrapping_mul(::core::mem::size_of::<f32>() as u64),
        );
    }
    if !((*model).grads_acts_memory).is_null() {
        memset(
            (*model).grads_acts_memory as *mut usize,
            0 as i32,
            ((*model).num_activations).wrapping_mul(::core::mem::size_of::<f32>() as u64),
        );
    }
}

pub unsafe extern "C" fn gpt2_backward(mut model: *mut GPT2) {
    if (*model).mean_loss == -1.0f32 {
        printf(b"Error: must forward with targets before backward\n\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    if ((*model).grads_memory).is_null() {
        (*model).grads_memory =
            malloc_and_point_parameters(&mut (*model).grads, ((*model).param_sizes).as_mut_ptr());
        (*model).grads_acts_memory = malloc_and_point_activations(
            &mut (*model).grads_acts,
            ((*model).act_sizes).as_mut_ptr(),
        );
        gpt2_zero_grad(model);
    }
    let mut B: size_t = (*model).batch_size as size_t;
    let mut T: size_t = (*model).seq_len as size_t;
    let mut V: size_t = (*model).config.vocab_size as size_t;
    let mut L: size_t = (*model).config.num_layers as size_t;
    let mut NH: size_t = (*model).config.num_heads as size_t;
    let mut C: size_t = (*model).config.channels as size_t;
    let mut params: ParameterTensors = (*model).params;
    let mut grads: ParameterTensors = (*model).grads;
    let mut acts: ActivationTensors = (*model).acts;
    let mut grads_acts: ActivationTensors = (*model).grads_acts;
    let mut dloss_mean: f32 = 1.0f32 / B.wrapping_mul(T) as f32;
    (0..(B.wrapping_mul(T) as usize)).into_iter().for_each(|i| {
        unsafe {
            *(grads_acts.losses.offset(i as isize)) = dloss_mean;
        }
    });
    crossentropy_softmax_backward(
        grads_acts.logits,
        grads_acts.losses,
        acts.probs,
        (*model).targets,
        B as i32,
        T as i32,
        V as i32,
    );
    matmul_backward(
        grads_acts.lnf,
        grads.wte,
        0 as *mut f32,
        grads_acts.logits,
        acts.lnf,
        params.wte,
        B as usize,
        T as usize,
        C as usize,
        V as usize,
    );
    let mut residual: *mut f32 = (acts.residual3).offset(
        L.wrapping_sub(1 as i32 as u64)
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(C) as isize,
    );
    let mut dresidual: *mut f32 = (grads_acts.residual3).offset(
        L.wrapping_sub(1 as i32 as u64)
            .wrapping_mul(B)
            .wrapping_mul(T)
            .wrapping_mul(C) as isize,
    );
    layernorm_backward(
        dresidual,
        grads.lnfw,
        grads.lnfb,
        grads_acts.lnf,
        residual,
        params.lnfw,
        acts.lnf_mean,
        acts.lnf_rstd,
        B as usize,
        T as usize,
        C as usize,
    );
    ((0..L).rev()).for_each(|l| {
        let residual = if l == 0 {
            acts.encoded
        } else {
            acts.residual3.offset(((l - 1) as u64 * B * T * C) as isize)
        };

        let dresidual = if l == 0 {
            grads_acts.encoded
        } else {
            grads_acts
                .residual3
                .offset(((l - 1) as u64 * B * T * C) as isize)
        };

        // Example of accessing a pointer for a specific layer parameter
        let l_ln1w = params.ln1w.offset((l as u64 * C) as isize);
        let mut l_qkvw: *mut f32 =
            (params.qkvw).offset((l * 3 as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut l_attprojw: *mut f32 =
            (params.attprojw).offset((l as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut l_ln2w: *mut f32 = (params.ln2w).offset((l as u64).wrapping_mul(C) as isize);
        let mut l_fcw: *mut f32 =
            (params.fcw).offset((l * 4 as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut l_fcprojw: *mut f32 = (params.fcprojw).offset(
            (l as u64)
                .wrapping_mul(C)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_ln1w: *mut f32 = (grads.ln1w).offset((l as u64).wrapping_mul(C) as isize);
        let mut dl_ln1b: *mut f32 = (grads.ln1b).offset((l as u64).wrapping_mul(C) as isize);
        let mut dl_qkvw: *mut f32 =
            (grads.qkvw).offset((l * 3 as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut dl_qkvb: *mut f32 = (grads.qkvb).offset((l * 3 as u64).wrapping_mul(C) as isize);
        let mut dl_attprojw: *mut f32 =
            (grads.attprojw).offset((l as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut dl_attprojb: *mut f32 =
            (grads.attprojb).offset((l as u64).wrapping_mul(C) as isize);
        let mut dl_ln2w: *mut f32 = (grads.ln2w).offset((l as u64).wrapping_mul(C) as isize);
        let mut dl_ln2b: *mut f32 = (grads.ln2b).offset((l as u64).wrapping_mul(C) as isize);
        let mut dl_fcw: *mut f32 =
            (grads.fcw).offset((l * 4 as u64).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut dl_fcb: *mut f32 = (grads.fcb).offset((l * 4 as u64).wrapping_mul(C) as isize);
        let mut dl_fcprojw: *mut f32 = (grads.fcprojw).offset(
            (l as u64)
                .wrapping_mul(C)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_fcprojb: *mut f32 = (grads.fcprojb).offset((l as u64).wrapping_mul(C) as isize);
        let mut l_ln1: *mut f32 =
            (acts.ln1).offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut l_ln1_mean: *mut f32 =
            (acts.ln1_mean).offset((l as u64).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_ln1_rstd: *mut f32 =
            (acts.ln1_rstd).offset((l as u64).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_qkv: *mut f32 = (acts.qkv).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(3 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut l_atty: *mut f32 =
            (acts.atty).offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut l_att: *mut f32 = (acts.att).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(NH)
                .wrapping_mul(T)
                .wrapping_mul(T) as isize,
        );
        let mut l_residual2: *mut f32 = (acts.residual2)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut l_ln2: *mut f32 =
            (acts.ln2).offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut l_ln2_mean: *mut f32 =
            (acts.ln2_mean).offset((l as u64).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_ln2_rstd: *mut f32 =
            (acts.ln2_rstd).offset((l as u64).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_fch: *mut f32 = (acts.fch).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut l_fch_gelu: *mut f32 = (acts.fch_gelu).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_ln1: *mut f32 = (grads_acts.ln1)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_qkv: *mut f32 = (grads_acts.qkv).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(3 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_atty: *mut f32 = (grads_acts.atty)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_preatt: *mut f32 = (grads_acts.preatt).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(NH)
                .wrapping_mul(T)
                .wrapping_mul(T) as isize,
        );
        let mut dl_att: *mut f32 = (grads_acts.att).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(NH)
                .wrapping_mul(T)
                .wrapping_mul(T) as isize,
        );
        let mut dl_attproj: *mut f32 = (grads_acts.attproj)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_residual2: *mut f32 = (grads_acts.residual2)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_ln2: *mut f32 = (grads_acts.ln2)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_fch: *mut f32 = (grads_acts.fch).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_fch_gelu: *mut f32 = (grads_acts.fch_gelu).offset(
            (l as u64)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as isize,
        );
        let mut dl_fcproj: *mut f32 = (grads_acts.fcproj)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        let mut dl_residual3: *mut f32 = (grads_acts.residual3)
            .offset((l as u64).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C) as isize);
        // Repeat similar offset calculations for other parameters as in the original code...

        // Functions are called here for the backward pass for each layer
        residual_backward(
            dl_residual2,
            dl_fcproj,
            dl_residual3,
            B.wrapping_mul(T).wrapping_mul(C) as usize,
        );
        matmul_backward(
            dl_fch_gelu,
            dl_fcprojw,
            dl_fcprojb,
            dl_fcproj,
            l_fch_gelu,
            l_fcprojw,
            B as usize,
            T as usize,
            (4 as i32 as u64).wrapping_mul(C) as usize,
            C as usize,
        );
        gelu_backward(
            dl_fch,
            l_fch,
            dl_fch_gelu,
            B.wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as usize,
        );
        matmul_backward(
            dl_ln2,
            dl_fcw,
            dl_fcb,
            dl_fch,
            l_ln2,
            l_fcw,
            B as usize,
            T as usize,
            C as usize,
            (4 as i32 as u64).wrapping_mul(C) as usize,
        );
        layernorm_backward(
            dl_residual2,
            dl_ln2w,
            dl_ln2b,
            dl_ln2,
            l_residual2,
            l_ln2w,
            l_ln2_mean,
            l_ln2_rstd,
            B as usize,
            T as usize,
            C as usize,
        );
        residual_backward(
            dresidual,
            dl_attproj,
            dl_residual2,
            B.wrapping_mul(T).wrapping_mul(C) as usize,
        );
        matmul_backward(
            dl_atty,
            dl_attprojw,
            dl_attprojb,
            dl_attproj,
            l_atty,
            l_attprojw,
            B as usize,
            T as usize,
            C as usize,
            C as usize,
        );
        attention_backward(
            dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B as usize, T as usize, C as usize,
            NH as usize,
        );
        matmul_backward(
            dl_ln1,
            dl_qkvw,
            dl_qkvb,
            dl_qkv,
            l_ln1,
            l_qkvw,
            B as usize,
            T as usize,
            C as usize,
            (3 as i32 as u64).wrapping_mul(C) as usize,
        );
        layernorm_backward(
            dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd,
            B as usize, T as usize, C as usize,
        );
    });
    encoder_backward(
        grads.wte,
        grads.wpe,
        grads_acts.encoded,
        (*model).inputs,
        B as usize,
        T as usize,
        C as usize,
    );
}

pub unsafe extern "C" fn gpt2_update(
    mut model: *mut GPT2,
    mut learning_rate: f32,
    mut beta1: f32,
    mut beta2: f32,
    mut eps: f32,
    mut weight_decay: f32,
    mut t: i32,
) {
    if ((*model).m_memory).is_null() {
        (*model).m_memory = calloc(
            (*model).num_parameters,
            ::core::mem::size_of::<f32>() as u64,
        ) as *mut f32;
        (*model).v_memory = calloc(
            (*model).num_parameters,
            ::core::mem::size_of::<f32>() as u64,
        ) as *mut f32;
    }
    (0..(*model).num_parameters as usize).into_iter().for_each(|i| {
        let param: f32 = *((*model).params_memory.offset(i as isize));
        let grad: f32 = *((*model).grads_memory.offset(i as isize));
        let m: f32 = beta1 * *((*model).m_memory.offset(i as isize)) + (1.0 - beta1) * grad;
        let v: f32 = beta2 * *((*model).v_memory.offset(i as isize)) + (1.0 - beta2) * grad * grad;
        let m_hat: f32 = m / (1.0 - powf(beta1, t as f32));
        let v_hat: f32 = v / (1.0 - powf(beta2, t as f32));

        *((*model).m_memory.offset(i as isize)) = m;
        *((*model).v_memory.offset(i as isize)) = v;
        *((*model).params_memory.offset(i as isize)) -=
            learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    });
}

pub unsafe extern "C" fn gpt2_free(mut model: *mut GPT2) {
    free((*model).params_memory as *mut usize);
    free((*model).grads_memory as *mut usize);
    free((*model).m_memory as *mut usize);
    free((*model).v_memory as *mut usize);
    free((*model).acts_memory as *mut usize);
    free((*model).grads_acts_memory as *mut usize);
    free((*model).inputs as *mut usize);
    free((*model).targets as *mut usize);
}

pub unsafe extern "C" fn dataloader_init(
    mut loader: *mut DataLoader,
    mut filename: *const u8,
    mut B: i32,
    mut T: i32,
) {
    (*loader).B = B;
    (*loader).T = T;
    (*loader).tokens_file = fopen(filename, b"rb\0" as *const u8 as *const u8);
    if ((*loader).tokens_file).is_null() {
        printf(b"Error opening tokens file\n\0" as *const u8 as *const u8);
        exit(1 as i32);
    }
    fseek((*loader).tokens_file, 0 as i32 as i64, 2 as i32);
    (*loader).file_size = ftell((*loader).tokens_file);
    fseek((*loader).tokens_file, 0 as i32 as i64, 0 as i32);
    if ((*loader).file_size as u64)
        < ((B * T + 1 as i32) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64)
    {
        printf(
            b"Error: file size is too small for the batch size and sequence length\n\0" as *const u8
                as *const u8,
        );
        exit(1 as i32);
    }
    (*loader).current_position = 0 as i32 as i64;
    (*loader).batch =
        malloc(((B * T + 1 as i32) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64))
            as *mut i32;
    (*loader).inputs = (*loader).batch;
    (*loader).targets = ((*loader).batch).offset(1 as i32 as isize);
    (*loader).num_batches = ((*loader).file_size as u64)
        .wrapping_div(((B * T) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64))
        as i32;
}

pub unsafe extern "C" fn dataloader_reset(mut loader: *mut DataLoader) {
    (*loader).current_position = 0 as i32 as i64;
}

pub unsafe extern "C" fn dataloader_next_batch(mut loader: *mut DataLoader) {
    let mut B: i32 = (*loader).B;
    let mut T: i32 = (*loader).T;
    if ((*loader).current_position as u64).wrapping_add(
        ((B * T + 1 as i32) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64),
    ) > (*loader).file_size as u64
    {
        (*loader).current_position = 0 as i32 as i64;
    }
    fseek((*loader).tokens_file, (*loader).current_position, 0 as i32);
    fread(
        (*loader).batch as *mut usize,
        ::core::mem::size_of::<i32>() as u64,
        (B * T + 1 as i32) as u64,
        (*loader).tokens_file,
    );
    (*loader).current_position = ((*loader).current_position as u64)
        .wrapping_add(((B * T) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64))
        as i64 as i64;
}

pub unsafe extern "C" fn dataloader_free(mut loader: *mut DataLoader) {
    fclose((*loader).tokens_file);
    free((*loader).batch as *mut usize);
}

pub unsafe extern "C" fn random_u32(mut state: *mut u64) -> u32 {
    *state ^= *state >> 12 as i32;
    *state ^= *state << 25 as i32;
    *state ^= *state >> 27 as i32;
    return ((*state).wrapping_mul(0x2545f4914f6cdd1d as u64) >> 32 as i32) as u32;
}

pub unsafe extern "C" fn random_f32(mut state: *mut u64) -> f32 {
    return (random_u32(state) >> 8 as i32) as f32 / 16777216.0f32;
}

pub unsafe extern "C" fn sample_mult(
    mut probabilities: *mut f32,
    mut n: i32,
    mut coin: f32,
) -> i32 {
    let mut cdf: f32 = 0.0f32;
    for i in 0..n {
        cdf += *probabilities.offset(i as isize);
        if coin < cdf {
            return i;
        }
    }

    return n - 1 as i32;
}

pub unsafe extern "C" fn safe_printf(mut piece: *const u8) {
    if piece.is_null() {
        return;
    }
    if *piece.offset(0 as i32 as isize) as i32 == '\0' as i32 {
        return;
    }
    if *piece.offset(1 as i32 as isize) as i32 == '\0' as i32 {
        let mut byte_val: u8 = *piece.offset(0 as i32 as isize) as u8;
        if !(*(*__ctype_b_loc()).offset(byte_val as i32 as isize) as i32
            & _ISprint as i32 as u16 as i32
            != 0
            || *(*__ctype_b_loc()).offset(byte_val as i32 as isize) as i32
                & _ISspace as i32 as u16 as i32
                != 0)
        {
            return;
        }
    }
    printf(b"%s\0" as *const u8 as *const u8, piece);
}

pub unsafe extern "C" fn tokenizer_init(mut tokenizer: *mut Tokenizer, mut filename: *const u8) {
    let mut file: *mut FILE = fopen(filename, b"rb\0" as *const u8 as *const u8);
    if file.is_null() {
        printf(b"---\n\0" as *const u8 as *const u8);
        printf(
            b"WARNING: Failed to open the tokenizer file %s\n\0" as *const u8 as *const u8,
            filename,
        );
        printf(
            b"The Tokenizer is a new feature added April 14 2024.\n\0" as *const u8 as *const u8,
        );
        printf(b"Re-run `python train_gpt2.py` to write it\n\0" as *const u8 as *const u8);
        printf(b"---\n\0" as *const u8 as *const u8);
        (*tokenizer).init_ok = 0 as i32;
        return;
    }
    let mut header: [uint32_t; 256] = [0; 256];
    fread(
        header.as_mut_ptr() as *mut usize,
        ::core::mem::size_of::<uint32_t>() as u64,
        256 as i32 as u64,
        file,
    );
    if header[0 as i32 as usize] == 20240328 as i32 as u32 {
    } else {
        __assert_fail(
            b"header[0] == 20240328\0" as *const u8 as *const u8,
            b"/home/linuxohos/demo2/train_gpt2.c\0" as *const u8 as *const u8,
            1105 as i32 as u32,
            (*::core::mem::transmute::<&[u8; 47], &[u8; 47]>(
                b"void tokenizer_init(Tokenizer *, const char *)\0",
            ))
            .as_ptr(),
        );
    }
    'c_12277: {
        if header[0 as i32 as usize] == 20240328 as i32 as u32 {
        } else {
            __assert_fail(
                b"header[0] == 20240328\0" as *const u8 as *const u8,
                b"/home/linuxohos/demo2/train_gpt2.c\0" as *const u8 as *const u8,
                1105 as i32 as u32,
                (*::core::mem::transmute::<&[u8; 47], &[u8; 47]>(
                    b"void tokenizer_init(Tokenizer *, const char *)\0",
                ))
                .as_ptr(),
            );
        }
    };
    if header[1 as i32 as usize] == 1 as i32 as u32 {
    } else {
        __assert_fail(
            b"header[1] == 1\0" as *const u8 as *const u8,
            b"/home/linuxohos/demo2/train_gpt2.c\0" as *const u8 as *const u8,
            1106 as i32 as u32,
            (*::core::mem::transmute::<&[u8; 47], &[u8; 47]>(
                b"void tokenizer_init(Tokenizer *, const char *)\0",
            ))
            .as_ptr(),
        );
    }
    'c_12233: {
        if header[1 as i32 as usize] == 1 as i32 as u32 {
        } else {
            __assert_fail(
                b"header[1] == 1\0" as *const u8 as *const u8,
                b"/home/linuxohos/demo2/train_gpt2.c\0" as *const u8 as *const u8,
                1106 as i32 as u32,
                (*::core::mem::transmute::<&[u8; 47], &[u8; 47]>(
                    b"void tokenizer_init(Tokenizer *, const char *)\0",
                ))
                .as_ptr(),
            );
        }
    };
    (*tokenizer).vocab_size = header[2 as i32 as usize];
    let mut length: u8 = 0;
    (*tokenizer).token_table = malloc(
        ((*tokenizer).vocab_size as u64).wrapping_mul(::core::mem::size_of::<*mut u8>() as u64),
    ) as *mut *mut u8;
    (0..(*tokenizer).vocab_size).into_iter().for_each(|i| {
        let mut length: u8 = 0;
        fread(
            &mut length as *mut u8 as *mut _,
            ::core::mem::size_of::<u8>() as u64,
            1,
            file,
        );
        assert!(length > 0, "length must be greater than 0");

        // Allocate memory for the token plus a null terminator
        let token_bytes = unsafe { malloc((length as usize + 1) as u64) as *mut u8 };
        if token_bytes.is_null() {
            panic!("Failed to allocate memory for token_bytes");
        }

        // Read the token data into the allocated buffer
        fread(
            token_bytes as *mut _,
            ::core::mem::size_of::<u8>() as u64,
            length as u64,
            file,
        );
        *token_bytes.offset(length as isize) = b'\0'; // Append null terminator
        *((*tokenizer).token_table.offset(i as isize)) = token_bytes;
    });

    fclose(file);
    (*tokenizer).init_ok = 1 as i32;
}

pub unsafe extern "C" fn tokenizer_decode(
    mut tokenizer: *mut Tokenizer,
    mut token_id: uint32_t,
) -> *const u8 {
    if (*tokenizer).init_ok == 0 as i32 {
        return 0 as *const u8;
    }
    if token_id < (*tokenizer).vocab_size {
        return *((*tokenizer).token_table).offset(token_id as isize);
    } else {
        printf(
            b"invalid token id %d!\n\0" as *const u8 as *const u8,
            token_id,
        );
        return 0 as *const u8;
    };
}

pub unsafe extern "C" fn tokenizer_free(mut tokenizer: *mut Tokenizer) {
    if (*tokenizer).init_ok != 0 {
        (0..(*tokenizer).vocab_size as usize).into_iter().for_each(|i| {
            free(*((*tokenizer).token_table.offset(i as isize)) as *mut usize);
        });
        free((*tokenizer).token_table as *mut usize);
    }
}

pub fn main() {
    let mut model: GPT2 = GPT2 {
        config: GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        },
        params: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        param_sizes: [0; 16],
        params_memory: 0 as *mut f32,
        num_parameters: 0,
        grads: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        grads_memory: 0 as *mut f32,
        m_memory: 0 as *mut f32,
        v_memory: 0 as *mut f32,
        acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        act_sizes: [0; 23],
        acts_memory: 0 as *mut f32,
        num_activations: 0,
        grads_acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        grads_acts_memory: 0 as *mut f32,
        batch_size: 0,
        seq_len: 0,
        inputs: 0 as *mut i32,
        targets: 0 as *mut i32,
        mean_loss: 0.,
    };
    
    unsafe {
        // build the GPT-2 model from a checkpoint
        gpt2_build_from_checkpoint(&mut model, b"gpt2_124M.bin\0" as *const u8 as *const u8);
        
        let mut tiny_stories_train: *const u8 =
            b"data/TinyStories_train.bin\0" as *const u8 as *const u8;
        let mut tiny_stories_val: *const u8 = b"data/TinyStories_val.bin\0" as *const u8 as *const u8;
        let mut tiny_shakespeare_train: *const u8 =
            b"data/tiny_shakespeare_train.bin\0" as *const u8 as *const u8;
        let mut tiny_shakespeare_val: *const u8 =
            b"data/tiny_shakespeare_val.bin\0" as *const u8 as *const u8;
        let mut train_tokens: *const u8 = if access(tiny_shakespeare_train, 0 as i32) != -(1 as i32) {
            tiny_shakespeare_train
        } else {
            tiny_stories_train
        };
        let mut val_tokens: *const u8 = if access(tiny_shakespeare_val, 0 as i32) != -(1 as i32) {
            tiny_shakespeare_val
        } else {
            tiny_stories_val
        };
        let mut B: i32 = 4 as i32;
        let mut T: i32 = 64 as i32;
        let mut train_loader: DataLoader = DataLoader {
            B: 0,
            T: 0,
            tokens_file: 0 as *mut FILE,
            file_size: 0,
            current_position: 0,
            batch: 0 as *mut i32,
            inputs: 0 as *mut i32,
            targets: 0 as *mut i32,
            num_batches: 0,
        };
        dataloader_init(&mut train_loader, train_tokens, B, T);
        printf(
            b"train dataset num_batches: %d\n\0" as *const u8 as *const u8,
            train_loader.num_batches,
        );
        let mut val_loader: DataLoader = DataLoader {
            B: 0,
            T: 0,
            tokens_file: 0 as *mut FILE,
            file_size: 0,
            current_position: 0,
            batch: 0 as *mut i32,
            inputs: 0 as *mut i32,
            targets: 0 as *mut i32,
            num_batches: 0,
        };
        dataloader_init(&mut val_loader, val_tokens, B, T);
        printf(
            b"val dataset num_batches: %d\n\0" as *const u8 as *const u8,
            val_loader.num_batches,
        );
        let mut val_num_batches: i32 = 5 as i32;
        let mut tokenizer: Tokenizer = Tokenizer {
            vocab_size: 0,
            token_table: 0 as *mut *mut u8,
            init_ok: 0,
        };
        tokenizer_init(
            &mut tokenizer,
            b"gpt2_tokenizer.bin\0" as *const u8 as *const u8,
        );
        let mut rng_state: u64 = 1337 as i32 as u64;
        let mut gen_tokens: *mut i32 =
            malloc(((B * T) as u64).wrapping_mul(::core::mem::size_of::<i32>() as u64)) as *mut i32;
        let genT: i32 = 64 as i32;
        let mut start: timespec = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let mut end: timespec = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        for step in 0..=40 {
            if step % 10 == 0 {
                let mut val_loss: f32 = 0.0;
                dataloader_reset(&mut val_loader);
                (0..val_num_batches).into_iter().for_each(|i| {
                    dataloader_next_batch(&mut val_loader);
                    gpt2_forward(
                        &mut model,
                        val_loader.inputs,
                        val_loader.targets,
                        B as size_t,
                        T as size_t,
                    );
                    val_loss += model.mean_loss;
                });
                val_loss /= val_num_batches as f32;
                println!("val loss {:.3}", val_loss); // Assuming use of Rust's print macros for simplicity
            }
    
            if step > 0 && step % 20 == 0 {
                (0..(B * T)).into_iter().for_each(|i| {
                    unsafe {
                        *gen_tokens.offset(i as isize) = 50256;
                    }
                });
                println!("generating:\n---");
                (1..genT).into_iter().for_each(|t| {
                    gpt2_forward(
                        &mut model,
                        gen_tokens,
                        std::ptr::null_mut(),
                        B as size_t,
                        T as size_t,
                    );
                    unsafe {
                        let probs =
                            (model.acts.probs).offset(((t - 1) * model.config.vocab_size) as isize);
                        let coin = random_f32(&mut rng_state);
                        let next_token = sample_mult(probs, model.config.vocab_size, coin);
                        *gen_tokens.offset(t as isize) = next_token;
                        if tokenizer.init_ok != 0 {
                            let token_str = tokenizer_decode(&mut tokenizer, next_token as u32);
                            safe_printf(token_str); // Assumes `safe_printf` correctly handles C strings.
                        } else {
                            println!("{}", next_token);
                        }
                    }
                });
                println!("\n---");
            }
    
            // Simulation of time measurement
            let start = std::time::Instant::now();
            dataloader_next_batch(&mut train_loader);
            gpt2_forward(
                &mut model,
                train_loader.inputs,
                train_loader.targets,
                B as size_t,
                T as size_t,
            );
            gpt2_zero_grad(&mut model);
            gpt2_backward(&mut model);
            gpt2_update(&mut model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
            let time_elapsed = start.elapsed();
            println!(
                "step {}: train loss {:.3} (took {:.3} ms)",
                step,
                model.mean_loss,
                time_elapsed.as_secs_f64() * 1000.0
            );
        }
    
        dataloader_free(&mut train_loader);
        dataloader_free(&mut val_loader);
        tokenizer_free(&mut tokenizer);
        gpt2_free(&mut model);
        free(gen_tokens as *mut usize);
    }
}