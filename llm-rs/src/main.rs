#![allow(
    dead_code,
    mutable_transmutes,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]
#![feature(extern_types, label_break_value)]
use std::sync::atomic::{AtomicPtr, Ordering};

#[allow(unused_imports)]
use rayon::prelude::*;
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
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
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: i32,
    pub _flags2: i32,
    pub _old_offset: __off_t,
    pub _cur_column: u16,
    pub _vtable_offset: u8,
    pub _shortbuf: [u8; 1],
    pub _lock: *mut usize,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
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

pub unsafe extern "C" fn encoder_forward(
    mut out: *mut f32,
    mut inp: *mut i32,
    mut wte: *mut f32,
    mut wpe: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    for b in 0..B {
        for t in 0..T {
            let mut out_bt: *mut f32 = out.offset((b * T * C) as isize).offset((t * C) as isize);
            let ix: i32 = *inp.offset((b * T + t) as isize);
            let wte_ix: *mut f32 = wte.offset((ix * C) as isize);
            let wpe_t: *mut f32 = wpe.offset((t * C) as isize);
            (0..C).into_iter().for_each(|i| {
                *out_bt.offset(i as isize) = *wte_ix.offset(i as isize) + *wpe_t.offset(i as isize);
            });
        }
    }
}

pub unsafe extern "C" fn encoder_backward(
    mut dwte: *mut f32,
    mut dwpe: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut i32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    for b in 0..B {
        for t in 0..T {
            let mut dout_bt: *mut f32 = dout.offset((b * T * C) as isize).offset((t * C) as isize);
            let ix: i32 = *inp.offset((b * T + t) as isize);
            let mut dwte_ix: *mut f32 = dwte.offset((ix * C) as isize);
            let mut dwpe_t: *mut f32 = dwpe.offset((t * C) as isize);
            (0..C).into_iter().for_each(|i| {
                let d: f32 = *dout_bt.offset(i as isize);
                *dwte_ix.offset(i as isize) += d;
                *dwpe_t.offset(i as isize) += d;
            });
        }
    }
}

pub unsafe extern "C" fn layernorm_forward(
    mut out: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    let mut eps: f32 = 1e-5f32;
    for b in 0..B {
        for t in 0..T {
            let mut x: *mut f32 = inp.offset((b * T * C) as isize).offset((t * C) as isize);

            // Compute the mean
            let mut m: f32 = 0.0;
            (0..C).into_iter().for_each(|i| {
                m += *x.offset(i as isize);
            });
            m /= C as f32;

            // Compute the variance
            let mut v: f32 = 0.0;
            (0..C).into_iter().for_each(|i_0| {
                let xshift: f32 = *x.offset(i_0 as isize) - m;
                v += xshift * xshift;
            });
            v /= C as f32;

            // Compute the standard deviation
            let s: f32 = 1.0 / (v + eps).sqrt();

            // Apply normalization, scale by weight, add bias, and write to output
            let mut out_bt: *mut f32 = out.offset((b * T * C) as isize).offset((t * C) as isize);
            (0..C).into_iter().for_each(|i_1| {
                let n: f32 = s * (*x.offset(i_1 as isize) - m);
                let o: f32 = n * *weight.offset(i_1 as isize) + *bias.offset(i_1 as isize);
                *out_bt.offset(i_1 as isize) = o;
            });

            // Save the computed mean and reciprocal of standard deviation
            *mean.offset((b * T + t) as isize) = m;
            *rstd.offset((b * T + t) as isize) = s;
        }
    }
}

pub unsafe extern "C" fn layernorm_backward(
    mut dinp: *mut f32,
    mut dweight: *mut f32,
    mut dbias: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    for b in 0..B {
        for t in 0..T {
            let mut dout_bt: *mut f32 = dout.offset((b * T * C) as isize).offset((t * C) as isize);
            let mut inp_bt: *mut f32 = inp.offset((b * T * C) as isize).offset((t * C) as isize);
            let mut dinp_bt: *mut f32 = dinp.offset((b * T * C) as isize).offset((t * C) as isize);
            let mean_bt: f32 = *mean.offset((b * T + t) as isize);
            let rstd_bt: f32 = *rstd.offset((b * T + t) as isize);

            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;

            // Compute intermediate sums
            (0..C).into_iter().for_each(|i| {
                let norm_bti: f32 = (*inp_bt.offset(i as isize) - mean_bt) * rstd_bt;
                let dnorm_i: f32 = *weight.offset(i as isize) * *dout_bt.offset(i as isize);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            });

            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Update gradients and normalized inputs
            (0..C).into_iter().for_each(|i_0| {
                let norm_bti_0: f32 = (*inp_bt.offset(i_0 as isize) - mean_bt) * rstd_bt;
                let dnorm_i_0: f32 = *weight.offset(i_0 as isize) * *dout_bt.offset(i_0 as isize);

                *dbias.offset(i_0 as isize) += *dout_bt.offset(i_0 as isize);
                *dweight.offset(i_0 as isize) += norm_bti_0 * *dout_bt.offset(i_0 as isize);

                let mut dval: f32 = dnorm_i_0 - dnorm_mean - (norm_bti_0 * dnorm_norm_mean);
                dval *= rstd_bt;
                *dinp_bt.offset(i_0 as isize) += dval;
            });
        }
    }
}

pub unsafe extern "C" fn matmul_forward_naive(
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: i32,
    T: i32,
    C: i32,
    OC: i32,
) {
    let out_ptr = AtomicPtr::new(out);
    let inp_ptr = AtomicPtr::new(inp as *mut f32);
    let weight_ptr = AtomicPtr::new(weight as *mut f32);
    let bias_ptr = AtomicPtr::new(bias as *mut f32);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let bt = b * T + t;
            let out_bt = out_ptr.load(Ordering::SeqCst).offset((bt * OC) as isize);
            for o in 0..OC {
                let mut val = if !bias_ptr.load(Ordering::SeqCst).is_null() {
                    *bias_ptr.load(Ordering::SeqCst).offset(o as isize)
                } else {
                    0.0
                };

                for i in 0..C {
                    val += *inp_ptr.load(Ordering::SeqCst).offset((bt * C + i) as isize)
                        * *weight_ptr.load(Ordering::SeqCst).offset((o * C + i) as isize);
                }

                *out_bt.offset(o as isize) = val;
            }
        });
    });
}

pub unsafe extern "C" fn matmul_forward(
    mut out: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut OC: i32,
) {
    const LOOP_UNROLL: i32 = 8;

    if B * T % LOOP_UNROLL != 0 {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    let out_ptr = AtomicPtr::new(out);
    let inp_ptr = AtomicPtr::new(inp as *mut f32);
    let weight_ptr = AtomicPtr::new(weight as *mut f32);
    let bias_ptr = AtomicPtr::new(bias as *mut f32);

    (0..B * T / LOOP_UNROLL).into_par_iter().for_each(|obt| {
        for o in 0..OC {
            let mut result: [f32; LOOP_UNROLL as usize] = [0.0; LOOP_UNROLL as usize];
            if !bias_ptr.load(Ordering::SeqCst).is_null() {
                for ibt in 0..LOOP_UNROLL {
                    result[ibt as usize] = *bias_ptr.load(Ordering::SeqCst).offset(o as isize);
                }
            }

            for i in 0..C {
                let w = *weight_ptr.load(Ordering::SeqCst).offset((i + o * C) as isize);
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt * LOOP_UNROLL + ibt;
                    result[ibt as usize] += *inp_ptr.load(Ordering::SeqCst).offset((bt * C + i) as isize) * w;
                }
            }

            for ibt in 0..LOOP_UNROLL {
                let bt = obt * LOOP_UNROLL + ibt;
                *out_ptr.load(Ordering::SeqCst).offset((bt * OC + o) as isize) = result[ibt as usize];
            }
        }
    });
}

pub unsafe extern "C" fn matmul_backward(
    mut dinp: *mut f32,
    mut dweight: *mut f32,
    mut dbias: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut OC: i32,
) {
    let dinp_ptr = AtomicPtr::new(dinp);
    let dweight_ptr = AtomicPtr::new(dweight);
    let dbias_ptr = AtomicPtr::new(dbias);
    let dout_ptr = AtomicPtr::new(dout as *mut f32);
    let inp_ptr = AtomicPtr::new(inp as *mut f32);
    let weight_ptr = AtomicPtr::new(weight as *mut f32);

    // Backward into inp first, parallelize over B,T
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dout_bt = dout_ptr.load(Ordering::SeqCst).offset((b * T * OC + t * OC) as isize);
            let dinp_bt = dinp_ptr.load(Ordering::SeqCst).offset((b * T * C + t * C) as isize);

            for o in 0..OC {
                let wrow = weight_ptr.load(Ordering::SeqCst).offset((o * C) as isize);
                let d = *dout_bt.offset(o as isize);

                for i in 0..C {
                    *dinp_bt.offset(i as isize) += *wrow.offset(i as isize) * d;
                }
            }
        });
    });

    // Backward into weight/bias, parallelize over output channels OC
    (0..OC).into_par_iter().for_each(|o| {
        for b in 0..B {
            for t in 0..T {
                let dout_bt = dout_ptr.load(Ordering::SeqCst).offset((b * T * OC + t * OC) as isize);
                let inp_bt = inp_ptr.load(Ordering::SeqCst).offset((b * T * C + t * C) as isize);
                let dwrow = dweight_ptr.load(Ordering::SeqCst).offset((o * C) as isize);
                let d = *dout_bt.offset(o as isize);

                if !dbias_ptr.load(Ordering::SeqCst).is_null() {
                    *dbias_ptr.load(Ordering::SeqCst).offset(o as isize) += d;
                }

                for i in 0..C {
                    *dwrow.offset(i as isize) += *inp_bt.offset(i as isize) * d;
                }
            }
        }
    });
}

pub unsafe extern "C" fn attention_forward(
    mut out: *mut f32,
    mut preatt: *mut f32,
    mut att: *mut f32,
    mut inp: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut NH: i32,
) {
    let C3: i32 = C * 3;
    let hs: i32 = C / NH;
    let scale: f32 = (1.0f64 / (hs as f32).sqrt() as f64) as f32;

    let out = AtomicPtr::new(out);
    let preatt = AtomicPtr::new(preatt);
    let att = AtomicPtr::new(att);
    let inp = AtomicPtr::new(inp);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                let query_t = inp
                    .load(Ordering::SeqCst)
                    .offset((b * T * C3 + t * C3 + h * hs) as isize);
                let preatt_bth = preatt
                    .load(Ordering::SeqCst)
                    .offset((b * NH * T * T + h * T * T + t * T) as isize);
                let att_bth = att
                    .load(Ordering::SeqCst)
                    .offset((b * NH * T * T + h * T * T + t * T) as isize);

                let mut maxval = -10000.0f32;

                for t2 in 0..=t {
                    let key_t2 = inp
                        .load(Ordering::SeqCst)
                        .offset((b * T * C3 + t2 * C3 + h * hs + C) as isize);
                    let mut val = 0.0f32;
                    (0..hs).into_iter().for_each(|i| {
                        val += *query_t.offset(i as isize) * *key_t2.offset(i as isize);
                    });
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.offset(t2 as isize) = val;
                }

                let mut expsum = 0.0f32;
                (0..=t).into_iter().for_each(|t2_0| {
                    let expv = (*preatt_bth.offset(t2_0 as isize) - maxval).exp();
                    expsum += expv;
                    *att_bth.offset(t2_0 as isize) = expv;
                });

                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };
                (0..T).into_iter().for_each(|t2_1| {
                    if t2_1 <= t {
                        *att_bth.offset(t2_1 as isize) *= expsum_inv;
                    } else {
                        *att_bth.offset(t2_1 as isize) = 0.0;
                    }
                });

                let out_bth = out
                    .load(Ordering::SeqCst)
                    .offset((b * T * C + t * C + h * hs) as isize);
                (0..hs).into_iter().for_each(|i_0| {
                    *out_bth.offset(i_0 as isize) = 0.0;
                });

                for t2_2 in 0..=t {
                    let value_t2 = inp
                        .load(Ordering::SeqCst)
                        .offset((b * T * C3 + t2_2 * C3 + h * hs + 2 * C) as isize);
                    let att_btht2 = *att_bth.offset(t2_2 as isize);
                    (0..hs).into_iter().for_each(|i_1| {
                        *out_bth.offset(i_1 as isize) += att_btht2 * *value_t2.offset(i_1 as isize);
                    });
                }
            });
        });
    });
}

pub unsafe extern "C" fn attention_backward(
    mut dinp: *mut f32,
    mut dpreatt: *mut f32,
    mut datt: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut att: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut NH: i32,
) {
    let mut C3: i32 = C * 3 as i32;
    let mut hs: i32 = C / NH;
    let mut scale: f32 = (1.0f64 / sqrtf(hs as f32) as f64) as f32;
    for b in 0..B {
        let mut t: i32 = 0 as i32;
        for t in 0..T {
            let mut h: i32 = 0 as i32;
            for h in 0..NH {
                let mut att_bth: *mut f32 = att
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut datt_bth: *mut f32 = datt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut dpreatt_bth: *mut f32 = dpreatt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut dquery_t: *mut f32 = dinp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut query_t: *mut f32 = inp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut dout_bth: *mut f32 = dout
                    .offset((b * T * C) as isize)
                    .offset((t * C) as isize)
                    .offset((h * hs) as isize);
                for t2 in 0..=t {
                    let mut value_t2: *mut f32 = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2) as isize);
                    let mut dvalue_t2: *mut f32 = dinp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2) as isize);
                    (0..hs).into_iter().for_each(|i| {
                        *datt_bth.offset(t2 as isize) +=
                            *value_t2.offset(i as isize) * *dout_bth.offset(i as isize);
                        *dvalue_t2.offset(i as isize) +=
                            *att_bth.offset(t2 as isize) * *dout_bth.offset(i as isize);
                    });
                }
                for t2_0 in 0..=t {
                    (0..=t).into_iter().for_each(|t3| {
                        let indicator: f32 = if t2_0 == t3 { 1.0f32 } else { 0.0f32 };
                        let local_derivative: f32 = unsafe {
                            *att_bth.offset(t2_0 as isize)
                                * (indicator - *att_bth.offset(t3 as isize))
                        };
                        *dpreatt_bth.offset(t3 as isize) +=
                            local_derivative * *datt_bth.offset(t2_0 as isize);
                    });
                }
                for t2_1 in 0..=t {
                    let mut key_t2: *mut f32 = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2_1 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut dkey_t2: *mut f32 = dinp
                        .offset((b * T * C3) as isize)
                        .offset((t2_1 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    (0..hs).into_iter().for_each(|i_0| {
                        *dquery_t.offset(i_0 as isize) += *key_t2.offset(i_0 as isize)
                            * *dpreatt_bth.offset(t2_1 as isize)
                            * scale;
                        *dkey_t2.offset(i_0 as isize) += *query_t.offset(i_0 as isize)
                            * *dpreatt_bth.offset(t2_1 as isize)
                            * scale;
                    });
                }
            }
        }
    }
}

pub unsafe extern "C" fn gelu_forward(mut out: *mut f32, mut inp: *mut f32, mut N: i32) {
    (0..N).into_iter().for_each(|i| {
        let mut x: f32 = unsafe { *inp.offset(i as isize) };
        let mut cube: f32 = 0.044715f32 * x * x * x;
        *out.offset(i as isize) =
            0.5f32 * x * (1.0f32 + tanhf(sqrtf(2.0f32 / 3.14159265358979323846f32) * (x + cube)));
    });
}

pub unsafe extern "C" fn gelu_backward(
    mut dinp: *mut f32,
    mut inp: *mut f32,
    mut dout: *mut f32,
    mut N: i32,
) {
    (0..N).into_iter().for_each(|i| {
        let x: f32 = unsafe { *inp.offset(i as isize) };
        let cube: f32 = 0.044715f32 * x * x * x;
        let tanh_arg: f32 = sqrtf((2.0f32 as f64 / 3.14159265358979323846f64) as f32) * (x + cube);
        let tanh_out: f32 = tanhf(tanh_arg);
        let coshf_out: f32 = coshf(tanh_arg);
        let sech_out: f32 = 1.0f32 / (coshf_out * coshf_out);
        let local_grad: f32 = 0.5f32 * (1.0f32 + tanh_out)
            + x * 0.5f32
                * sech_out
                * sqrtf((2.0f32 as f64 / 3.14159265358979323846f64) as f32)
                * (1.0f32 + 3.0f32 * 0.044715f32 * x * x);
        *dinp.offset(i as isize) += local_grad * *dout.offset(i as isize);
    });
}

pub unsafe extern "C" fn residual_forward(
    mut out: *mut f32,
    mut inp1: *mut f32,
    mut inp2: *mut f32,
    mut N: i32,
) {
    (0..N).into_iter().for_each(|i| {
        *out.offset(i as isize) = *inp1.offset(i as isize) + *inp2.offset(i as isize);
    });
}

pub unsafe extern "C" fn residual_backward(
    mut dinp1: *mut f32,
    mut dinp2: *mut f32,
    mut dout: *mut f32,
    mut N: i32,
) {
    (0..N).into_iter().for_each(|i| {
        *dinp1.offset(i as isize) += *dout.offset(i as isize);
        *dinp2.offset(i as isize) += *dout.offset(i as isize);
    });
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
        B as i32,
        T as i32,
        C as i32,
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
            l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B as i32, T as i32, C as i32,
        );
        matmul_forward(
            l_qkv,
            l_ln1,
            l_qkvw,
            l_qkvb,
            B as i32,
            T as i32,
            C as i32,
            (3 as u64 * C) as i32,
        );
        attention_forward(
            l_atty, l_preatt, l_att, l_qkv, B as i32, T as i32, C as i32, NH as i32,
        );
        matmul_forward(
            l_attproj, l_atty, l_attprojw, l_attprojb, B as i32, T as i32, C as i32, C as i32,
        );
        residual_forward(l_residual2, residual, l_attproj, (B * T * C) as i32);
        layernorm_forward(
            l_ln2,
            l_ln2_mean,
            l_ln2_rstd,
            l_residual2,
            l_ln2w,
            l_ln2b,
            B as i32,
            T as i32,
            C as i32,
        );
        matmul_forward(
            l_fch,
            l_ln2,
            l_fcw,
            l_fcb,
            B as i32,
            T as i32,
            C as i32,
            (4 * C) as i32,
        );
        gelu_forward(l_fch_gelu, l_fch, (B * T * 4 * C) as i32);
        matmul_forward(
            l_fcproj,
            l_fch_gelu,
            l_fcprojw,
            l_fcprojb,
            B as i32,
            T as i32,
            (4 * C) as i32,
            C as i32,
        );
        residual_forward(l_residual3, l_residual2, l_fcproj, (B * T * C) as i32);
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
        B as i32,
        T as i32,
        C as i32,
    );
    matmul_forward(
        acts.logits,
        acts.lnf,
        params.wte,
        0 as *mut f32,
        B as i32,
        T as i32,
        C as i32,
        V as i32,
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
        B as i32,
        T as i32,
        C as i32,
        V as i32,
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
        B as i32,
        T as i32,
        C as i32,
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
            B.wrapping_mul(T).wrapping_mul(C) as i32,
        );
        matmul_backward(
            dl_fch_gelu,
            dl_fcprojw,
            dl_fcprojb,
            dl_fcproj,
            l_fch_gelu,
            l_fcprojw,
            B as i32,
            T as i32,
            (4 as i32 as u64).wrapping_mul(C) as i32,
            C as i32,
        );
        gelu_backward(
            dl_fch,
            l_fch,
            dl_fch_gelu,
            B.wrapping_mul(T)
                .wrapping_mul(4 as i32 as u64)
                .wrapping_mul(C) as i32,
        );
        matmul_backward(
            dl_ln2,
            dl_fcw,
            dl_fcb,
            dl_fch,
            l_ln2,
            l_fcw,
            B as i32,
            T as i32,
            C as i32,
            (4 as i32 as u64).wrapping_mul(C) as i32,
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
            B as i32,
            T as i32,
            C as i32,
        );
        residual_backward(
            dresidual,
            dl_attproj,
            dl_residual2,
            B.wrapping_mul(T).wrapping_mul(C) as i32,
        );
        matmul_backward(
            dl_atty,
            dl_attprojw,
            dl_attprojb,
            dl_attproj,
            l_atty,
            l_attprojw,
            B as i32,
            T as i32,
            C as i32,
            C as i32,
        );
        attention_backward(
            dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B as i32, T as i32, C as i32,
            NH as i32,
        );
        matmul_backward(
            dl_ln1,
            dl_qkvw,
            dl_qkvb,
            dl_qkv,
            l_ln1,
            l_qkvw,
            B as i32,
            T as i32,
            C as i32,
            (3 as i32 as u64).wrapping_mul(C) as i32,
        );
        layernorm_backward(
            dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd,
            B as i32, T as i32, C as i32,
        );
    });
    encoder_backward(
        grads.wte,
        grads.wpe,
        grads_acts.encoded,
        (*model).inputs,
        B as i32,
        T as i32,
        C as i32,
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
unsafe fn main_0() -> i32 {
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
    gpt2_build_from_checkpoint(&mut model, b"gpt2_124M.bin\0" as *const u8 as *const u8);
    let mut tiny_stories_train: *const u8 =
        b"data/tinystories/TinyStories_train.bin\0" as *const u8 as *const u8;
    let mut tiny_stories_val: *const u8 = b"data/tinystories/TinyStories_val.bin\0" as *const u8 as *const u8;
    let mut tiny_shakespeare_train: *const u8 =
        b"data/tinyshakespeare/tiny_shakespeare_train.bin\0" as *const u8 as *const u8;
    let mut tiny_shakespeare_val: *const u8 =
        b"data/tinyshakespeare/tiny_shakespeare_val.bin\0" as *const u8 as *const u8;
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
    return 0 as i32;
}
pub fn main() {
    unsafe { ::std::process::exit(main_0() as i32) }
}
