#![allow(dead_code, mutable_transmutes, non_camel_case_types, non_snake_case, non_upper_case_globals, unused_assignments, unused_mut)]
#![feature(extern_types)]
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
    fn fclose(__stream: *mut FILE) -> libc::c_int;
    fn fopen(_: *const libc::c_char, _: *const libc::c_char) -> *mut FILE;
    fn printf(_: *const libc::c_char, _: ...) -> libc::c_int;
    fn fread(
        _: *mut libc::c_void,
        _: libc::c_ulong,
        _: libc::c_ulong,
        _: *mut FILE,
    ) -> libc::c_ulong;
    fn fseek(
        __stream: *mut FILE,
        __off: libc::c_long,
        __whence: libc::c_int,
    ) -> libc::c_int;
    fn ftell(__stream: *mut FILE) -> libc::c_long;
    fn calloc(_: libc::c_ulong, _: libc::c_ulong) -> *mut libc::c_void;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn exit(_: libc::c_int) -> !;
    fn free(__ptr: *mut libc::c_void);
    fn coshf(_: libc::c_float) -> libc::c_float;
    fn tanhf(_: libc::c_float) -> libc::c_float;
    fn expf(_: libc::c_float) -> libc::c_float;
    fn logf(_: libc::c_float) -> libc::c_float;
    fn powf(_: libc::c_float, _: libc::c_float) -> libc::c_float;
    fn sqrtf(_: libc::c_float) -> libc::c_float;
    fn clock_gettime(__clock_id: clockid_t, __tp: *mut timespec) -> libc::c_int;
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn access(__name: *const libc::c_char, __type: libc::c_int) -> libc::c_int;
}
pub type size_t = libc::c_ulong;
pub type __off_t = libc::c_long;
pub type __off64_t = libc::c_long;
pub type __time_t = libc::c_long;
pub type __clockid_t = libc::c_int;
pub type __syscall_slong_t = libc::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: libc::c_int,
    pub _IO_read_ptr: *mut libc::c_char,
    pub _IO_read_end: *mut libc::c_char,
    pub _IO_read_base: *mut libc::c_char,
    pub _IO_write_base: *mut libc::c_char,
    pub _IO_write_ptr: *mut libc::c_char,
    pub _IO_write_end: *mut libc::c_char,
    pub _IO_buf_base: *mut libc::c_char,
    pub _IO_buf_end: *mut libc::c_char,
    pub _IO_save_base: *mut libc::c_char,
    pub _IO_backup_base: *mut libc::c_char,
    pub _IO_save_end: *mut libc::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: libc::c_int,
    pub _flags2: libc::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: libc::c_ushort,
    pub _vtable_offset: libc::c_schar,
    pub _shortbuf: [libc::c_char; 1],
    pub _lock: *mut libc::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut libc::c_void,
    pub __pad5: size_t,
    pub _mode: libc::c_int,
    pub _unused2: [libc::c_char; 20],
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParameterTensors {
    pub wte: *mut libc::c_float,
    pub wpe: *mut libc::c_float,
    pub ln1w: *mut libc::c_float,
    pub ln1b: *mut libc::c_float,
    pub qkvw: *mut libc::c_float,
    pub qkvb: *mut libc::c_float,
    pub attprojw: *mut libc::c_float,
    pub attprojb: *mut libc::c_float,
    pub ln2w: *mut libc::c_float,
    pub ln2b: *mut libc::c_float,
    pub fcw: *mut libc::c_float,
    pub fcb: *mut libc::c_float,
    pub fcprojw: *mut libc::c_float,
    pub fcprojb: *mut libc::c_float,
    pub lnfw: *mut libc::c_float,
    pub lnfb: *mut libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ActivationTensors {
    pub encoded: *mut libc::c_float,
    pub ln1: *mut libc::c_float,
    pub ln1_mean: *mut libc::c_float,
    pub ln1_rstd: *mut libc::c_float,
    pub qkv: *mut libc::c_float,
    pub atty: *mut libc::c_float,
    pub preatt: *mut libc::c_float,
    pub att: *mut libc::c_float,
    pub attproj: *mut libc::c_float,
    pub residual2: *mut libc::c_float,
    pub ln2: *mut libc::c_float,
    pub ln2_mean: *mut libc::c_float,
    pub ln2_rstd: *mut libc::c_float,
    pub fch: *mut libc::c_float,
    pub fch_gelu: *mut libc::c_float,
    pub fcproj: *mut libc::c_float,
    pub residual3: *mut libc::c_float,
    pub lnf: *mut libc::c_float,
    pub lnf_mean: *mut libc::c_float,
    pub lnf_rstd: *mut libc::c_float,
    pub logits: *mut libc::c_float,
    pub probs: *mut libc::c_float,
    pub losses: *mut libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2Config {
    pub max_seq_len: libc::c_int,
    pub vocab_size: libc::c_int,
    pub num_layers: libc::c_int,
    pub num_heads: libc::c_int,
    pub channels: libc::c_int,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2 {
    pub config: GPT2Config,
    pub params: ParameterTensors,
    pub param_sizes: [size_t; 16],
    pub params_memory: *mut libc::c_float,
    pub num_parameters: libc::c_int,
    pub grads: ParameterTensors,
    pub grads_memory: *mut libc::c_float,
    pub m_memory: *mut libc::c_float,
    pub v_memory: *mut libc::c_float,
    pub acts: ActivationTensors,
    pub act_sizes: [size_t; 23],
    pub acts_memory: *mut libc::c_float,
    pub num_activations: libc::c_int,
    pub grads_acts: ActivationTensors,
    pub grads_acts_memory: *mut libc::c_float,
    pub batch_size: libc::c_int,
    pub seq_len: libc::c_int,
    pub inputs: *mut libc::c_int,
    pub targets: *mut libc::c_int,
    pub mean_loss: libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DataLoader {
    pub B: libc::c_int,
    pub T: libc::c_int,
    pub tokens_file: *mut FILE,
    pub file_size: libc::c_long,
    pub current_position: libc::c_long,
    pub batch: *mut libc::c_int,
    pub inputs: *mut libc::c_int,
    pub targets: *mut libc::c_int,
    pub num_batches: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn encoder_forward(
    mut out: *mut libc::c_float,
    mut inp: *mut libc::c_int,
    mut wte: *mut libc::c_float,
    mut wpe: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut out_bt: *mut libc::c_float = out
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut ix: libc::c_int = *inp.offset((b * T + t) as isize);
            let mut wte_ix: *mut libc::c_float = wte.offset((ix * C) as isize);
            let mut wpe_t: *mut libc::c_float = wpe.offset((t * C) as isize);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                *out_bt
                    .offset(
                        i as isize,
                    ) = *wte_ix.offset(i as isize) + *wpe_t.offset(i as isize);
                i += 1;
                i;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn encoder_backward(
    mut dwte: *mut libc::c_float,
    mut dwpe: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut inp: *mut libc::c_int,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut dout_bt: *mut libc::c_float = dout
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut ix: libc::c_int = *inp.offset((b * T + t) as isize);
            let mut dwte_ix: *mut libc::c_float = dwte.offset((ix * C) as isize);
            let mut dwpe_t: *mut libc::c_float = dwpe.offset((t * C) as isize);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                let mut d: libc::c_float = *dout_bt.offset(i as isize);
                *dwte_ix.offset(i as isize) += d;
                *dwpe_t.offset(i as isize) += d;
                i += 1;
                i;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn layernorm_forward(
    mut out: *mut libc::c_float,
    mut mean: *mut libc::c_float,
    mut rstd: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut weight: *mut libc::c_float,
    mut bias: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut eps: libc::c_float = 1e-5f32;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut x: *mut libc::c_float = inp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut m: libc::c_float = 0.0f32;
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                m += *x.offset(i as isize);
                i += 1;
                i;
            }
            m = m / C as libc::c_float;
            let mut v: libc::c_float = 0.0f32;
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while i_0 < C {
                let mut xshift: libc::c_float = *x.offset(i_0 as isize) - m;
                v += xshift * xshift;
                i_0 += 1;
                i_0;
            }
            v = v / C as libc::c_float;
            let mut s: libc::c_float = 1.0f32 / sqrtf(v + eps);
            let mut out_bt: *mut libc::c_float = out
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut i_1: libc::c_int = 0 as libc::c_int;
            while i_1 < C {
                let mut n: libc::c_float = s * (*x.offset(i_1 as isize) - m);
                let mut o: libc::c_float = n * *weight.offset(i_1 as isize)
                    + *bias.offset(i_1 as isize);
                *out_bt.offset(i_1 as isize) = o;
                i_1 += 1;
                i_1;
            }
            *mean.offset((b * T + t) as isize) = m;
            *rstd.offset((b * T + t) as isize) = s;
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn layernorm_backward(
    mut dinp: *mut libc::c_float,
    mut dweight: *mut libc::c_float,
    mut dbias: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut weight: *mut libc::c_float,
    mut mean: *mut libc::c_float,
    mut rstd: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut dout_bt: *mut libc::c_float = dout
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut inp_bt: *mut libc::c_float = inp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut dinp_bt: *mut libc::c_float = dinp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut mean_bt: libc::c_float = *mean.offset((b * T + t) as isize);
            let mut rstd_bt: libc::c_float = *rstd.offset((b * T + t) as isize);
            let mut dnorm_mean: libc::c_float = 0.0f32;
            let mut dnorm_norm_mean: libc::c_float = 0.0f32;
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                let mut norm_bti: libc::c_float = (*inp_bt.offset(i as isize) - mean_bt)
                    * rstd_bt;
                let mut dnorm_i: libc::c_float = *weight.offset(i as isize)
                    * *dout_bt.offset(i as isize);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
                i += 1;
                i;
            }
            dnorm_mean = dnorm_mean / C as libc::c_float;
            dnorm_norm_mean = dnorm_norm_mean / C as libc::c_float;
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while i_0 < C {
                let mut norm_bti_0: libc::c_float = (*inp_bt.offset(i_0 as isize)
                    - mean_bt) * rstd_bt;
                let mut dnorm_i_0: libc::c_float = *weight.offset(i_0 as isize)
                    * *dout_bt.offset(i_0 as isize);
                *dbias.offset(i_0 as isize) += *dout_bt.offset(i_0 as isize);
                *dweight.offset(i_0 as isize)
                    += norm_bti_0 * *dout_bt.offset(i_0 as isize);
                let mut dval: libc::c_float = 0.0f32;
                dval += dnorm_i_0;
                dval -= dnorm_mean;
                dval -= norm_bti_0 * dnorm_norm_mean;
                dval *= rstd_bt;
                *dinp_bt.offset(i_0 as isize) += dval;
                i_0 += 1;
                i_0;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn matmul_forward(
    mut out: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut weight: *mut libc::c_float,
    mut bias: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut OC: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut out_bt: *mut libc::c_float = out
                .offset((b * T * OC) as isize)
                .offset((t * OC) as isize);
            let mut inp_bt: *mut libc::c_float = inp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut o: libc::c_int = 0 as libc::c_int;
            while o < OC {
                let mut val: libc::c_float = if !bias.is_null() {
                    *bias.offset(o as isize)
                } else {
                    0.0f32
                };
                let mut wrow: *mut libc::c_float = weight.offset((o * C) as isize);
                let mut i: libc::c_int = 0 as libc::c_int;
                while i < C {
                    val += *inp_bt.offset(i as isize) * *wrow.offset(i as isize);
                    i += 1;
                    i;
                }
                *out_bt.offset(o as isize) = val;
                o += 1;
                o;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn matmul_backward(
    mut dinp: *mut libc::c_float,
    mut dweight: *mut libc::c_float,
    mut dbias: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut weight: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut OC: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut dout_bt: *mut libc::c_float = dout
                .offset((b * T * OC) as isize)
                .offset((t * OC) as isize);
            let mut dinp_bt: *mut libc::c_float = dinp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut o: libc::c_int = 0 as libc::c_int;
            while o < OC {
                let mut wrow: *mut libc::c_float = weight.offset((o * C) as isize);
                let mut d: libc::c_float = *dout_bt.offset(o as isize);
                let mut i: libc::c_int = 0 as libc::c_int;
                while i < C {
                    *dinp_bt.offset(i as isize) += *wrow.offset(i as isize) * d;
                    i += 1;
                    i;
                }
                o += 1;
                o;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
    let mut o_0: libc::c_int = 0 as libc::c_int;
    while o_0 < OC {
        let mut b_0: libc::c_int = 0 as libc::c_int;
        while b_0 < B {
            let mut t_0: libc::c_int = 0 as libc::c_int;
            while t_0 < T {
                let mut dout_bt_0: *mut libc::c_float = dout
                    .offset((b_0 * T * OC) as isize)
                    .offset((t_0 * OC) as isize);
                let mut inp_bt: *mut libc::c_float = inp
                    .offset((b_0 * T * C) as isize)
                    .offset((t_0 * C) as isize);
                let mut dwrow: *mut libc::c_float = dweight.offset((o_0 * C) as isize);
                let mut d_0: libc::c_float = *dout_bt_0.offset(o_0 as isize);
                if !dbias.is_null() {
                    *dbias.offset(o_0 as isize) += d_0;
                }
                let mut i_0: libc::c_int = 0 as libc::c_int;
                while i_0 < C {
                    *dwrow.offset(i_0 as isize) += *inp_bt.offset(i_0 as isize) * d_0;
                    i_0 += 1;
                    i_0;
                }
                t_0 += 1;
                t_0;
            }
            b_0 += 1;
            b_0;
        }
        o_0 += 1;
        o_0;
    }
}
#[no_mangle]
pub unsafe extern "C" fn attention_forward(
    mut out: *mut libc::c_float,
    mut preatt: *mut libc::c_float,
    mut att: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut NH: libc::c_int,
) {
    let mut C3: libc::c_int = C * 3 as libc::c_int;
    let mut hs: libc::c_int = C / NH;
    let mut scale: libc::c_float = (1.0f64
        / sqrtf(hs as libc::c_float) as libc::c_double) as libc::c_float;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut h: libc::c_int = 0 as libc::c_int;
            while h < NH {
                let mut query_t: *mut libc::c_float = inp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut preatt_bth: *mut libc::c_float = preatt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut att_bth: *mut libc::c_float = att
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut maxval: libc::c_float = -10000.0f32;
                let mut t2: libc::c_int = 0 as libc::c_int;
                while t2 <= t {
                    let mut key_t2: *mut libc::c_float = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut val: libc::c_float = 0.0f32;
                    let mut i: libc::c_int = 0 as libc::c_int;
                    while i < hs {
                        val += *query_t.offset(i as isize) * *key_t2.offset(i as isize);
                        i += 1;
                        i;
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.offset(t2 as isize) = val;
                    t2 += 1;
                    t2;
                }
                let mut expsum: libc::c_float = 0.0f32;
                let mut t2_0: libc::c_int = 0 as libc::c_int;
                while t2_0 <= t {
                    let mut expv: libc::c_float = expf(
                        *preatt_bth.offset(t2_0 as isize) - maxval,
                    );
                    expsum += expv;
                    *att_bth.offset(t2_0 as isize) = expv;
                    t2_0 += 1;
                    t2_0;
                }
                let mut expsum_inv: libc::c_float = if expsum == 0.0f32 {
                    0.0f32
                } else {
                    1.0f32 / expsum
                };
                let mut t2_1: libc::c_int = 0 as libc::c_int;
                while t2_1 < T {
                    if t2_1 <= t {
                        *att_bth.offset(t2_1 as isize) *= expsum_inv;
                    } else {
                        *att_bth.offset(t2_1 as isize) = 0.0f32;
                    }
                    t2_1 += 1;
                    t2_1;
                }
                let mut out_bth: *mut libc::c_float = out
                    .offset((b * T * C) as isize)
                    .offset((t * C) as isize)
                    .offset((h * hs) as isize);
                let mut i_0: libc::c_int = 0 as libc::c_int;
                while i_0 < hs {
                    *out_bth.offset(i_0 as isize) = 0.0f32;
                    i_0 += 1;
                    i_0;
                }
                let mut t2_2: libc::c_int = 0 as libc::c_int;
                while t2_2 <= t {
                    let mut value_t2: *mut libc::c_float = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2_2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2 as libc::c_int) as isize);
                    let mut att_btht2: libc::c_float = *att_bth.offset(t2_2 as isize);
                    let mut i_1: libc::c_int = 0 as libc::c_int;
                    while i_1 < hs {
                        *out_bth.offset(i_1 as isize)
                            += att_btht2 * *value_t2.offset(i_1 as isize);
                        i_1 += 1;
                        i_1;
                    }
                    t2_2 += 1;
                    t2_2;
                }
                h += 1;
                h;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn attention_backward(
    mut dinp: *mut libc::c_float,
    mut dpreatt: *mut libc::c_float,
    mut datt: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut att: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut NH: libc::c_int,
) {
    let mut C3: libc::c_int = C * 3 as libc::c_int;
    let mut hs: libc::c_int = C / NH;
    let mut scale: libc::c_float = (1.0f64
        / sqrtf(hs as libc::c_float) as libc::c_double) as libc::c_float;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut h: libc::c_int = 0 as libc::c_int;
            while h < NH {
                let mut att_bth: *mut libc::c_float = att
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut datt_bth: *mut libc::c_float = datt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut dpreatt_bth: *mut libc::c_float = dpreatt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut dquery_t: *mut libc::c_float = dinp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut query_t: *mut libc::c_float = inp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut dout_bth: *mut libc::c_float = dout
                    .offset((b * T * C) as isize)
                    .offset((t * C) as isize)
                    .offset((h * hs) as isize);
                let mut t2: libc::c_int = 0 as libc::c_int;
                while t2 <= t {
                    let mut value_t2: *mut libc::c_float = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2 as libc::c_int) as isize);
                    let mut dvalue_t2: *mut libc::c_float = dinp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2 as libc::c_int) as isize);
                    let mut i: libc::c_int = 0 as libc::c_int;
                    while i < hs {
                        *datt_bth.offset(t2 as isize)
                            += *value_t2.offset(i as isize)
                                * *dout_bth.offset(i as isize);
                        *dvalue_t2.offset(i as isize)
                            += *att_bth.offset(t2 as isize)
                                * *dout_bth.offset(i as isize);
                        i += 1;
                        i;
                    }
                    t2 += 1;
                    t2;
                }
                let mut t2_0: libc::c_int = 0 as libc::c_int;
                while t2_0 <= t {
                    let mut t3: libc::c_int = 0 as libc::c_int;
                    while t3 <= t {
                        let mut indicator: libc::c_float = if t2_0 == t3 {
                            1.0f32
                        } else {
                            0.0f32
                        };
                        let mut local_derivative: libc::c_float = *att_bth
                            .offset(t2_0 as isize)
                            * (indicator - *att_bth.offset(t3 as isize));
                        *dpreatt_bth.offset(t3 as isize)
                            += local_derivative * *datt_bth.offset(t2_0 as isize);
                        t3 += 1;
                        t3;
                    }
                    t2_0 += 1;
                    t2_0;
                }
                let mut t2_1: libc::c_int = 0 as libc::c_int;
                while t2_1 <= t {
                    let mut key_t2: *mut libc::c_float = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2_1 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut dkey_t2: *mut libc::c_float = dinp
                        .offset((b * T * C3) as isize)
                        .offset((t2_1 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut i_0: libc::c_int = 0 as libc::c_int;
                    while i_0 < hs {
                        *dquery_t.offset(i_0 as isize)
                            += *key_t2.offset(i_0 as isize)
                                * *dpreatt_bth.offset(t2_1 as isize) * scale;
                        *dkey_t2.offset(i_0 as isize)
                            += *query_t.offset(i_0 as isize)
                                * *dpreatt_bth.offset(t2_1 as isize) * scale;
                        i_0 += 1;
                        i_0;
                    }
                    t2_1 += 1;
                    t2_1;
                }
                h += 1;
                h;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn gelu_forward(
    mut out: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        let mut x: libc::c_float = *inp.offset(i as isize);
        let mut cube: libc::c_float = 0.044715f32 * x * x * x;
        *out
            .offset(
                i as isize,
            ) = 0.5f32 * x
            * (1.0f32
                + tanhf(
                    sqrtf(
                        (2.0f32 as libc::c_double / 3.14159265358979323846f64)
                            as libc::c_float,
                    ) * (x + cube),
                ));
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn gelu_backward(
    mut dinp: *mut libc::c_float,
    mut inp: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        let mut x: libc::c_float = *inp.offset(i as isize);
        let mut cube: libc::c_float = 0.044715f32 * x * x * x;
        let mut tanh_arg: libc::c_float = sqrtf(
            (2.0f32 as libc::c_double / 3.14159265358979323846f64) as libc::c_float,
        ) * (x + cube);
        let mut tanh_out: libc::c_float = tanhf(tanh_arg);
        let mut coshf_out: libc::c_float = coshf(tanh_arg);
        let mut sech_out: libc::c_float = 1.0f32 / (coshf_out * coshf_out);
        let mut local_grad: libc::c_float = 0.5f32 * (1.0f32 + tanh_out)
            + x * 0.5f32 * sech_out
                * sqrtf(
                    (2.0f32 as libc::c_double / 3.14159265358979323846f64)
                        as libc::c_float,
                ) * (1.0f32 + 3.0f32 * 0.044715f32 * x * x);
        *dinp.offset(i as isize) += local_grad * *dout.offset(i as isize);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn residual_forward(
    mut out: *mut libc::c_float,
    mut inp1: *mut libc::c_float,
    mut inp2: *mut libc::c_float,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        *out.offset(i as isize) = *inp1.offset(i as isize) + *inp2.offset(i as isize);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn residual_backward(
    mut dinp1: *mut libc::c_float,
    mut dinp2: *mut libc::c_float,
    mut dout: *mut libc::c_float,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        *dinp1.offset(i as isize) += *dout.offset(i as isize);
        *dinp2.offset(i as isize) += *dout.offset(i as isize);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn softmax_forward(
    mut probs: *mut libc::c_float,
    mut logits: *mut libc::c_float,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut V: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut logits_bt: *mut libc::c_float = logits
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut probs_bt: *mut libc::c_float = probs
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut maxval: libc::c_float = -10000.0f32;
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < V {
                if *logits_bt.offset(i as isize) > maxval {
                    maxval = *logits_bt.offset(i as isize);
                }
                i += 1;
                i;
            }
            let mut sum: libc::c_float = 0.0f32;
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while i_0 < V {
                *probs_bt
                    .offset(
                        i_0 as isize,
                    ) = expf(*logits_bt.offset(i_0 as isize) - maxval);
                sum += *probs_bt.offset(i_0 as isize);
                i_0 += 1;
                i_0;
            }
            let mut i_1: libc::c_int = 0 as libc::c_int;
            while i_1 < V {
                *probs_bt.offset(i_1 as isize) /= sum;
                i_1 += 1;
                i_1;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn crossentropy_forward(
    mut losses: *mut libc::c_float,
    mut probs: *mut libc::c_float,
    mut targets: *mut libc::c_int,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut V: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut probs_bt: *mut libc::c_float = probs
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut ix: libc::c_int = *targets.offset((b * T + t) as isize);
            *losses.offset((b * T + t) as isize) = -logf(*probs_bt.offset(ix as isize));
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn crossentropy_softmax_backward(
    mut dlogits: *mut libc::c_float,
    mut dlosses: *mut libc::c_float,
    mut probs: *mut libc::c_float,
    mut targets: *mut libc::c_int,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut V: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut dlogits_bt: *mut libc::c_float = dlogits
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut probs_bt: *mut libc::c_float = probs
                .offset((b * T * V) as isize)
                .offset((t * V) as isize);
            let mut dloss: libc::c_float = *dlosses.offset((b * T + t) as isize);
            let mut ix: libc::c_int = *targets.offset((b * T + t) as isize);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < V {
                let mut p: libc::c_float = *probs_bt.offset(i as isize);
                let mut indicator: libc::c_float = if i == ix { 1.0f32 } else { 0.0f32 };
                *dlogits_bt.offset(i as isize) += (p - indicator) * dloss;
                i += 1;
                i;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn malloc_and_point_parameters(
    mut params: *mut ParameterTensors,
    mut param_sizes: *mut size_t,
) -> *mut libc::c_float {
    let mut num_parameters: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 16 as libc::c_int as libc::c_ulong {
        num_parameters = (num_parameters as libc::c_ulong)
            .wrapping_add(*param_sizes.offset(i as isize)) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    let mut params_memory: *mut libc::c_float = malloc(
        num_parameters
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    let mut ptrs: [*mut *mut libc::c_float; 16] = [
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
    let mut params_memory_iterator: *mut libc::c_float = params_memory;
    let mut i_0: size_t = 0 as libc::c_int as size_t;
    while i_0 < 16 as libc::c_int as libc::c_ulong {
        *ptrs[i_0 as usize] = params_memory_iterator;
        params_memory_iterator = params_memory_iterator
            .offset(*param_sizes.offset(i_0 as isize) as isize);
        i_0 = i_0.wrapping_add(1);
        i_0;
    }
    return params_memory;
}
#[no_mangle]
pub unsafe extern "C" fn malloc_and_point_activations(
    mut acts: *mut ActivationTensors,
    mut act_sizes: *mut size_t,
) -> *mut libc::c_float {
    let mut num_activations: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 23 as libc::c_int as libc::c_ulong {
        num_activations = (num_activations as libc::c_ulong)
            .wrapping_add(*act_sizes.offset(i as isize)) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    let mut acts_memory: *mut libc::c_float = malloc(
        num_activations
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    let mut ptrs: [*mut *mut libc::c_float; 23] = [
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
    let mut acts_memory_iterator: *mut libc::c_float = acts_memory;
    let mut i_0: size_t = 0 as libc::c_int as size_t;
    while i_0 < 23 as libc::c_int as libc::c_ulong {
        *ptrs[i_0 as usize] = acts_memory_iterator;
        acts_memory_iterator = acts_memory_iterator
            .offset(*act_sizes.offset(i_0 as isize) as isize);
        i_0 = i_0.wrapping_add(1);
        i_0;
    }
    return acts_memory;
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_build_from_checkpoint(
    mut model: *mut GPT2,
    mut checkpoint_path: *mut libc::c_char,
) {
    let mut model_file: *mut FILE = fopen(
        checkpoint_path,
        b"rb\0" as *const u8 as *const libc::c_char,
    );
    if model_file.is_null() {
        printf(b"Error opening model file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    let mut model_header: [libc::c_int; 256] = [0; 256];
    fread(
        model_header.as_mut_ptr() as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        256 as libc::c_int as libc::c_ulong,
        model_file,
    );
    if model_header[0 as libc::c_int as usize] != 20240326 as libc::c_int {
        printf(b"Bad magic model file\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    if model_header[1 as libc::c_int as usize] != 1 as libc::c_int {
        printf(b"Bad version in model file\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    let mut maxT: libc::c_int = 0;
    let mut V: libc::c_int = 0;
    let mut L: libc::c_int = 0;
    let mut NH: libc::c_int = 0;
    let mut C: libc::c_int = 0;
    maxT = model_header[2 as libc::c_int as usize];
    (*model).config.max_seq_len = maxT;
    V = model_header[3 as libc::c_int as usize];
    (*model).config.vocab_size = V;
    L = model_header[4 as libc::c_int as usize];
    (*model).config.num_layers = L;
    NH = model_header[5 as libc::c_int as usize];
    (*model).config.num_heads = NH;
    C = model_header[6 as libc::c_int as usize];
    (*model).config.channels = C;
    printf(b"[GPT-2]\n\0" as *const u8 as *const libc::c_char);
    printf(b"max_seq_len: %d\n\0" as *const u8 as *const libc::c_char, maxT);
    printf(b"vocab_size: %d\n\0" as *const u8 as *const libc::c_char, V);
    printf(b"num_layers: %d\n\0" as *const u8 as *const libc::c_char, L);
    printf(b"num_heads: %d\n\0" as *const u8 as *const libc::c_char, NH);
    printf(b"channels: %d\n\0" as *const u8 as *const libc::c_char, C);
    (*model).param_sizes[0 as libc::c_int as usize] = (V * C) as size_t;
    (*model).param_sizes[1 as libc::c_int as usize] = (maxT * C) as size_t;
    (*model).param_sizes[2 as libc::c_int as usize] = (L * C) as size_t;
    (*model).param_sizes[3 as libc::c_int as usize] = (L * C) as size_t;
    (*model)
        .param_sizes[4 as libc::c_int
        as usize] = (L * (3 as libc::c_int * C) * C) as size_t;
    (*model)
        .param_sizes[5 as libc::c_int as usize] = (L * (3 as libc::c_int * C)) as size_t;
    (*model).param_sizes[6 as libc::c_int as usize] = (L * C * C) as size_t;
    (*model).param_sizes[7 as libc::c_int as usize] = (L * C) as size_t;
    (*model).param_sizes[8 as libc::c_int as usize] = (L * C) as size_t;
    (*model).param_sizes[9 as libc::c_int as usize] = (L * C) as size_t;
    (*model)
        .param_sizes[10 as libc::c_int
        as usize] = (L * (4 as libc::c_int * C) * C) as size_t;
    (*model)
        .param_sizes[11 as libc::c_int
        as usize] = (L * (4 as libc::c_int * C)) as size_t;
    (*model)
        .param_sizes[12 as libc::c_int
        as usize] = (L * C * (4 as libc::c_int * C)) as size_t;
    (*model).param_sizes[13 as libc::c_int as usize] = (L * C) as size_t;
    (*model).param_sizes[14 as libc::c_int as usize] = C as size_t;
    (*model).param_sizes[15 as libc::c_int as usize] = C as size_t;
    let mut num_parameters: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 16 as libc::c_int as libc::c_ulong {
        num_parameters = (num_parameters as libc::c_ulong)
            .wrapping_add((*model).param_sizes[i as usize]) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    printf(
        b"num_parameters: %zu\n\0" as *const u8 as *const libc::c_char,
        num_parameters,
    );
    (*model).num_parameters = num_parameters as libc::c_int;
    (*model)
        .params_memory = malloc_and_point_parameters(
        &mut (*model).params,
        ((*model).param_sizes).as_mut_ptr(),
    );
    fread(
        (*model).params_memory as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        num_parameters,
        model_file,
    );
    fclose(model_file);
    (*model).acts_memory = 0 as *mut libc::c_float;
    (*model).grads_memory = 0 as *mut libc::c_float;
    (*model).m_memory = 0 as *mut libc::c_float;
    (*model).v_memory = 0 as *mut libc::c_float;
    (*model).grads_acts_memory = 0 as *mut libc::c_float;
    (*model).inputs = 0 as *mut libc::c_int;
    (*model).targets = 0 as *mut libc::c_int;
    (*model).batch_size = 0 as libc::c_int;
    (*model).seq_len = 0 as libc::c_int;
    (*model).mean_loss = -1.0f32;
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_forward(
    mut model: *mut GPT2,
    mut inputs: *mut libc::c_int,
    mut targets: *mut libc::c_int,
    mut B: libc::c_int,
    mut T: libc::c_int,
) {
    if ((*model).params_memory).is_null() {
        printf(
            b"Error: model was not initialized properly.\n\0" as *const u8
                as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    let mut V: libc::c_int = (*model).config.vocab_size;
    let mut L: libc::c_int = (*model).config.num_layers;
    let mut NH: libc::c_int = (*model).config.num_heads;
    let mut C: libc::c_int = (*model).config.channels;
    if ((*model).acts_memory).is_null() {
        (*model).batch_size = B;
        (*model).seq_len = T;
        (*model).act_sizes[0 as libc::c_int as usize] = (B * T * C) as size_t;
        (*model).act_sizes[1 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[2 as libc::c_int as usize] = (L * B * T) as size_t;
        (*model).act_sizes[3 as libc::c_int as usize] = (L * B * T) as size_t;
        (*model)
            .act_sizes[4 as libc::c_int
            as usize] = (L * B * T * 3 as libc::c_int * C) as size_t;
        (*model).act_sizes[5 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[6 as libc::c_int as usize] = (L * B * NH * T * T) as size_t;
        (*model).act_sizes[7 as libc::c_int as usize] = (L * B * NH * T * T) as size_t;
        (*model).act_sizes[8 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[9 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[10 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[11 as libc::c_int as usize] = (L * B * T) as size_t;
        (*model).act_sizes[12 as libc::c_int as usize] = (L * B * T) as size_t;
        (*model)
            .act_sizes[13 as libc::c_int
            as usize] = (L * B * T * 4 as libc::c_int * C) as size_t;
        (*model)
            .act_sizes[14 as libc::c_int
            as usize] = (L * B * T * 4 as libc::c_int * C) as size_t;
        (*model).act_sizes[15 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[16 as libc::c_int as usize] = (L * B * T * C) as size_t;
        (*model).act_sizes[17 as libc::c_int as usize] = (B * T * C) as size_t;
        (*model).act_sizes[18 as libc::c_int as usize] = (B * T) as size_t;
        (*model).act_sizes[19 as libc::c_int as usize] = (B * T) as size_t;
        (*model).act_sizes[20 as libc::c_int as usize] = (B * T * V) as size_t;
        (*model).act_sizes[21 as libc::c_int as usize] = (B * T * V) as size_t;
        (*model).act_sizes[22 as libc::c_int as usize] = (B * T) as size_t;
        let mut num_activations: size_t = 0 as libc::c_int as size_t;
        let mut i: size_t = 0 as libc::c_int as size_t;
        while i < 23 as libc::c_int as libc::c_ulong {
            num_activations = (num_activations as libc::c_ulong)
                .wrapping_add((*model).act_sizes[i as usize]) as size_t as size_t;
            i = i.wrapping_add(1);
            i;
        }
        printf(
            b"num_activations: %zu\n\0" as *const u8 as *const libc::c_char,
            num_activations,
        );
        (*model).num_activations = num_activations as libc::c_int;
        (*model)
            .acts_memory = malloc_and_point_activations(
            &mut (*model).acts,
            ((*model).act_sizes).as_mut_ptr(),
        );
        (*model)
            .inputs = malloc(
            ((B * T) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        ) as *mut libc::c_int;
        (*model)
            .targets = malloc(
            ((B * T) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        ) as *mut libc::c_int;
    } else if B > (*model).batch_size || T > (*model).seq_len {
        printf(
            b"Error: batch size or sequence length is inadequately large\n\0"
                as *const u8 as *const libc::c_char,
        );
        printf(
            b"Model: B=%d T=%d, Desired: B=%d T=%d\n\0" as *const u8
                as *const libc::c_char,
            (*model).batch_size,
            (*model).seq_len,
            B,
            T,
        );
        exit(1 as libc::c_int);
    }
    memcpy(
        (*model).inputs as *mut libc::c_void,
        inputs as *const libc::c_void,
        ((B * T) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    );
    if !targets.is_null() {
        memcpy(
            (*model).targets as *mut libc::c_void,
            targets as *const libc::c_void,
            ((B * T) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        );
    }
    let mut params: ParameterTensors = (*model).params;
    let mut acts: ActivationTensors = (*model).acts;
    let mut residual: *mut libc::c_float = 0 as *mut libc::c_float;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
    let mut l: libc::c_int = 0 as libc::c_int;
    while l < L {
        residual = if l == 0 as libc::c_int {
            acts.encoded
        } else {
            (acts.residual3).offset(((l - 1 as libc::c_int) * B * T * C) as isize)
        };
        let mut l_ln1w: *mut libc::c_float = (params.ln1w).offset((l * C) as isize);
        let mut l_ln1b: *mut libc::c_float = (params.ln1b).offset((l * C) as isize);
        let mut l_qkvw: *mut libc::c_float = (params.qkvw)
            .offset((l * 3 as libc::c_int * C * C) as isize);
        let mut l_qkvb: *mut libc::c_float = (params.qkvb)
            .offset((l * 3 as libc::c_int * C) as isize);
        let mut l_attprojw: *mut libc::c_float = (params.attprojw)
            .offset((l * C * C) as isize);
        let mut l_attprojb: *mut libc::c_float = (params.attprojb)
            .offset((l * C) as isize);
        let mut l_ln2w: *mut libc::c_float = (params.ln2w).offset((l * C) as isize);
        let mut l_ln2b: *mut libc::c_float = (params.ln2b).offset((l * C) as isize);
        let mut l_fcw: *mut libc::c_float = (params.fcw)
            .offset((l * 4 as libc::c_int * C * C) as isize);
        let mut l_fcb: *mut libc::c_float = (params.fcb)
            .offset((l * 4 as libc::c_int * C) as isize);
        let mut l_fcprojw: *mut libc::c_float = (params.fcprojw)
            .offset((l * C * 4 as libc::c_int * C) as isize);
        let mut l_fcprojb: *mut libc::c_float = (params.fcprojb)
            .offset((l * C) as isize);
        let mut l_ln1: *mut libc::c_float = (acts.ln1).offset((l * B * T * C) as isize);
        let mut l_ln1_mean: *mut libc::c_float = (acts.ln1_mean)
            .offset((l * B * T) as isize);
        let mut l_ln1_rstd: *mut libc::c_float = (acts.ln1_rstd)
            .offset((l * B * T) as isize);
        let mut l_qkv: *mut libc::c_float = (acts.qkv)
            .offset((l * B * T * 3 as libc::c_int * C) as isize);
        let mut l_atty: *mut libc::c_float = (acts.atty)
            .offset((l * B * T * C) as isize);
        let mut l_preatt: *mut libc::c_float = (acts.preatt)
            .offset((l * B * NH * T * T) as isize);
        let mut l_att: *mut libc::c_float = (acts.att)
            .offset((l * B * NH * T * T) as isize);
        let mut l_attproj: *mut libc::c_float = (acts.attproj)
            .offset((l * B * T * C) as isize);
        let mut l_residual2: *mut libc::c_float = (acts.residual2)
            .offset((l * B * T * C) as isize);
        let mut l_ln2: *mut libc::c_float = (acts.ln2).offset((l * B * T * C) as isize);
        let mut l_ln2_mean: *mut libc::c_float = (acts.ln2_mean)
            .offset((l * B * T) as isize);
        let mut l_ln2_rstd: *mut libc::c_float = (acts.ln2_rstd)
            .offset((l * B * T) as isize);
        let mut l_fch: *mut libc::c_float = (acts.fch)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut l_fch_gelu: *mut libc::c_float = (acts.fch_gelu)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut l_fcproj: *mut libc::c_float = (acts.fcproj)
            .offset((l * B * T * C) as isize);
        let mut l_residual3: *mut libc::c_float = (acts.residual3)
            .offset((l * B * T * C) as isize);
        layernorm_forward(
            l_ln1,
            l_ln1_mean,
            l_ln1_rstd,
            residual,
            l_ln1w,
            l_ln1b,
            B,
            T,
            C,
        );
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 as libc::c_int * C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(
            l_ln2,
            l_ln2_mean,
            l_ln2_rstd,
            l_residual2,
            l_ln2w,
            l_ln2b,
            B,
            T,
            C,
        );
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 as libc::c_int * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 as libc::c_int * C);
        matmul_forward(
            l_fcproj,
            l_fch_gelu,
            l_fcprojw,
            l_fcprojb,
            B,
            T,
            4 as libc::c_int * C,
            C,
        );
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        l += 1;
        l;
    }
    residual = (acts.residual3).offset(((L - 1 as libc::c_int) * B * T * C) as isize);
    layernorm_forward(
        acts.lnf,
        acts.lnf_mean,
        acts.lnf_rstd,
        residual,
        params.lnfw,
        params.lnfb,
        B,
        T,
        C,
    );
    matmul_forward(
        acts.logits,
        acts.lnf,
        params.wte,
        0 as *mut libc::c_float,
        B,
        T,
        C,
        V,
    );
    softmax_forward(acts.probs, acts.logits, B, T, V);
    if !targets.is_null() {
        crossentropy_forward(
            (*model).acts.losses,
            (*model).acts.probs,
            targets,
            B,
            T,
            V,
        );
        let mut mean_loss: libc::c_float = 0.0f32;
        let mut i_0: libc::c_int = 0 as libc::c_int;
        while i_0 < B * T {
            mean_loss += *((*model).acts.losses).offset(i_0 as isize);
            i_0 += 1;
            i_0;
        }
        mean_loss /= (B * T) as libc::c_float;
        (*model).mean_loss = mean_loss;
    } else {
        (*model).mean_loss = -1.0f32;
    };
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_zero_grad(mut model: *mut GPT2) {
    if !((*model).grads_memory).is_null() {
        memset(
            (*model).grads_memory as *mut libc::c_void,
            0 as libc::c_int,
            ((*model).num_parameters as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
        );
    }
    if !((*model).grads_acts_memory).is_null() {
        memset(
            (*model).grads_acts_memory as *mut libc::c_void,
            0 as libc::c_int,
            ((*model).num_activations as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_backward(mut model: *mut GPT2) {
    if (*model).mean_loss == -1.0f32 {
        printf(
            b"Error: must forward with targets before backward\n\0" as *const u8
                as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    if ((*model).grads_memory).is_null() {
        (*model)
            .grads_memory = malloc_and_point_parameters(
            &mut (*model).grads,
            ((*model).param_sizes).as_mut_ptr(),
        );
        (*model)
            .grads_acts_memory = malloc_and_point_activations(
            &mut (*model).grads_acts,
            ((*model).act_sizes).as_mut_ptr(),
        );
        gpt2_zero_grad(model);
    }
    let mut B: libc::c_int = (*model).batch_size;
    let mut T: libc::c_int = (*model).seq_len;
    let mut V: libc::c_int = (*model).config.vocab_size;
    let mut L: libc::c_int = (*model).config.num_layers;
    let mut NH: libc::c_int = (*model).config.num_heads;
    let mut C: libc::c_int = (*model).config.channels;
    let mut params: ParameterTensors = (*model).params;
    let mut grads: ParameterTensors = (*model).grads;
    let mut acts: ActivationTensors = (*model).acts;
    let mut grads_acts: ActivationTensors = (*model).grads_acts;
    let mut dloss_mean: libc::c_float = 1.0f32 / (B * T) as libc::c_float;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < B * T {
        *(grads_acts.losses).offset(i as isize) = dloss_mean;
        i += 1;
        i;
    }
    crossentropy_softmax_backward(
        grads_acts.logits,
        grads_acts.losses,
        acts.probs,
        (*model).targets,
        B,
        T,
        V,
    );
    matmul_backward(
        grads_acts.lnf,
        grads.wte,
        0 as *mut libc::c_float,
        grads_acts.logits,
        acts.lnf,
        params.wte,
        B,
        T,
        C,
        V,
    );
    let mut residual: *mut libc::c_float = (acts.residual3)
        .offset(((L - 1 as libc::c_int) * B * T * C) as isize);
    let mut dresidual: *mut libc::c_float = (grads_acts.residual3)
        .offset(((L - 1 as libc::c_int) * B * T * C) as isize);
    layernorm_backward(
        dresidual,
        grads.lnfw,
        grads.lnfb,
        grads_acts.lnf,
        residual,
        params.lnfw,
        acts.lnf_mean,
        acts.lnf_rstd,
        B,
        T,
        C,
    );
    let mut l: libc::c_int = L - 1 as libc::c_int;
    while l >= 0 as libc::c_int {
        residual = if l == 0 as libc::c_int {
            acts.encoded
        } else {
            (acts.residual3).offset(((l - 1 as libc::c_int) * B * T * C) as isize)
        };
        dresidual = if l == 0 as libc::c_int {
            grads_acts.encoded
        } else {
            (grads_acts.residual3).offset(((l - 1 as libc::c_int) * B * T * C) as isize)
        };
        let mut l_ln1w: *mut libc::c_float = (params.ln1w).offset((l * C) as isize);
        let mut l_qkvw: *mut libc::c_float = (params.qkvw)
            .offset((l * 3 as libc::c_int * C * C) as isize);
        let mut l_attprojw: *mut libc::c_float = (params.attprojw)
            .offset((l * C * C) as isize);
        let mut l_ln2w: *mut libc::c_float = (params.ln2w).offset((l * C) as isize);
        let mut l_fcw: *mut libc::c_float = (params.fcw)
            .offset((l * 4 as libc::c_int * C * C) as isize);
        let mut l_fcprojw: *mut libc::c_float = (params.fcprojw)
            .offset((l * C * 4 as libc::c_int * C) as isize);
        let mut dl_ln1w: *mut libc::c_float = (grads.ln1w).offset((l * C) as isize);
        let mut dl_ln1b: *mut libc::c_float = (grads.ln1b).offset((l * C) as isize);
        let mut dl_qkvw: *mut libc::c_float = (grads.qkvw)
            .offset((l * 3 as libc::c_int * C * C) as isize);
        let mut dl_qkvb: *mut libc::c_float = (grads.qkvb)
            .offset((l * 3 as libc::c_int * C) as isize);
        let mut dl_attprojw: *mut libc::c_float = (grads.attprojw)
            .offset((l * C * C) as isize);
        let mut dl_attprojb: *mut libc::c_float = (grads.attprojb)
            .offset((l * C) as isize);
        let mut dl_ln2w: *mut libc::c_float = (grads.ln2w).offset((l * C) as isize);
        let mut dl_ln2b: *mut libc::c_float = (grads.ln2b).offset((l * C) as isize);
        let mut dl_fcw: *mut libc::c_float = (grads.fcw)
            .offset((l * 4 as libc::c_int * C * C) as isize);
        let mut dl_fcb: *mut libc::c_float = (grads.fcb)
            .offset((l * 4 as libc::c_int * C) as isize);
        let mut dl_fcprojw: *mut libc::c_float = (grads.fcprojw)
            .offset((l * C * 4 as libc::c_int * C) as isize);
        let mut dl_fcprojb: *mut libc::c_float = (grads.fcprojb)
            .offset((l * C) as isize);
        let mut l_ln1: *mut libc::c_float = (acts.ln1).offset((l * B * T * C) as isize);
        let mut l_ln1_mean: *mut libc::c_float = (acts.ln1_mean)
            .offset((l * B * T) as isize);
        let mut l_ln1_rstd: *mut libc::c_float = (acts.ln1_rstd)
            .offset((l * B * T) as isize);
        let mut l_qkv: *mut libc::c_float = (acts.qkv)
            .offset((l * B * T * 3 as libc::c_int * C) as isize);
        let mut l_atty: *mut libc::c_float = (acts.atty)
            .offset((l * B * T * C) as isize);
        let mut l_att: *mut libc::c_float = (acts.att)
            .offset((l * B * NH * T * T) as isize);
        let mut l_residual2: *mut libc::c_float = (acts.residual2)
            .offset((l * B * T * C) as isize);
        let mut l_ln2: *mut libc::c_float = (acts.ln2).offset((l * B * T * C) as isize);
        let mut l_ln2_mean: *mut libc::c_float = (acts.ln2_mean)
            .offset((l * B * T) as isize);
        let mut l_ln2_rstd: *mut libc::c_float = (acts.ln2_rstd)
            .offset((l * B * T) as isize);
        let mut l_fch: *mut libc::c_float = (acts.fch)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut l_fch_gelu: *mut libc::c_float = (acts.fch_gelu)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut dl_ln1: *mut libc::c_float = (grads_acts.ln1)
            .offset((l * B * T * C) as isize);
        let mut dl_qkv: *mut libc::c_float = (grads_acts.qkv)
            .offset((l * B * T * 3 as libc::c_int * C) as isize);
        let mut dl_atty: *mut libc::c_float = (grads_acts.atty)
            .offset((l * B * T * C) as isize);
        let mut dl_preatt: *mut libc::c_float = (grads_acts.preatt)
            .offset((l * B * NH * T * T) as isize);
        let mut dl_att: *mut libc::c_float = (grads_acts.att)
            .offset((l * B * NH * T * T) as isize);
        let mut dl_attproj: *mut libc::c_float = (grads_acts.attproj)
            .offset((l * B * T * C) as isize);
        let mut dl_residual2: *mut libc::c_float = (grads_acts.residual2)
            .offset((l * B * T * C) as isize);
        let mut dl_ln2: *mut libc::c_float = (grads_acts.ln2)
            .offset((l * B * T * C) as isize);
        let mut dl_fch: *mut libc::c_float = (grads_acts.fch)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut dl_fch_gelu: *mut libc::c_float = (grads_acts.fch_gelu)
            .offset((l * B * T * 4 as libc::c_int * C) as isize);
        let mut dl_fcproj: *mut libc::c_float = (grads_acts.fcproj)
            .offset((l * B * T * C) as isize);
        let mut dl_residual3: *mut libc::c_float = (grads_acts.residual3)
            .offset((l * B * T * C) as isize);
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
        matmul_backward(
            dl_fch_gelu,
            dl_fcprojw,
            dl_fcprojb,
            dl_fcproj,
            l_fch_gelu,
            l_fcprojw,
            B,
            T,
            4 as libc::c_int * C,
            C,
        );
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 as libc::c_int * C);
        matmul_backward(
            dl_ln2,
            dl_fcw,
            dl_fcb,
            dl_fch,
            l_ln2,
            l_fcw,
            B,
            T,
            C,
            4 as libc::c_int * C,
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
            B,
            T,
            C,
        );
        residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
        matmul_backward(
            dl_atty,
            dl_attprojw,
            dl_attprojb,
            dl_attproj,
            l_atty,
            l_attprojw,
            B,
            T,
            C,
            C,
        );
        attention_backward(
            dl_qkv,
            dl_preatt,
            dl_att,
            dl_atty,
            l_qkv,
            l_att,
            B,
            T,
            C,
            NH,
        );
        matmul_backward(
            dl_ln1,
            dl_qkvw,
            dl_qkvb,
            dl_qkv,
            l_ln1,
            l_qkvw,
            B,
            T,
            C,
            3 as libc::c_int * C,
        );
        layernorm_backward(
            dresidual,
            dl_ln1w,
            dl_ln1b,
            dl_ln1,
            residual,
            l_ln1w,
            l_ln1_mean,
            l_ln1_rstd,
            B,
            T,
            C,
        );
        l -= 1;
        l;
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, (*model).inputs, B, T, C);
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_update(
    mut model: *mut GPT2,
    mut learning_rate: libc::c_float,
    mut beta1: libc::c_float,
    mut beta2: libc::c_float,
    mut eps: libc::c_float,
    mut weight_decay: libc::c_float,
    mut t: libc::c_int,
) {
    if ((*model).m_memory).is_null() {
        (*model)
            .m_memory = calloc(
            (*model).num_parameters as libc::c_ulong,
            ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        ) as *mut libc::c_float;
        (*model)
            .v_memory = calloc(
            (*model).num_parameters as libc::c_ulong,
            ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        ) as *mut libc::c_float;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < (*model).num_parameters {
        let mut param: libc::c_float = *((*model).params_memory).offset(i as isize);
        let mut grad: libc::c_float = *((*model).grads_memory).offset(i as isize);
        let mut m: libc::c_float = beta1 * *((*model).m_memory).offset(i as isize)
            + (1.0f32 - beta1) * grad;
        let mut v: libc::c_float = beta2 * *((*model).v_memory).offset(i as isize)
            + (1.0f32 - beta2) * grad * grad;
        let mut m_hat: libc::c_float = m / (1.0f32 - powf(beta1, t as libc::c_float));
        let mut v_hat: libc::c_float = v / (1.0f32 - powf(beta2, t as libc::c_float));
        *((*model).m_memory).offset(i as isize) = m;
        *((*model).v_memory).offset(i as isize) = v;
        *((*model).params_memory).offset(i as isize)
            -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_free(mut model: *mut GPT2) {
    free((*model).params_memory as *mut libc::c_void);
    free((*model).grads_memory as *mut libc::c_void);
    free((*model).m_memory as *mut libc::c_void);
    free((*model).v_memory as *mut libc::c_void);
    free((*model).acts_memory as *mut libc::c_void);
    free((*model).grads_acts_memory as *mut libc::c_void);
    free((*model).inputs as *mut libc::c_void);
    free((*model).targets as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_init(
    mut loader: *mut DataLoader,
    mut filename: *mut libc::c_char,
    mut B: libc::c_int,
    mut T: libc::c_int,
) {
    (*loader).B = B;
    (*loader).T = T;
    (*loader).tokens_file = fopen(filename, b"rb\0" as *const u8 as *const libc::c_char);
    if ((*loader).tokens_file).is_null() {
        printf(b"Error opening tokens file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    fseek((*loader).tokens_file, 0 as libc::c_int as libc::c_long, 2 as libc::c_int);
    (*loader).file_size = ftell((*loader).tokens_file);
    fseek((*loader).tokens_file, 0 as libc::c_int as libc::c_long, 0 as libc::c_int);
    if ((*loader).file_size as libc::c_ulong)
        < ((B * T + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
    {
        printf(
            b"Error: file size is too small for the batch size and sequence length\n\0"
                as *const u8 as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    (*loader).current_position = 0 as libc::c_int as libc::c_long;
    (*loader)
        .batch = malloc(
        ((B * T + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    (*loader).inputs = (*loader).batch;
    (*loader).targets = ((*loader).batch).offset(1 as libc::c_int as isize);
    (*loader)
        .num_batches = ((*loader).file_size as libc::c_ulong)
        .wrapping_div(
            ((B * T) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        ) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_reset(mut loader: *mut DataLoader) {
    (*loader).current_position = 0 as libc::c_int as libc::c_long;
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_next_batch(mut loader: *mut DataLoader) {
    let mut B: libc::c_int = (*loader).B;
    let mut T: libc::c_int = (*loader).T;
    if ((*loader).current_position as libc::c_ulong)
        .wrapping_add(
            ((B * T + 1 as libc::c_int) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        ) > (*loader).file_size as libc::c_ulong
    {
        (*loader).current_position = 0 as libc::c_int as libc::c_long;
    }
    fseek((*loader).tokens_file, (*loader).current_position, 0 as libc::c_int);
    fread(
        (*loader).batch as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        (B * T + 1 as libc::c_int) as libc::c_ulong,
        (*loader).tokens_file,
    );
    (*loader)
        .current_position = ((*loader).current_position as libc::c_ulong)
        .wrapping_add(
            ((B * T) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        ) as libc::c_long as libc::c_long;
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_free(mut loader: *mut DataLoader) {
    fclose((*loader).tokens_file);
    free((*loader).batch as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn random_u32(mut state: *mut libc::c_ulonglong) -> libc::c_uint {
    *state ^= *state >> 12 as libc::c_int;
    *state ^= *state << 25 as libc::c_int;
    *state ^= *state >> 27 as libc::c_int;
    return ((*state).wrapping_mul(0x2545f4914f6cdd1d as libc::c_ulonglong)
        >> 32 as libc::c_int) as libc::c_uint;
}
#[no_mangle]
pub unsafe extern "C" fn random_f32(mut state: *mut libc::c_ulonglong) -> libc::c_float {
    return (random_u32(state) >> 8 as libc::c_int) as libc::c_float / 16777216.0f32;
}
#[no_mangle]
pub unsafe extern "C" fn sample_mult(
    mut probabilities: *mut libc::c_float,
    mut n: libc::c_int,
    mut coin: libc::c_float,
) -> libc::c_int {
    let mut cdf: libc::c_float = 0.0f32;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < n {
        cdf += *probabilities.offset(i as isize);
        if coin < cdf {
            return i;
        }
        i += 1;
        i;
    }
    return n - 1 as libc::c_int;
}
unsafe fn main_0() -> libc::c_int {
    let mut model: GPT2 = GPT2 {
        config: GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        },
        params: ParameterTensors {
            wte: 0 as *mut libc::c_float,
            wpe: 0 as *mut libc::c_float,
            ln1w: 0 as *mut libc::c_float,
            ln1b: 0 as *mut libc::c_float,
            qkvw: 0 as *mut libc::c_float,
            qkvb: 0 as *mut libc::c_float,
            attprojw: 0 as *mut libc::c_float,
            attprojb: 0 as *mut libc::c_float,
            ln2w: 0 as *mut libc::c_float,
            ln2b: 0 as *mut libc::c_float,
            fcw: 0 as *mut libc::c_float,
            fcb: 0 as *mut libc::c_float,
            fcprojw: 0 as *mut libc::c_float,
            fcprojb: 0 as *mut libc::c_float,
            lnfw: 0 as *mut libc::c_float,
            lnfb: 0 as *mut libc::c_float,
        },
        param_sizes: [0; 16],
        params_memory: 0 as *mut libc::c_float,
        num_parameters: 0,
        grads: ParameterTensors {
            wte: 0 as *mut libc::c_float,
            wpe: 0 as *mut libc::c_float,
            ln1w: 0 as *mut libc::c_float,
            ln1b: 0 as *mut libc::c_float,
            qkvw: 0 as *mut libc::c_float,
            qkvb: 0 as *mut libc::c_float,
            attprojw: 0 as *mut libc::c_float,
            attprojb: 0 as *mut libc::c_float,
            ln2w: 0 as *mut libc::c_float,
            ln2b: 0 as *mut libc::c_float,
            fcw: 0 as *mut libc::c_float,
            fcb: 0 as *mut libc::c_float,
            fcprojw: 0 as *mut libc::c_float,
            fcprojb: 0 as *mut libc::c_float,
            lnfw: 0 as *mut libc::c_float,
            lnfb: 0 as *mut libc::c_float,
        },
        grads_memory: 0 as *mut libc::c_float,
        m_memory: 0 as *mut libc::c_float,
        v_memory: 0 as *mut libc::c_float,
        acts: ActivationTensors {
            encoded: 0 as *mut libc::c_float,
            ln1: 0 as *mut libc::c_float,
            ln1_mean: 0 as *mut libc::c_float,
            ln1_rstd: 0 as *mut libc::c_float,
            qkv: 0 as *mut libc::c_float,
            atty: 0 as *mut libc::c_float,
            preatt: 0 as *mut libc::c_float,
            att: 0 as *mut libc::c_float,
            attproj: 0 as *mut libc::c_float,
            residual2: 0 as *mut libc::c_float,
            ln2: 0 as *mut libc::c_float,
            ln2_mean: 0 as *mut libc::c_float,
            ln2_rstd: 0 as *mut libc::c_float,
            fch: 0 as *mut libc::c_float,
            fch_gelu: 0 as *mut libc::c_float,
            fcproj: 0 as *mut libc::c_float,
            residual3: 0 as *mut libc::c_float,
            lnf: 0 as *mut libc::c_float,
            lnf_mean: 0 as *mut libc::c_float,
            lnf_rstd: 0 as *mut libc::c_float,
            logits: 0 as *mut libc::c_float,
            probs: 0 as *mut libc::c_float,
            losses: 0 as *mut libc::c_float,
        },
        act_sizes: [0; 23],
        acts_memory: 0 as *mut libc::c_float,
        num_activations: 0,
        grads_acts: ActivationTensors {
            encoded: 0 as *mut libc::c_float,
            ln1: 0 as *mut libc::c_float,
            ln1_mean: 0 as *mut libc::c_float,
            ln1_rstd: 0 as *mut libc::c_float,
            qkv: 0 as *mut libc::c_float,
            atty: 0 as *mut libc::c_float,
            preatt: 0 as *mut libc::c_float,
            att: 0 as *mut libc::c_float,
            attproj: 0 as *mut libc::c_float,
            residual2: 0 as *mut libc::c_float,
            ln2: 0 as *mut libc::c_float,
            ln2_mean: 0 as *mut libc::c_float,
            ln2_rstd: 0 as *mut libc::c_float,
            fch: 0 as *mut libc::c_float,
            fch_gelu: 0 as *mut libc::c_float,
            fcproj: 0 as *mut libc::c_float,
            residual3: 0 as *mut libc::c_float,
            lnf: 0 as *mut libc::c_float,
            lnf_mean: 0 as *mut libc::c_float,
            lnf_rstd: 0 as *mut libc::c_float,
            logits: 0 as *mut libc::c_float,
            probs: 0 as *mut libc::c_float,
            losses: 0 as *mut libc::c_float,
        },
        grads_acts_memory: 0 as *mut libc::c_float,
        batch_size: 0,
        seq_len: 0,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        mean_loss: 0.,
    };
    gpt2_build_from_checkpoint(
        &mut model,
        b"gpt2_124M.bin\0" as *const u8 as *const libc::c_char as *mut libc::c_char,
    );
    let mut tiny_stories_train: *mut libc::c_char = b"data/TinyStories_train.bin\0"
        as *const u8 as *const libc::c_char as *mut libc::c_char;
    let mut tiny_stories_val: *mut libc::c_char = b"data/TinyStories_val.bin\0"
        as *const u8 as *const libc::c_char as *mut libc::c_char;
    let mut tiny_shakespeare_train: *mut libc::c_char = b"data/tiny_shakespeare_train.bin\0"
        as *const u8 as *const libc::c_char as *mut libc::c_char;
    let mut tiny_shakespeare_val: *mut libc::c_char = b"data/tiny_shakespeare_val.bin\0"
        as *const u8 as *const libc::c_char as *mut libc::c_char;
    let mut train_tokens: *mut libc::c_char = if access(
        tiny_shakespeare_train,
        0 as libc::c_int,
    ) != -(1 as libc::c_int)
    {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let mut val_tokens: *mut libc::c_char = if access(
        tiny_shakespeare_val,
        0 as libc::c_int,
    ) != -(1 as libc::c_int)
    {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };
    let mut B: libc::c_int = 4 as libc::c_int;
    let mut T: libc::c_int = 64 as libc::c_int;
    let mut train_loader: DataLoader = DataLoader {
        B: 0,
        T: 0,
        tokens_file: 0 as *mut FILE,
        file_size: 0,
        current_position: 0,
        batch: 0 as *mut libc::c_int,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        num_batches: 0,
    };
    dataloader_init(&mut train_loader, train_tokens, B, T);
    printf(
        b"train dataset num_batches: %d\n\0" as *const u8 as *const libc::c_char,
        train_loader.num_batches,
    );
    let mut val_loader: DataLoader = DataLoader {
        B: 0,
        T: 0,
        tokens_file: 0 as *mut FILE,
        file_size: 0,
        current_position: 0,
        batch: 0 as *mut libc::c_int,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        num_batches: 0,
    };
    dataloader_init(&mut val_loader, val_tokens, B, T);
    printf(
        b"val dataset num_batches: %d\n\0" as *const u8 as *const libc::c_char,
        val_loader.num_batches,
    );
    let mut val_num_batches: libc::c_int = 10 as libc::c_int;
    let mut rng_state: libc::c_ulonglong = 1337 as libc::c_int as libc::c_ulonglong;
    let gen_max_length: libc::c_int = 64 as libc::c_int;
    let mut gen_tokens: [libc::c_int; 64] = [0; 64];
    let mut start: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
    let mut end: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
    let mut step: libc::c_int = 0 as libc::c_int;
    while step <= 40 as libc::c_int {
        if step % 10 as libc::c_int == 0 as libc::c_int {
            let mut val_loss: libc::c_float = 0.0f32;
            dataloader_reset(&mut val_loader);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < val_num_batches {
                dataloader_next_batch(&mut val_loader);
                gpt2_forward(&mut model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
                i += 1;
                i;
            }
            val_loss /= val_num_batches as libc::c_float;
            printf(
                b"val loss %f\n\0" as *const u8 as *const libc::c_char,
                val_loss as libc::c_double,
            );
        }
        if step > 0 as libc::c_int && step % 20 as libc::c_int == 0 as libc::c_int {
            gen_tokens[0 as libc::c_int as usize] = 50256 as libc::c_int;
            let mut t: libc::c_int = 1 as libc::c_int;
            while t < gen_max_length {
                gpt2_forward(
                    &mut model,
                    gen_tokens.as_mut_ptr(),
                    0 as *mut libc::c_int,
                    1 as libc::c_int,
                    t,
                );
                let mut probs: *mut libc::c_float = (model.acts.probs)
                    .offset(((t - 1 as libc::c_int) * model.config.vocab_size) as isize);
                let mut coin: libc::c_float = random_f32(&mut rng_state);
                let mut next_token: libc::c_int = sample_mult(
                    probs,
                    model.config.vocab_size,
                    coin,
                );
                gen_tokens[t as usize] = next_token;
                t += 1;
                t;
            }
            printf(b"generated: \0" as *const u8 as *const libc::c_char);
            let mut t_0: libc::c_int = 0 as libc::c_int;
            while t_0 < gen_max_length {
                printf(
                    b"%d \0" as *const u8 as *const libc::c_char,
                    gen_tokens[t_0 as usize],
                );
                t_0 += 1;
                t_0;
            }
            printf(b"\n\0" as *const u8 as *const libc::c_char);
        }
        clock_gettime(1 as libc::c_int, &mut start);
        dataloader_next_batch(&mut train_loader);
        gpt2_forward(&mut model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&mut model);
        gpt2_backward(&mut model);
        gpt2_update(
            &mut model,
            1e-4f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.0f32,
            step + 1 as libc::c_int,
        );
        clock_gettime(1 as libc::c_int, &mut end);
        let mut time_elapsed_s: libc::c_double = (end.tv_sec - start.tv_sec)
            as libc::c_double + (end.tv_nsec - start.tv_nsec) as libc::c_double / 1e9f64;
        printf(
            b"step %d: train loss %f (took %f ms)\n\0" as *const u8
                as *const libc::c_char,
            step,
            model.mean_loss as libc::c_double,
            time_elapsed_s * 1000 as libc::c_int as libc::c_double,
        );
        step += 1;
        step;
    }
    dataloader_free(&mut train_loader);
    dataloader_free(&mut val_loader);
    gpt2_free(&mut model);
    return 0 as libc::c_int;
}
pub fn main() {
    unsafe { ::std::process::exit(main_0() as i32) }
}
