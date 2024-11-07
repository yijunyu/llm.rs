use rayon::prelude::*;
use std::f32::consts::PI;

use crate::send_ptr::SendPtr;

const LOOP_UNROLL: usize = 8;

// ----------------------------------------------------------------------------
// All the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size
// ----------------------------------------------------------------------------

/// Computes the forward pass for the encoder, combining token and positional embeddings.
///
/// # Arguments
///
/// * `out` - Output tensor for combined embeddings.
/// * `inp` - Input tensor containing token indices.
/// * `wte` - Token embedding matrix.
/// * `wpe` - Positional embedding matrix.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Embedding dimension.
pub unsafe fn encoder_forward(
    out: SendPtr<f32>,
    inp: SendPtr<i32>,
    wte: SendPtr<f32>,
    wpe: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let out = out;
            let inp = inp;
            let wte = wte;
            let wpe = wpe;

            let out_bt = out.ptr.add(b * T * C + t * C);
            let ix = *inp.ptr.add(b * T + t) as usize;
            let wte_ix = wte.ptr.add(ix * C);
            let wpe_t = wpe.ptr.add(t * C);

            for i in 0..C {
                *out_bt.add(i) = *wte_ix.add(i) + *wpe_t.add(i);
            }
        });
    });
}

/// Computes the backward pass for the encoder, updating gradients for token and position embeddings.
///
/// # Arguments
///
/// * `dwte` - Gradient of the token embedding matrix.
/// * `dwpe` - Gradient of the positional embedding matrix.
/// * `dout` - Gradient of the output tensor.
/// * `inp` - Input tensor containing token indices.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Embedding dimension.
pub unsafe fn encoder_backward(
    dwte: SendPtr<f32>,
    dwpe: SendPtr<f32>,
    dout: SendPtr<f32>,
    inp: SendPtr<i32>,
    B: usize,
    T: usize,
    C: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dwte = dwte;
            let dwpe = dwpe;
            let dout = dout;
            let inp = inp;

            let dout_bt = dout.ptr.add(b * T * C + t * C);
            let ix = *inp.ptr.add(b * T + t) as usize;
            let dwte_ix = dwte.ptr.add(ix * C);
            let dwpe_t = dwpe.ptr.add(t * C);

            for i in 0..C {
                let d = *dout_bt.add(i);
                *dwte_ix.add(i) += d;
                *dwpe_t.add(i) += d;
            }
        });
    });
}

/// Computes the forward pass for Layer Normalization, producing normalized output,
/// and caching mean and reciprocal standard deviation.
///
/// # Arguments
///
/// * `out` - Output tensor for the normalized result.
/// * `mean` - Buffer to store the mean values.
/// * `rstd` - Buffer to store the reciprocal of the standard deviation.
/// * `inp` - Input tensor.
/// * `weight` - Weight vector for scaling.
/// * `bias` - Bias vector for shifting.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
///
/// # Note
///
/// Reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
pub unsafe fn layernorm_forward(
    out: SendPtr<f32>,
    mean: SendPtr<f32>,
    rstd: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    bias: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
) {
    let eps: f32 = 1e-5;

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let out = out;
            let mean = mean;
            let rstd = rstd;
            let inp = inp;
            let weight = weight;
            let bias = bias;

            // Calculate the base address for inp[b,t,:]
            let x = inp.ptr.add(b * T * C + t * C);

            // Calculate the mean
            let mut m: f32 = 0.0;
            for i in 0..C {
                m += *x.add(i);
            }
            m /= C as f32;

            // Calculate the variance
            let mut v: f32 = 0.0;
            for i in 0..C {
                let xshift = *x.add(i) - m;
                v += xshift * xshift;
            }
            v /= C as f32;

            // Calculate the rstd (reciprocal standard deviation)
            let s: f32 = 1.0 / (v + eps).sqrt();

            // Calculate the base address for out[b,t,:]
            let out_bt = out.ptr.add(b * T * C + t * C);
            for i in 0..C {
                let n = s * (*x.add(i) - m); // Normalize
                let o = n * *weight.ptr.add(i) + *bias.ptr.add(i); // Scale and shift
                *out_bt.add(i) = o; // Write
            }

            // Cache the mean and rstd for the backward pass
            *mean.ptr.add(b * T + t) = m;
            *rstd.ptr.add(b * T + t) = s;
        });
    });
}

/// Computes the backward pass for Layer Normalization, updating gradients for inputs,
/// weights, and biases.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `dweight` - Gradient of the weight vector.
/// * `dbias` - Gradient of the bias vector.
/// * `dout` - Gradient of the output tensor.
/// * `inp` - Input tensor.
/// * `weight` - Weight vector.
/// * `mean` - Mean of the input tensor across the normalization axis.
/// * `rstd` - Reciprocal of the standard deviation of the input tensor.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
pub unsafe fn layernorm_backward(
    dinp: SendPtr<f32>,
    dweight: SendPtr<f32>,
    dbias: SendPtr<f32>,
    dout: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    mean: SendPtr<f32>,
    rstd: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dinp = dinp;
            let dweight = dweight;
            let dbias = dbias;
            let dout = dout;
            let inp = inp;
            let weight = weight;
            let mean = mean;
            let rstd = rstd;

            // Calculate the base addresses
            let dout_bt = dout.ptr.add(b * T * C + t * C);
            let inp_bt = inp.ptr.add(b * T * C + t * C);
            let dinp_bt = dinp.ptr.add(b * T * C + t * C);
            let mean_bt = *mean.ptr.add(b * T + t);
            let rstd_bt = *rstd.ptr.add(b * T + t);

            // First: two reduce operations
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.ptr.add(i) * *dout_bt.add(i);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Now iterate again and accumulate all the gradients
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.ptr.add(i) * *dout_bt.add(i);

                // Gradient contribution to bias
                *dbias.ptr.add(i) += *dout_bt.add(i);

                // Gradient contribution to weight
                *dweight.ptr.add(i) += norm_bti * *dout_bt.add(i);

                // Gradient contribution to input
                let mut dval: f32 = 0.0;
                dval += dnorm_i; // Term 1
                dval -= dnorm_mean; // Term 2
                dval -= norm_bti * dnorm_norm_mean; // Term 3
                dval *= rstd_bt; // Final scale
                *dinp_bt.add(i) += dval;
            }
        });
    });
}

/// Naive implementation of the forward pass for matrix multiplication, producing the output tensor.
///
/// # Arguments
///
/// * `out` - Output tensor for the matrix multiplication result.
/// * `inp` - Input tensor.
/// * `weight` - Weight matrix.
/// * `bias` - Bias vector.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Input feature dimension.
/// * `OC` - Output feature dimension or output channels.
///
/// # Note
///
/// This is the most naive implementation of matrix multiplication that serves as an algorithmic reference, and as a fallback for unfriendly input shapes inside matmul_forward().
pub unsafe fn matmul_forward_naive(
    out: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    bias: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // Create a parallel iterator over the batch dimension
    (0..B).into_par_iter().for_each(|b| {
        // Create a parallel iterator over the sequence length
        (0..T).into_par_iter().for_each(|t| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let out = out;
            let inp = inp;
            let weight = weight;
            let bias = bias;

            let bt = b * T + t;
            // Iterate over the output channels
            for o in 0..OC {
                // Initialize the output value with the bias if provided, otherwise 0.0
                let mut val = if !bias.ptr.is_null() {
                    *bias.ptr.add(o)
                } else {
                    0.0f32
                };
                // Perform the dot product
                for i in 0..C {
                    val += *inp.ptr.add(bt * C + i) * *weight.ptr.add(o * C + i);
                }
                // Store the result
                *out.ptr.add(bt * OC + o) = val;
            }
        });
    });
}

pub unsafe fn matmul_forward_fast(
    out: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    bias: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    matrixmultiply::sgemm(
        B * T,
        C,
        OC,
        1.0,
        inp.ptr,
        C as isize,
        1,
        weight.ptr,
        1,
        C as isize,
        0.0,
        out.ptr,
        OC as isize,
        1,
    );

    if !bias.ptr.is_null() {
        for bt in 0..B * T {
            for o in 0..OC {
                *out.ptr.add(bt * OC + o) += *bias.ptr.add(o);
            }
        }
    }
}

/// Computes the forward pass for matrix multiplication, producing the output tensor.
///
/// # Arguments
///
/// * `out` - Output tensor for the matrix multiplication result.
/// * `inp` - Input tensor.
/// * `weight` - Weight matrix.
/// * `bias` - Bias vector.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Input feature dimension.
/// * `OC` - Output feature dimension or output channels.
///
/// # Note
///
/// Most of the running time is spent here and in matmul_backward, therefore, the implementation below is very mildly optimized.
/// This function is otherwise identical to that of matmul_forward_naive().
pub unsafe fn matmul_forward(
    out: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    bias: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    return matmul_forward_fast(out, inp, weight, bias, B, T, C, OC);
    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // Parallelize the outer loop using Rayon
    (0..B * T)
        .into_par_iter()
        .step_by(LOOP_UNROLL)
        .for_each(|obt| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let out = out;
            let inp = inp;
            let weight = weight;
            let bias = bias;

            for o in 0..OC {
                // Initialize the result array with bias if present
                let mut result = [0.0f32; LOOP_UNROLL];
                for ibt in 0..LOOP_UNROLL {
                    result[ibt] = if !bias.ptr.is_null() {
                        *bias.ptr.add(o)
                    } else {
                        0.0f32
                    };
                }

                // Cache the weight value and compute dot products
                for i in 0..C {
                    let w = *weight.ptr.add(i + o * C);
                    for ibt in 0..LOOP_UNROLL {
                        let bt = obt + ibt;
                        result[ibt] += *inp.ptr.add(bt * C + i) * w;
                    }
                }

                // Write results back to the output matrix
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt + ibt;
                    *out.ptr.add(bt * OC + o) = result[ibt];
                }
            }
        });
}

pub unsafe fn matmul_backward_fast(
    dinp: SendPtr<f32>,
    dweight: SendPtr<f32>,
    dbias: SendPtr<f32>,
    dout: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    matrixmultiply::sgemm(
        B * T,
        OC,
        C,
        1.0,
        dout.ptr,
        OC as isize,
        1,
        weight.ptr,
        C as isize,
        1,
        1.0,
        dinp.ptr,
        C as isize,
        1,
    );

    matrixmultiply::sgemm(
        OC,
        B * T,
        C,
        1.0,
        dout.ptr,
        1,
        OC as isize,
        inp.ptr,
        C as isize,
        1,
        1.0,
        dweight.ptr,
        C as isize,
        1,
    );

    if !dbias.ptr.is_null() {
        for bt in 0..B * T {
            for o in 0..OC {
                *dbias.ptr.add(o) += *dout.ptr.add(bt * OC + o);
            }
        }
    }
}

/// Computes the backward pass for matrix multiplication, updating gradients for inputs,
/// weights, and biases.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `dweight` - Gradient of the weight matrix.
/// * `dbias` - Gradient of the bias vector.
/// * `dout` - Gradient of the output tensor.
/// * `inp` - Input tensor.
/// * `weight` - Weight matrix.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Input feature dimension.
/// * `OC` - Output feature dimension.
///
/// # Note
///
/// Most of the running time is spent here and in matmul_forward.
/// This backward could be done in a single "round" of loops but that doesn't afford an efficient parallelization strategy.
pub unsafe fn matmul_backward(
    dinp: SendPtr<f32>,
    dweight: SendPtr<f32>,
    dbias: SendPtr<f32>,
    dout: SendPtr<f32>,
    inp: SendPtr<f32>,
    weight: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    return matmul_backward_fast(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        // Parallelize over B and T for input gradient computation
        (0..B).into_par_iter().for_each(|b| {
            (0..T).into_par_iter().for_each(|t| {
                let dout = dout;
                let dinp = dinp;
                let weight = weight;

                let dout_bt = dout.ptr.add(b * T * OC + t * OC);
                let dinp_bt = dinp.ptr.add(b * T * C + t * C);

                for o in 0..OC {
                    let wrow = weight.ptr.add(o * C);
                    let d = *dout_bt.add(o);
                    for i in 0..C {
                        *dinp_bt.add(i) += *wrow.add(i) * d;
                    }
                }
            });
        });
    } else {
        // Parallelize over B and T for input gradient computation
        (0..B * T)
            .into_par_iter()
            .step_by(LOOP_UNROLL)
            .for_each(|obt| {
                let dout = dout;
                let dinp = dinp;
                let weight = weight;

                for o in 0..OC {
                    for ibt in 0..LOOP_UNROLL {
                        let bt = obt + ibt;
                        let dout_bt = dout.ptr.add(bt * OC + o);
                        let dinp_bt = dinp.ptr.add(bt * C);

                        let wrow = weight.ptr.add(o * C);
                        let d = *dout_bt;

                        for i in 0..C {
                            *dinp_bt.add(i) += *wrow.add(i) * d;
                        }
                    }
                }
            });
    }

    // Parallelize over output channels for weight and bias gradient computation
    (0..OC).into_par_iter().for_each(|o| {
        for b in 0..B {
            for t in 0..T {
                let dout = dout;
                let inp = inp;
                let dweight = dweight;
                let dbias = dbias;

                let dout_bt = dout.ptr.add(b * T * OC + t * OC);
                let inp_bt = inp.ptr.add(b * T * C + t * C);
                let dwrow = dweight.ptr.add(o * C);

                let d = *dout_bt.add(o);
                if !dbias.ptr.is_null() {
                    *dbias.ptr.add(o) += d;
                }
                for i in 0..C {
                    *dwrow.add(i) += *inp_bt.add(i) * d;
                }
            }
        }
    });
}

/// Naive implementation of the forward pass for multi-head attention, generating output and storing attention scores.
///
/// # Arguments
///
/// * `out` - Output tensor for attention results.
/// * `preatt` - Pre-attention scores.
/// * `att` - Post-attention scores.
/// * `inp` - Input tensor containing query, key, and value vectors.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
/// * `NH` - Number of attention heads.
pub unsafe fn attention_forward_naive(
    out: SendPtr<f32>,
    preatt: SendPtr<f32>,
    att: SendPtr<f32>,
    inp: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                let out = out;
                let preatt = preatt;
                let att = att;
                let inp = inp;

                let query_t = inp.ptr.add(b * T * C3 + t * C3 + h * hs);
                let preatt_bth = preatt.ptr.add(b * NH * T * T + h * T * T + t * T);
                let att_bth = att.ptr.add(b * NH * T * T + h * T * T + t * T);

                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=t {
                    let key_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C);
                    let mut val = 0.0;
                    for i in 0..hs {
                        val += *query_t.add(i) * *key_t2.add(i);
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.add(t2) = val;
                }

                let mut expsum = 0.0;
                for t2 in 0..=t {
                    let expv = (*preatt_bth.add(t2) - maxval).exp();
                    expsum += expv;
                    *att_bth.add(t2) = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                for t2 in 0..T {
                    if t2 <= t {
                        *att_bth.add(t2) *= expsum_inv;
                    } else {
                        *att_bth.add(t2) = 0.0;
                    }
                }

                let out_bth = out.ptr.add(b * T * C + t * C + h * hs);
                for i in 0..hs {
                    *out_bth.add(i) = 0.0;
                }
                for t2 in 0..=t {
                    let value_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);
                    let att_btht2 = *att_bth.add(t2);
                    for i in 0..hs {
                        *out_bth.add(i) += att_btht2 * *value_t2.add(i);
                    }
                }
            });
        });
    });
}

/// Computes the forward pass for multi-head attention, generating output and storing attention scores.
///
/// # Arguments
///
/// * `out` - Output tensor for attention results.
/// * `preatt` - Pre-attention scores.
/// * `att` - Post-attention scores.
/// * `inp` - Input tensor containing query, key, and value vectors.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
/// * `NH` - Number of attention heads.
pub unsafe fn attention_forward(
    out: SendPtr<f32>,
    preatt: SendPtr<f32>,
    att: SendPtr<f32>,
    inp: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        attention_forward_naive(out, preatt, att, inp, B, T, C, NH);
        return;
    }

    (0..B * T)
        .into_par_iter()
        .step_by(LOOP_UNROLL)
        .for_each(|obt| {
            let out = out;
            let preatt = preatt;
            let att = att;
            let inp = inp;

            for h in 0..NH {
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt + ibt;
                    let t = bt % T;
                    let b = bt / T;

                    let query_t = inp.ptr.add(b * T * C3 + t * C3 + h * hs);
                    let preatt_bth = preatt.ptr.add(b * NH * T * T + h * T * T + t * T);
                    let att_bth = att.ptr.add(b * NH * T * T + h * T * T + t * T);

                    let mut maxval = f32::NEG_INFINITY;
                    for t2 in 0..=t {
                        let key_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C);
                        let mut val = 0.0;
                        for i in 0..hs {
                            val += *query_t.add(i) * *key_t2.add(i);
                        }
                        val *= scale;
                        if val > maxval {
                            maxval = val;
                        }
                        *preatt_bth.add(t2) = val;
                    }

                    let mut expsum = 0.0;
                    for t2 in 0..=t {
                        let expv = (*preatt_bth.add(t2) - maxval).exp();
                        expsum += expv;
                        *att_bth.add(t2) = expv;
                    }
                    let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                    for t2 in 0..T {
                        if t2 <= t {
                            *att_bth.add(t2) *= expsum_inv;
                        } else {
                            *att_bth.add(t2) = 0.0;
                        }
                    }

                    let out_bth = out.ptr.add(b * T * C + t * C + h * hs);
                    for i in 0..hs {
                        *out_bth.add(i) = 0.0;
                    }
                    for t2 in 0..=t {
                        let value_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);
                        let att_btht2 = *att_bth.add(t2);
                        for i in 0..hs {
                            *out_bth.add(i) += att_btht2 * *value_t2.add(i);
                        }
                    }
                }
            }
        });
}

/// Naive implementation of the backward pass for attention mechanisms, updating gradients for inputs,
/// pre-attention weights, and attention weights.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `dpreatt` - Gradient of the pre-attention weights.
/// * `datt` - Gradient of the attention weights.
/// * `dout` - Gradient of the output tensor.
/// * `inp` - Input tensor.
/// * `att` - Attention weights.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
/// * `NH` - Number of attention heads.
pub unsafe fn attention_backward_naive(
    dinp: SendPtr<f32>,
    dpreatt: SendPtr<f32>,
    datt: SendPtr<f32>,
    dout: SendPtr<f32>,
    inp: SendPtr<f32>,
    att: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                let dinp = dinp;
                let dpreatt = dpreatt;
                let datt = datt;
                let dout = dout;
                let inp = inp;
                let att = att;

                let att_bth = att.ptr.add(b * NH * T * T + h * T * T + t * T);
                let datt_bth = datt.ptr.add(b * NH * T * T + h * T * T + t * T);
                let dpreatt_bth = dpreatt.ptr.add(b * NH * T * T + h * T * T + t * T);
                let dquery_t = dinp.ptr.add(b * T * C3 + t * C3 + h * hs);
                let query_t = inp.ptr.add(b * T * C3 + t * C3 + h * hs);

                // Backward pass 4: through the value accumulation
                let dout_bth = dout.ptr.add(b * T * C + t * C + h * hs);
                for t2 in 0..=t {
                    let value_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                    let dvalue_t2 = dinp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                    for i in 0..hs {
                        *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                        *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                    }
                }

                // Backward pass 2 & 3: the softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                        *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                    }
                }

                // Backward pass 1: the query @ key matmul
                for t2 in 0..=t {
                    let key_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    let dkey_t2 = dinp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    for i in 0..hs {
                        *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                        *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                    }
                }
            });
        });
    });
}

/// Computes the backward pass for attention mechanisms, updating gradients for inputs,
/// pre-attention weights, and attention weights.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `dpreatt` - Gradient of the pre-attention weights.
/// * `datt` - Gradient of the attention weights.
/// * `dout` - Gradient of the output tensor.
/// * `inp` - Input tensor.
/// * `att` - Attention weights.
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
/// * `NH` - Number of attention heads.
pub unsafe fn attention_backward(
    dinp: SendPtr<f32>,
    dpreatt: SendPtr<f32>,
    datt: SendPtr<f32>,
    dout: SendPtr<f32>,
    att: SendPtr<f32>,
    inp: SendPtr<f32>,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        attention_backward_naive(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
        return;
    }

    (0..B * T)
        .into_par_iter()
        .step_by(LOOP_UNROLL)
        .for_each(|obt| {
            let dinp = dinp;
            let dpreatt = dpreatt;
            let datt = datt;
            let dout = dout;
            let inp = inp;
            let att = att;

            for ibt in 0..LOOP_UNROLL {
                let bt = obt + ibt;
                let b = bt / T;
                let t = bt % T;

                for h in 0..NH {
                    let att_bth = att.ptr.add(b * NH * T * T + h * T * T + t * T);
                    let datt_bth = datt.ptr.add(b * NH * T * T + h * T * T + t * T);
                    let dpreatt_bth = dpreatt.ptr.add(b * NH * T * T + h * T * T + t * T);
                    let dquery_t = dinp.ptr.add(b * T * C3 + t * C3 + h * hs);
                    let query_t = inp.ptr.add(b * T * C3 + t * C3 + h * hs);

                    // Backward pass 4: through the value accumulation
                    let dout_bth = dout.ptr.add(b * T * C + t * C + h * hs);
                    for t2 in 0..=t {
                        let value_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                        let dvalue_t2 = dinp.ptr.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                        for i in 0..hs {
                            *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                            *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                        }
                    }

                    // Backward pass 2 & 3: the softmax
                    for t2 in 0..=t {
                        for t3 in 0..=t {
                            let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                            let local_derivative =
                                *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                            *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                        }
                    }

                    // Backward pass 1: the query @ key matmul
                    for t2 in 0..=t {
                        let key_t2 = inp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                        let dkey_t2 = dinp.ptr.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                        for i in 0..hs {
                            *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                            *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                        }
                    }
                }
            }
        });
}

/// Applies the GELU activation function to the input tensor.
///
/// # Arguments
///
/// * `out` - Output tensor to store the GELU results.
/// * `inp` - Input tensor.
/// * `N` - Number of elements.
pub unsafe fn gelu_forward(out: SendPtr<f32>, inp: SendPtr<f32>, N: usize) {
    (0..N).into_par_iter().for_each(|i| {
        let out = out;
        let inp = inp;

        // Load the input value
        let x = *inp.ptr.add(i);
        // Calculate the cubic term
        let cube = 0.044715 * x * x * x;
        // Apply the GeLU function
        *out.ptr.add(i) = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());
    });
}

/// Computes the gradient of the GELU activation function.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `inp` - Input tensor.
/// * `dout` - Gradient of the output tensor.
/// * `N` - Number of elements.
pub unsafe fn gelu_backward(dinp: SendPtr<f32>, inp: SendPtr<f32>, dout: SendPtr<f32>, N: usize) {
    let gelu_scaling_factor = (2.0 / PI).sqrt();

    (0..N).into_par_iter().for_each(|i| {
        let dinp = dinp;
        let inp = inp;
        let dout = dout;

        // Load the input value
        let x = *inp.ptr.add(i);
        let dout_val = *dout.ptr.add(i);

        // Compute the cubic term
        let cube = 0.044715 * x * x * x;

        // Compute the argument and the output of the tanh function
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();

        // Compute the hyperbolic cosine and sech (hyperbolic secant)
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);

        // Compute the local gradient
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);

        // Accumulate the gradient into dinp
        *dinp.ptr.add(i) += local_grad * dout_val;
    });
}

/// Adds two input tensors element-wise and stores the result in the output tensor.
///
/// # Arguments
///
/// * `out` - Output tensor to store the result.
/// * `inp1` - First input tensor.
/// * `inp2` - Second input tensor.
/// * `N` - Number of elements.
pub unsafe fn residual_forward(
    out: SendPtr<f32>,
    inp1: SendPtr<f32>,
    inp2: SendPtr<f32>,
    N: usize,
) {
    (0..N).into_par_iter().for_each(|i| {
        let out = out;
        let inp1 = inp1;
        let inp2 = inp2;

        // Perform element-wise addition
        *out.ptr.add(i) = *inp1.ptr.add(i) + *inp2.ptr.add(i);
    });
}

/// Accumulates gradients for two input tensors using the gradient of the output tensor.
///
/// # Arguments
///
/// * `dinp1` - Gradient of the first input tensor.
/// * `dinp2` - Gradient of the second input tensor.
/// * `dout` - Gradient of the output tensor.
/// * `N` - Number of elements.
pub unsafe fn residual_backward(
    dinp1: SendPtr<f32>,
    dinp2: SendPtr<f32>,
    dout: SendPtr<f32>,
    N: usize,
) {
    (0..N).into_par_iter().for_each(|i| {
        let dinp1 = dinp1;
        let dinp2 = dinp2;
        let dout = dout;

        // Update the gradients for the inputs
        *dinp1.ptr.add(i) += *dout.ptr.add(i);
        *dinp2.ptr.add(i) += *dout.ptr.add(i);
    });
}

/// Computes the softmax probabilities from logits in parallel.
///
/// # Arguments
///
/// * `probs` - Output probabilities (B, T, Vp).
/// * `logits` - Input unnormalized log probabilities (B, T, Vp).
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `V` - Real vocabulary size.
/// * `Vp` - Padded vocabulary size.
pub unsafe fn softmax_forward(
    probs: SendPtr<f32>,
    logits: SendPtr<f32>,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let probs = probs;
            let logits = logits;

            // Calculate the base addresses
            let logits_bt = logits.ptr.add(b * T * Vp + t * Vp);
            let probs_bt = probs.ptr.add(b * T * Vp + t * Vp);

            // Calculate maxval for numerical stability
            let mut maxval = f32::NEG_INFINITY;
            for i in 0..V {
                let logit = *logits_bt.add(i);
                if logit > maxval {
                    maxval = logit;
                }
            }

            // Calculate softmax numerator and denominator (sum)
            let mut sum = 0.0;
            for i in 0..V {
                let exp_val = (logits_bt.add(i).read() - maxval).exp();
                probs_bt.add(i).write(exp_val);
                sum += exp_val;
            }

            // Normalize the probabilities
            for i in 0..V {
                probs_bt.add(i).write(probs_bt.add(i).read() / sum);
            }

            // Set padded dimensions to zero
            for i in V..Vp {
                probs_bt.add(i).write(0.0);
            }
        });
    });
}

/// Computes the cross-entropy losses from probabilities and targets.
///
/// # Arguments
///
/// * `losses` - Output losses (B, T).
/// * `probs` - Input probabilities (B, T, Vp).
/// * `targets` - Target indices (B, T).
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `Vp` - Padded vocabulary size.
pub unsafe fn crossentropy_forward(
    losses: SendPtr<f32>,
    probs: SendPtr<f32>,
    targets: SendPtr<i32>,
    B: usize,
    T: usize,
    Vp: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let losses = losses;
            let probs = probs;
            let targets = targets;

            // Calculate the base address for probs
            let probs_bt = probs.ptr.add(b * T * Vp + t * Vp);

            // Get the target index
            let ix = *targets.ptr.add(b * T + t) as usize;

            // Compute the cross-entropy loss and store it
            *losses.ptr.add(b * T + t) = -probs_bt.add(ix).read().ln();
        });
    });
}

/// Backward pass through both softmax and cross-entropy loss.
///
/// # Arguments
///
/// * `dlogits` - Gradient of the logits (B, T, Vp).
/// * `dlosses` - Gradient of the losses (B, T).
/// * `probs` - Probabilities (B, T, Vp).
/// * `targets` - Target indices (B, T).
/// * `B` - Batch size.
/// * `T` - Sequence length.
/// * `V` - Real vocabulary size.
/// * `Vp` - Padded vocabulary size.
pub unsafe fn crossentropy_softmax_backward(
    dlogits: SendPtr<f32>,
    dlosses: SendPtr<f32>,
    probs: SendPtr<f32>,
    targets: SendPtr<i32>,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dlogits = dlogits;
            let dlosses = dlosses;
            let probs = probs;
            let targets = targets;

            // Calculate the base addresses
            let dlogits_bt = dlogits.ptr.add(b * T * Vp + t * Vp);
            let probs_bt = probs.ptr.add(b * T * Vp + t * Vp);
            let dloss = *dlosses.ptr.add(b * T + t);
            let ix = *targets.ptr.add(b * T + t) as usize;

            // Loop only to V, leaving padded dimensions untouched
            for i in 0..V {
                let p = *probs_bt.add(i);
                let indicator = if i == ix { 1.0 } else { 0.0 };
                *dlogits_bt.add(i) += (p - indicator) * dloss;
            }
        });
    });
}
