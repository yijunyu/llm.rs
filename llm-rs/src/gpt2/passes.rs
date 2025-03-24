use std::f32::consts::PI;

// ----------------------------------------------------------------------------
// All the individual layers' forward and backward passes
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
pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            let out_bt = &mut out[bt * C..(bt + 1) * C];
            let ix = inp[bt] as usize;
            let wte_ix = &wte[ix * C..(ix + 1) * C];
            let wpe_t = &wpe[t * C..(t + 1) * C];

            for i in 0..C {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
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
pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            let dout_bt = &dout[bt * C..(bt + 1) * C];
            let ix = inp[bt] as usize;
            let dwte_ix = &mut dwte[ix * C..(ix + 1) * C];
            let dwpe_t = &mut dwpe[t * C..(t + 1) * C];

            for i in 0..C {
                let d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
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
pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    let eps: f32 = 1e-5;

    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            // Calculate the base address for inp[b,t,:]
            let x = &inp[bt * C..(bt + 1) * C];

            // Calculate the mean
            let mut m: f32 = 0.0;
            for i in 0..C {
                m += x[i];
            }
            m /= C as f32;

            // Calculate the variance
            let mut v: f32 = 0.0;
            for i in 0..C {
                let xshift = x[i] - m;
                v += xshift * xshift;
            }
            v /= C as f32;

            // Calculate the rstd (reciprocal standard deviation)
            let s: f32 = 1.0 / (v + eps).sqrt();

            // Calculate the base address for out[b,t,:]
            let out_bt = &mut out[bt * C..(bt + 1) * C];
            for i in 0..C {
                let n = s * (x[i] - m); // Normalize
                let o = n * weight[i] + bias[i]; // Scale and shift
                out_bt[i] = o; // Write
            }

            // Cache the mean and rstd for the backward pass
            mean[bt] = m;
            rstd[bt] = s;
        }
    }
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
pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            // Calculate the base addresses
            let dout_bt = &dout[bt * C..(bt + 1) * C];
            let inp_bt = &inp[bt * C..(bt + 1) * C];
            let dinp_bt = &mut dinp[bt * C..(bt + 1) * C];
            let mean_bt = mean[bt];
            let rstd_bt = rstd[bt];

            // First: two reduce operations
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Now iterate again and accumulate all the gradients
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];

                // Gradient contribution to bias
                dbias[i] += dout_bt[i];

                // Gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];

                // Gradient contribution to input
                let mut dval: f32 = 0.0;
                dval += dnorm_i; // Term 1
                dval -= dnorm_mean; // Term 2
                dval -= norm_bti * dnorm_norm_mean; // Term 3
                dval *= rstd_bt; // Final scale
                dinp_bt[i] += dval;
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
pub fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // Create a parallel iterator over the batch dimension
    for b in 0..B {
        // Create a parallel iterator over the sequence length
        for t in 0..T {
            let bt = b * T + t;

            // Iterate over the output channels
            for o in 0..OC {
                // Initialize the output value with the bias if provided, otherwise 0.0
                let mut val = if !bias.is_empty() {
                    bias[o]
                } else {
                    0.0f32
                };
                // Perform the dot product
                for i in 0..C {
                    val += inp[bt * C + i] * weight[o * C + i];
                }
                // Store the result
                out[bt * OC + o] = val;
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
pub fn matmul_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            let dout_bt = &dout[bt * OC..(bt + 1) * OC];
            let dinp_bt = &mut dinp[bt * C..(bt + 1) * C];

            for o in 0..OC {
                let wrow = &weight[o * C..(o + 1) * C];
                let d = dout_bt[o];
                for i in 0..C {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    for o in 0..OC {
        for b in 0..B {
            for t in 0..T {
                let bt = b * T + t;

                let dout_bt = &dout[bt * OC..(bt + 1) * OC];
                let inp_bt = &inp[bt * C..(bt + 1) * C];
                let dwrow = &mut dweight[o * C..(o + 1) * C];

                let d = dout_bt[o];
                if !dbias.is_empty() {
                    dbias[o] += d;
                }
                for i in 0..C {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

/// Forward pass for multi-head attention, generating output and storing attention scores.
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
pub fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let bth = b * NH * T + h * T + t;
                let query_offset = b * T * C3 + t * C3 + h * hs;

                let query_t = &inp[query_offset..query_offset + hs];
                let preatt_bth = &mut preatt[bth * T..(bth + 1) * T];
                let att_bth = &mut att[bth * T..(bth + 1) * T];

                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=t {
                    let key_offset = b * T * C3 + t2 * C3 + h * hs + C;

                    let key_t2 = &inp[key_offset..key_offset + hs];
                    let mut val = 0.0;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }

                let mut expsum = 0.0;
                for t2 in 0..=t {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                for t2 in 0..T {
                    if t2 <= t {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0;
                    }
                }

                let out_bth = &mut out[b * T * C + t * C + h * hs..]; // ++++++++++++++++++++++ QUESTIONABLE ++++++++++++++++++++++
                for i in 0..hs {
                    out_bth[i] = 0.0;
                }

                for t2 in 0..=t {
                    let value_offset = b * T * C3 + t2 * C3 + h * hs + 2 * C;

                    let value_t2 = &inp[value_offset..value_offset + hs];
                    let att_btht2 = att_bth[t2];

                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

/// Backward pass for attention mechanisms, updating gradients for inputs,
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
pub fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let bth = b * NH * T * T + h * T * T + t * T;

                let att_bth = &att[bth..bth + T];
                let datt_bth = &mut datt[bth..bth + T];
                let dpreatt_bth = &mut dpreatt[bth..bth + T];
                let dquery_t_offset = b * T * C3 + t * C3 + h * hs;
                let query_t = &inp[b * T * C3 + t * C3 + h * hs..b * T * C3 + t * C3 + (h + 1) * hs];

                // Backward pass 4: through the value accumulation
                let dout_bth = &dout[b * T * C + t * C + h * hs..];
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + 2 * C..];
                    let dvalue_t2 = &mut dinp[b * T * C3 + t2 * C3 + h * hs + 2 * C..];

                    for i in 0..hs {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // Backward pass 2 & 3: the softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }
                
                // Backward pass 1: the query @ key matmul
                for t2 in 0..=t {
                    let key_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C..];
                    let dkey_t2_offset = b * T * C3 + t2 * C3 + h * hs + C;

                    for i in 0..hs {
                        dinp[dquery_t_offset + i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dinp[dkey_t2_offset + i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

/// Applies the GELU activation function to the input tensor.
///
/// # Arguments
///
/// * `out` - Output tensor to store the GELU results.
/// * `inp` - Input tensor.
/// * `N` - Number of elements.
pub fn gelu_forward(
    out: &mut [f32], 
    inp: &[f32], 
    N: usize,
) {
    for i in 0..N {
        // Load the input value
        let x = inp[i];
        // Calculate the cubic term
        let cube = 0.044715 * x * x * x;
        // Apply the GeLU function
        out[i] = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());
    }
}

/// Computes the gradient of the GELU activation function.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `inp` - Input tensor.
/// * `dout` - Gradient of the output tensor.
/// * `N` - Number of elements.
pub fn gelu_backward(
    dinp: &mut [f32], 
    inp: &[f32], 
    dout: &[f32], 
    N: usize,
) {
    let gelu_scaling_factor = (2.0 / PI).sqrt();

    for i in 0..N {
        // Load the input value
        let x = inp[i];
        let dout_val = dout[i];

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
        dinp[i] += local_grad * dout_val;
    }
}

/// Adds two input tensors element-wise and stores the result in the output tensor.
///
/// # Arguments
///
/// * `out` - Output tensor to store the result.
/// * `inp1` - First input tensor.
/// * `inp2` - Second input tensor.
/// * `N` - Number of elements.
pub fn residual_forward(
    out: &mut [f32],
    inp1: &[f32],
    inp2: &[f32],
    N: usize,
) {
    for i in 0..N {
        // Perform element-wise addition
        out[i] = inp1[i] + inp2[i];
    }
}

/// Accumulates gradients for two input tensors using the gradient of the output tensor.
///
/// # Arguments
///
/// * `dinp1` - Gradient of the first input tensor.
/// * `dinp2` - Gradient of the second input tensor.
/// * `dout` - Gradient of the output tensor.
/// * `N` - Number of elements.
pub fn residual_backward(
    dinp1: &mut [f32],
    dinp2: &mut [f32],
    dout: &[f32],
    N: usize,
) {
    for i in 0..N {
        // Update the gradients for the inputs
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
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
pub fn softmax_forward(
    probs: &mut [f32],
    logits: &[f32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            // Calculate the base addresses
            let logits_bt = &logits[bt * Vp..(bt + 1) * Vp];
            let probs_bt = &mut probs[bt * Vp..(bt + 1) * Vp];

            // Calculate maxval for numerical stability
            let mut maxval = f32::NEG_INFINITY;
            for i in 0..V {
                let logit = logits_bt[i];
                if logit > maxval {
                    maxval = logit;
                }
            }

            // Calculate softmax numerator and denominator (sum)
            let mut sum = 0.0;
            for i in 0..V {
                let exp_val = (logits_bt[i] - maxval).exp();
                probs_bt[i] = exp_val;
                sum += exp_val;
            }

            // Normalize the probabilities
            for i in 0..V {
                probs_bt[i] = probs_bt[i] / sum;
            }

            // Set padded dimensions to zero
            for i in V..Vp {
                probs_bt[i] = 0.0;
            }
        }
    }
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
pub fn crossentropy_forward(
    losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            // Calculate the base address for probs
            let probs_bt = &probs[bt * Vp..(bt + 1) * Vp];

            // Get the target index
            let ix = targets[bt] as usize;

            // Compute the cross-entropy loss and store it
            losses[bt] = -probs_bt[ix].ln();
        }
    }
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
pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;

            // Calculate the base addresses
            let dlogits_bt = &mut dlogits[bt * Vp..(bt + 1) * Vp];
            let probs_bt = &probs[bt * Vp..(bt + 1) * Vp];
            let dloss = dlosses[bt];
            let ix = targets[bt] as usize;

            // Loop only to V, leaving padded dimensions untouched
            for i in 0..V {
                let p = probs_bt[i];
                let indicator = if i == ix { 1.0 } else { 0.0 };
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}
