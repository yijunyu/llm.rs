use rayon::prelude::*;
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
/// * `T` - Sequence length.
/// * `C` - Embedding dimension.
pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    T: usize,
    C: usize,
) {
    out.par_chunks_mut(C).enumerate().for_each(|(bt, out_chunk)| {
        let t = bt % T;

        let ix = inp[bt] as usize;
        let wte_slice = &wte[ix * C..(ix + 1) * C];
        let wpe_slice = &wpe[t * C..(t + 1) * C];

        // Sum the token and positional embeddings for this position
        for i in 0..C {
            out_chunk[i] = wte_slice[i] + wpe_slice[i];
        }
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
pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    (0..B).into_iter().for_each(|b| {
        (0..T).into_iter().for_each(|t| {
            let bt = b * T + t;

            let dout_slice = &dout[bt * C..(bt + 1) * C];
            let ix = inp[bt] as usize;
            let dwte_slice = &mut dwte[ix * C..(ix + 1) * C];
            let dwpe_slice = &mut dwpe[t * C..(t + 1) * C];

            for i in 0..C {
                let d = dout_slice[i];
                dwpe_slice[i] += d;
                dwte_slice[i] += d;
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
    C: usize,
) {
    let eps: f32 = 1e-5;

    out.par_chunks_mut(C).zip_eq(mean.into_par_iter()).zip_eq(rstd.into_par_iter()).enumerate().for_each(|(bt, ((out_chunk, mean_bt), rstd_bt))| {
        // Find the corresponding slice in inp
        let inp_slice = &inp[bt * C..(bt + 1) * C]; // Sliced batch `b`, time step `t`

        // Step 1: Compute mean
        let mean_value = inp_slice.iter().copied().sum::<f32>() / C as f32;

        // Step 2: Compute variance and standard deviation
        let variance = inp_slice.iter().map(|&x| (x - mean_value).powi(2)).sum::<f32>() / C as f32;
        let rstd_value = (variance + eps).sqrt().recip();

        // Step 3: Normalize and apply scaling and bias
        inp_slice.iter().enumerate().for_each(|(i, &x)| {
            let normalized = (x - mean_value) * rstd_value; // Normalize
            out_chunk[i] = normalized * weight[i] + bias[i]; // Scale and shift
        });

        // Step 4: Store mean and reciprocal standard deviation for backprop
        *mean_bt = mean_value;
        *rstd_bt = rstd_value;
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
    C: usize,
) {
    // Use par_chunks_mut for efficient parallel iteration
    let (acc_dweight, acc_dbias) = dinp.par_chunks_mut(C).enumerate().fold(
        // Initial accumulator values: zero-initialized vectors for dweight and dbias
        || (vec![0.0; C], vec![0.0; C]),
        |(mut acc_dweight, mut acc_dbias), (bt, dinp_chunk)| {
            let dout_slice = &dout[bt * C..(bt + 1) * C];
            let inp_slice = &inp[bt * C..(bt + 1) * C];

            let mean_value = mean[bt]; // Mean for the current batch/time
            let rstd_value = rstd[bt]; // Reciprocal standard deviation for the current batch/time

            // First: two reduce operations
            let mut dnorm_mean = 0.0;
            let mut dnorm_norm_mean = 0.0;

            for i in 0..C {
                let norm_bti = (inp_slice[i] - mean_value) * rstd_value;
                let dnorm_i = weight[i] * dout_slice[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }

            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Now iterate again and accumulate all the gradients
            for i in 0..C {
                let output_grad = dout_slice[i];
                let norm_bti = (inp_slice[i] - mean_value) * rstd_value;
                let dnorm_i = weight[i] * output_grad;

                // Gradient contribution to bias
                acc_dbias[i] += output_grad;

                // Gradient contribution to weight
                acc_dweight[i] += norm_bti * output_grad;

                // Gradient contribution to input
                let mut dval = 0.0;
                dval += dnorm_i; // Term 1
                dval -= dnorm_mean; // Term 2
                dval -= norm_bti * dnorm_norm_mean; // Term 3
                dval *= rstd_value; // Final scale
                dinp_chunk[i] += dval;
            }

            // Return the updated accumulator (dweight, dbias)
            (acc_dweight, acc_dbias)
        },
    ).reduce(
        // Identity for reduction: we start with zeroed accumulators
        || (vec![0.0; C], vec![0.0; C]),
        |(acc_dweight1, acc_dbias1), (acc_dweight2, acc_dbias2)| {
            // Combine the results of different threads
            let mut combined_dweight = acc_dweight1;
            let mut combined_dbias = acc_dbias1;

            for i in 0..C {
                combined_dweight[i] += acc_dweight2[i];
                combined_dbias[i] += acc_dbias2[i];
            }

            (combined_dweight, combined_dbias)
        }
    );

    // After the fold, accumulate the results into dweight and dbias
    for i in 0..C {
        dweight[i] += acc_dweight[i];
        dbias[i] += acc_dbias[i];
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
    unsafe {
        matrixmultiply::sgemm(
            B * T,
            C,
            OC,
            1.0,
            inp.as_ptr(),
            C as isize,
            1,
            weight.as_ptr(),
            1,
            C as isize,
            0.0,
            out.as_mut_ptr(),
            OC as isize,
            1,
        );
    }

    if !bias.is_empty() {
        for bt in 0..B * T {
            for o in 0..OC {
                out[bt * OC + o] += bias[o];
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
    unsafe {
        matrixmultiply::sgemm(
            B * T,
            OC,
            C,
            1.0,
            dout.as_ptr(),
            OC as isize,
            1,
            weight.as_ptr(),
            C as isize,
            1,
            1.0,
            dinp.as_mut_ptr(),
            C as isize,
            1,
        );
    
        matrixmultiply::sgemm(
            OC,
            B * T,
            C,
            1.0,
            dout.as_ptr(),
            1,
            OC as isize,
            inp.as_ptr(),
            C as isize,
            1,
            1.0,
            dweight.as_mut_ptr(),
            C as isize,
            1,
        );
    }

    if !dbias.is_empty() {
        for bt in 0..B * T {
            for o in 0..OC {
                dbias[o] += dout[bt * OC + o];
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
/// * `T` - Sequence length.
/// * `C` - Feature dimension.
/// * `NH` - Number of attention heads.
pub fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    out.par_chunks_mut(hs).zip_eq(preatt.par_chunks_mut(T)).zip_eq(att.par_chunks_mut(T)).enumerate().for_each(|(bth, ((out_chunk, preatt_chunk), att_chunk))| {
        let b = bth / (T * NH); // Batch index
        let t = (bth / NH) % T; // Time step index
        let h = bth % NH;       // Attention head index

        // Compute index for accessing query vector in `inp`
        let query_offset = b * T * C3 + t * C3 + h * hs;
        let query_t = &inp[query_offset..query_offset + hs];
        
        // Step 1: Compute unnormalized attention scores (dot product with scaling)
        let mut maxval = f32::NEG_INFINITY;
        for t2 in 0..=t {
            let key_offset = b * T * C3 + t2 * C3 + h * hs + C; // Offset for key vector
            let key_t2 = &inp[key_offset..key_offset + hs];

            let val = query_t.iter().zip(key_t2).map(|(&q, &k)| q * k).sum::<f32>() * scale;
            maxval = maxval.max(val);
            preatt_chunk[t2] = val;
        }

        // Step 2: Compute softmax over attention scores
        let mut expsum = 0.0;
        for t2 in 0..=t {
            let expv = (preatt_chunk[t2] - maxval).exp();
            expsum += expv;
            att_chunk[t2] = expv;
        }
        let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

        for t2 in 0..T {
            att_chunk[t2] = if t2 <= t { att_chunk[t2] * expsum_inv } else { 0.0 };
        }

        // Step 3: Compute weighted sum of values for attention output
        for i in 0..hs {
            out_chunk[i] = 0.0;
        }
        for t2 in 0..=t {
            let value_offset = b * T * C3 + t2 * C3 + h * hs + 2 * C; // Offset for value vector
            let value_t2 = &inp[value_offset..value_offset + hs];
            let att_weight = att_chunk[t2];

            for (o, &v) in out_chunk.iter_mut().zip(value_t2) {
                *o += att_weight * v;
            }
        }
    });
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
pub fn gelu_forward(
    out: &mut [f32], 
    inp: &[f32], 
) {
    out.into_par_iter().enumerate().for_each(|(i, out_val)| {
        // Load the input value
        let x = inp[i];

        // Calculate the cubic term
        let cube = 0.044715 * x * x * x;

        // Apply the GeLU function
        *out_val = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());
    });
}

/// Computes the gradient of the GELU activation function.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor.
/// * `inp` - Input tensor.
/// * `dout` - Gradient of the output tensor.
pub fn gelu_backward(
    dinp: &mut [f32], 
    inp: &[f32], 
    dout: &[f32], 
) {
    let gelu_scaling_factor = (2.0 / PI).sqrt();

    dinp.into_par_iter().enumerate().for_each(|(i, dinp_val)| {
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
        *dinp_val += local_grad * dout_val;
    });
}

/// Adds two input tensors element-wise and stores the result in the output tensor.
///
/// # Arguments
///
/// * `out` - Output tensor to store the result.
/// * `inp1` - First input tensor.
/// * `inp2` - Second input tensor.
pub fn residual_forward(
    out: &mut [f32],
    inp1: &[f32],
    inp2: &[f32],
) {
    out.into_par_iter().enumerate().for_each(|(i, out_val)| {
        // Perform element-wise addition
        *out_val = inp1[i] + inp2[i];
    });
}

/// Accumulates gradients for two input tensors using the gradient of the output tensor.
///
/// # Arguments
///
/// * `dinp1` - Gradient of the first input tensor.
/// * `dinp2` - Gradient of the second input tensor.
/// * `dout` - Gradient of the output tensor.
pub fn residual_backward(
    dinp1: &mut [f32],
    dinp2: &mut [f32],
    dout: &[f32],
) {
    dinp1.into_par_iter().zip_eq(dinp2.into_par_iter()).enumerate().for_each(|(i, (dinp1_val, dinp2_val))| {
        // Update the gradients for the inputs
        *dinp1_val += dout[i];
        *dinp2_val += dout[i];
    });
}

/// Computes the softmax probabilities from logits in parallel.
///
/// # Arguments
///
/// * `probs` - Output probabilities (B, T, Vp).
/// * `logits` - Input unnormalized log probabilities (B, T, Vp).
/// * `V` - Real vocabulary size.
/// * `Vp` - Padded vocabulary size.
pub fn softmax_forward(
    probs: &mut [f32],
    logits: &[f32],
    V: usize,
    Vp: usize,
) {
    probs.par_chunks_mut(Vp).enumerate().for_each(|(bt, probs_chunk)| {
        // Calculate the base addresses
        let logits_slice = &logits[bt * Vp..(bt + 1) * Vp];

        // Calculate maxval for numerical stability
        let mut maxval = f32::NEG_INFINITY;
        for i in 0..V {
            let logit = logits_slice[i];
            if logit > maxval {
                maxval = logit;
            }
        }

        // Calculate softmax numerator and denominator (sum)
        let mut sum = 0.0;
        for i in 0..V {
            let exp_val = (logits_slice[i] - maxval).exp();
            probs_chunk[i] = exp_val;
            sum += exp_val;
        }

        // Normalize the probabilities
        for i in 0..V {
            probs_chunk[i] = probs_chunk[i] / sum;
        }

        // Set padded dimensions to zero
        for i in V..Vp {
            probs_chunk[i] = 0.0;
        }
    });
}

/// Computes the cross-entropy losses from probabilities and targets.
///
/// # Arguments
///
/// * `losses` - Output losses (B, T).
/// * `probs` - Input probabilities (B, T, Vp).
/// * `targets` - Target indices (B, T).
/// * `T` - Sequence length.
/// * `Vp` - Padded vocabulary size.
pub fn crossentropy_forward(
    losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    T: usize,
    Vp: usize,
) {
    losses.into_par_iter().enumerate().for_each(|(i, loss_val)| {
        // Calculate the batch index (b) and time step index (t) from the flat index `i`
        let b = i / T; // integer division gives the batch index
        let t = i % T; // remainder gives the time step index

        let bt = b * T + t;

        // Calculate the base index for the current batch and time step in probs
        let probs_bt = &probs[bt * Vp..(bt + 1) * Vp];

        // Get the target index for the current batch and time step
        let ix = targets[bt] as usize;

        // Compute the cross-entropy loss for the current batch and time step
        *loss_val = -probs_bt[ix].ln();
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
/// * `V` - Real vocabulary size.
/// * `Vp` - Padded vocabulary size.
pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    V: usize,
    Vp: usize,
) {
    dlogits.par_chunks_mut(Vp).enumerate().for_each(|(bt, dlogits_chunk)| {
        // Calculate the base addresses
        let probs_slice = &probs[bt * Vp..(bt + 1) * Vp];
        let dloss = dlosses[bt];
        let ix = targets[bt] as usize;

        // Loop only to V, leaving padded dimensions untouched
        for i in 0..V {
            let p = probs_slice[i];
            let indicator = if i == ix { 1.0 } else { 0.0 };
            dlogits_chunk[i] += (p - indicator) * dloss;
        }
    });
}
