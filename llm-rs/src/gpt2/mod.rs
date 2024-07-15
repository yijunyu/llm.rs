mod activation_tensors;
mod parameter_tensors;
mod passes;

use core::slice;
use std::alloc::{self, Layout};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::mem;
use std::ptr::{self, null_mut};

use activation_tensors::*;
use parameter_tensors::*;
use passes::*;

use crate::send_ptr::SendPtr;

#[derive(Debug, Clone, PartialEq)]
pub struct GPT2Config {
    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Padded vocabulary size.
    pub padded_vocab_size: usize,

    /// Number of layers.
    pub num_layers: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of channels.
    pub channels: usize,
}

impl GPT2Config {
    /// Creates a new GPT2Config instance.
    ///
    /// # Returns
    ///
    /// A new `GPT2Config` instance.
    fn new() -> Self {
        GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        }
    }
}

pub struct GPT2 {
    /// Model configuration.
    pub config: GPT2Config,

    /// The weights (parameters) of the model.
    pub params: ParameterTensors,

    /// Sizes of the model parameters.
    pub param_sizes: [usize; NUM_PARAMETER_TENSORS],

    /// Memory block containing all model parameters.
    pub params_memory: SendPtr<f32>,

    /// Total number of parameters.
    pub num_parameters: usize,

    /// Gradients of the weights.
    pub grads: ParameterTensors,

    /// Memory block containing all gradients of the model parameters.
    pub grads_memory: SendPtr<f32>,

    /// Buffer for the AdamW optimizer.
    pub m_memory: SendPtr<f32>,

    /// Buffer for the AdamW optimizer.
    pub v_memory: SendPtr<f32>,

    /// The activations of the model.
    pub acts: ActivationTensors,

    /// Sizes of the model activations.
    pub act_sizes: [usize; NUM_ACTIVATION_TENSORS],

    /// Memory block containing all activations.
    pub acts_memory: SendPtr<f32>,

    /// Total number of activations.
    pub num_activations: usize,

    /// Gradients of the activations.
    pub grads_acts: ActivationTensors,

    /// Memory block containing all gradients of the activations.
    pub grads_acts_memory: SendPtr<f32>,

    /// The batch size (B) of the current forward pass
    pub batch_size: usize,

    /// The sequence length (T) of the current forward pass
    pub seq_len: usize,

    /// The input tokens for the current forward pass
    pub inputs: SendPtr<i32>,

    /// The target tokens for the current forward pass
    pub targets: SendPtr<i32>,

    /// After a forward pass with targets, will be populated with the mean loss
    pub mean_loss: f32,
}

impl GPT2 {
    /// Creates a new GPT-2 model instance from a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the checkpoint file containing model parameters and configuration.
    ///
    /// # Returns
    ///
    /// A new `GPT2` model instance.
    pub fn new(checkpoint_path: &Path) -> Self {
        let mut model = GPT2 {
            config: GPT2Config::new(),
            params: ParameterTensors::new(),
            param_sizes: [0; NUM_PARAMETER_TENSORS],
            params_memory: SendPtr::new(null_mut()),
            num_parameters: 0,
            grads: ParameterTensors::new(),
            grads_memory: SendPtr::new(null_mut()),
            m_memory: SendPtr::new(null_mut()),
            v_memory: SendPtr::new(null_mut()),
            acts: ActivationTensors::new(),
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: SendPtr::new(null_mut()),
            num_activations: 0,
            grads_acts: ActivationTensors::new(),
            grads_acts_memory: SendPtr::new(null_mut()),
            inputs: SendPtr::new(null_mut()),
            targets: SendPtr::new(null_mut()),
            batch_size: 0,
            seq_len: 0,
            mean_loss: -1.0,
        };

        // Read model from a checkpoint file
        let mut model_file = File::open(checkpoint_path).unwrap_or_else(|_| {
            panic!("Error opening model file");
        });

        // Read model header
        let mut model_header = [0; 256];
        model_file
            .read_exact(unsafe {
                slice::from_raw_parts_mut(
                    model_header.as_mut_ptr() as *mut u8,
                    model_header.len() * mem::size_of::<i32>(),
                )
            })
            .expect("Failed to read model header");

        // Check magic number and version
        if model_header[0] != 20240326 {
            panic!("Bad magic model file");
        }
        if model_header[1] != 3 {
            panic!("Bad version in model file\n---> HINT: try to re-run `python train_gpt2.py`");
        }

        // Read in hyperparameters
        let maxT = model_header[2] as usize;
        let V = model_header[3] as usize;
        let L = model_header[4] as usize;
        let NH = model_header[5] as usize;
        let C = model_header[6] as usize;
        let Vp = model_header[7] as usize;
        model.config = GPT2Config {
            max_seq_len: maxT,
            vocab_size: V,
            padded_vocab_size: Vp,
            num_layers: L,
            num_heads: NH,
            channels: C,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {}", maxT);
        println!("vocab_size: {}", V);
        println!("padded_vocab_size: {}", Vp);
        println!("num_layers: {}", L);
        println!("num_heads: {}", NH);
        println!("channels: {}", C);

        // Allocate space for all the parameters and read them in
        model.param_sizes[0] = Vp * C; // wte
        model.param_sizes[1] = maxT * C; // wpe
        model.param_sizes[2] = L * C; // ln1w
        model.param_sizes[3] = L * C; // ln1b
        model.param_sizes[4] = L * (3 * C) * C; // qkvw
        model.param_sizes[5] = L * (3 * C); // qkvb
        model.param_sizes[6] = L * C * C; // attprojw
        model.param_sizes[7] = L * C; // attprojb
        model.param_sizes[8] = L * C; // ln2w
        model.param_sizes[9] = L * C; // ln2b
        model.param_sizes[10] = L * (4 * C) * C; // fcw
        model.param_sizes[11] = L * (4 * C); // fcb
        model.param_sizes[12] = L * C * (4 * C); // fcprojw
        model.param_sizes[13] = L * C; // fcprojb
        model.param_sizes[14] = C; // lnfw
        model.param_sizes[15] = C; // lnfb

        // Count the number of parameters
        let num_parameters: usize = model.param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);
        model.num_parameters = num_parameters;

        // Read in all the parameters from file
        unsafe {
            model.params_memory = model.params.alloc_and_point_parameters(&model.param_sizes);
            model_file
                .read_exact(slice::from_raw_parts_mut(
                    model.params_memory.ptr as *mut u8,
                    num_parameters * mem::size_of::<f32>(),
                ))
                .expect("Failed to read parameters");
        }

        model
    }

    /// Performs the forward pass for a GPT-2 model, computing token embeddings, attention layers,
    /// and optionally the loss if targets are provided.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the GPT-2 model containing parameters and buffers.
    /// * `inputs` - Input tensor containing token indices.
    /// * `targets` - Target tensor containing token indices for loss calculation (optional).
    /// * `B` - Batch size.
    /// * `T` - Sequence length.
    pub fn forward(&mut self, inputs: SendPtr<i32>, targets: SendPtr<i32>, B: usize, T: usize) {
        // Ensure the model was initialized or error out
        if self.params_memory.ptr.is_null() {
            panic!("Error: model was not initialized properly.");
        }

        // Convenience parameters
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // Validate inputs, all indices must be in the range [0, V)
        unsafe {
            for i in 0..(B * T) {
                assert!((*inputs.ptr.add(i) >= 0 && *inputs.ptr.add(i) < V as i32));
                if !targets.ptr.is_null() {
                    assert!((*targets.ptr.add(i) >= 0 && *targets.ptr.add(i) < V as i32));
                }
            }
        }

        // Allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory.ptr.is_null() {
            // Record the current B, T as well
            self.batch_size = B;
            self.seq_len = T;

            // Allocate space for activations
            self.act_sizes[0] = B * T * C; // encoded
            self.act_sizes[1] = L * B * T * C; // ln1
            self.act_sizes[2] = L * B * T; // ln1_mean
            self.act_sizes[3] = L * B * T; // ln1_rstd
            self.act_sizes[4] = L * B * T * 3 * C; // qkv
            self.act_sizes[5] = L * B * T * C; // atty
            self.act_sizes[6] = L * B * NH * T * T; // preatt
            self.act_sizes[7] = L * B * NH * T * T; // att
            self.act_sizes[8] = L * B * T * C; // attproj
            self.act_sizes[9] = L * B * T * C; // residual2
            self.act_sizes[10] = L * B * T * C; // ln2
            self.act_sizes[11] = L * B * T; // ln2_mean
            self.act_sizes[12] = L * B * T; // ln2_rstd
            self.act_sizes[13] = L * B * T * 4 * C; // fch
            self.act_sizes[14] = L * B * T * 4 * C; // fch_gelu
            self.act_sizes[15] = L * B * T * C; // fcproj
            self.act_sizes[16] = L * B * T * C; // residual3
            self.act_sizes[17] = B * T * C; // lnf
            self.act_sizes[18] = B * T; // lnf_mean
            self.act_sizes[19] = B * T; // lnf_rstd
            self.act_sizes[20] = B * T * Vp; // logits
            self.act_sizes[21] = B * T * Vp; // probs
            self.act_sizes[22] = B * T; // losses

            let num_activations: usize = self.act_sizes.iter().sum();
            println!("num_activations: {}", num_activations);
            self.num_activations = num_activations;

            unsafe {
                self.acts_memory = self.acts.alloc_and_point_activations(&self.act_sizes);

                // Create memory for caching inputs and targets
                let input_layout = Layout::array::<i32>(B * T).expect("Failed to create layout");
                self.inputs.ptr = alloc::alloc(input_layout) as *mut i32;
                self.targets.ptr = alloc::alloc(input_layout) as *mut i32; // might be unused if we never have targets but it's small
            }
        } else {
            // Validate B, T is consistent with how we've allocated the memory before
            if B != self.batch_size || T != self.seq_len {
                panic!(
                    "Model: B={} T={}, Desired: B={} T={}",
                    self.batch_size, self.seq_len, B, T
                );
            }
        }

        // Cache the inputs/targets
        unsafe {
            ptr::copy_nonoverlapping(inputs.ptr, self.inputs.ptr, B * T);
            if !targets.ptr.is_null() {
                ptr::copy_nonoverlapping(targets.ptr, self.targets.ptr, B * T);
            }
        }

        // Forward pass
        let params = &self.params;
        let acts = &mut self.acts;
        let mut residual: SendPtr<f32> = SendPtr::new(null_mut());

        unsafe {
            encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

            for l in 0..L {
                residual.ptr = if l == 0 {
                    acts.encoded.ptr
                } else {
                    acts.residual3.ptr.add((l - 1) * B * T * C)
                };

                // Get the pointers of the weights for this layer
                let l_ln1w = SendPtr::new(params.ln1w.ptr.add(l * C));
                let l_ln1b = SendPtr::new(params.ln1b.ptr.add(l * C));
                let l_qkvw = SendPtr::new(params.qkvw.ptr.add(l * 3 * C * C));
                let l_qkvb = SendPtr::new(params.qkvb.ptr.add(l * 3 * C));
                let l_attprojw = SendPtr::new(params.attprojw.ptr.add(l * C * C));
                let l_attprojb = SendPtr::new(params.attprojb.ptr.add(l * C));
                let l_ln2w = SendPtr::new(params.ln2w.ptr.add(l * C));
                let l_ln2b = SendPtr::new(params.ln2b.ptr.add(l * C));
                let l_fcw = SendPtr::new(params.fcw.ptr.add(l * 4 * C * C));
                let l_fcb = SendPtr::new(params.fcb.ptr.add(l * 4 * C));
                let l_fcprojw = SendPtr::new(params.fcprojw.ptr.add(l * C * 4 * C));
                let l_fcprojb = SendPtr::new(params.fcprojb.ptr.add(l * C));

                // Get the pointers of the activations for this layer
                let l_ln1 = SendPtr::new(acts.ln1.ptr.add(l * B * T * C));
                let l_ln1_mean = SendPtr::new(acts.ln1_mean.ptr.add(l * B * T));
                let l_ln1_rstd = SendPtr::new(acts.ln1_rstd.ptr.add(l * B * T));
                let l_qkv = SendPtr::new(acts.qkv.ptr.add(l * B * T * 3 * C));
                let l_atty = SendPtr::new(acts.atty.ptr.add(l * B * T * C));
                let l_preatt = SendPtr::new(acts.preatt.ptr.add(l * B * NH * T * T));
                let l_att = SendPtr::new(acts.att.ptr.add(l * B * NH * T * T));
                let l_attproj = SendPtr::new(acts.attproj.ptr.add(l * B * T * C));
                let l_residual2 = SendPtr::new(acts.residual2.ptr.add(l * B * T * C));
                let l_ln2 = SendPtr::new(acts.ln2.ptr.add(l * B * T * C));
                let l_ln2_mean = SendPtr::new(acts.ln2_mean.ptr.add(l * B * T));
                let l_ln2_rstd = SendPtr::new(acts.ln2_rstd.ptr.add(l * B * T));
                let l_fch = SendPtr::new(acts.fch.ptr.add(l * B * T * 4 * C));
                let l_fch_gelu = SendPtr::new(acts.fch_gelu.ptr.add(l * B * T * 4 * C));
                let l_fcproj = SendPtr::new(acts.fcproj.ptr.add(l * B * T * C));
                let l_residual3 = SendPtr::new(acts.residual3.ptr.add(l * B * T * C));

                // Now do the forward pass
                layernorm_forward(
                    l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C,
                );
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
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
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
            }

            residual.ptr = acts.residual3.ptr.add((L - 1) * B * T * C); // last residual is in residual3
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
            matmul_forward(acts.logits, acts.lnf, params.wte, SendPtr::new(null_mut()), B, T, C, Vp);
            softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

            // Forward the cross-entropy loss function if we have the targets
            if !targets.ptr.is_null() {
                crossentropy_forward(self.acts.losses, self.acts.probs, targets, B, T, Vp);
                // Evaluate the mean loss
                let mut mean_loss = 0.0;
                for i in 0..(B * T) {
                    mean_loss += *self.acts.losses.ptr.add(i);
                }
                mean_loss /= (B * T) as f32;
                self.mean_loss = mean_loss;
            } else {
                // If we don't have targets, we don't have a loss
                self.mean_loss = -1.0;
            }
        }
    }

    /// Performs the backward pass for the GPT2 model.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub unsafe fn backward(&mut self) {
        // Double-check we forwarded previously, with targets
        if self.mean_loss == -1.0 {
            panic!("Error: must forward with targets before backward");
        }

        // Lazily allocate memory for gradients if needed
        if self.grads_memory.ptr.is_null() {
            self.grads_memory = self.grads.alloc_and_point_parameters(&self.param_sizes);
            self.grads_acts_memory = self.grads_acts.alloc_and_point_activations(&self.act_sizes);
            self.zero_grad();
        }

        // Convenience shortcuts
        let B = self.batch_size;
        let T = self.seq_len;
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // Start backpropagation
        let params = &self.params;
        let grads = &mut self.grads;
        let acts = &self.acts;
        let grads_acts = &mut self.grads_acts;

        // Kick off the chain rule by filling in dlosses with 1.0 / (B * T)
        let dloss_mean = 1.0 / (B * T) as f32;
        for i in 0..(B * T) {
            *grads_acts.losses.ptr.add(i) = dloss_mean;
        }

        crossentropy_softmax_backward(
            grads_acts.logits,
            grads_acts.losses,
            acts.probs,
            self.targets,
            B,
            T,
            V,
            Vp,
        );
        matmul_backward(
            grads_acts.lnf,
            grads.wte,
            SendPtr::new(null_mut()),
            grads_acts.logits,
            acts.lnf,
            params.wte,
            B,
            T,
            C,
            Vp,
        );
        let mut residual = SendPtr::new(acts.residual3.ptr.add((L - 1) * B * T * C)); // last layer's residual
        let mut dresidual = SendPtr::new(grads_acts.residual3.ptr.add((L - 1) * B * T * C)); // write to last layer's residual
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

        for l in (0..L).rev() {
            residual.ptr = if l == 0 {
                acts.encoded.ptr
            } else {
                acts.residual3.ptr.add((l - 1) * B * T * C)
            };
            dresidual.ptr = if l == 0 {
                grads_acts.encoded.ptr
            } else {
                grads_acts.residual3.ptr.add((l - 1) * B * T * C)
            };

            // Get the pointers of the weights for this layer
            let l_ln1w = SendPtr::new(params.ln1w.ptr.add(l * C));
            let l_qkvw = SendPtr::new(params.qkvw.ptr.add(l * 3 * C * C));
            let l_attprojw = SendPtr::new(params.attprojw.ptr.add(l * C * C));
            let l_ln2w = SendPtr::new(params.ln2w.ptr.add(l * C));
            let l_fcw = SendPtr::new(params.fcw.ptr.add(l * 4 * C * C));
            let l_fcprojw = SendPtr::new(params.fcprojw.ptr.add(l * C * 4 * C));

            // Get the pointers of the gradients of the weights for this layer
            let dl_ln1w = SendPtr::new(grads.ln1w.ptr.add(l * C));
            let dl_ln1b = SendPtr::new(grads.ln1b.ptr.add(l * C));
            let dl_qkvw = SendPtr::new(grads.qkvw.ptr.add(l * 3 * C * C));
            let dl_qkvb = SendPtr::new(grads.qkvb.ptr.add(l * 3 * C));
            let dl_attprojw = SendPtr::new(grads.attprojw.ptr.add(l * C * C));
            let dl_attprojb = SendPtr::new(grads.attprojb.ptr.add(l * C));
            let dl_ln2w = SendPtr::new(grads.ln2w.ptr.add(l * C));
            let dl_ln2b = SendPtr::new(grads.ln2b.ptr.add(l * C));
            let dl_fcw = SendPtr::new(grads.fcw.ptr.add(l * 4 * C * C));
            let dl_fcb = SendPtr::new(grads.fcb.ptr.add(l * 4 * C));
            let dl_fcprojw = SendPtr::new(grads.fcprojw.ptr.add(l * C * 4 * C));
            let dl_fcprojb = SendPtr::new(grads.fcprojb.ptr.add(l * C));

            // Get the pointers of the activations for this layer
            let l_ln1 = SendPtr::new(acts.ln1.ptr.add(l * B * T * C));
            let l_ln1_mean = SendPtr::new(acts.ln1_mean.ptr.add(l * B * T));
            let l_ln1_rstd = SendPtr::new(acts.ln1_rstd.ptr.add(l * B * T));
            let l_qkv = SendPtr::new(acts.qkv.ptr.add(l * B * T * 3 * C));
            let l_atty = SendPtr::new(acts.atty.ptr.add(l * B * T * C));
            let l_att = SendPtr::new(acts.att.ptr.add(l * B * NH * T * T));
            let l_residual2 = SendPtr::new(acts.residual2.ptr.add(l * B * T * C));
            let l_ln2 = SendPtr::new(acts.ln2.ptr.add(l * B * T * C));
            let l_ln2_mean = SendPtr::new(acts.ln2_mean.ptr.add(l * B * T));
            let l_ln2_rstd = SendPtr::new(acts.ln2_rstd.ptr.add(l * B * T));
            let l_fch = SendPtr::new(acts.fch.ptr.add(l * B * T * 4 * C));
            let l_fch_gelu = SendPtr::new(acts.fch_gelu.ptr.add(l * B * T * 4 * C));

            // Get the pointers of the gradients of the activations for this layer
            let dl_ln1 = SendPtr::new(grads_acts.ln1.ptr.add(l * B * T * C));
            let dl_qkv = SendPtr::new(grads_acts.qkv.ptr.add(l * B * T * 3 * C));
            let dl_atty = SendPtr::new(grads_acts.atty.ptr.add(l * B * T * C));
            let dl_preatt = SendPtr::new(grads_acts.preatt.ptr.add(l * B * NH * T * T));
            let dl_att = SendPtr::new(grads_acts.att.ptr.add(l * B * NH * T * T));
            let dl_attproj = SendPtr::new(grads_acts.attproj.ptr.add(l * B * T * C));
            let dl_residual2 = SendPtr::new(grads_acts.residual2.ptr.add(l * B * T * C));
            let dl_ln2 = SendPtr::new(grads_acts.ln2.ptr.add(l * B * T * C));
            let dl_fch = SendPtr::new(grads_acts.fch.ptr.add(l * B * T * 4 * C));
            let dl_fch_gelu = SendPtr::new(grads_acts.fch_gelu.ptr.add(l * B * T * 4 * C));
            let dl_fcproj = SendPtr::new(grads_acts.fcproj.ptr.add(l * B * T * C));
            let dl_residual3 = SendPtr::new(grads_acts.residual3.ptr.add(l * B * T * C));

            // Backprop this layer
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
                4 * C,
                C,
            );
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
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
                dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH,
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
                3 * C,
            );
            layernorm_backward(
                dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B,
                T, C,
            );
        }
        encoder_backward(
            grads.wte,
            grads.wpe,
            grads_acts.encoded,
            self.inputs,
            B,
            T,
            C,
        );
    }

    /// Sets all gradients in the model to zero.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub unsafe fn zero_grad(&mut self) {
        if !self.grads_memory.ptr.is_null() {
            // Create a slice from the grads_memory pointer
            let grads_slice = slice::from_raw_parts_mut(self.grads_memory.ptr, self.num_parameters);
            // Set all elements in the grads_slice to 0
            ptr::write_bytes(grads_slice.as_mut_ptr(), 0, self.num_parameters);
        }

        if !self.grads_acts_memory.ptr.is_null() {
            // Create a slice from the grads_acts_memory pointer
            let grads_acts_slice =
                slice::from_raw_parts_mut(self.grads_acts_memory.ptr, self.num_activations);
            // Set all elements in the grads_acts_slice to 0
            ptr::write_bytes(grads_acts_slice.as_mut_ptr(), 0, self.num_activations);
        }
    }

    /// Updates the GPT2 model parameters using AdamW optimization.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    /// * `learning_rate` - Learning rate.
    /// * `beta1` - Exponential decay rate for the first moment estimates.
    /// * `beta2` - Exponential decay rate for the second moment estimates.
    /// * `eps` - Small constant for numerical stability.
    /// * `weight_decay` - Weight decay coefficient.
    /// * `t` - Time step.
    pub unsafe fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: usize,
    ) {
        // Lazily allocate the memory for m_memory and v_memory
        if self.m_memory.ptr.is_null() {
            let m_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.m_memory = SendPtr::new(alloc::alloc_zeroed(m_layout) as *mut f32);
        }
        if self.v_memory.ptr.is_null() {
            let v_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.v_memory = SendPtr::new(alloc::alloc_zeroed(v_layout) as *mut f32);
        }

        // Iterate over the parameters and update using AdamW
        for i in 0..self.num_parameters {
            let param = *self.params_memory.ptr.add(i);
            let grad = *self.grads_memory.ptr.add(i);

            // Update the first moment (momentum)
            let m = beta1 * *self.m_memory.ptr.add(i) + (1.0 - beta1) * grad;
            // Update the second moment (RMSprop)
            let v = beta2 * *self.v_memory.ptr.add(i) + (1.0 - beta2) * grad * grad;
            // Bias-correct both moments
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // Update m and v in the model
            *self.m_memory.ptr.add(i) = m;
            *self.v_memory.ptr.add(i) = v;

            // Update the parameters
            *self.params_memory.ptr.add(i) -=
                learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }

    /// Frees the memory allocated for the GPT2 model.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub unsafe fn free(&mut self) {
        unsafe fn free_memory<T>(send_ptr: SendPtr<T>, num_elements: usize) {
            if !send_ptr.ptr.is_null() {
                let layout = Layout::array::<T>(num_elements).expect("Layout error");
                alloc::dealloc(send_ptr.ptr as *mut u8, layout);
            }
        }

        // Deallocate memory for model parameters
        free_memory(self.params_memory, self.num_parameters);
        free_memory(self.grads_memory, self.num_parameters);
        free_memory(self.m_memory, self.num_parameters);
        free_memory(self.v_memory, self.num_parameters);

        // Deallocate memory for model activations
        free_memory(self.acts_memory, self.num_activations);
        free_memory(self.grads_acts_memory, self.num_activations);

        // Deallocate memory for inputs and targets
        free_memory(self.inputs, self.batch_size * self.seq_len);
        free_memory(self.targets, self.batch_size * self.seq_len);

        // Set pointers to null after deallocation
        self.params_memory = SendPtr::new(null_mut());
        self.grads_memory = SendPtr::new(null_mut());
        self.m_memory = SendPtr::new(null_mut());
        self.v_memory = SendPtr::new(null_mut());
        self.acts_memory = SendPtr::new(null_mut());
        self.grads_acts_memory = SendPtr::new(null_mut());
        self.inputs = SendPtr::new(null_mut());
        self.targets = SendPtr::new(null_mut());
    }
}
