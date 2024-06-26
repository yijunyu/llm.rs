mod parameter_tensors;
mod activation_tensors;
mod util;

use std::alloc::{self, alloc, Layout};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::{ptr, mem};
use core::slice;

use parameter_tensors::*;
use activation_tensors::*;
use util::*;

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
    pub params_memory: *mut f32,

    /// Total number of parameters.
    pub num_parameters: usize,

    /// Gradients of the weights.
    pub grads: ParameterTensors,

    /// Memory block containing all gradients of the model parameters.
    pub grads_memory: *mut f32,

    /// Buffer for the AdamW optimizer.
    pub m_memory: *mut f32,

    /// Buffer for the AdamW optimizer.
    pub v_memory: *mut f32,

    /// The activations of the model.
    pub acts: ActivationTensors,

    /// Sizes of the model activations.
    pub act_sizes: [usize; NUM_ACTIVATION_TENSORS],

    /// Memory block containing all activations.
    pub acts_memory: *mut f32,

    /// Total number of activations.
    pub num_activations: usize,

    /// Gradients of the activations.
    pub grads_acts: ActivationTensors,

    /// Memory block containing all gradients of the activations.
    pub grads_acts_memory: *mut f32,

    /// The batch size (B) of the current forward pass
    pub batch_size: usize,

    /// The sequence length (T) of the current forward pass
    pub seq_len: usize,

    /// The input tokens for the current forward pass
    pub inputs: *mut i32,

    /// The target tokens for the current forward pass
    pub targets: *mut i32,

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
            params_memory: std::ptr::null_mut(),
            num_parameters: 0,
            grads: ParameterTensors::new(),
            grads_memory: std::ptr::null_mut(),
            m_memory: std::ptr::null_mut(),
            v_memory: std::ptr::null_mut(),
            acts: ActivationTensors::new(),
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: std::ptr::null_mut(),
            num_activations: 0,
            grads_acts: ActivationTensors::new(),
            grads_acts_memory: std::ptr::null_mut(),
            inputs: std::ptr::null_mut(),
            targets: std::ptr::null_mut(),
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
        model_file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                model_header.as_mut_ptr() as *mut u8,
                model_header.len() * mem::size_of::<i32>(),
            )
        }).expect("Failed to read model header");

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
            model_file.read_exact(
                slice::from_raw_parts_mut(
                    model.params_memory as *mut u8,
                    num_parameters * mem::size_of::<f32>(),
                )
            ).expect("Failed to read parameters");
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
    pub fn forward(
        &mut self,
        inputs: *const i32,
        targets: *const i32,
        B: usize,
        T: usize,
    ) {
        // Ensure the model was initialized or error out
        if self.params_memory.is_null() {
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
                assert!((*inputs.add(i) >= 0 && *inputs.add(i) < V as i32));
                if !targets.is_null() {
                    assert!((*targets.add(i) >= 0 && *targets.add(i) < V as i32));
                }
            }
        }

        // Allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory.is_null() {
            // Record the current B, T as well
            self.batch_size = B;
            self.seq_len = T;

            // Allocate space for activations
            self.act_sizes[0] = B * T * C; // encoded
            self.act_sizes[1] = L * B * T * C; // ln1
            self.act_sizes[2] = L * B * T;  // ln1_mean
            self.act_sizes[3] = L * B * T;  // ln1_rstd
            self.act_sizes[4] = L * B * T * 3 * C; // qkv
            self.act_sizes[5] = L * B * T * C;  // atty
            self.act_sizes[6] = L * B * NH * T * T;  // preatt
            self.act_sizes[7] = L * B * NH * T * T;  // att
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
                self.inputs = alloc(input_layout) as *mut i32;
                self.targets = alloc(input_layout) as *mut i32; // might be unused if we never have targets but it's small
            }
        } else {
            // Validate B, T is consistent with how we've allocated the memory before
            if B != self.batch_size || T != self.seq_len {
                panic!("Model: B={} T={}, Desired: B={} T={}", self.batch_size, self.seq_len, B, T);
            }
        }

        // Cache the inputs/targets
        unsafe {
            ptr::copy_nonoverlapping(inputs, self.inputs, B * T);
            if !targets.is_null() {
                ptr::copy_nonoverlapping(targets, self.targets, B * T);
            }
        }

        // Forward pass
        let params = &self.params;
        let acts = &mut self.acts;
        let mut residual: *mut f32;

        unsafe {
            encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

            for l in 0..L {
                residual = if l == 0 { acts.encoded } else { acts.residual3.add((l - 1) * B * T * C) };

                // Get the pointers of the weights for this layer
                let l_ln1w = params.ln1w.add(l * C);
                let l_ln1b = params.ln1b.add(l * C);
                let l_qkvw = params.qkvw.add(l * 3 * C * C);
                let l_qkvb = params.qkvb.add(l * 3 * C);
                let l_attprojw = params.attprojw.add(l * C * C);
                let l_attprojb = params.attprojb.add(l * C);
                let l_ln2w = params.ln2w.add(l * C);
                let l_ln2b = params.ln2b.add(l * C);
                let l_fcw = params.fcw.add(l * 4 * C * C);
                let l_fcb = params.fcb.add(l * 4 * C);
                let l_fcprojw = params.fcprojw.add(l * C * 4 * C);
                let l_fcprojb = params.fcprojb.add(l * C);

                // Get the pointers of the activations for this layer
                let l_ln1 = acts.ln1.add(l * B * T * C);
                let l_ln1_mean = acts.ln1_mean.add(l * B * T);
                let l_ln1_rstd = acts.ln1_rstd.add(l * B * T);
                let l_qkv = acts.qkv.add(l * B * T * 3 * C);
                let l_atty = acts.atty.add(l * B * T * C);
                let l_preatt = acts.preatt.add(l * B * NH * T * T);
                let l_att = acts.att.add(l * B * NH * T * T);
                let l_attproj = acts.attproj.add(l * B * T * C);
                let l_residual2 = acts.residual2.add(l * B * T * C);
                let l_ln2 = acts.ln2.add(l * B * T * C);
                let l_ln2_mean = acts.ln2_mean.add(l * B * T);
                let l_ln2_rstd = acts.ln2_rstd.add(l * B * T);
                let l_fch = acts.fch.add(l * B * T * 4 * C);
                let l_fch_gelu = acts.fch_gelu.add(l * B * T * 4 * C);
                let l_fcproj = acts.fcproj.add(l * B * T * C);
                let l_residual3 = acts.residual3.add(l * B * T * C);

                // Now do the forward pass
                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
                attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
                matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
                residual_forward(l_residual2, residual, l_attproj, B * T * C);
                layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
            }

            residual = acts.residual3.add((L - 1) * B * T * C); // last residual is in residual3
            layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
            matmul_forward(acts.logits, acts.lnf, params.wte, ptr::null(), B, T, C, Vp);
            softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

            // Forward the cross-entropy loss function if we have the targets
            if !targets.is_null() {
                crossentropy_forward(self.acts.losses, self.acts.probs, targets, B, T, Vp);
                // Evaluate the mean loss
                let mut mean_loss = 0.0;
                for i in 0..(B * T) {
                    mean_loss += *self.acts.losses.add(i);
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
        if self.grads_memory.is_null() {
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
            *grads_acts.losses.add(i) = dloss_mean;
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
            ptr::null_mut(),
            grads_acts.logits,
            acts.lnf,
            params.wte,
            B,
            T,
            C,
            Vp,
        );
        let mut residual = acts.residual3.add((L - 1) * B * T * C); // last layer's residual
        let mut dresidual = grads_acts.residual3.add((L - 1) * B * T * C); // write to last layer's residual
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
            residual = if l == 0 {
                acts.encoded
            } else {
                acts.residual3.add((l - 1) * B * T * C)
            };
            dresidual = if l == 0 {
                grads_acts.encoded
            } else {
                grads_acts.residual3.add((l - 1) * B * T * C)
            };

            // Get the pointers of the weights for this layer
            let l_ln1w = params.ln1w.add(l * C);
            let l_qkvw = params.qkvw.add(l * 3 * C * C);
            let l_attprojw = params.attprojw.add(l * C * C);
            let l_ln2w = params.ln2w.add(l * C);
            let l_fcw = params.fcw.add(l * 4 * C * C);
            let l_fcprojw = params.fcprojw.add(l * C * 4 * C);

            // Get the pointers of the gradients of the weights for this layer
            let dl_ln1w = grads.ln1w.add(l * C);
            let dl_ln1b = grads.ln1b.add(l * C);
            let dl_qkvw = grads.qkvw.add(l * 3 * C * C);
            let dl_qkvb = grads.qkvb.add(l * 3 * C);
            let dl_attprojw = grads.attprojw.add(l * C * C);
            let dl_attprojb = grads.attprojb.add(l * C);
            let dl_ln2w = grads.ln2w.add(l * C);
            let dl_ln2b = grads.ln2b.add(l * C);
            let dl_fcw = grads.fcw.add(l * 4 * C * C);
            let dl_fcb = grads.fcb.add(l * 4 * C);
            let dl_fcprojw = grads.fcprojw.add(l * C * 4 * C);
            let dl_fcprojb = grads.fcprojb.add(l * C);

            // Get the pointers of the activations for this layer
            let l_ln1 = acts.ln1.add(l * B * T * C);
            let l_ln1_mean = acts.ln1_mean.add(l * B * T);
            let l_ln1_rstd = acts.ln1_rstd.add(l * B * T);
            let l_qkv = acts.qkv.add(l * B * T * 3 * C);
            let l_atty = acts.atty.add(l * B * T * C);
            let l_att = acts.att.add(l * B * NH * T * T);
            let l_residual2 = acts.residual2.add(l * B * T * C);
            let l_ln2 = acts.ln2.add(l * B * T * C);
            let l_ln2_mean = acts.ln2_mean.add(l * B * T);
            let l_ln2_rstd = acts.ln2_rstd.add(l * B * T);
            let l_fch = acts.fch.add(l * B * T * 4 * C);
            let l_fch_gelu = acts.fch_gelu.add(l * B * T * 4 * C);

            // Get the pointers of the gradients of the activations for this layer
            let dl_ln1 = grads_acts.ln1.add(l * B * T * C);
            let dl_qkv = grads_acts.qkv.add(l * B * T * 3 * C);
            let dl_atty = grads_acts.atty.add(l * B * T * C);
            let dl_preatt = grads_acts.preatt.add(l * B * NH * T * T);
            let dl_att = grads_acts.att.add(l * B * NH * T * T);
            let dl_attproj = grads_acts.attproj.add(l * B * T * C);
            let dl_residual2 = grads_acts.residual2.add(l * B * T * C);
            let dl_ln2 = grads_acts.ln2.add(l * B * T * C);
            let dl_fch = grads_acts.fch.add(l * B * T * 4 * C);
            let dl_fch_gelu = grads_acts.fch_gelu.add(l * B * T * 4 * C);
            let dl_fcproj = grads_acts.fcproj.add(l * B * T * C);
            let dl_residual3 = grads_acts.residual3.add(l * B * T * C);

            // Backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        }
        encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, self.inputs, B, T, C);
    }

    /// Sets all gradients in the model to zero.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub unsafe fn zero_grad(&mut self) {
        if !self.grads_memory.is_null() {
            // Create a slice from the grads_memory pointer
            let grads_slice = slice::from_raw_parts_mut(self.grads_memory, self.num_parameters);
            // Set all elements in the grads_slice to 0
            ptr::write_bytes(grads_slice.as_mut_ptr(), 0, self.num_parameters);
        }

        if !self.grads_acts_memory.is_null() {
            // Create a slice from the grads_acts_memory pointer
            let grads_acts_slice = slice::from_raw_parts_mut(self.grads_acts_memory, self.num_activations);
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
        if self.m_memory.is_null() {
            let m_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.m_memory = alloc::alloc_zeroed(m_layout) as *mut f32;
        }
        if self.v_memory.is_null() {
            let v_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.v_memory = alloc::alloc_zeroed(v_layout) as *mut f32;
        }

        // Iterate over the parameters and update using AdamW
        for i in 0..self.num_parameters {
            let param = *self.params_memory.add(i);
            let grad = *self.grads_memory.add(i);

            // Update the first moment (momentum)
            let m = beta1 * *self.m_memory.add(i) + (1.0 - beta1) * grad;
            // Update the second moment (RMSprop)
            let v = beta2 * *self.v_memory.add(i) + (1.0 - beta2) * grad * grad;
            // Bias-correct both moments
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // Update m and v in the model
            *self.m_memory.add(i) = m;
            *self.v_memory.add(i) = v;

            // Update the parameters
            *self.params_memory.add(i) -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }

    /// Frees the memory allocated for the GPT2 model.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub unsafe fn free(&mut self) {
        unsafe fn free_memory<T>(ptr: *mut T, num_elements: usize) {
            if !ptr.is_null() {
                let layout = Layout::array::<T>(num_elements).expect("Layout error");
                alloc::dealloc(ptr as *mut u8, layout);
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
        self.params_memory = ptr::null_mut();
        self.grads_memory = ptr::null_mut();
        self.m_memory = ptr::null_mut();
        self.v_memory = ptr::null_mut();
        self.acts_memory = ptr::null_mut();
        self.grads_acts_memory = ptr::null_mut();
        self.inputs = ptr::null_mut();
        self.targets = ptr::null_mut();
    }
}