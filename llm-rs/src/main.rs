#![allow(
    non_snake_case,
)]

pub mod dataloader;

use std::f32::consts::PI;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::alloc::{self, alloc, Layout};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::{ptr, mem};
use core::slice;
use rayon::prelude::*;
use std::time::Instant;
use std::io::{self, Write};
use dataloader::DataLoader;

const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;
const BATCH_SIZE: usize = 4;
const SEQ_LENGTH: usize = 64;

// ----------------------------------------------------------------------------
// GPT-2 model definition
// ----------------------------------------------------------------------------

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
    /// New initialization of GPT2 Config
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

#[derive(Debug, Clone, Copy)]
pub struct ParameterTensors {
    /// Token embeddings (V, C).
    pub wte: *mut f32,

    /// Position embeddings (maxT, C).
    pub wpe: *mut f32,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: *mut f32,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: *mut f32,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: *mut f32,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: *mut f32,

    /// Attention projection weights (L, C, C).
    pub attprojw: *mut f32,

    /// Attention projection biases (L, C).
    pub attprojb: *mut f32,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: *mut f32,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: *mut f32,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: *mut f32,

    /// Fully connected biases (L, 4*C).
    pub fcb: *mut f32,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: *mut f32,

    /// Fully connected projection biases (L, C).
    pub fcprojb: *mut f32,

    /// Final layer normalization weights (C).
    pub lnfw: *mut f32,

    /// Final layer normalization biases (C).
    pub lnfb: *mut f32,
}

impl ParameterTensors {
    /// New initialization of Parameter Tensors
    fn new() -> Self {
        ParameterTensors {
            wte: ptr::null_mut(),
            wpe: ptr::null_mut(),
            ln1w: ptr::null_mut(),
            ln1b: ptr::null_mut(),
            qkvw: ptr::null_mut(),
            qkvb: ptr::null_mut(),
            attprojw: ptr::null_mut(),
            attprojb: ptr::null_mut(),
            ln2w: ptr::null_mut(),
            ln2b: ptr::null_mut(),
            fcw: ptr::null_mut(),
            fcb: ptr::null_mut(),
            fcprojw: ptr::null_mut(),
            fcprojb: ptr::null_mut(),
            lnfw: ptr::null_mut(),
            lnfb: ptr::null_mut(),
        }
    }

    // Allocates memory for model parameters and sets their pointers within a `ParameterTensors` structure.
    ///
    /// # Arguments
    ///
    /// * `params` - Pointer to the `ParameterTensors` structure where the parameter tensor pointers will be set.
    /// * `param_sizes` - Array of sizes for each parameter tensor.
    ///
    /// # Returns
    ///
    /// * Pointer to the allocated memory for parameters.
    pub unsafe fn alloc_and_point_parameters(&mut self, param_sizes: &[usize; NUM_PARAMETER_TENSORS]) -> *mut f32 {
        // Calculate the total size needed
        let num_parameters: usize = param_sizes.iter().sum();

        // Allocate memory for all parameters
        let layout = Layout::array::<f32>(num_parameters).expect("Layout error");
        let params_memory = alloc(layout) as *mut f32;

        // Check for successful allocation
        if params_memory.is_null() {
            panic!("Memory allocation failed");
        }

        // Assign the tensors to the allocated memory
        let mut params_memory_iterator = params_memory;
        let mut ptrs: [&mut *mut f32; NUM_PARAMETER_TENSORS] = [
            &mut self.wte, &mut self.wpe, &mut self.ln1w, &mut self.ln1b, &mut self.qkvw,
            &mut self.qkvb, &mut self.attprojw, &mut self.attprojb, &mut self.ln2w,
            &mut self.ln2b, &mut self.fcw, &mut self.fcb, &mut self.fcprojw, &mut self.fcprojb,
            &mut self.lnfw, &mut self.lnfb
        ];

        for (i, ptr) in ptrs.iter_mut().enumerate() {
            **ptr = params_memory_iterator;
            params_memory_iterator = params_memory_iterator.add(param_sizes[i]);
        }

        params_memory
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActivationTensors {
    /// Encoded (B, T, C)
    pub encoded: *mut f32,
    
    /// Layer normalization 1 (L, B, T, C)
    pub ln1: *mut f32,

    /// Layer normalization 1 mean (L, B, T)
    pub ln1_mean: *mut f32,

    /// Layer normalization 1 reciprocal std (L, B, T)
    pub ln1_rstd: *mut f32,

    /// Query, Key, Value (L, B, T, 3*C)
    pub qkv: *mut f32,

    /// Attention output (L, B, T, C)
    pub atty: *mut f32,

    /// Pre-attention scores (L, B, NH, T, T)
    pub preatt: *mut f32,

    /// Attention scores (L, B, NH, T, T)
    pub att: *mut f32,

    /// Attention projection (L, B, T, C)
    pub attproj: *mut f32,

    /// Second residual connection (L, B, T, C)
    pub residual2: *mut f32,

    /// Layer normalization 2 (L, B, T, C)
    pub ln2: *mut f32,

    /// Layer normalization 2 mean (L, B, T)
    pub ln2_mean: *mut f32,

    /// Layer normalization 2 reciprocal std (L, B, T)
    pub ln2_rstd: *mut f32,

    /// Fully connected hidden (L, B, T, 4*C)
    pub fch: *mut f32,

    /// Fully connected hidden GELU activation (L, B, T, 4*C)
    pub fch_gelu: *mut f32,

    /// Fully connected projection (L, B, T, C)
    pub fcproj: *mut f32,

    /// Third residual connection (L, B, T, C)
    pub residual3: *mut f32,

    /// Final layer normalization (B, T, C)
    pub lnf: *mut f32,

    /// Final layer normalization mean (B, T)
    pub lnf_mean: *mut f32,

    /// Final layer normalization reciprocal std (B, T)
    pub lnf_rstd: *mut f32,

    /// Logits (B, T, V)
    pub logits: *mut f32,

    /// Probabilities (B, T, V)
    pub probs: *mut f32,

    /// Losses (B, T)
    pub losses: *mut f32,
}

impl ActivationTensors {
    /// New initialization of Activation Tensors
    fn new() -> Self {
        ActivationTensors {
            encoded: std::ptr::null_mut(),
            ln1: std::ptr::null_mut(),
            ln1_mean: std::ptr::null_mut(),
            ln1_rstd: std::ptr::null_mut(),
            qkv: std::ptr::null_mut(),
            atty: std::ptr::null_mut(),
            preatt: std::ptr::null_mut(),
            att: std::ptr::null_mut(),
            attproj: std::ptr::null_mut(),
            residual2: std::ptr::null_mut(),
            ln2: std::ptr::null_mut(),
            ln2_mean: std::ptr::null_mut(),
            ln2_rstd: std::ptr::null_mut(),
            fch: std::ptr::null_mut(),
            fch_gelu: std::ptr::null_mut(),
            fcproj: std::ptr::null_mut(),
            residual3: std::ptr::null_mut(),
            lnf: std::ptr::null_mut(),
            lnf_mean: std::ptr::null_mut(),
            lnf_rstd: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
            probs: std::ptr::null_mut(),
            losses: std::ptr::null_mut(),
        }
    }

    /// Allocates memory for activation tensors and sets their pointers within a `ActivationTensors` structure.
    ///
    /// # Arguments
    ///
    /// * `acts` - Pointer to the `ActivationTensors` structure where the activation tensor pointers will be set.
    /// * `act_sizes` - Array of sizes for each activation tensor.
    ///
    /// # Returns
    ///
    /// * Pointer to the allocated memory for activations.
    pub unsafe fn alloc_and_point_activations(&mut self, act_sizes: &[usize; NUM_ACTIVATION_TENSORS]) -> *mut f32 {
        // Calculate the total size needed
        let num_activations: usize = act_sizes.iter().sum();

        // Allocate memory for all activations
        let layout = Layout::array::<f32>(num_activations).expect("Layout error");
        let acts_memory = alloc(layout) as *mut f32;

        // Check for successful allocation
        if acts_memory.is_null() {
            panic!("Memory allocation failed");
        }

        // Assign the tensors to the allocated memory
        let mut acts_memory_iterator = acts_memory;
        let mut ptrs: [*mut *mut f32; NUM_ACTIVATION_TENSORS] = [
            &mut self.encoded, &mut self.ln1, &mut self.ln1_mean, &mut self.ln1_rstd, &mut self.qkv, &mut self.atty,
            &mut self.preatt, &mut self.att, &mut self.attproj, &mut self.residual2, &mut self.ln2, &mut self.ln2_mean,
            &mut self.ln2_rstd, &mut self.fch, &mut self.fch_gelu, &mut self.fcproj, &mut self.residual3, &mut self.lnf,
            &mut self.lnf_mean, &mut self.lnf_rstd, &mut self.logits, &mut self.probs, &mut self.losses
        ];

        // Assign slices of the allocated memory to each tensor
        for (i, ptr) in ptrs.iter_mut().enumerate() {
            **ptr = acts_memory_iterator;
            acts_memory_iterator = acts_memory_iterator.add(act_sizes[i]);
        }
    
        acts_memory
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
    /// New initialization of GPT2
    fn new(checkpoint_path: &Path) -> Self {
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
    out: *mut f32,
    inp: *const i32,
    wte: *const f32,
    wpe: *const f32,
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
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
    dwte: *mut f32,
    dwpe: *mut f32,
    dout: *const f32,
    inp: *const i32,
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            // Calculate the base address for dout[b,t,:]
            let dout_bt = dout.add(b * T * C + t * C);
            // Get the token index at inp[b, t]
            let ix = *inp.add(b * T + t) as usize;
            // Calculate the base address for dwte[ix,:]
            let dwte_ix = dwte.add(ix * C);
            // Calculate the base address for dwpe[t,:]
            let dwpe_t = dwpe.add(t * C);
            // Accumulate the gradients from dout into dwte and dwpe
            for i in 0..C {
                let d = *dout_bt.add(i);
                *dwte_ix.add(i) += d;
                *dwpe_t.add(i) += d;
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
    let eps: f32 = 1e-5;

    for b in 0..B {
        for t in 0..T {
            // Calculate the base address for inp[b,t,:]
            let x = inp.add(b * T * C + t * C);

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
            let out_bt = out.add(b * T * C + t * C);
            for i in 0..C {
                let n = s * (*x.add(i) - m); // Normalize
                let o = n * *weight.add(i) + *bias.add(i); // Scale and shift
                *out_bt.add(i) = o; // Write
            }

            // Cache the mean and rstd for the backward pass
            *mean.add(b * T + t) = m;
            *rstd.add(b * T + t) = s;
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
    for b in 0..B {
        for t in 0..T {
            // Calculate the base addresses
            let dout_bt = dout.add(b * T * C + t * C);
            let inp_bt = inp.add(b * T * C + t * C);
            let dinp_bt = dinp.add(b * T * C + t * C);
            let mean_bt = *mean.add(b * T + t);
            let rstd_bt = *rstd.add(b * T + t);

            // First: two reduce operations
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
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
                let mut dval: f32 = 0.0;
                dval += dnorm_i; // Term 1
                dval -= dnorm_mean; // Term 2
                dval -= norm_bti * dnorm_norm_mean; // Term 3
                dval *= rstd_bt; // Final scale
                *dinp_bt.add(i) += dval;
            }
        }
    }
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
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    let out_atomic = AtomicPtr::new(out);
    let inp_atomic = AtomicPtr::new(inp as *mut f32);
    let weight_atomic = AtomicPtr::new(weight as *mut f32);
    let bias_atomic = AtomicPtr::new(bias as *mut f32);

    // Create a parallel iterator over the batch dimension
    (0..B).into_par_iter().for_each(|b| {
        // Create a parallel iterator over the sequence length
        (0..T).into_par_iter().for_each(|t| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let out_raw = out_atomic.load(Ordering::SeqCst);
            let inp_raw = inp_atomic.load(Ordering::SeqCst);
            let weight_raw = weight_atomic.load(Ordering::SeqCst);
            let bias_raw = bias_atomic.load(Ordering::SeqCst);

            let bt = b * T + t;
            // Iterate over the output channels
            for o in 0..OC {
                // Initialize the output value with the bias if provided, otherwise 0.0
                let mut val = if !bias_raw.is_null() {
                    *bias_raw.add(o)
                } else {
                    0.0f32
                };
                // Perform the dot product
                for i in 0..C {
                    val += *inp_raw.add(bt * C + i) * *weight_raw.add(o * C + i);
                }
                // Store the result
                *out_raw.add(bt * OC + o) = val;
            }
        });
    });
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
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    bias: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    const LOOP_UNROLL: usize = 8;
    let out_atomic = AtomicPtr::new(out);
    let inp_atomic = AtomicPtr::new(inp as *mut f32);
    let weight_atomic = AtomicPtr::new(weight as *mut f32);
    let bias_atomic = AtomicPtr::new(bias as *mut f32);

    // Fallback to naive implementation if B * T is not a multiple of LOOP_UNROLL
    if (B * T) % LOOP_UNROLL != 0 {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // Parallelize the outer loop using Rayon
    (0..B * T).into_par_iter().step_by(LOOP_UNROLL).for_each(|obt| {
        // Load the AtomicPtr values into raw pointers for the current scope
        let out_raw = out_atomic.load(Ordering::SeqCst);
        let inp_raw = inp_atomic.load(Ordering::SeqCst);
        let weight_raw = weight_atomic.load(Ordering::SeqCst);
        let bias_raw = bias_atomic.load(Ordering::SeqCst);

        for o in 0..OC {
            // Initialize the result array with bias if present
            let mut result = [0.0f32; LOOP_UNROLL];
            for ibt in 0..LOOP_UNROLL {
                result[ibt] = if !bias_raw.is_null() { *bias_raw.add(o) } else { 0.0f32 };
            }

            // Cache the weight value and compute dot products
            for i in 0..C {
                let w = *weight_raw.add(i + o * C);
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt + ibt;
                    result[ibt] += *inp_raw.add(bt * C + i) * w;
                }
            }

            // Write results back to the output matrix
            for ibt in 0..LOOP_UNROLL {
                let bt = obt + ibt;
                *out_raw.add(bt * OC + o) = result[ibt];
            }
        }
    });
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
/// This backward could be done in a single "round" of loops but that doesn't afford an efficient parallelization strategy
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
    let dinp_atomic = AtomicPtr::new(dinp);
    let dweight_atomic = AtomicPtr::new(dweight);
    let dbias_atomic = AtomicPtr::new(dbias);
    let dout_atomic = AtomicPtr::new(dout as *mut f32);
    let inp_atomic = AtomicPtr::new(inp as *mut f32);
    let weight_atomic = AtomicPtr::new(weight as *mut f32);

    // Parallelize over B and T for input gradient computation
    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let dout_raw = dout_atomic.load(Ordering::SeqCst);
            let dinp_raw = dinp_atomic.load(Ordering::SeqCst);
            let weight_raw = weight_atomic.load(Ordering::SeqCst);

            // Calculate the base addresses for dout and dinp slices
            let dout_bt = dout_raw.add(b * T * OC + t * OC);
            let dinp_bt = dinp_raw.add(b * T * C + t * C);

            for o in 0..OC {
                let wrow = weight_raw.add(o * C);
                let d = *dout_bt.add(o);
                for i in 0..C {
                    *dinp_bt.add(i) += *wrow.add(i) * d;
                }
            }
        });
    });

    // Parallelize over output channels for weight and bias gradient computation
    (0..OC).into_par_iter().for_each(|o| {
        for b in 0..B {
            for t in 0..T {
                // Load the AtomicPtr values into raw pointers for the current scope
                let dout_raw = dout_atomic.load(Ordering::SeqCst);
                let inp_raw = inp_atomic.load(Ordering::SeqCst);
                let dweight_raw = dweight_atomic.load(Ordering::SeqCst);
                let dbias_raw = dbias_atomic.load(Ordering::SeqCst);

                // Calculate the base addresses for dout and inp slices
                let dout_bt = dout_raw.add(b * T * OC + t * OC);
                let inp_bt = inp_raw.add(b * T * C + t * C);
                let dwrow = dweight_raw.add(o * C);

                let d = *dout_bt.add(o);
                // Update dbias if not null
                if !dbias_raw.is_null() {
                    *dbias_raw.add(o) += d;
                }
                // Update dweight
                for i in 0..C {
                    *dwrow.add(i) += *inp_bt.add(i) * d;
                }
            }
        }
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
    out: *mut f32,
    preatt: *mut f32,
    att: *mut f32,
    inp: *const f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // feature dimension scaled by 3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // scale for dot product

    let out_atomic = AtomicPtr::new(out);
    let preatt_atomic = AtomicPtr::new(preatt);
    let att_atomic = AtomicPtr::new(att);
    let inp_atomic = AtomicPtr::new(inp as *mut f32);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                // Load the AtomicPtr values into raw pointers for the current scope
                let out_raw = out_atomic.load(Ordering::SeqCst);
                let preatt_raw = preatt_atomic.load(Ordering::SeqCst);
                let att_raw = att_atomic.load(Ordering::SeqCst);
                let inp_raw = inp_atomic.load(Ordering::SeqCst);

                // Calculate the base addresses
                let query_t = inp_raw.add(b * T * C3 + t * C3 + h * hs);
                let preatt_bth = preatt_raw.add(b * NH * T * T + h * T * T + t * T);
                let att_bth = att_raw.add(b * NH * T * T + h * T * T + t * T);

                // Pass 1: Calculate query dot key and maxval
                let mut maxval = f32::NEG_INFINITY; // Using f32::NEG_INFINITY for better initial value
                for t2 in 0..=t {
                    let key_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + C); // +C for key
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

                // Pass 2: Calculate the exp and keep track of sum
                let mut expsum = 0.0;
                for t2 in 0..=t {
                    let expv = (*preatt_bth.add(t2) - maxval).exp();
                    expsum += expv;
                    *att_bth.add(t2) = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: Normalize to get the softmax
                for t2 in 0..T {
                    if t2 <= t {
                        *att_bth.add(t2) *= expsum_inv;
                    } else {
                        *att_bth.add(t2) = 0.0;
                    }
                }

                // Pass 4: Accumulate weighted values into the output of attention
                let out_bth = out_raw.add(b * T * C + t * C + h * hs);
                for i in 0..hs {
                    *out_bth.add(i) = 0.0;
                }
                for t2 in 0..=t {
                    let value_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 for value
                    let att_btht2 = *att_bth.add(t2);
                    for i in 0..hs {
                        *out_bth.add(i) += att_btht2 * *value_t2.add(i);
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

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_bth = att.add(b * NH * T * T + h * T * T + t * T);
                let datt_bth = datt.add(b * NH * T * T + h * T * T + t * T);
                let dpreatt_bth = dpreatt.add(b * NH * T * T + h * T * T + t * T);
                let dquery_t = dinp.add(b * T * C3 + t * C3 + h * hs);
                let query_t = inp.add(b * T * C3 + t * C3 + h * hs);

                // Backward pass 4: through the value accumulation
                let dout_bth = dout.add(b * T * C + t * C + h * hs);
                for t2 in 0..=t {
                    let value_t2 = inp.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
                    let dvalue_t2 = dinp.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 because it's value
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
                    let key_t2 = inp.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    let dkey_t2 = dinp.add(b * T * C3 + t2 * C3 + h * hs + C); // +C because it's key
                    for i in 0..hs {
                        *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                        *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
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
pub unsafe fn gelu_forward(
    out: *mut f32, 
    inp: *const f32, 
    N: usize
) {
    // Process each element
    for i in 0..N {
        // Load the input value
        let x = *inp.add(i);
        // Calculate the cubic term
        let cube = 0.044715 * x * x * x;
        // Apply the GeLU function
        *out.add(i) = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());
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
pub unsafe fn gelu_backward(
    dinp: *mut f32,
    inp: *const f32,
    dout: *const f32,
    N: usize,
) {
    let gelu_scaling_factor = (2.0 / PI).sqrt();

    for i in 0..N {
        // Load the input value
        let x = *inp.add(i);
        let dout_val = *dout.add(i);
        
        // Compute the cubic term
        let cube = 0.044715 * x * x * x;
        
        // Compute the argument and the output of the tanh function
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        
        // Compute the hyperbolic cosine and sech (hyperbolic secant)
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);
        
        // Compute the local gradient
        let local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        
        // Accumulate the gradient into dinp
        *dinp.add(i) += local_grad * dout_val;
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
pub unsafe fn residual_forward(
    out: *mut f32,
    inp1: *const f32,
    inp2: *const f32,
    N: usize,
) {
    for i in 0..N {
        *out.add(i) = *inp1.add(i) + *inp2.add(i);
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
pub unsafe fn residual_backward(
    dinp1: *mut f32,
    dinp2: *mut f32,
    dout: *const f32,
    N: usize,
) {
    for i in 0..N {
        *dinp1.add(i) += *dout.add(i);
        *dinp2.add(i) += *dout.add(i);
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
pub unsafe fn softmax_forward(
    probs: *mut f32,
    logits: *const f32,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    let probs_atomic = AtomicPtr::new(probs);
    let logits_atomic = AtomicPtr::new(logits as *mut f32);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            // Load the AtomicPtr values into raw pointers for the current scope
            let probs_raw = probs_atomic.load(Ordering::SeqCst);
            let logits_raw = logits_atomic.load(Ordering::SeqCst);

            // Calculate the base addresses
            let logits_bt = logits_raw.add(b * T * Vp + t * Vp);
            let probs_bt = probs_raw.add(b * T * Vp + t * Vp);

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
    losses: *mut f32,
    probs: *const f32,
    targets: *const i32,
    B: usize,
    T: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            // Calculate the base address for probs
            let probs_bt = probs.add(b * T * Vp + t * Vp);

            // Get the target index
            let ix = *targets.add(b * T + t) as usize;

            // Compute the cross-entropy loss and store it
            *losses.add(b * T + t) = -probs_bt.add(ix).read().ln();
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
pub unsafe fn crossentropy_softmax_backward(
    dlogits: *mut f32,
    dlosses: *const f32,
    probs: *const f32,
    targets: *const i32,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            // Calculate the base addresses
            let dlogits_bt = dlogits.add(b * T * Vp + t * Vp);
            let probs_bt = probs.add(b * T * Vp + t * Vp);
            let dloss = *dlosses.add(b * T + t);
            let ix = *targets.add(b * T + t) as usize;

            // Loop only to V, leaving padded dimensions untouched
            for i in 0..V {
                let p = *probs_bt.add(i);
                let indicator = if i == ix { 1.0 } else { 0.0 };
                *dlogits_bt.add(i) += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Testing
// ----------------------------------------------------------------------------

/// Generates a random `u32` using the xorshift* algorithm.
///
/// # Arguments
///
/// * `state` - A mutable reference to the RNG state.
///
/// # Returns
///
/// A random `u32` value.
fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545F4914F6CDD1D) >> 32).try_into().unwrap()
}

/// Generates a random `f32` in the range [0, 1).
///
/// # Arguments
///
/// * `state` - A mutable reference to the RNG state.
///
/// # Returns
///
/// A random `f32` value in the range [0, 1).
fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0
}

/// Samples an index from the given probabilities.
///
/// # Arguments
///
/// * `probabilities` - A raw pointer to an array of probabilities. They must sum to 1.
/// * `n` - The number of probabilities.
/// * `coin` - A random number in the range [0, 1).
///
/// # Returns
///
/// The sampled index based on the given probabilities.
fn sample_mult(
    probabilities: *const f32, 
    n: usize, 
    coin: f32,
) -> usize {
    unsafe {
        let mut cdf = 0.0;
        for i in 0..n {
            cdf += *probabilities.add(i);
            if coin < cdf {
                return i;
            }
        }
        n - 1 // in case of rounding errors
    }
}

// ----------------------------------------------------------------------------
// Main training loop
// ----------------------------------------------------------------------------

pub fn main() {
    // Initialize the GPT-2 model from a checkpoint
    let checkpoint_path = Path::new("gpt2_124M.bin");
    let mut model = GPT2::new(checkpoint_path);

    // Build DataLoaders from token files
    let tiny_stories_train = Path::new("data/TinyStories_train.bin");
    let tiny_stories_val = Path::new("data/TinyStories_val.bin");
    let tiny_shakespeare_train = Path::new("data/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val = Path::new("data/tiny_shakespeare_val.bin");

    let train_tokens = if tiny_shakespeare_train.exists() {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let val_tokens = if tiny_shakespeare_val.exists() {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };

    let B = BATCH_SIZE;
    let T = SEQ_LENGTH;

    unsafe {
        let mut train_loader = DataLoader::new(train_tokens, B, T);
        let mut val_loader = DataLoader::new(val_tokens, B, T);
        println!("train dataset num_batches: {}", train_loader.num_batches);
        println!("val dataset num_batches: {}", val_loader.num_batches);

        let val_num_batches = 5;

        // Initialize the Tokenizer
        // let mut tokenizer = Tokenizer::new();
        // tokenizer.init("gpt2_tokenizer.bin");

        // Memory for generating samples
        let mut rng_state: u64 = 1337;
        let gen_tokens_layout = Layout::array::<i32>(B * T).expect("Failed to create layout");
        let gen_tokens = alloc(gen_tokens_layout) as *mut i32;
        let genT = 64;

        // Training loop
        for step in 0..=40 {
            // Estimate validation loss periodically
            if step % 10 == 0 {
                let mut val_loss = 0.0;
                val_loader.reset();
                for _ in 0..val_num_batches {
                    val_loader.next_batch();
                    model.forward(val_loader.inputs, val_loader.targets, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches as f32;
                println!("val loss {}", val_loss);
            }

            // Generate text periodically
            if step > 0 && step % 20 == 0 {
                for i in 0..B * T {
                    *gen_tokens.add(i) = 50256;
                }
                println!("generating:\n---");
                for t in 1..genT {
                    model.forward(gen_tokens, ptr::null(), B, T);
                    let probs = model.acts.probs.add((t - 1) * model.config.padded_vocab_size);
                    let coin = random_f32(&mut rng_state);
                    let next_token = sample_mult(probs, model.config.vocab_size, coin) as u32;
                    *gen_tokens.add(t) = next_token as i32;
                    // if tokenizer.init_ok {
                    //     let token_str = tokenizer.decode(next_token);
                    //     safe_printf(token_str.unwrap());
                    // } else {
                    //     print!("{} ", next_token);
                    // }
                    io::stdout().flush().unwrap();
                }
                println!("\n---");
            }

            // Training step
            let start = Instant::now();
            train_loader.next_batch();
            model.forward(train_loader.inputs, train_loader.targets, B, T);
            model.zero_grad();
            model.backward();
            model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
            let duration = start.elapsed();
            println!("step {}: train loss {} (took {:.2} ms)", step, model.mean_loss, duration.as_secs_f64() * 1000.0);
        }

        // Free resources
        train_loader.free();
        val_loader.free();
        // drop(tokenizer);
        model.free();
        alloc::dealloc(gen_tokens as *mut u8, gen_tokens_layout);
    }
}