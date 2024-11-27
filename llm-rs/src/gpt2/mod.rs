mod activation_tensors;
mod parameter_tensors;
mod passes;

use core::{panic, slice};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::mem;

use activation_tensors::*;
use parameter_tensors::*;
use passes::*;

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
    pub params_memory: Vec<f32>,

    /// Total number of parameters.
    pub num_parameters: usize,

    /// Gradients of the weights.
    pub grads: ParameterTensors,

    /// Memory block containing all gradients of the model parameters.
    pub grads_memory: Vec<f32>,

    /// Buffer for the AdamW optimizer.
    pub m_memory: Vec<f32>,

    /// Buffer for the AdamW optimizer.
    pub v_memory: Vec<f32>,

    /// The activations of the model.
    pub acts: ActivationTensors,

    /// Sizes of the model activations.
    pub act_sizes: [usize; NUM_ACTIVATION_TENSORS],

    /// Memory block containing all activations.
    pub acts_memory: Vec<f32>,

    /// Total number of activations.
    pub num_activations: usize,

    /// Gradients of the activations.
    pub grads_acts: ActivationTensors,

    /// Memory block containing all gradients of the activations.
    pub grads_acts_memory: Vec<f32>,

    /// The batch size (B) of the current forward pass
    pub batch_size: usize,

    /// The sequence length (T) of the current forward pass
    pub seq_len: usize,

    /// The input tokens for the current forward pass
    pub inputs: Vec<i32>,

    /// The target tokens for the current forward pass
    pub targets: Vec<i32>,

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
            params_memory: Vec::new(),
            num_parameters: 0,
            grads: ParameterTensors::new(),
            grads_memory: Vec::new(),
            m_memory: Vec::new(),
            v_memory: Vec::new(),
            acts: ActivationTensors::new(),
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: Vec::new(),
            num_activations: 0,
            grads_acts: ActivationTensors::new(),
            grads_acts_memory: Vec::new(),
            inputs: Vec::new(),
            targets: Vec::new(),
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
        model.params_memory = model.params.alloc_and_point_parameters(&model.param_sizes);

        // Calculate the total number of bytes needed
        let num_bytes = model.params_memory.len() * mem::size_of::<f32>();

        // Temporary byte buffer for reading the raw data from file
        let mut byte_buffer: Vec<u8> = vec![0; num_bytes];
        
        // Read the exact number of bytes from the file
        model_file.read_exact(&mut byte_buffer).unwrap();

        // Convert bytes to f32s and fill params_memory
        for (i, chunk) in byte_buffer.chunks_exact(4).enumerate() {
            model.params_memory[i] = f32::from_ne_bytes(chunk.try_into().unwrap());
        }

        let mut offset = 0;
        let mut assign_slice = |vec_field: &mut Vec<f32>, size: usize| {
            *vec_field = model.params_memory[offset..offset + size].to_vec();
            offset += size;
        };

        assign_slice(&mut model.params.wte, model.param_sizes[0]);
        assign_slice(&mut model.params.wpe, model.param_sizes[1]);
        assign_slice(&mut model.params.ln1w, model.param_sizes[2]);
        assign_slice(&mut model.params.ln1b, model.param_sizes[3]);
        assign_slice(&mut model.params.qkvw, model.param_sizes[4]);
        assign_slice(&mut model.params.qkvb, model.param_sizes[5]);
        assign_slice(&mut model.params.attprojw, model.param_sizes[6]);
        assign_slice(&mut model.params.attprojb, model.param_sizes[7]);
        assign_slice(&mut model.params.ln2w, model.param_sizes[8]);
        assign_slice(&mut model.params.ln2b, model.param_sizes[9]);
        assign_slice(&mut model.params.fcw, model.param_sizes[10]);
        assign_slice(&mut model.params.fcb, model.param_sizes[11]);
        assign_slice(&mut model.params.fcprojw, model.param_sizes[12]);
        assign_slice(&mut model.params.fcprojb, model.param_sizes[13]);
        assign_slice(&mut model.params.lnfw, model.param_sizes[14]);
        assign_slice(&mut model.params.lnfb, model.param_sizes[15]);
            
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
        inputs: &[i32], 
        targets: &[i32], 
        B: usize, 
        T: usize
    ) {
        // Ensure the model was initialized or error out
        if self.params_memory.is_empty() {
            panic!("Error: model was not initialized properly.");
        }

        // Convenience parameters
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // Validate inputs, all indices must be in the range [0, V)
        for i in 0..(B * T) {
            assert!((inputs[i] >= 0 && inputs[i] < V as i32));
            if !targets.is_empty() {
                assert!((targets[i] >= 0 && targets[i] < V as i32));
            }
        }

        // Allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory.is_empty() {
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

            self.acts_memory = self.acts.alloc_and_point_activations(&self.act_sizes);
        } else {
            // Validate B, T is consistent with how we've allocated the memory before
            if B != self.batch_size || T != self.seq_len {
                panic!(
                    "Model: B={} T={}, Desired: B={} T={}",
                    self.batch_size, self.seq_len, B, T
                );
            }
        }

        // Allocate space for caching inputs and targets as Vec<i32>
        self.inputs = inputs.to_vec(); // Cache the inputs
        self.targets = targets.to_vec(); // Cache the targets if provided

        // Forward pass
        let params = &self.params;
        let acts = &mut self.acts;

        encoder_forward(&mut acts.encoded, &inputs, &params.wte, &params.wpe, T, C);
        
        for l in 0..L {
            // Get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[l * C..(l + 1) * C];
            let l_ln1b = &params.ln1b[l * C..(l + 1) * C];
            let l_qkvw = &params.qkvw[l * 3 * C * C..(l + 1) * 3 * C * C];
            let l_qkvb = &params.qkvb[l * 3 * C..(l + 1) * 3 * C];
            let l_attprojw = &params.attprojw[l * C * C..(l + 1) * C * C];
            let l_attprojb = &params.attprojb[l * C..(l + 1) * C];
            let l_ln2w = &params.ln2w[l * C..(l + 1) * C];
            let l_ln2b = &params.ln2b[l * C..(l + 1) * C];
            let l_fcw = &params.fcw[l * 4 * C * C..(l + 1) * 4 * C * C];
            let l_fcb = &params.fcb[l * 4 * C..(l + 1) * 4 * C];
            let l_fcprojw = &params.fcprojw[l * C * 4 * C..(l + 1) * C * 4 * C];
            let l_fcprojb = &params.fcprojb[l * C..(l + 1) * C];

            // Get the pointers of the activations for this layer
            let mut l_ln1 = &mut acts.ln1[l * B * T * C..(l + 1) * B * T * C];
            let mut l_ln1_mean = &mut acts.ln1_mean[l * B * T..(l + 1) * B * T];
            let mut l_ln1_rstd = &mut acts.ln1_rstd[l * B * T..(l + 1) * B * T];
            let mut l_qkv = &mut acts.qkv[l * B * T * 3 * C..(l + 1) * B * T * 3 * C];
            let mut l_atty = &mut acts.atty[l * B * T * C..(l + 1) * B * T * C];
            let mut l_preatt = &mut acts.preatt[l * B * NH * T * T..(l + 1) * B * NH * T * T];
            let mut l_att = &mut acts.att[l * B * NH * T * T..(l + 1) * B * NH * T * T];
            let mut l_attproj = &mut acts.attproj[l * B * T * C..(l + 1) * B * T * C];
            let mut l_residual2 = &mut acts.residual2[l * B * T * C..(l + 1) * B * T * C];
            let mut l_ln2 = &mut acts.ln2[l * B * T * C..(l + 1) * B * T * C];
            let mut l_ln2_mean = &mut acts.ln2_mean[l * B * T..(l + 1) * B * T];
            let mut l_ln2_rstd = &mut acts.ln2_rstd[l * B * T..(l + 1) * B * T];
            let mut l_fch = &mut acts.fch[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];
            let mut l_fch_gelu = &mut acts.fch_gelu[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];
            let mut l_fcproj = &mut acts.fcproj[l * B * T * C..(l + 1) * B * T * C];

            let (left_residual, right_residual) = acts.residual3.split_at_mut(l * B * T * C);
            let mut l_residual3 = &mut right_residual[..B * T * C];

            let mut prev_residual = if l == 0 {
                &mut acts.encoded
            } else {
                &mut left_residual[(l - 1) * B * T * C..]
            };

            // Now do the forward pass
            layernorm_forward(&mut l_ln1, &mut l_ln1_mean, &mut l_ln1_rstd, &prev_residual, l_ln1w, l_ln1b, C,);
            matmul_forward(&mut l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            attention_forward(&mut l_atty, &mut l_preatt, &mut l_att, l_qkv, T, C, NH);
            matmul_forward(&mut l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(&mut l_residual2, &mut prev_residual, l_attproj);
            layernorm_forward(&mut l_ln2, &mut l_ln2_mean, &mut l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, C);
            matmul_forward(&mut l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(&mut l_fch_gelu, l_fch);
            matmul_forward(&mut l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(&mut l_residual3, l_residual2, l_fcproj);
        }

        layernorm_forward(&mut acts.lnf, &mut acts.lnf_mean, &mut acts.lnf_rstd, &acts.residual3[(L - 1) * B * T * C..L * B * T * C], &params.lnfw, &params.lnfb, C);
        matmul_forward(&mut acts.logits, &acts.lnf, &params.wte, &Vec::new(), B, T, C, Vp);
        softmax_forward(&mut acts.probs, &acts.logits, V, Vp);

        // Forward the cross-entropy loss function if we have the targets
        if !targets.is_empty() {
            crossentropy_forward(&mut self.acts.losses, &self.acts.probs, targets, T, Vp);

            // Evaluate the mean loss
            let mean_loss = self.acts.losses.iter().sum::<f32>() / (B * T) as f32;
            self.mean_loss = mean_loss;
        } else {
            // If we don't have targets, we don't have a loss
            self.mean_loss = -1.0;
        }
    }

    /// Performs the backward pass for the GPT2 model.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub fn backward(&mut self) {
        // Double-check we forwarded previously, with targets
        if self.mean_loss == -1.0 {
            panic!("Error: must forward with targets before backward");
        }

        // Lazily allocate memory for gradients if needed
        if self.grads_memory.is_empty() {
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
            grads_acts.losses[i] = dloss_mean;
        }

        crossentropy_softmax_backward(&mut grads_acts.logits, &grads_acts.losses, &acts.probs, &self.targets, V, Vp);
        matmul_backward(&mut grads_acts.lnf, &mut grads.wte, &mut Vec::new(), &grads_acts.logits, &acts.lnf, &params.wte, B, T, C, Vp);

        let mut residual = &acts.residual3[(L - 1) * B * T * C..L * B * T * C]; // last layer's residual
        let mut dresidual = &mut grads_acts.residual3[(L - 1) * B * T * C..L * B * T * C]; // write to last layer's residual

        // println!("Before:\n dresidual: {}\n grads.lnfw: {}\n grads.lnfb: {}\n grads_acts.lnf: {}\n residual: {}\n params.lnfw: {}\n acts.lnf_mean: {}\n acts.lnf_rstd: {}\n C: {}", dresidual[0], grads.lnfw[0], grads.lnfb[0], grads_acts.lnf[0], residual[0], params.lnfw[0], acts.lnf_mean[0], acts.lnf_rstd[0], C);
        layernorm_backward(&mut dresidual, &mut grads.lnfw, &mut grads.lnfb, &grads_acts.lnf, &residual, &params.lnfw, &acts.lnf_mean, &acts.lnf_rstd, C);
        // println!("After:\n dresidual: {}\n grads.lnfw: {}\n grads.lnfb: {}\n grads_acts.lnf: {}\n residual: {}\n params.lnfw: {}\n acts.lnf_mean: {}\n acts.lnf_rstd: {}\n C: {}\n", dresidual[0], grads.lnfw[0], grads.lnfb[0], grads_acts.lnf[0], residual[0], params.lnfw[0], acts.lnf_mean[0], acts.lnf_rstd[0], C);

        for l in (0..L).rev() {
            // Get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[l * C..(l + 1) * C];
            let l_qkvw = &params.qkvw[l * 3 * C * C..(l + 1) * 3 * C * C];
            let l_attprojw = &params.attprojw[l * C * C..(l + 1) * C * C];
            let l_ln2w = &params.ln2w[l * C..(l + 1) * C];
            let l_fcw = &params.fcw[l * 4 * C * C..(l + 1) * 4 * C * C];
            let l_fcprojw = &params.fcprojw[l * C * 4 * C..(l + 1) * C * 4 * C];

            // Get the pointers of the gradients of the weights for this layer
            let mut dl_ln1w = &mut grads.ln1w[l * C..(l + 1) * C];
            let mut dl_ln1b = &mut grads.ln1b[l * C..(l + 1) * C];
            let mut dl_qkvw = &mut grads.qkvw[l * 3 * C * C..(l + 1) * 3 * C * C];
            let mut dl_qkvb = &mut grads.qkvb[l * 3 * C..(l + 1) * 3 * C];
            let mut dl_attprojw = &mut grads.attprojw[l * C * C..(l + 1) * C * C];
            let mut dl_attprojb = &mut grads.attprojb[l * C..(l + 1) * C];
            let mut dl_ln2w = &mut grads.ln2w[l * C..(l + 1) * C];
            let mut dl_ln2b = &mut grads.ln2b[l * C..(l + 1) * C];
            let mut dl_fcw = &mut grads.fcw[l * 4 * C * C..(l + 1) * 4 * C * C];
            let mut dl_fcb = &mut grads.fcb[l * 4 * C..(l + 1) * 4 * C];
            let mut dl_fcprojw = &mut grads.fcprojw[l * C * 4 * C..(l + 1) * C * 4 * C];
            let mut dl_fcprojb = &mut grads.fcprojb[l * C..(l + 1) * C];

            // Get the pointers of the activations for this layer
            let l_ln1 = &acts.ln1[l * B * T * C..(l + 1) * B * T * C];
            let l_ln1_mean = &acts.ln1_mean[l * B * T..(l + 1) * B * T];
            let l_ln1_rstd = &acts.ln1_rstd[l * B * T..(l + 1) * B * T];
            let l_qkv = &acts.qkv[l * B * T * 3 * C..(l + 1) * B * T * 3 * C];
            let l_atty = &acts.atty[l * B * T * C..(l + 1) * B * T * C];
            let l_att = &acts.att[l * B * NH * T * T..(l + 1) * B * NH * T * T];
            let l_residual2 = &acts.residual2[l * B * T * C..(l + 1) * B * T * C];
            let l_ln2 = &acts.ln2[l * B * T * C..(l + 1) * B * T * C];
            let l_ln2_mean = &acts.ln2_mean[l * B * T..(l + 1) * B * T];
            let l_ln2_rstd = &acts.ln2_rstd[l * B * T..(l + 1) * B * T];
            let l_fch = &acts.fch[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];
            let l_fch_gelu = &acts.fch_gelu[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];

            // Get the pointers of the gradients of the activations for this layer
            let mut dl_ln1 = &mut grads_acts.ln1[l * B * T * C..(l + 1) * B * T * C];
            let mut dl_qkv = &mut grads_acts.qkv[l * B * T * 3 * C..(l + 1) * B * T * 3 * C];
            let mut dl_atty = &mut grads_acts.atty[l * B * T * C..(l + 1) * B * T * C];
            let mut dl_preatt = &mut grads_acts.preatt[l * B * NH * T * T..(l + 1) * B * NH * T * T];
            let mut dl_att = &mut grads_acts.att[l * B * NH * T * T..(l + 1) * B * NH * T * T];
            let mut dl_attproj = &mut grads_acts.attproj[l * B * T * C..(l + 1) * B * T * C];
            let mut dl_residual2 = &mut grads_acts.residual2[l * B * T * C..(l + 1) * B * T * C];
            let mut dl_ln2 = &mut grads_acts.ln2[l * B * T * C..(l + 1) * B * T * C];
            let mut dl_fch = &mut grads_acts.fch[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];
            let mut dl_fch_gelu = &mut grads_acts.fch_gelu[l * B * T * 4 * C..(l + 1) * B * T * 4 * C];
            let mut dl_fcproj = &mut grads_acts.fcproj[l * B * T * C..(l + 1) * B * T * C];
            
            let (left_residual, right_residual) = grads_acts.residual3.split_at_mut(l * B * T * C);
            let dl_residual3 = &mut right_residual[..B * T * C];

            residual = if l == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(l - 1) * B * T * C..l * B *T * C]
            };
            dresidual = if l == 0 {
                &mut grads_acts.encoded
            } else {
                &mut left_residual[(l - 1) * B * T * C..]
            };

            // Backprop this layer
            residual_backward(&mut dl_residual2, &mut dl_fcproj, &dl_residual3);
            matmul_backward(&mut dl_fch_gelu, &mut dl_fcprojw, &mut dl_fcprojb, &dl_fcproj, &l_fch_gelu, &l_fcprojw, B, T, 4 * C, C);
            gelu_backward(&mut dl_fch, &l_fch, &dl_fch_gelu);
            matmul_backward(&mut dl_ln2, &mut dl_fcw, &mut dl_fcb, &dl_fch, &l_ln2, &l_fcw, B, T, C, 4 * C);
            layernorm_backward(&mut dl_residual2, &mut dl_ln2w, &mut dl_ln2b, &dl_ln2, &l_residual2, &l_ln2w, &l_ln2_mean, &l_ln2_rstd, C);
            residual_backward(&mut dresidual, &mut dl_attproj, &dl_residual2);
            matmul_backward(&mut dl_atty, &mut dl_attprojw, &mut dl_attprojb, &dl_attproj, &l_atty, &l_attprojw, B, T, C, C);
            // attention_backward(&mut dl_qkv, &mut dl_preatt, &mut dl_att, &dl_atty, &l_qkv, &l_att, T, C, NH,);
            matmul_backward(&mut dl_ln1, &mut dl_qkvw, &mut dl_qkvb, &dl_qkv, &l_ln1, &l_qkvw, B, T, C, 3 * C);
            layernorm_backward(&mut dresidual, &mut dl_ln1w, &mut dl_ln1b, &dl_ln1, &residual, &l_ln1w, &l_ln1_mean, &l_ln1_rstd, C);
        }
        // encoder_backward(&mut grads.wte, &mut grads.wpe, &grads_acts.encoded, &self.inputs, T, C);
    }

    /// Sets all gradients in the model to zero.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub fn zero_grad(&mut self) {
        if !self.grads_memory.is_empty() {
            // Set all elements in the grads_memory vector to 0
            self.grads_memory.fill(0.0);
        }

        if !self.grads_acts_memory.is_empty() {
            // Set all elements in the grads_acts_memory vector to 0
            self.grads_acts_memory.fill(0.0);
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
    pub fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: usize,
    ) {
        // Lazily allocate the memory for m_memory and v_memory
        if self.m_memory.is_empty() {
            self.m_memory = vec![0.0; self.num_parameters];
        }
        if self.v_memory.is_empty() {
            self.v_memory = vec![0.0; self.num_parameters];
        }

        // Iterate over the parameters and update using AdamW
        for i in 0..self.num_parameters {
            let param = self.params_memory[i];
            let grad = self.grads_memory[i];

            // Update the first moment (momentum)
            let m = beta1 * self.m_memory[i] + (1.0 - beta1) * grad;
            // Update the second moment (RMSprop)
            let v = beta2 * self.v_memory[i] + (1.0 - beta2) * grad * grad;
            // Bias-correct both moments
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // Update m and v in the model
            self.m_memory[i] = m;
            self.v_memory[i] = v;

            // Update the parameters
            self.params_memory[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }

    /// Frees the memory allocated for the GPT2 model.
    ///
    /// # Arguments
    ///
    /// * `model` - The GPT2 model.
    pub fn free(&mut self) {
        // Clear the Vecs, which automatically frees their memory
        self.params_memory.clear();
        self.grads_memory.clear();
        self.m_memory.clear();
        self.v_memory.clear();
        self.acts_memory.clear();
        self.grads_acts_memory.clear();
        self.inputs.clear();
        self.targets.clear();
    }
}
