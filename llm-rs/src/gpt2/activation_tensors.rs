use std::ptr::null_mut;
use std::alloc::{alloc, Layout};

pub const NUM_ACTIVATION_TENSORS: usize = 23;

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
    /// Creates a new ActivationTensors instance.
    ///
    /// # Returns
    ///
    /// A new `ActivationTensors` instance.
    pub fn new() -> Self {
        ActivationTensors {
            encoded: null_mut(),
            ln1: null_mut(),
            ln1_mean: null_mut(),
            ln1_rstd: null_mut(),
            qkv: null_mut(),
            atty: null_mut(),
            preatt: null_mut(),
            att: null_mut(),
            attproj: null_mut(),
            residual2: null_mut(),
            ln2: null_mut(),
            ln2_mean: null_mut(),
            ln2_rstd: null_mut(),
            fch: null_mut(),
            fch_gelu: null_mut(),
            fcproj: null_mut(),
            residual3: null_mut(),
            lnf: null_mut(),
            lnf_mean: null_mut(),
            lnf_rstd: null_mut(),
            logits: null_mut(),
            probs: null_mut(),
            losses: null_mut(),
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