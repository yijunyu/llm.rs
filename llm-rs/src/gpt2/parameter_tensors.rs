use std::alloc::{alloc, Layout};
use std::ptr::null_mut;

pub const NUM_PARAMETER_TENSORS: usize = 16;

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
    /// Creates a new ParameterTensors instance.
    ///
    /// # Returns
    ///
    /// A new `ParameterTensors` instance.
    pub fn new() -> Self {
        ParameterTensors {
            wte: null_mut(),
            wpe: null_mut(),
            ln1w: null_mut(),
            ln1b: null_mut(),
            qkvw: null_mut(),
            qkvb: null_mut(),
            attprojw: null_mut(),
            attprojb: null_mut(),
            ln2w: null_mut(),
            ln2b: null_mut(),
            fcw: null_mut(),
            fcb: null_mut(),
            fcprojw: null_mut(),
            fcprojb: null_mut(),
            lnfw: null_mut(),
            lnfb: null_mut(),
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
    pub unsafe fn alloc_and_point_parameters(
        &mut self,
        param_sizes: &[usize; NUM_PARAMETER_TENSORS],
    ) -> *mut f32 {
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
        let mut ptrs: [*mut *mut f32; NUM_PARAMETER_TENSORS] = [
            &mut self.wte,
            &mut self.wpe,
            &mut self.ln1w,
            &mut self.ln1b,
            &mut self.qkvw,
            &mut self.qkvb,
            &mut self.attprojw,
            &mut self.attprojb,
            &mut self.ln2w,
            &mut self.ln2b,
            &mut self.fcw,
            &mut self.fcb,
            &mut self.fcprojw,
            &mut self.fcprojb,
            &mut self.lnfw,
            &mut self.lnfb,
        ];

        for (i, ptr) in ptrs.iter_mut().enumerate() {
            **ptr = params_memory_iterator;
            params_memory_iterator = params_memory_iterator.add(param_sizes[i]);
        }

        params_memory
    }
}
