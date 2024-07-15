use std::alloc::{alloc, Layout};
use std::ptr::null_mut;

use crate::send_ptr::SendPtr;

pub const NUM_PARAMETER_TENSORS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct ParameterTensors {
    /// Token embeddings (V, C).
    pub wte: SendPtr<f32>,

    /// Position embeddings (maxT, C).
    pub wpe: SendPtr<f32>,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: SendPtr<f32>,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: SendPtr<f32>,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: SendPtr<f32>,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: SendPtr<f32>,

    /// Attention projection weights (L, C, C).
    pub attprojw: SendPtr<f32>,

    /// Attention projection biases (L, C).
    pub attprojb: SendPtr<f32>,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: SendPtr<f32>,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: SendPtr<f32>,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: SendPtr<f32>,

    /// Fully connected biases (L, 4*C).
    pub fcb: SendPtr<f32>,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: SendPtr<f32>,

    /// Fully connected projection biases (L, C).
    pub fcprojb: SendPtr<f32>,

    /// Final layer normalization weights (C).
    pub lnfw: SendPtr<f32>,

    /// Final layer normalization biases (C).
    pub lnfb: SendPtr<f32>,
}

impl ParameterTensors {
    /// Creates a new ParameterTensors instance.
    ///
    /// # Returns
    ///
    /// A new `ParameterTensors` instance.
    pub fn new() -> Self {
        ParameterTensors {
            wte: SendPtr::new(null_mut()),
            wpe: SendPtr::new(null_mut()),
            ln1w: SendPtr::new(null_mut()),
            ln1b: SendPtr::new(null_mut()),
            qkvw: SendPtr::new(null_mut()),
            qkvb: SendPtr::new(null_mut()),
            attprojw: SendPtr::new(null_mut()),
            attprojb: SendPtr::new(null_mut()),
            ln2w: SendPtr::new(null_mut()),
            ln2b: SendPtr::new(null_mut()),
            fcw: SendPtr::new(null_mut()),
            fcb: SendPtr::new(null_mut()),
            fcprojw: SendPtr::new(null_mut()),
            fcprojb: SendPtr::new(null_mut()),
            lnfw: SendPtr::new(null_mut()),
            lnfb: SendPtr::new(null_mut()),
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
    ) -> SendPtr<f32> {
        // Calculate the total size needed
        let num_parameters: usize = param_sizes.iter().sum();

        // Allocate memory for all parameters
        let layout = Layout::array::<f32>(num_parameters).expect("Layout error");
        let params_memory: SendPtr<f32> = SendPtr::new(alloc(layout) as *mut f32);

        // Check for successful allocation
        if params_memory.ptr.is_null() {
            panic!("Memory allocation failed");
        }

        // Assign the tensors to the allocated memory
        let mut params_memory_iterator = params_memory;
        let mut ptrs: [*mut SendPtr<f32>; NUM_PARAMETER_TENSORS] = [
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
            params_memory_iterator.ptr = params_memory_iterator.ptr.add(param_sizes[i]);
        }

        params_memory
    }
}
