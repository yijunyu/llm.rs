pub const NUM_PARAMETER_TENSORS: usize = 16;

#[derive(Debug, Clone)]
pub struct ParameterTensors {
    /// Token embeddings (V, C).
    pub wte: Vec<f32>,

    /// Position embeddings (maxT, C).
    pub wpe: Vec<f32>,

    /// Layer normalization weights for the first layer (L, C).
    pub ln1w: Vec<f32>,

    /// Layer normalization biases for the first layer (L, C).
    pub ln1b: Vec<f32>,

    /// Query, Key, Value weights (L, 3*C, C).
    pub qkvw: Vec<f32>,

    /// Query, Key, Value biases (L, 3*C).
    pub qkvb: Vec<f32>,

    /// Attention projection weights (L, C, C).
    pub attprojw: Vec<f32>,

    /// Attention projection biases (L, C).
    pub attprojb: Vec<f32>,

    /// Layer normalization weights for the second layer (L, C).
    pub ln2w: Vec<f32>,

    /// Layer normalization biases for the second layer (L, C).
    pub ln2b: Vec<f32>,

    /// Fully connected weights (L, 4*C, C).
    pub fcw: Vec<f32>,

    /// Fully connected biases (L, 4*C).
    pub fcb: Vec<f32>,

    /// Fully connected projection weights (L, C, 4*C).
    pub fcprojw: Vec<f32>,

    /// Fully connected projection biases (L, C).
    pub fcprojb: Vec<f32>,

    /// Final layer normalization weights (C).
    pub lnfw: Vec<f32>,

    /// Final layer normalization biases (C).
    pub lnfb: Vec<f32>,
}

impl ParameterTensors {
    /// Creates a new ParameterTensors instance.
    ///
    /// # Returns
    ///
    /// A new `ParameterTensors` instance.
    pub fn new() -> Self {
        ParameterTensors {
            wte: Vec::new(),
            wpe: Vec::new(),
            ln1w: Vec::new(),
            ln1b: Vec::new(),
            qkvw: Vec::new(),
            qkvb: Vec::new(),
            attprojw: Vec::new(),
            attprojb: Vec::new(),
            ln2w: Vec::new(),
            ln2b: Vec::new(),
            fcw: Vec::new(),
            fcb: Vec::new(),
            fcprojw: Vec::new(),
            fcprojb: Vec::new(),
            lnfw: Vec::new(),
            lnfb: Vec::new(),
        }
    }

    // Allocates memory for model parameters and sets their pointers within a `ParameterTensors` structure.
    ///
    /// # Arguments
    ///
    /// * `param_sizes` - Array of sizes for each parameter tensor.
    pub fn alloc_and_point_parameters(
        &mut self,
        param_sizes: &[usize; NUM_PARAMETER_TENSORS],
    ) -> Vec<f32> {
        let mut all_params = Vec::new();

        // Assign the vectors to their respective sizes
        let mut fields: [&mut Vec<f32>; NUM_PARAMETER_TENSORS] = [
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

        for (i, field) in fields.iter_mut().enumerate() {
            // Resize each vector based on the corresponding size in `param_sizes`.
            field.resize(param_sizes[i], 0.0);

            all_params.extend_from_slice(&field);
        }

        all_params
    }
}
