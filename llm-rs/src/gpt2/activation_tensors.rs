pub const NUM_ACTIVATION_TENSORS: usize = 23;

#[derive(Debug, Clone)]
pub struct ActivationTensors {
    /// Encoded (B, T, C)
    pub encoded: Vec<f32>,

    /// Layer normalization 1 (L, B, T, C)
    pub ln1: Vec<f32>,

    /// Layer normalization 1 mean (L, B, T)
    pub ln1_mean: Vec<f32>,

    /// Layer normalization 1 reciprocal std (L, B, T)
    pub ln1_rstd: Vec<f32>,

    /// Query, Key, Value (L, B, T, 3*C)
    pub qkv: Vec<f32>,

    /// Attention output (L, B, T, C)
    pub atty: Vec<f32>,

    /// Pre-attention scores (L, B, NH, T, T)
    pub preatt: Vec<f32>,

    /// Attention scores (L, B, NH, T, T)
    pub att: Vec<f32>,

    /// Attention projection (L, B, T, C)
    pub attproj: Vec<f32>,

    /// Second residual connection (L, B, T, C)
    pub residual2: Vec<f32>,

    /// Layer normalization 2 (L, B, T, C)
    pub ln2: Vec<f32>,

    /// Layer normalization 2 mean (L, B, T)
    pub ln2_mean: Vec<f32>,

    /// Layer normalization 2 reciprocal std (L, B, T)
    pub ln2_rstd: Vec<f32>,

    /// Fully connected hidden (L, B, T, 4*C)
    pub fch: Vec<f32>,

    /// Fully connected hidden GELU activation (L, B, T, 4*C)
    pub fch_gelu: Vec<f32>,

    /// Fully connected projection (L, B, T, C)
    pub fcproj: Vec<f32>,

    /// Third residual connection (L, B, T, C)
    pub residual3: Vec<f32>,

    /// Final layer normalization (B, T, C)
    pub lnf: Vec<f32>,

    /// Final layer normalization mean (B, T)
    pub lnf_mean: Vec<f32>,

    /// Final layer normalization reciprocal std (B, T)
    pub lnf_rstd: Vec<f32>,

    /// Logits (B, T, V)
    pub logits: Vec<f32>,

    /// Probabilities (B, T, V)
    pub probs: Vec<f32>,

    /// Losses (B, T)
    pub losses: Vec<f32>,
}

impl ActivationTensors {
    /// Creates a new ActivationTensors instance.
    ///
    /// # Returns
    ///
    /// A new `ActivationTensors` instance.
    pub fn new() -> Self {
        ActivationTensors {
            encoded: Vec::new(),
            ln1: Vec::new(),
            ln1_mean: Vec::new(),
            ln1_rstd: Vec::new(),
            qkv: Vec::new(),
            atty: Vec::new(),
            preatt: Vec::new(),
            att: Vec::new(),
            attproj: Vec::new(),
            residual2: Vec::new(),
            ln2: Vec::new(),
            ln2_mean: Vec::new(),
            ln2_rstd: Vec::new(),
            fch: Vec::new(),
            fch_gelu: Vec::new(),
            fcproj: Vec::new(),
            residual3: Vec::new(),
            lnf: Vec::new(),
            lnf_mean: Vec::new(),
            lnf_rstd: Vec::new(),
            logits: Vec::new(),
            probs: Vec::new(),
            losses: Vec::new(),
        }
    }

    /// Allocates memory for activation tensors and sets their pointers within a `ActivationTensors` structure.
    ///
    /// # Arguments
    ///
    /// * `act_sizes` - Array of sizes for each activation tensor.
    pub fn alloc_and_point_activations(
        &mut self,
        act_sizes: &[usize; NUM_ACTIVATION_TENSORS],
    ) -> Vec<f32> {
        let mut all_acts = Vec::new();

        // Assign the vectors to their respective sizes
        let mut fields: [&mut Vec<f32>; NUM_ACTIVATION_TENSORS] = [
            &mut self.encoded,
            &mut self.ln1,
            &mut self.ln1_mean,
            &mut self.ln1_rstd,
            &mut self.qkv,
            &mut self.atty,
            &mut self.preatt,
            &mut self.att,
            &mut self.attproj,
            &mut self.residual2,
            &mut self.ln2,
            &mut self.ln2_mean,
            &mut self.ln2_rstd,
            &mut self.fch,
            &mut self.fch_gelu,
            &mut self.fcproj,
            &mut self.residual3,
            &mut self.lnf,
            &mut self.lnf_mean,
            &mut self.lnf_rstd,
            &mut self.logits,
            &mut self.probs,
            &mut self.losses,
        ];

        // Assign slices of the allocated memory to each tensor
        for (i, field) in fields.iter_mut().enumerate() {
            // Resize each vector based on the corresponding size in `param_sizes`.
            field.resize(act_sizes[i], 0.0);

            all_acts.extend_from_slice(&field);
        }

        all_acts
    }
}
