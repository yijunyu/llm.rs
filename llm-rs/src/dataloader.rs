use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct DataLoader {
    // ----------------------------------------------------------------------------
    // Hyperparameters
    // ----------------------------------------------------------------------------
    /// Batch size
    pub B: usize,

    /// Sequence length
    pub T: usize,

    // ----------------------------------------------------------------------------
    // Input handling and its state
    // ----------------------------------------------------------------------------
    /// File for tokens
    pub tokens_file: Option<File>,

    /// File size
    pub file_size: u64,

    /// Current position in the file
    pub current_position: u64,

    // ----------------------------------------------------------------------------
    // Output memory
    // ----------------------------------------------------------------------------
    /// Pointer to batch memory
    pub batch: Vec<i32>,

    /// Pointer to input tokens
    pub inputs: Vec<i32>,

    /// Pointer to target tokens
    pub targets: Vec<i32>,

    // ----------------------------------------------------------------------------
    // Convenience variables
    // ----------------------------------------------------------------------------
    /// Number of batches
    pub num_batches: usize,
}

impl DataLoader {
    /// Creates a new DataLoader instance.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the tokens file.
    /// * `B` - Batch size.
    /// * `T` - Sequence length.
    ///
    /// # Returns
    ///
    /// A new `DataLoader` instance.
    pub fn new(
        filename: &Path, 
        B: usize, 
        T: usize
    ) -> Self {
        let mut loader = DataLoader {
            B,
            T,
            tokens_file: None,
            file_size: 0,
            current_position: 0,
            batch: Vec::new(),
            inputs: Vec::new(),
            targets: Vec::new(),
            num_batches: 0,
        };

        loader.tokens_file = match File::open(filename) {
            Ok(file) => Some(file),
            Err(_) => {
                panic!("Error opening tokens file");
            }
        };

        // Determine the file size
        if let Some(file) = &mut loader.tokens_file {
            if let Err(_) = file.seek(SeekFrom::End(0)) {
                panic!("Error seeking to end of tokens file");
            }

            loader.file_size = match file.metadata() {
                Ok(metadata) => metadata.len(),
                Err(_) => {
                    panic!("Error getting file size");
                }
            };

            if let Err(_) = file.seek(SeekFrom::Start(0)) {
                panic!("Error seeking to start of tokens file");
            }
        }

        let batch_size = B * T + 1;

        if loader.file_size < (batch_size * std::mem::size_of::<i32>()) as u64 {
            panic!("Error: file size is too small for the batch size and sequence length");
        }
        loader.current_position = 0; // Start at the beginning

        // Allocate space for B*T + 1 integers to store the inputs and targets
        loader.batch = vec![0; batch_size];  // Initialize with `batch_size` elements, filled with 0

        // Populate inputs and targets as slices into the batch vector
        loader.inputs = loader.batch[0..B * T].to_vec(); // First B*T elements are inputs
        loader.targets = loader.batch[1..batch_size].to_vec(); // Targets are shifted by one

        loader.num_batches = (loader.file_size as usize) / (B * T * std::mem::size_of::<i32>());

        loader
    }

    /// Resets the DataLoader to start from the beginning of the file.
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Loads the next batch of data into the DataLoader's memory.
    pub fn next_batch(&mut self) {
        let B = self.B;
        let T = self.T;

        let batch_size = B * T + 1;

        // If we are at the end of the file, loop back to the beginning
        if self.current_position + (batch_size * std::mem::size_of::<i32>()) as u64
            > self.file_size
        {
            self.current_position = 0;
        }

        // Read the B*T+1 integers from the file into batch
        if let Some(tokens_file) = &mut self.tokens_file {
            // seek to the current position in the file
            tokens_file
                .seek(SeekFrom::Start(self.current_position))
                .expect("Seek Failed");

            // read B*T+1 integers from the file into batch
            let mut buffer = vec![0; batch_size * std::mem::size_of::<i32>()];
            tokens_file.read_exact(&mut buffer).expect("Read Failed");

            // Convert the buffer into i32 and populate the batch
            for (i, chunk) in buffer.chunks_exact(std::mem::size_of::<i32>()).enumerate() {
                let value = i32::from_ne_bytes(chunk.try_into().unwrap());
                self.batch[i] = value;  // Copy into the batch vector
            }

            // Now, set inputs and targets to appropriate slices of the batch
            self.inputs = self.batch[0..B * T].to_vec(); // Inputs are the first B*T elements
            self.targets = self.batch[1..batch_size].to_vec(); // Targets are the B*T elements shifted by one

            // Advance the current position by B*T integers
            self.current_position += (B * T) as u64 * std::mem::size_of::<i32>() as u64;
        } else {
            panic!("File is not open");
        }
    }

    /// Frees the memory allocated by the DataLoader.
    pub fn free(&mut self) {
        // Close the file safely
        if let Some(file) = self.tokens_file.take() {
            drop(file); // File will be closed when dropped
        }

        // Clear the batch vector and its contents
        self.batch.clear();
        self.inputs.clear();
        self.targets.clear();
    }
}
