use std::alloc::Layout;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::{mem, ptr};

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
    pub batch: *mut i32,

    /// Pointer to input tokens
    pub inputs: *mut i32,

    /// Pointer to target tokens
    pub targets: *mut i32,

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
    pub fn new(filename: &Path, B: usize, T: usize) -> Self {
        let mut loader = DataLoader {
            B,
            T,
            tokens_file: None,
            file_size: 0,
            current_position: 0,
            batch: ptr::null_mut(),
            inputs: ptr::null_mut(),
            targets: ptr::null_mut(),
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

        if loader.file_size < ((B * T + 1) * std::mem::size_of::<i32>()) as u64 {
            panic!("Error: file size is too small for the batch size and sequence length");
        }
        loader.current_position = 0; // Start at the beginning

        // Allocate space for B*T + 1 integers to store the inputs and targets
        unsafe {
            let layout =
                Layout::array::<i32>((B * T + 1) * mem::size_of::<i32>()).expect("Layout error");
            loader.batch = std::alloc::alloc(layout) as *mut i32;
            loader.inputs = loader.batch;
            loader.targets = loader.batch.add(1); // Targets are shifted by one
            loader.num_batches = (loader.file_size as usize) / (B * T * std::mem::size_of::<i32>());
        }

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

        // If we are at the end of the file, loop back to the beginning
        if self.current_position + ((B * T + 1) * std::mem::size_of::<i32>()) as u64
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
            let mut buffer = vec![0; (B * T + 1) * std::mem::size_of::<i32>()];
            tokens_file.read_exact(&mut buffer).expect("Read Failed");

            // copy the buffer into the batch pointer
            unsafe {
                let batch_slice = std::slice::from_raw_parts_mut(self.batch, B * T + 1);
                for (i, chunk) in buffer.chunks_exact(std::mem::size_of::<i32>()).enumerate() {
                    batch_slice[i] = i32::from_ne_bytes(chunk.try_into().unwrap());
                }
            }

            // advance the current position by B*T integers
            self.current_position += (B * T) as u64 * std::mem::size_of::<i32>() as u64;
        } else {
            panic!("File is not open");
        }
    }

    /// Frees the memory allocated by the DataLoader.
    pub fn free(&mut self) {
        if let Some(file) = self.tokens_file.take() {
            drop(file); // Close the file by dropping it
        }
        unsafe {
            if !self.batch.is_null() {
                let layout =
                    std::alloc::Layout::array::<i32>(self.B * self.T + 1).expect("Layout error");
                std::alloc::dealloc(self.batch as *mut u8, layout);
            }
        }
        self.batch = ptr::null_mut();
        self.inputs = ptr::null_mut();
        self.targets = ptr::null_mut();
    }
}
