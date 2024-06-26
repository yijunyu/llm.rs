use std::path::Path;
use std::fs::File;
use std::io::Read;

pub struct Tokenizer {
    vocab_size: u32,
    token_table: Vec<String>,
    pub init_ok: bool,
}

impl Tokenizer {
    /// Creates a new Tokenizer instance from a file.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the tokenizer file.
    ///
    /// # Returns
    ///
    /// A new `Tokenizer` instance.
    pub fn new(filename: &Path) -> Self {
        let mut tokenizer = Tokenizer {
            vocab_size: 0,
            token_table: Vec::new(),
            init_ok: false,
        };

        let mut file = match File::open(filename) {
            Ok(file) => file,
            Err(_) => {
                panic!(
                    "---\n
                    WARNING: Failed to open the tokenizer file\n
                    The Tokenizer is a new feature added April 14 2024.\n
                    Re-run `python train_gpt2.py` to write it\n
                    ---"
                );
            }
        };

        let mut header = [0; 256];
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                header.as_mut_ptr() as *mut u8,
                header.len() * std::mem::size_of::<u32>(),
            )
        }).expect("Failed to read header");

        // Check magic number and version
        if header[0] != 20240328 {
            panic!("Bad magic tokenizer file");
        }
        if header[1] != 2 {
            panic!("Bad version in tokenizer file")
        }

        tokenizer.vocab_size = header[2];

        for _ in 0..tokenizer.vocab_size {
            let mut length = [0];
            file.read_exact(&mut length).expect("Failed to read token length");

            assert!(length[0] > 0); // Every token should be at least one character
            let mut token_bytes = vec![0u8; length[0] as usize];
            file.read_exact(&mut token_bytes).expect("Failed to read token bytes");
            let token = match String::from_utf8(token_bytes) {
                Ok(token) => token,
                Err(_) => String::new(),
            };
            
            tokenizer.token_table.push(token);
        }

        tokenizer.init_ok = true;

        tokenizer
    }

    /// Decodes a token ID into its corresponding string.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID to decode.
    ///
    /// # Returns
    ///
    /// The corresponding string for the token ID.
    pub fn decode(
        &mut self,
        token_id: u32,
    ) -> &str{
        if !self.init_ok {
            ""
        } else if token_id < self.vocab_size {
            &self.token_table[token_id as usize]
        } else {
            println!("invalid token id {}!", token_id);
            ""
        }
    }

    /// Frees the resources allocated by the Tokenizer.
    pub fn free(&mut self) {
        if self.init_ok != false {
            self.token_table.clear();
            self.init_ok = false;
        }
    }
}

/// Safely prints a string if it contains valid ASCII characters.
///
/// # Arguments
///
/// * `piece` - The string slice to print.
pub unsafe fn safe_print(piece: &str) {
    if piece.is_empty() {
        return;
    }

    let bytes = piece.as_bytes();
    if bytes.len() == 1 {
        let byte_val = bytes[0];
        if !(byte_val.is_ascii_graphic() || byte_val.is_ascii_whitespace()) {
            return; // weird byte, don't print it
        }
    }

    // Print the string if it is valid
    print!("{}", piece);
}