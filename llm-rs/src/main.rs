#![allow(non_snake_case)]

pub mod dataloader;
pub mod gpt2;
pub mod tokenizer;

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use dataloader::DataLoader;
use gpt2::*;
use tokenizer::*;

const BATCH_SIZE: usize = 4;
const SEQ_LENGTH: usize = 64;

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
    ((*state).wrapping_mul(0x2545F4914F6CDD1D) >> 32)
        .try_into()
        .unwrap()
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
    probabilities: &[f32], 
    n: usize, 
    coin: f32
) -> usize {
    let mut cdf = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if coin < cdf {
            return i;
        }
    }
    
    n - 1 // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Main training loop
// ----------------------------------------------------------------------------

pub fn main() {
    let mut lock = io::stdout().lock();

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

    // Build DataLoader instances
    let mut train_loader = DataLoader::new(train_tokens, BATCH_SIZE, SEQ_LENGTH);
    let mut val_loader = DataLoader::new(val_tokens, BATCH_SIZE, SEQ_LENGTH);
    writeln!(lock, "train dataset num_batches: {}", train_loader.num_batches).unwrap();
    writeln!(lock, "val dataset num_batches: {}", val_loader.num_batches).unwrap();

    let val_num_batches = 5;

    // Initialize the Tokenizer
    let tokenizer_path = Path::new("gpt2_tokenizer.bin");
    let mut tokenizer = Tokenizer::new(tokenizer_path);

    // Memory for generating samples
    let mut rng_state: u64 = 1337;
    let mut gen_tokens = vec![50256; BATCH_SIZE * SEQ_LENGTH]; // Initialize with the EOS token (50256)
    let genT = 64;

    // Training loop
    for step in 0..=40 {
        // Estimate validation loss periodically
        if step % 10 == 0 {
            let mut val_loss = 0.0;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                model.forward(&val_loader.inputs, &val_loader.targets, BATCH_SIZE, SEQ_LENGTH);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            writeln!(lock, "val loss {}", val_loss).unwrap();
        }

        // Generate text periodically
        if step > 0 && step % 20 == 0 {
            // Reset the gen_tokens with the EOS token (50256) for each batch
            for i in 0..BATCH_SIZE * SEQ_LENGTH {
                gen_tokens[i] = 50256;
            }

            writeln!(lock, "generating:\n---").unwrap();

            for t in 1..genT {
                model.forward(&gen_tokens, &Vec::new(), BATCH_SIZE, SEQ_LENGTH);
                let probs = model
                    .acts
                    .probs
                    .as_slice()
                    .get((t - 1) * model.config.padded_vocab_size..t * model.config.padded_vocab_size)
                    .unwrap();

                let coin = random_f32(&mut rng_state);
                let next_token = sample_mult(&probs, model.config.vocab_size, coin) as u32;
                gen_tokens[t] = next_token as i32;
                if tokenizer.init_ok {
                    let token_str = tokenizer.decode(next_token);
                    safe_print(token_str);
                } else {
                    write!(lock, "{} ", next_token).unwrap();
                }
                // io::stdout().flush().unwrap();
            }
            writeln!(lock, "\n---").unwrap();
        }

        // Training step
        let start = Instant::now();
        train_loader.next_batch();
        model.forward(&train_loader.inputs, &train_loader.targets, BATCH_SIZE, SEQ_LENGTH);
        model.zero_grad();
        model.backward();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        let duration = start.elapsed();
        writeln!(lock, "step {}: train loss {:.6} (took {:.2} ms)",
            step,
            model.mean_loss,
            duration.as_secs_f64() * 1000.0
        ).unwrap();
    }
}
