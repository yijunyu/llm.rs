pub use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

pub fn generate_random_slice(len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut slice = Vec::with_capacity(len);

    // Fill the slice with random values
    for _ in 0..len {
        slice.push(rng.gen());
    }

    slice
}