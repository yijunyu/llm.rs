pub use criterion::{criterion_group, criterion_main, Criterion};
use rand::{distributions::{Distribution, Standard}, Rng};

pub fn generate_random_slice<T>(len: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng = rand::thread_rng();
    let mut slice = Vec::with_capacity(len);

    for _ in 0..len {
        slice.push(Standard.sample(&mut rng));
    }

    slice
}