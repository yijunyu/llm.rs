#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::crossentropy_forward;

pub struct CrossentropyForwardInputs {
    pub losses: usize,
    pub probs: usize,
    pub targets: usize,
    pub B: usize,
    pub T: usize,
    pub Vp: usize,
}

fn benchmark_crossentropy_forward(c: &mut Criterion) {
    let inputs = vec![
        CrossentropyForwardInputs {
            losses: 256,
            probs: 12877824,
            targets: 256,
            B: 4,
            T: 64,
            Vp: 50304,
        }
    ];

    for input in inputs {
        let mut losses = generate_random_slice(input.losses);
        let probs = generate_random_slice(input.probs);
        let targets = generate_random_slice(input.targets);

        c.bench_function("crossentropy_forward", |b| {
            b.iter(|| {
                crossentropy_forward(
                    &mut losses, &probs, &targets, input.B, input.T, input.Vp
                );
            });
        });

        c.bench_function("crossentropy_forward updated", |b| {
            b.iter(|| {
                crossentropy_forward_updated(
                    &mut losses, &probs, &targets, input.B, input.T, input.Vp
                );
            });
        });
    }
}

pub fn crossentropy_forward_updated(
    mut losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    Vp: usize,
) {
    // Placeholder
    crossentropy_forward(
        &mut losses, &probs, &targets, B, T, Vp
    );
}

criterion_group!(benches, benchmark_crossentropy_forward);
criterion_main!(benches);