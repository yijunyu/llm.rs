#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::softmax_forward;

pub struct SoftmaxForwardInputs {
    pub probs: usize,
    pub logits: usize,
    pub B: usize,
    pub T: usize,
    pub V: usize,
    pub Vp: usize,
}

fn benchmark_softmax_forward(c: &mut Criterion) {
    let inputs = vec![
        SoftmaxForwardInputs {
            probs: 12877824,
            logits: 12877824,
            B: 4,
            T: 64,
            V: 50257,
            Vp: 50304,
        }
    ];

    for input in inputs {
        let mut probs = generate_random_slice(input.probs);
        let logits = generate_random_slice(input.logits);

        c.bench_function("softmax_forward", |b| {
            b.iter(|| {
                softmax_forward(
                    &mut probs, &logits, input.B, input.T, input.V, input.Vp
                );
            });
        });

        c.bench_function("softmax_forward updated", |b| {
            b.iter(|| {
                softmax_forward_updated(
                    &mut probs, &logits, input.V, input.Vp, input.B, input.T
                );
            });
        });
    }
}

pub fn softmax_forward_updated(
    mut probs: &mut [f32],
    logits: &[f32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    // Placeholder
    softmax_forward(
        &mut probs, &logits, B, T, V, Vp
    );
}

criterion_group!(benches, benchmark_softmax_forward);
criterion_main!(benches);