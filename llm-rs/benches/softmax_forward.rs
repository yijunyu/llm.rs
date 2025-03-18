#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::softmax_forward;

pub struct SoftmaxForwardInputs {
    probs: usize,
    logits: usize,
    V: usize,
    Vp: usize,
}

fn benchmark_softmax_forward(c: &mut Criterion) {
    let inputs = vec![
        SoftmaxForwardInputs {
            probs: 12877824,
            logits: 12877824,
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
                    &mut probs, &logits, input.V, input.Vp
                );
            });
        });

        c.bench_function("softmax_forward updated", |b| {
            b.iter(|| {
                softmax_forward_updated(
                    &mut probs, &logits, input.V, input.Vp
                );
            });
        });
    }
}

pub fn softmax_forward_updated(
    mut probs: &mut [f32],
    logits: &[f32],
    V: usize,
    Vp: usize,
) {
    // Placeholder
    softmax_forward(
        &mut probs, &logits, V, Vp
    );
}

criterion_group!(benches, benchmark_softmax_forward);
criterion_main!(benches);