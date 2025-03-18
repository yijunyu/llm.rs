#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::crossentropy_softmax_backward;

pub struct CrossentropySoftmaxBacwardInputs {
    dlogits: usize,
    dlosses: usize,
    probs: usize,
    targets: usize,
    V: usize,
    Vp: usize,
}

fn benchmark_crossentropy_softmax_backward(c: &mut Criterion) {
    let inputs = vec![
        CrossentropySoftmaxBacwardInputs {
            dlogits: 12877824,
            dlosses: 256,
            probs: 12877824,
            targets: 256,
            V: 50257,
            Vp: 50304,
        }
    ];

    for input in inputs {
        let mut dlogits = generate_random_slice(input.dlogits);
        let dlosses = generate_random_slice(input.dlosses);
        let probs = generate_random_slice(input.probs);
        let targets = generate_random_slice(input.targets);

        c.bench_function("crossentropy_softmax_backward", |b| {
            b.iter(|| {
                crossentropy_softmax_backward(
                    &mut dlogits, &dlosses, &probs, &targets, input.V, input.Vp
                );
            });
        });

        c.bench_function("crossentropy_softmax_backward updated", |b| {
            b.iter(|| {
                crossentropy_softmax_backward_updated(
                    &mut dlogits, &dlosses, &probs, &targets, input.V, input.Vp
                );
            });
        });
    }
}

pub fn crossentropy_softmax_backward_updated(
    mut dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    V: usize,
    Vp: usize,
) {
    // Placeholder
    crossentropy_softmax_backward(
        &mut dlogits, &dlosses, &probs, &targets, V, Vp
    );
}

criterion_group!(benches, benchmark_crossentropy_softmax_backward);
criterion_main!(benches);