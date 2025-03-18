#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::residual_forward;

pub struct ResidualForwardInputs {
    out: usize,
    inp1: usize,
    inp2: usize,
}

fn benchmark_residual_forward(c: &mut Criterion) {
    let inputs = vec![
        ResidualForwardInputs {
            out: 196608, 
            inp1: 196608, 
            inp2: 196608,
        }
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let inp1 = generate_random_slice(input.inp1);
        let inp2 = generate_random_slice(input.inp2);

        c.bench_function("residual_forward", |b| {
            b.iter(|| {
                residual_forward(
                    &mut out, &inp1, &inp2
                );
            });
        });

        c.bench_function("residual_forward updated", |b| {
            b.iter(|| {
                residual_forward_updated(
                    &mut out, &inp1, &inp2
                );
            });
        });
    }
}

pub fn residual_forward_updated(
    mut out: &mut [f32], 
    inp1: &[f32], 
    inp2: &[f32],
) {
    // Placeholder
    residual_forward(
        &mut out, &inp1, &inp2
    );
}

criterion_group!(benches, benchmark_residual_forward);
criterion_main!(benches);