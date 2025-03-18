#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::residual_backward;

pub struct ResidualBackwardInputs {
    dinp1: usize,
    dinp2: usize,
    dout: usize,
}

fn benchmark_residual_backward(c: &mut Criterion) {
    let inputs = vec![
        ResidualBackwardInputs {
            dinp1: 196608, 
            dinp2: 196608, 
            dout: 196608,
        }
    ];

    for input in inputs {
        let mut dinp1 = generate_random_slice(input.dinp1);
        let mut dinp2 = generate_random_slice(input.dinp2);
        let dout = generate_random_slice(input.dout);

        c.bench_function("residual_backward", |b| {
            b.iter(|| {
                residual_backward(
                    &mut dinp1, &mut dinp2, &dout
                );
            });
        });

        c.bench_function("residual_backward updated", |b| {
            b.iter(|| {
                residual_backward_updated(
                    &mut dinp1, &mut dinp2, &dout
                );
            });
        });
    }
}

pub fn residual_backward_updated(
    mut dinp1: &mut [f32], 
    mut dinp2: &mut [f32], 
    dout: &[f32],
) {
    // Placeholder
    residual_backward(
        &mut dinp1, &mut dinp2, &dout
    );
}

criterion_group!(benches, benchmark_residual_backward);
criterion_main!(benches);