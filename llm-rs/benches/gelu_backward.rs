#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::gelu_backward;

pub struct GeluBackwardInputs {
    pub dinp: usize,
    pub inp: usize,
    pub dout: usize,
    pub N: usize,
}

fn benchmark_gelu_backward(c: &mut Criterion) {
    let inputs = vec![
        GeluBackwardInputs {
            dinp: 786432, 
            inp: 786432,
            dout: 786432, 
            N: 786432,
        }
    ];

    for input in inputs {
        let mut dinp = generate_random_slice(input.dinp);
        let inp = generate_random_slice(input.inp);
        let dout = generate_random_slice(input.dout);

        c.bench_function("gelu_backward", |b| {
            b.iter(|| {
                gelu_backward(
                    &mut dinp, &inp, &dout, input.N
                );
            });
        });

        c.bench_function("gelu_backward updated", |b| {
            b.iter(|| {
                gelu_backward_updated(
                    &mut dinp, &inp, &dout, input.N
                );
            });
        });
    }
}

pub fn gelu_backward_updated(
    mut dinp: &mut [f32], 
    inp: &[f32], 
    dout: &[f32],
    N: usize,
) {
    // Placeholder
    gelu_backward(
        &mut dinp, &inp, &dout, N
    );
}

criterion_group!(benches, benchmark_gelu_backward);
criterion_main!(benches);