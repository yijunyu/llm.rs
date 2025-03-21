#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::gelu_forward;

pub struct GeluForwardInputs {
    pub out: usize,
    pub inp: usize,
    pub N: usize,
}

fn benchmark_gelu_forward(c: &mut Criterion) {
    let inputs = vec![
        GeluForwardInputs {
            out: 786432, 
            inp: 786432, 
            N: 786432,
        }
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let inp = generate_random_slice(input.inp);

        c.bench_function("gelu_forward", |b| {
            b.iter(|| {
                gelu_forward(
                    &mut out, &inp, input.N
                );
            });
        });

        c.bench_function("gelu_forward updated", |b| {
            b.iter(|| {
                gelu_forward_updated(
                    &mut out, &inp, input.N
                );
            });
        });
    }
}

pub fn gelu_forward_updated(
    mut out: &mut [f32], 
    inp: &[f32], 
    N: usize,
) {
    // Placeholder
    gelu_forward(
        &mut out, &inp, N
    );
}

criterion_group!(benches, benchmark_gelu_forward);
criterion_main!(benches);