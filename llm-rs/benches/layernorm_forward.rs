#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::layernorm_forward;

pub struct LayernormForwardInputs {
    pub out: usize,
    pub mean: usize,
    pub rstd: usize,
    pub inp: usize,
    pub weight: usize,
    pub bias: usize,
    pub C: usize,
}

fn benchmark_layernorm_forward(c: &mut Criterion) {
    let inputs = vec![
        LayernormForwardInputs {
            out: 196608,
            mean: 256,
            rstd: 256,
            inp: 196608,
            weight: 768,
            bias: 768,
            C: 768,
        }
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let mut mean = generate_random_slice(input.mean);
        let mut rstd = generate_random_slice(input.rstd);
        let inp = generate_random_slice(input.inp);
        let weight = generate_random_slice(input.weight);
        let bias = generate_random_slice(input.bias);

        c.bench_function("layernorm_forward", |b| {
            b.iter(|| {
                layernorm_forward(
                    &mut out, &mut mean, &mut rstd, &inp, &weight, &bias, input.C
                );
            });
        });

        c.bench_function("layernorm_forward updated", |b| {
            b.iter(|| {
                layernorm_forward_updated(
                    &mut out, &mut mean, &mut rstd, &inp, &weight, &bias, input.C
                );
            });
        });
    }
}

pub fn layernorm_forward_updated(
    mut out: &mut [f32],
    mut mean: &mut [f32],
    mut rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    C: usize,
) {
    // Placeholder
    layernorm_forward(
        &mut out, &mut mean, &mut rstd, &inp, &weight, &bias, C
    );
}

criterion_group!(benches, benchmark_layernorm_forward);
criterion_main!(benches);