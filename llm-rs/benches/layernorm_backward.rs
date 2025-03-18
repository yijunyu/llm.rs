#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::layernorm_backward;

pub struct LayernormBackwardInputs {
    pub dinp: usize,
    pub dweight: usize,
    pub dbias: usize,
    pub dout: usize,
    pub inp: usize,
    pub weight: usize,
    pub mean: usize,
    pub rstd: usize,
    pub C: usize,
}

fn benchmark_layernorm_backward(c: &mut Criterion) {
    let inputs = vec![
        LayernormBackwardInputs {
            dinp: 196608,
            dweight: 768,
            dbias: 768,
            dout: 196608,
            inp: 196608,
            weight: 768,
            mean: 256,
            rstd: 256,
            C: 768,
        }
    ];

    for input in inputs {
        let mut dinp = generate_random_slice(input.dinp);
        let mut dweight = generate_random_slice(input.dweight);
        let mut dbias = generate_random_slice(input.dbias);
        let dout = generate_random_slice(input.dout);
        let inp = generate_random_slice(input.inp);
        let weight = generate_random_slice(input.weight);
        let mean = generate_random_slice(input.mean);
        let rstd = generate_random_slice(input.rstd);

        c.bench_function("layernorm_backward", |b| {
            b.iter(|| {
                layernorm_backward(
                    &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, &mean, &rstd, input.C
                );
            });
        });

        c.bench_function("layernorm_backward updated", |b| {
            b.iter(|| {
                layernorm_backward_updated(
                    &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, &mean, &rstd, input.C
                );
            });
        });
    }
}

pub fn layernorm_backward_updated(
    mut dinp: &mut [f32],
    mut dweight: &mut [f32],
    mut dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    C: usize,
) {
    // Placeholder
    layernorm_backward(
        &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, &mean, &rstd, C
    );
}

criterion_group!(benches, benchmark_layernorm_backward);
criterion_main!(benches);