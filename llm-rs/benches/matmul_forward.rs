use criterion::{criterion_group, criterion_main, Criterion};
use llm_rs::gpt2::passes::*;
use rand::Rng;

fn benchmark_matmul_forward(c: &mut Criterion) {
    let inputs = vec![
        MatMulInputs {
            out: 196608,
            inp: 196608,
            weight: 589824,
            bias: 768,
            B: 4,
            T: 64,
            C: 768,
            OC: 768,
        },
        MatMulInputs {
            out: 786432,
            inp: 196608,
            weight: 2359296,
            bias: 3072,
            B: 4,
            T: 64,
            C: 768,
            OC: 3072,
        },
        MatMulInputs {
            out: 196608,
            inp: 786432,
            weight: 2359296,
            bias: 768,
            B: 4,
            T: 64,
            C: 3072,
            OC: 768,
        },
        MatMulInputs {
            out: 12877824,
            inp: 196608,
            weight: 38633472,
            bias: 0,
            B: 4,
            T: 64,
            C: 768,
            OC: 50304,
        },
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let inp = generate_random_slice(input.inp);
        let weight = generate_random_slice(input.weight);
        let bias = generate_random_slice(input.bias);

        c.bench_function("matmul_forward", |b| {
            b.iter(|| {
                matmul_forward(
                    &mut out, &inp, &weight, &bias, input.B, input.T, input.C, input.OC,
                );
            });
        });
    }
}

criterion_group!(benches, benchmark_matmul_forward);
criterion_main!(benches);

fn generate_random_slice(len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut slice = Vec::with_capacity(len);

    // Fill the slice with random values
    for _ in 0..len {
        slice.push(rng.gen());
    }

    slice
}

pub struct MatMulInputs {
    pub out: usize,
    pub inp: usize,
    pub weight: usize,
    pub bias: usize,
    pub B: usize,
    pub T: usize,
    pub C: usize,
    pub OC: usize,
}
