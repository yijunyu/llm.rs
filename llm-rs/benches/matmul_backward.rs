#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::matmul_backward;

pub struct MatmulBackwardInputs {
    pub dinp: usize,
    pub dweight: usize,
    pub dbias: usize,
    pub dout: usize,
    pub inp: usize,
    pub weight: usize,
    pub B: usize,
    pub T: usize,
    pub C: usize,
    pub OC: usize,
}

fn benchmark_matmul_backward(c: &mut Criterion) {
    let inputs = vec![
        MatmulBackwardInputs {
            dinp: 196608,
            dweight: 38633472,
            dbias: 0,
            dout: 12877824,
            inp: 196608,
            weight: 38633472,
            B: 4,
            T: 64,
            C: 768,
            OC: 50304,
        },
        MatmulBackwardInputs {
            dinp: 786432,
            dweight: 2359296,
            dbias: 768,
            dout: 196608,
            inp: 786432,
            weight: 2359296,
            B: 4,
            T: 64,
            C: 3072,
            OC: 768,
        },
        MatmulBackwardInputs {
            dinp: 196608,
            dweight: 2359296,
            dbias: 3072,
            dout: 786432,
            inp: 196608,
            weight: 2359296,
            B: 4,
            T: 64,
            C: 768,
            OC: 3072,
        },
        MatmulBackwardInputs {
            dinp: 196608,
            dweight: 589824,
            dbias: 768,
            dout: 196608,
            inp: 196608,
            weight: 589824,
            B: 4,
            T: 64,
            C: 768,
            OC: 768,
        },
    ];

    for input in inputs {
        let mut dinp = generate_random_slice(input.dinp);
        let mut dweight = generate_random_slice(input.dweight);
        let mut dbias = generate_random_slice(input.dbias);
        let dout = generate_random_slice(input.dout);
        let inp = generate_random_slice(input.inp);
        let weight = generate_random_slice(input.weight);

        c.bench_function("matmul_backward", |b| {
            b.iter(|| {
                matmul_backward(
                    &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, input.B, input.T, input.C, input.OC
                );
            });
        });
        
        c.bench_function("matmul_backward updated", |b| {
            b.iter(|| {
                matmul_backward_updated(
                    &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, input.B, input.T, input.C, input.OC
                );
            });
        });
    }
}

pub fn matmul_backward_updated(
    mut dinp: &mut [f32],
    mut dweight: &mut [f32],
    mut dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // Placeholder
    matmul_backward(
        &mut dinp, &mut dweight, &mut dbias, &dout, &inp, &weight, B, T, C, OC
    )
}

criterion_group!(benches, benchmark_matmul_backward);
criterion_main!(benches);
