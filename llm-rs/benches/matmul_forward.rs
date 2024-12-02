use criterion::{criterion_group, criterion_main, Criterion};
use llm_rs::gpt2::passes::*;
use rand::Rng;
use ndarray::{ArrayView2, ArrayView1, ArrayViewMut2};

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
        let B = input.B;
        let T = input.T;
        let C = input.C;
        let OC = input.OC;
        // Prepare slices into 2D views
        let inp_view = ArrayView2::from_shape((B * T, C), &inp).expect("Input shape mismatch");
        let weight_view = ArrayView2::from_shape((C, OC), &weight).expect("Weight shape mismatch");
        let bias_view = ArrayView1::from(&bias);

        // The mutable output view must be recreated for each iteration, but allocation happens here
        let mut out_view = ArrayViewMut2::from_shape((B * T, OC), &mut out)
            .expect("Output shape mismatch");

        c.bench_function("matmul_forward ndarray", |b| {
            b.iter(|| {
                matmul_forward_ndarray( &mut out_view, &inp_view, &weight_view, &bias_view);
            });
        });
        c.bench_function("matmul_forward", |b| {
            b.iter(|| {
                matmul_forward(
                    &mut out, &inp, &weight, &bias, input.B, input.T, input.C, input.OC,
                );
            });
        });
    }
}

pub fn matmul_forward_ndarray(
    out_view: &mut ArrayViewMut2<f32>,
    inp_view: &ArrayView2<f32>,
    weight_view: &ArrayView2<f32>,
    bias_view: &ArrayView1<f32>,
) {
    // Perform matrix multiplication and assign the result to the mutable view
    out_view.assign(&inp_view.dot(weight_view));

    // Add bias if present
    if !bias_view.is_empty() {
        for mut row in out_view.outer_iter_mut() {
            row += bias_view;
        }
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
