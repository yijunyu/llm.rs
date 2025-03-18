#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::encoder_forward;

pub struct EncoderForwardInputs {
    pub out: usize,
    pub inp: usize,
    pub wte: usize,
    pub wpe: usize,
    pub T: usize,
    pub C: usize,
}

fn benchmark_encoder_forward(c: &mut Criterion) {
    let inputs = vec![
        EncoderForwardInputs {
            out: 196608,
            inp: 256,
            wte: 38633472,
            wpe: 786432,
            T: 64,
            C: 768,
        }
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let inp = generate_random_slice(input.inp);
        let wte = generate_random_slice(input.wte);
        let wpe = generate_random_slice(input.wpe);

        c.bench_function("encoder_forward", |b| {
            b.iter(|| {
                encoder_forward(
                    &mut out, &inp, &wte, &wpe, input.T, input.C
                );
            });
        });

        c.bench_function("encoder_forward updated", |b| {
            b.iter(|| {
                encoder_forward_updated(
                    &mut out, &inp, &wte, &wpe, input.T, input.C
                );
            });
        });
    }
}

pub fn encoder_forward_updated(
    mut out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    T: usize,
    C: usize,
) {
    // Placeholder
    encoder_forward(
        &mut out, &inp, &wte, &wpe, T, C
    );
}

criterion_group!(benches, benchmark_encoder_forward);
criterion_main!(benches);