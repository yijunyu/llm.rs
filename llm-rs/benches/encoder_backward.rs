#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::encoder_backward;

pub struct EncoderBackwardInputs {
    pub dwte: usize,
    pub dwpe: usize,
    pub dout: usize,
    pub inp: usize,
    pub B: usize,
    pub T: usize,
    pub C: usize,
}

fn benchmark_encoder_backward(c: &mut Criterion) {
    let inputs = vec![
        EncoderBackwardInputs {
            dwte: 38633472,
            dwpe: 786432,
            dout: 196608,
            inp: 256,
            B: 4,
            T: 64,
            C: 768,
        }
    ];

    for input in inputs {
        let mut dwte = generate_random_slice(input.dwte);
        let mut dwpe = generate_random_slice(input.dwpe);
        let dout = generate_random_slice(input.dout);
        let inp = generate_random_slice(input.inp);

        c.bench_function("encoder_backward", |b| {
            b.iter(|| {
                encoder_backward(
                    &mut dwte, &mut dwpe, &dout, &inp, input.B, input.T, input.C
                );
            });
        });

        c.bench_function("encoder_backward updated", |b| {
            b.iter(|| {
                encoder_backward_updated(
                    &mut dwte, &mut dwpe, &dout, &inp, input.B, input.T, input.C
                );
            });
        });
    }
}

pub fn encoder_backward_updated(
    mut dwte: &mut [f32],
    mut dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    // Placeholder
    encoder_backward(
        &mut dwte, &mut dwpe, &dout, &inp, B, T, C
    );
}

criterion_group!(benches, benchmark_encoder_backward);
criterion_main!(benches);