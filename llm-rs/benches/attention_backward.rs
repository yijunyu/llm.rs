#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::attention_backward;

pub struct AttentionBackwardInputs {
    pub dinp: usize,
    pub dpreatt: usize,
    pub datt: usize,
    pub dout: usize,
    pub inp: usize,
    pub att: usize,
    pub B: usize,
    pub T: usize,
    pub C: usize,
    pub NH: usize,
}

fn benchmark_attention_backward(c: &mut Criterion) {
    let inputs = vec![
        AttentionBackwardInputs {
            dinp: 589824,
            dpreatt: 196608,
            datt: 196608,
            dout: 196608,
            inp: 589824,
            att: 196608,
            B: 4,
            T: 64,
            C: 768,
            NH: 12, 
        }
    ];

    for input in inputs {
        let mut dinp = generate_random_slice(input.dinp);
        let mut dpreatt = generate_random_slice(input.dpreatt);
        let mut datt = generate_random_slice(input.datt);
        let dout = generate_random_slice(input.dout);
        let inp = generate_random_slice(input.inp);
        let att = generate_random_slice(input.att);

        c.bench_function("attention_backward", |b| {
            b.iter(|| {
                attention_backward(
                    &mut dinp, &mut dpreatt, &mut datt, &dout, &inp, &att, input.B, input.T, input.C, input.NH
                );
            });
        });

        c.bench_function("attention_backward updated", |b| {
            b.iter(|| {
                attention_backward_updated(
                    &mut dinp, &mut dpreatt, &mut datt, &dout, &inp, &att, input.B, input.T, input.C, input.NH
                );
            });
        });
    }
}

pub fn attention_backward_updated(
    mut dinp: &mut [f32],
    mut dpreatt: &mut [f32],
    mut datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // Placeholder
    attention_backward(
        &mut dinp, &mut dpreatt, &mut datt, &dout, &inp, &att, B, T, C, NH
    );
}

criterion_group!(benches, benchmark_attention_backward);
criterion_main!(benches);