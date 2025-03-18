#![allow(non_snake_case)]
mod bench_base;

use bench_base::*;
use llm_rs::gpt2::passes::attention_forward;

pub struct AttentionForwardInputs {
    out: usize,
    preatt: usize,
    att: usize,
    inp: usize,
    T: usize,
    C: usize,
    NH: usize,
}

fn benchmark_attention_forward(c: &mut Criterion) {
    let inputs = vec![
        AttentionForwardInputs {
            out: 196608, 
            preatt: 196608, 
            att: 196608, 
            inp: 589824, 
            T: 64, 
            C: 768, 
            NH: 12, 
        }
    ];

    for input in inputs {
        let mut out = generate_random_slice(input.out);
        let mut preatt = generate_random_slice(input.preatt);
        let mut att = generate_random_slice(input.att);
        let inp = generate_random_slice(input.inp);

        c.bench_function("attention_forward", |b| {
            b.iter(|| {
                attention_forward(
                    &mut out, &mut preatt, &mut att, &inp, input.T, input.C, input.NH
                );
            });
        });

        c.bench_function("attention_forward updated", |b| {
            b.iter(|| {
                attention_forward_updated(
                    &mut out, &mut preatt, &mut att, &inp, input.T, input.C, input.NH
                );
            });
        });
    }
}

pub fn attention_forward_updated(
    mut out: &mut [f32],
    mut preatt: &mut [f32],
    mut att: &mut [f32],
    inp: &[f32],
    T: usize,
    C: usize,
    NH: usize,
) {
    // Placeholder
    attention_forward(
        &mut out, &mut preatt, &mut att, &inp, T, C, NH
    );
}

criterion_group!(benches, benchmark_attention_forward);
criterion_main!(benches);