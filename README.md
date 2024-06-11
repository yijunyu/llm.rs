# llm.rs

Migration of Karpathy's [llm.c](https://github.com/karpathy/llm.c) project into Rust

## Development Process

The development steps taken to migrate llm.c into Rust

### 1. Utilizing [c2rust](https://github.com/immunant)

Using c2rust, train_gpt2.c was translated from Karpathy's [llm.c](https://github.com/karpathy/llm.c) project to Rust.

### 2. Utilizing [GPT4](https://chat.openai.com)

Although the transpilation of c2rust was successful, all the for loops have been turned into while loops.

Using GPT-4, we are able to convert all the while loops back into for loops.

### 3. Utilizing [Mate](https://github.com/trusted-programming/mate)

Furthermore, using Mate, we converted some of these for loops into iter() functions using the Rayon library.

### 4. Manual Updates

Currently, the project is undergoing manual updates to find performance improvements

## Performance

The system that all the testing is done on is an Intel Core i7-9700 8-core CPU. Currently this implementation is still slower than the C version on this system which can be seen from the breakdown below:
- **Rust**: **6.038s** on average per step
- **C**: **2.447s** on average per step

## Quick Start

Install python dependencies, output tokenized dataset:

```bash
make setup
```

Run the training script:

```bash
make train
```

This will run `cargo build --release` from the llm-rs cargo project after which the binary will be copied into the main project folder.

## TODO

- [ ] Migrate the testing script
- [ ] Fix tinystories dataset download
- [ ] Implement SIMD for improved speed
