use criterion::{criterion_group, criterion_main, Criterion};
use llm_rs::dataloader::DataLoader;
use std::path::Path;

fn benchmark_data_loaders(c: &mut Criterion) {
    const BATCH_SIZE: usize = 4;
    const SEQ_LENGTH: usize = 64;

    // Paths to data files
    let tiny_stories_train = Path::new("../data/TinyStories_train.bin");
    let tiny_stories_val = Path::new(".//data/TinyStories_val.bin");
    let tiny_shakespeare_train = Path::new("../data/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val = Path::new("../data/tiny_shakespeare_val.bin");

    // Check if at least one file exists
    let files_exist = tiny_stories_train.exists()
        || tiny_stories_val.exists()
        || tiny_shakespeare_train.exists()
        || tiny_shakespeare_val.exists();

    // Skip benchmarks if no files are available
    if !files_exist {
        eprintln!("No data files found. Skipping benchmarks.");
        return;
    }

    // Select train and validation tokens
    let train_tokens = if tiny_shakespeare_train.exists() {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let val_tokens = if tiny_shakespeare_val.exists() {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };

    // Benchmark initialization of `DataLoader` for training
    c.bench_function("DataLoader::new Train Initialization", |b| {
        b.iter(|| {
            let _train_loader = DataLoader::new(train_tokens, BATCH_SIZE, SEQ_LENGTH);
        });
    });

    // Benchmark initialization of `DataLoader` for validation
    c.bench_function("DataLoader::new Validation Initialization", |b| {
        b.iter(|| {
            let _val_loader = DataLoader::new(val_tokens, BATCH_SIZE, SEQ_LENGTH);
        });
    });

    // Build DataLoader instances for `next_batch` benchmarking
    let mut train_loader = DataLoader::new(train_tokens, BATCH_SIZE, SEQ_LENGTH);
    let mut val_loader = DataLoader::new(val_tokens, BATCH_SIZE, SEQ_LENGTH);

    // Benchmark `next_batch` for training
    c.bench_function("Train DataLoader::next_batch", |b| {
        b.iter(|| {
            train_loader.next_batch();
        });
    });

    // Benchmark `next_batch` for validation
    c.bench_function("Validation DataLoader::next_batch", |b| {
        b.iter(|| {
            val_loader.next_batch();
        });
    });

    // Clean up memory after benchmarks
    train_loader.free();
    val_loader.free();
}

criterion_group!(benches, benchmark_data_loaders);
criterion_main!(benches);
