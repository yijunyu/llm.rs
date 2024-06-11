.PHONY:	setup install preprocess train run

install:
	pip install -r requirements.txt

preprocess:
	python prepro_tinyshakespeare.py

train:	llm-rs/src/main.rs
	cd llm-rs && cargo build --release && cp target/release/llm-rs ../train_gpt2_rs

setup:	install preprocess

all:	setup train_gpt2_rs

clean:
	rm -f train_gpt2_rs
