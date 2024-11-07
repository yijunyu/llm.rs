.PHONY:	setup install preprocess train run

all: train

install:
	pip install -r requirements.txt

preprocess:
	./prepro_tinyshakespeare.py
	./train_gpt2.py

train:	llm-rs/src/main.rs
	cd llm-rs && cargo build --release && cp target/release/llm-rs ../train
	./train

setup:	install preprocess

all:	setup train

clean:
	rm -f train
