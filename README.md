# Migrating Karpathy's [llm.c](https://github.com/karpathy/llm.c) project into Rust

## Using [c2rust](https://github.com/immunant)

```bash
	make train_gpt2rs
```

Then run `train_gpt2rs` instead of `train_gpt2`. As a result, it took on average 3.626s to 
run a step, which is comparable to the C counterpart (3.658s per step). 

Placing the output code into "result.py" and run

```bash
    python result.py
```
we get the following result.

```
<|endoftext|>Come palm thy back, yet, thou
Colonius: or in good Faith
Of thine faith shall we see thee. Sharif,
O my heart, pardon all your sins and ruin your health together
With the general, and sleep with a letter.
Now, that nobler, yet and
```

Notably, the C output is exactly the same. Well done, C2Rust!
