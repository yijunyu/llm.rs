# Migrating Karpathy's [llm.c](https://github.com/karpathy/llm.c) project into Rust

## Using [c2rust](https://github.com/immunant)

```bash
	make train_gpt2rs
```

Then run `train_gpt2rs` instead of `train_gpt2`. As a result, it took on average 42.798s to 
run a step, which is comparable to the C counterpart (9.585s per step w/o OpenMP and 5.543s per step with 8 threads of OpenMP).  

The reason for that is OpenMP has not been used in the Rust version. That's why we will need a parallel version of this
generated Rust code.

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
## Using [GPT4](https://chat.openai.com)

Although the transpilation of c2rust was successful, all the for loops have been turned into while loops.

Using GPT-4, we are able to convert all the while loops back into for loops.

## Using [Mate](https://github.com/trusted-programming/mate) and Rayon library

Furthermore, we use Mate to convert some of these for loops into iter() functions using the Rayon library.
In this way, the resulting code has 39.501s per step, or 10% better performance than the vanilla for loops. 

## Post Code Restructuring
15.332s on average per step