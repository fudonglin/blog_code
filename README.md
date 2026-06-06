# Blog Code

A collection of minimal, from-scratch implementations that accompany [my blog posts](https://fudonglin.github.io/). Each package focuses on a single core idea in modern machine learning, with an emphasis on clarity and readable, well-commented code for learning purposes.



## Collections

### [`bpe`](./bpe) — Byte Pair Encoding Tokenizer

A from-scratch implementation of the **Byte Pair Encoding (BPE)** tokenizer from [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) in the style of GPT-2. It operates at the byte level, learns subword merges from a corpus, follows the GPT-2 convention of marking word boundaries with `Ġ`, and guarantees no out-of-vocabulary tokens. This is the tokenization scheme behind most modern LLMs (GPT, LLaMA, Mistral, Qwen).

### [`transformer`](./transformer) — Transformer from Scratch

An unofficial PyTorch implementation of the **Transformer architecture** from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). It builds each component step-by-step — positional encoding, multi-head attention, position-wise feedforward networks, and the encoder/decoder stacks — and includes runnable training and inference examples for a machine translation task.

### [`dpo`](./dpo) — Direct Preference Optimization

A minimal PyTorch implementation of **Direct Preference Optimization (DPO)**, a simple alternative to RLHF for aligning language models with human preferences. It covers the two core pieces — computing per-sequence log-probabilities and the DPO preference loss — and ships with a self-contained demo. Based on [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290).

### [`grpo`](./grpo) — Group Relative Policy Optimization

A minimal PyTorch implementation of **Group Relative Policy Optimization (GRPO)**, a memory-efficient variant of PPO that drops the value network and instead normalizes rewards within a group of sampled completions. It covers the core pieces — per-token log-probabilities, group-relative advantages, the KL-to-reference penalty, and the clipped surrogate loss — and ships with a self-contained demo. Based on [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300).



## License

MIT License. Feel free to use, modify, and share.
