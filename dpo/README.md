# Direct Preference Optimization (DPO) 

This repository offers a minimal, clean PyTorch implementation of Direct Preference Optimization (DPO). Before deep dive the code, I highly recommend reading the original paper, [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290). The code is intentionally compact and heavily commented for learning and educational purposes.



## Overview

**DPO** is a simple alternative to RLHF (Reinforcement Learning from Human Feedback) for aligning language models with human preferences. Given a dataset of preference pairs — a prompt `x`, a chosen (preferred) response `y_w`, and a rejected response `y_l` — DPO fine-tunes the model to increase the likelihood of chosen responses relative to rejected ones.

The key insight is that the RLHF objective has a closed-form optimal solution, which lets us skip training a separate reward model and skip RL altogether. Instead, the policy itself is optimized directly with a simple classification-style loss.

Each training step relies on two models:

- the policy model $\pi_\theta$ — the model being fine-tuned, and
- the reference model $\pi_\text{ref}$ — a frozen copy of the starting model that keeps the policy from drifting too far.



## Why DPO?

Classic RLHF is powerful but cumbersome:

- it requires training a separate reward model,
- it relies on on-policy RL (e.g., PPO), which is unstable and sensitive to hyperparameters,
- it needs to sample from the policy during training, which is slow.

DPO addresses these issues by:

- using preference pairs directly, with no explicit reward model,
- replacing RL with a single supervised loss,
- requiring only four forward passes (policy and reference, on chosen and rejected) per batch.

This simplicity is why DPO and its variants have become a popular choice for aligning modern LLMs.



## The DPO Loss

DPO defines an implicit reward in terms of the log-ratio between the policy and the reference model:

```math
r(x, y) \varpropto  \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}.
```

The training objective is then to make the chosen response score higher than the rejected one, using a logistic (Bradley–Terry) loss:

```math
\mathcal{L}_\text{DPO} = -\log \sigma \Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \Big).
```

Here, $y_w$ and $y_l$ are the chosen and rejected responses, $\sigma(\cdot)$ is the sigmoid function, and $\beta$ is a temperature that controls how strongly the policy is pushed away from the reference (typically in the range `0.1` to `0.5`).



### Step 1: Sequence Log-Probabilities

Before computing the loss, we reduce raw model logits of shape `(batch_size, seq_length, vocab_size)` down to a single log-probability per example, i.e., $\log \pi(y \mid x)$. Prompt and padding tokens are marked with `-1` and ignored, so only the response tokens are scored.

```python
def get_batch_logps(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, seq_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -1 are ignored. Shape: (batch_size, seq_length)

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """

    assert logits.shape[:-1] == labels.shape

    # Shift by one so logits[:, i] aligns with labels[:, i] (next-token prediction).
    labels = labels[:, 1:].clone()  # (B, T-1)
    logits = logits[:, :-1, :]  # (B, T-1, V)

    # Positions to score; -1 marks prompt/pad tokens to ignore.
    loss_mask = (labels != -1)

    # Replace -1 with a dummy id so gather() gets valid indices; masked out later.
    labels[labels == -1] = 0

    # Pick the log-prob of the ground-truth token at each position. Shape: (B, T-1).
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # Sum log-probs over scored tokens = log P(answer | prompt). Shape: (B,).
    return (per_token_logps * loss_mask).sum(-1)
```



### Step 2: Preference Loss

With the four log-probabilities in hand (policy and reference, on chosen and rejected), the DPO loss is a few lines. The function also returns the implicit rewards for the chosen and rejected responses, which are detached and used only for logging.

```python
def preference_loss(policy_chosen_logps: Tensor,
                    policy_rejected_logps: Tensor,
                    reference_chosen_logps: Tensor,
                    reference_rejected_logps: Tensor,
                    beta: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
    """
    # Log-ratio (chosen vs rejected) under the policy and reference models.
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # How much more the policy prefers chosen-over-rejected than the reference does.
    logits = pi_logratios - ref_logratios

    # DPO loss: -log σ(β · logits). Eq. 7 of the DPO paper. Shape: (B,).
    losses = -F.logsigmoid(beta * logits)

    # Implicit DPO rewards r(x,y) = β · log(π_policy / π_ref). Detached: logging only.
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
```

> 🔍 The `losses`, `chosen_rewards`, and `rejected_rewards` returned here are per-example tensors of shape `(B,)`. During training you typically optimize `losses.mean()`, while the rewards are handy for monitoring the gap between chosen and rejected responses.



## Quick Start

### Installation

```shell
git clone https://github.com/fudonglin/blog_code.git
cd blog_code/dpo
```



### Run the Demo

The script includes a self-contained demo that simulates the four forward passes with random logits, then walks through each step end-to-end:

```shell
python dpo.py
```

The demo:

1. simulates policy and reference outputs for chosen/rejected responses, marking prompt tokens with `-1`,
2. reduces the `(B, T, V)` logits to per-example log-probabilities via `get_batch_logps`, and
3. computes the DPO loss and implicit rewards via `preference_loss`.



### Use in Your Own Code

```python
import torch
from dpo.dpo import get_batch_logps, preference_loss

beta = 0.1

# logits: (batch_size, seq_length, vocab_size) from each forward pass
# labels: (batch_size, seq_length), with prompt/pad tokens set to -1
policy_chosen_logps     = get_batch_logps(policy_chosen_logits, chosen_labels)
policy_rejected_logps   = get_batch_logps(policy_rejected_logits, rejected_labels)
reference_chosen_logps  = get_batch_logps(reference_chosen_logits, chosen_labels)
reference_rejected_logps = get_batch_logps(reference_rejected_logits, rejected_labels)

losses, chosen_rewards, rejected_rewards = preference_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=beta,
)

loss = losses.mean()
loss.backward()
```



## License

MIT License.
Feel free to use, modify, and share.
