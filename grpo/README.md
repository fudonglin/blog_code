# Group Relative Policy Optimization (GRPO)

This repository offers a minimal, clean PyTorch implementation of Group Relative Policy Optimization (GRPO). Before deep dive the code, I highly recommend reading the original paper, [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300). The code is intentionally compact and heavily commented for learning and educational purposes.



## Overview

**GRPO** is a reinforcement learning algorithm for aligning language models, introduced as a memory-efficient variant of PPO. Given a prompt `q`, the policy samples a *group* of `G` completions, each completion is scored with an outcome reward, and the model is updated to favor completions that scored above the group average.

The key insight is that PPO's value network — which estimates the baseline used to compute advantages — can be dropped entirely. Instead, GRPO uses the mean reward of the group as the baseline. This is where the name comes from: advantages are *relative* to the other completions in the same *group*.

Each training step relies on three sets of log-probabilities:

- the policy model $\pi_\theta$ — the model being fine-tuned,
- the old policy $\pi_{\theta_\text{old}}$ — the snapshot that generated the completions, used for the importance ratio, and
- the reference model $\pi_\text{ref}$ — a frozen copy of the starting model that keeps the policy from drifting too far.



## Why GRPO?

Classic PPO is powerful but expensive:

- it trains a separate value network, roughly doubling memory and compute,
- the value network is itself hard to fit for long, sparse-reward sequences,
- tuning the critic adds another source of instability.

GRPO addresses these issues by:

- replacing the learned value baseline with the group's mean reward,
- keeping PPO's clipped surrogate objective for stable updates,
- adding the KL-to-reference term directly to the loss rather than into the reward.

This simplicity, paired with strong results on reasoning tasks, is why GRPO has become a popular choice for RL fine-tuning of modern LLMs.



## The GRPO Loss

For each completion $o_i$ in a group of $G$, GRPO normalizes the outcome reward within the group to form the advantage:

```math
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}.
```

The advantage is shared across all tokens of the completion. The objective is then PPO's clipped surrogate, with a KL penalty pulling the policy toward the reference:

```math
\mathcal{L}_\text{GRPO} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big[ \min\big( \rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) A_i \big) - \beta\, \mathbb{D}_\text{KL}\big[\pi_\theta \,\|\, \pi_\text{ref}\big] \Big].
```

Here, $\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ is the per-token importance ratio, $\epsilon$ is the clipping range (typically `0.2`), and $\beta$ weights the KL penalty (typically around `0.1`).



### Step 1: Per-Token Log-Probabilities

Before computing the loss, we reduce raw model logits of shape `(batch, seq_length, vocab_size)` down to the log-probability of each *realized* completion token, i.e., $\log \pi(o_t \mid q, o_{<t})$. We do this once per model (policy, old policy, reference).

```python
def get_per_token_logps(logits: Tensor, completions: Tensor) -> Tensor:
    """Gather the log-probability of each generated token under the model.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch, seq_length, vocab_size)
        completions: Token ids that were actually generated. Shape: (batch, seq_length)

    Returns:
        Per-token log-probabilities log π(o_t | q, o_<t). Shape: (batch, seq_length)
    """
    assert logits.shape[:-1] == completions.shape

    # Pick the log-prob of the realized token at each position. Shape: (batch, seq_length).
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=completions.unsqueeze(2)
    ).squeeze(2)
    return per_token_logps
```



### Step 2: Group-Relative Advantages

This is the heart of GRPO. Rather than a learned critic, the baseline is the mean reward of the `G` completions sampled for the same prompt. We whiten the rewards within each group, then add a token axis so the advantage broadcasts over every token of its completion.

```python
def compute_group_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """Normalize outcome rewards within each prompt's group of completions.

    Args:
        rewards: Scalar outcome reward per completion. Shape: (B * G,)
        group_size: Number of completions sampled per prompt (G).

    Returns:
        Group-normalized advantages, broadcastable over tokens. Shape: (B * G, 1)
    """
    # Reshape so each row is one prompt's group of G rewards.
    grouped = rewards.view(-1, group_size)  # (B, G)

    # Whiten within the group: subtract the mean, divide by the std.
    mean = grouped.mean(dim=1, keepdim=True)  # (B, 1)
    std = grouped.std(dim=1, keepdim=True)    # (B, 1)
    advantages = (grouped - mean) / (std + 1e-8)  # (B, G)

    # Flatten back to per-completion and add a token axis for broadcasting.
    return advantages.view(-1, 1)  # (B * G, 1)
```



### Step 3: KL Divergence

GRPO regularizes the policy toward the reference model with a per-token KL penalty. It uses the low-variance, unbiased **k3** estimator, which is always non-negative.

```python
def compute_kl_divergence(per_token_logps: Tensor, ref_per_token_logps: Tensor) -> Tensor:
    """Unbiased (k3) per-token estimator of KL(π_θ || π_ref); always ≥ 0.

    Args:
        per_token_logps: Policy log-probs log π_θ(o_t). Shape: (batch, seq_length)
        ref_per_token_logps: Reference log-probs log π_ref(o_t). Shape: (batch, seq_length)

    Returns:
        Per-token KL estimate. Shape: (batch, seq_length)
    """
    # log(π_ref / π_θ); positive when the reference favors the token more than the policy.
    log_ratio = ref_per_token_logps - per_token_logps

    # k3 estimator: exp(r) - r - 1 ≥ 0, low-variance and unbiased for the KL.
    return torch.exp(log_ratio) - log_ratio - 1
```



### Step 4: GRPO Loss

With per-token log-probabilities, advantages, and the KL term in hand, the loss combines PPO's clipped surrogate with the KL penalty, then masked-averages over completion tokens and over the batch.

```python
def grpo_loss(per_token_logps: Tensor,
              old_per_token_logps: Tensor,
              ref_per_token_logps: Tensor,
              advantages: Tensor,
              completion_mask: Tensor,
              kl_beta: float,
              eps: float) -> Tuple[Tensor, Tensor]:
    """Compute the GRPO loss for a batch of completions.

    Args:
        per_token_logps: Current policy log-probs log π_θ(o_t). Shape: (B * G, L)
        old_per_token_logps: Generation-time policy log-probs log π_θ_old(o_t). Shape: (B * G, L)
        ref_per_token_logps: Frozen reference log-probs log π_ref(o_t). Shape: (B * G, L)
        advantages: Group-normalized advantages. Shape: (B * G, 1)
        completion_mask: 1 for real tokens, 0 for padding. Shape: (B * G, L)
        kl_beta: Weight of the KL-to-reference penalty.
        eps: Clipping range for the importance ratio (e.g. 0.2 → [0.8, 1.2]).

    Returns:
        A tuple of two tensors: (loss, kl).
    """
    # PPO-style importance ratio π_θ / π_θ_old, per token. Shape: (B * G, L).
    policy_ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(policy_ratio, min=1.0 - eps, max=1.0 + eps)

    # Clipped surrogate objective; take the pessimistic (min) branch. Shape: (B * G, L).
    surrogate = torch.min(advantages * policy_ratio, advantages * clipped_ratio)

    # Per-token KL penalty that keeps the policy close to the reference. Shape: (B * G, L).
    kl = compute_kl_divergence(per_token_logps, ref_per_token_logps)

    # Maximize surrogate − β·KL ⇒ minimize its negative. Shape: (B * G, L).
    per_token_loss = -(surrogate - kl_beta * kl)

    # Average over valid completion tokens, then over the batch. Scalar.
    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()

    return loss, kl
```

> 🔍 The advantage $A_i$ is the same for every token in a completion — GRPO uses *outcome* rewards, so all tokens share credit for the final score. The `completion_mask` ensures padding tokens contribute neither to the loss nor to the per-sequence token count.



## Quick Start

### Installation

```shell
git clone https://github.com/fudonglin/blog_code.git
cd blog_code/grpo
```



### Run the Demo

The script includes a self-contained demo that simulates the rollouts and forward passes with random logits and rewards, then walks through each step end-to-end:

```shell
python grpo.py
```

The demo:

1. simulates `G` completions per prompt with outcome rewards and a padding mask,
2. reduces the `(B*G, L, V)` logits to per-token log-probabilities for the policy, old policy, and reference via `get_per_token_logps`,
3. whitens the rewards within each group via `compute_group_advantages`, and
4. computes the clipped surrogate plus KL penalty via `grpo_loss`.



### Use in Your Own Code

```python
import torch
from grpo.grpo import get_per_token_logps, compute_group_advantages, grpo_loss

kl_beta = 0.1
eps = 0.2

# logits: (B*G, L, V) from each forward pass
# completions: (B*G, L) token ids that were generated
# rewards: (B*G,) outcome reward per completion
# completion_mask: (B*G, L), 1 for real tokens and 0 for padding
per_token_logps     = get_per_token_logps(policy_logits, completions)
old_per_token_logps = get_per_token_logps(old_logits, completions)
ref_per_token_logps = get_per_token_logps(ref_logits, completions)

advantages = compute_group_advantages(rewards, group_size=G)

loss, kl = grpo_loss(
    per_token_logps,
    old_per_token_logps,
    ref_per_token_logps,
    advantages,
    completion_mask,
    kl_beta=kl_beta,
    eps=eps,
)

loss.backward()
```



## License

MIT License.
Feel free to use, modify, and share.
