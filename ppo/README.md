# Proximal Policy Optimization (PPO)

This repository offers a minimal, clean PyTorch implementation of Proximal Policy Optimization (PPO) for RLHF-style language-model post-training. Before diving into the code, I recommend reading the original paper, [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347), and the RLHF recipe in [Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325). The code is intentionally compact and heavily commented for learning and educational purposes.



## Overview

**PPO** is the reinforcement learning algorithm behind classic RLHF. Given a prompt, the policy generates a completion, a reward model scores it, and the policy is updated to favor higher-scoring completions — all while a clipped objective keeps each update close to the policy that generated the data.

Each training step relies on three sets of log-probabilities plus the critic:

- the policy model $\pi_\theta$ — the model being fine-tuned (the *actor*),
- the old policy $\pi_{\theta_\text{old}}$ — the snapshot that generated the completions, used for the importance ratio,
- the reference model $\pi_\text{ref}$ — a frozen copy of the starting model that keeps the policy from drifting too far, and
- the value model $V_\theta$ — the critic that predicts expected return at each token.



## Why PPO?

PPO became the default for RLHF because it is stable and sample-efficient:

- the clipped surrogate prevents destructively large policy updates,
- the learned value baseline reduces gradient variance, which matters for long, sparse-reward sequences,
- GAE gives a single knob ($\lambda$) to trade bias against variance.

The cost is complexity: PPO needs a critic (roughly doubling memory and compute). 



## The PPO Pipeline

RLHF-style PPO turns a single outcome score into a dense, per-token learning signal, then optimizes a clipped objective.

### Step 1: Per-Token Log-Probabilities

Before computing anything, we reduce raw model logits of shape `(batch, seq_length, vocab_size)` to the log-probability of each *realized* completion token. We do this once per model (policy, old policy, reference).

```python
def get_per_token_logps(logits: Tensor, completions: Tensor) -> Tensor:
    assert logits.shape[:-1] == completions.shape

    # Shift by one so logits[:, t] aligns with completions[:, t+1] (next-token prediction).
    completions = completions[:, 1:]   # (batch, seq_length - 1)
    logits = logits[:, :-1, :]         # (batch, seq_length - 1, vocab_size)

    # Pick the log-prob of the realized token at each position. Shape: (batch, seq_length - 1).
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=completions.unsqueeze(2)
    ).squeeze(2)
    return per_token_logps
```



### Step 2: Per-Token Rewards (score − β·KL)

The reward model emits a single scalar per completion. PPO converts it into a per-token reward by paying a small per-token KL penalty against the reference and adding the outcome score once, at the last real token. PPO uses the ordinary log-ratio (k1) KL estimator, $\log(\pi_\theta / \pi_\text{ref})$.

```python
def compute_token_rewards(scores, per_token_logps, ref_per_token_logps, completion_mask, kl_beta):
    # Per-token KL penalty keeps the policy close to the reference. Shape: (B, L).
    kl = compute_kl_divergence(per_token_logps, ref_per_token_logps)
    token_rewards = -kl_beta * kl

    # Index of the last real (unmasked) token in each completion. Shape: (B,).
    last_idx = completion_mask.sum(-1).long() - 1
    rows = torch.arange(token_rewards.size(0))

    # Add the outcome score only at the final token; intermediate tokens get KL alone.
    token_rewards[rows, last_idx] += scores
    return token_rewards * completion_mask
```



### Step 3: Generalized Advantage Estimation (GAE)

With a learned value baseline $V(s_t)$, GAE computes advantages by walking backward through the sequence:

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad A_t = \delta_t + \gamma \lambda\, A_{t+1}.
```

The value targets (returns) are then $R_t = A_t + V(s_t)$.

```python
def gae(rewards, values, gamma=1.0, lam=1.0):
    gen_length = rewards.shape[1]
    advantages_reversed = []
    A_t = 0.0

    # Walk backwards: A_t = δ_t + γλ·A_{t+1}, with δ_t = r_t + γ·V_{t+1} − V_t.
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        A_t = delta + gamma * lam * A_t
        advantages_reversed.append(A_t)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    return advantages
```

Advantages are then whitened (mean-zero, unit-variance) over all valid tokens in the batch for stable updates, and the value targets are recovered as `returns = advantages + values`.



### Step 4: PPO Loss

The loss combines two clipped terms. The **policy** loss is PPO's clipped surrogate using the per-token importance ratio $\rho_{t} = \frac{\pi_\theta(o_t)}{\pi_{\theta_\text{old}}(o_t)}$. The **value** loss regresses the critic toward the GAE returns, also clipped to limit each update.

```math
\mathcal{L}^{\text{policy}} = -\,\mathbb{E}_t\Big[ \min\big( \rho_t A_t,\ \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) A_t \big) \Big]
```

```math
\mathcal{L}^{\text{value}} = \mathbb{E}_t\Big[ \max\big( (V_t - R_t)^2,\ (V_t^{\text{clip}} - R_t)^2 \big) \Big], \qquad \mathcal{L} = \mathcal{L}^{\text{policy}} + c_v\, \mathcal{L}^{\text{value}}.
```

```python
def ppo_loss(per_token_logps, old_per_token_logps, advantages, returns,
             values, old_values, completion_mask, eps, value_eps, vf_coef):
    # Policy loss: PPO clipped surrogate.
    policy_ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(policy_ratio, min=1.0 - eps, max=1.0 + eps)
    per_token_policy_loss = -torch.min(advantages * policy_ratio, advantages * clipped_ratio)

    # Value loss: clipped regression toward the GAE returns.
    values_clipped = old_values + torch.clamp(values - old_values, -value_eps, value_eps)
    per_token_value_loss = torch.max((values - returns) ** 2, (values_clipped - returns) ** 2)

    def masked_mean(x):
        return ((x * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()

    policy_loss = masked_mean(per_token_policy_loss)
    value_loss = masked_mean(per_token_value_loss)
    loss = policy_loss + vf_coef * value_loss
    return loss, policy_loss, value_loss
```



## Quick Start

### Installation

```shell
git clone https://github.com/fudonglin/blog_code.git
cd blog_code/ppo
```



### Run the Demo

The script includes a self-contained demo that simulates the rollouts, forward passes, and critic outputs with random logits, rewards, and values, then walks through each step end-to-end:

```shell
python ppo.py
```

The demo:

1. simulates completions with outcome scores and a padding mask,
2. reduces the `(B, L, V)` logits to per-token log-probabilities for the policy, old policy, and reference via `get_per_token_logps`,
3. builds dense per-token rewards (`score − β·KL`) via `compute_token_rewards`,
4. computes advantages and value targets via `gae` and whitens the advantages, and
5. computes the clipped policy and value losses via `ppo_loss`.



### Use in Your Own Code

```python
import torch
from ppo.ppo import (
    get_per_token_logps, compute_token_rewards,
    gae, masked_whiten, ppo_loss,
)

kl_beta, eps, value_eps, vf_coef = 0.1, 0.2, 0.2, 0.5
gamma, lam = 1.0, 0.95

# logits: (B, L, V) from each forward pass
# completions: (B, L) token ids that were generated
# scores: (B,) outcome reward per completion
# values / old_values: (B, L-1) critic outputs
# completion_mask: (B, L-1), 1 for real tokens and 0 for padding
per_token_logps     = get_per_token_logps(policy_logits, completions)
old_per_token_logps = get_per_token_logps(old_logits, completions)
ref_per_token_logps = get_per_token_logps(ref_logits, completions)

token_rewards = compute_token_rewards(
    scores, per_token_logps, ref_per_token_logps, completion_mask, kl_beta
)
advantages = gae(token_rewards, values, gamma=gamma, lam=lam)
returns = advantages + values
advantages = masked_whiten(advantages, completion_mask)

loss, policy_loss, value_loss = ppo_loss(
    per_token_logps, old_per_token_logps, advantages, returns,
    values, old_values, completion_mask,
    eps=eps, value_eps=value_eps, vf_coef=vf_coef,
)

loss.backward()
```



## License

MIT License.
Feel free to use, modify, and share.
