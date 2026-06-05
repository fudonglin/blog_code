import torch
import torch.nn.functional as F
from typing import Tuple

from torch import Tensor


# ---------------------------------------------------------------------------
# GRPO functions
# ---------------------------------------------------------------------------
def get_per_token_logps(logits: Tensor, completions: Tensor) -> Tensor:
    """Gather the log-probability of each generated token under the model.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch, seq_length, vocab_size)
        completions: Token ids that were actually generated. Shape: (batch, seq_length)

    Returns:
        Per-token log-probabilities log π(o_t | q, o_<t). Shape: (batch, seq_length)

    Note:
        For clarity this toy version assumes logits are already aligned with the
        completion tokens. A real setup predicts the next token, so you would use
        logits[:, :-1] against completions[:, 1:] (the same shift as in DPO).
    """
    assert logits.shape[:-1] == completions.shape

    # Pick the log-prob of the realized token at each position. Shape: (batch, seq_length).
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=completions.unsqueeze(2)
    ).squeeze(2)
    return per_token_logps


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


def compute_group_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """Normalize outcome rewards within each prompt's group of completions.

    GRPO drops the value network: the baseline is simply the mean reward of the
    G completions sampled for the same prompt. This is the "Group Relative" part.

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
        loss is the scalar GRPO objective to minimize.
        kl is the per-token KL-to-reference estimate (for logging). Shape: (B * G, L)
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)

    B = 2            # number of prompts
    G = 4            # completions sampled per prompt (group size)
    L = 10           # completion length (tokens)
    V = 50257        # vocab size (GPT-2 tokenizer)
    kl_beta = 0.1    # weight of the KL-to-reference penalty
    eps = 0.2        # clipping range for the importance ratio → [0.8, 1.2]

    BG = B * G       # total completions in the batch

    # ------------------------------------------------------------------
    # Step 1: Simulate rollouts. Sample G completions per prompt and score them.
    # ------------------------------------------------------------------
    banner("Step 1: simulated rollouts and rewards")

    # Token ids that were "generated" for each of the B*G completions.
    completions = torch.randint(0, V, (BG, L))  # (B*G, L)

    # Outcome reward per completion (e.g. from a reward model or verifier).
    rewards = torch.randn(BG)  # (B*G,)

    # Variable-length completions: mask out padding past each completion's length.
    lengths = torch.randint(L // 2, L + 1, (BG,))                 # (B*G,)
    positions = torch.arange(L).unsqueeze(0)                      # (1, L)
    completion_mask = (positions < lengths.unsqueeze(1)).float()  # (B*G, L)

    print(f"completions     : {tuple(completions.shape)}  # (B*G, L)")
    print(f"rewards         : {tuple(rewards.shape)}        # (B*G,)")
    print(f"tokens per completion : {completion_mask.sum(-1).int().tolist()}")

    # ------------------------------------------------------------------
    # Step 2: Forward passes → per-token log-probabilities.
    # Three models: current policy π_θ, generation-time policy π_θ_old, reference π_ref.
    # ------------------------------------------------------------------
    banner("Step 2: get_per_token_logps → log π(o_t) per token")

    policy_logits = torch.randn(BG, L, V)  # π_θ   (being optimized)
    old_logits = torch.randn(BG, L, V)     # π_θ_old (frozen snapshot from generation)
    ref_logits = torch.randn(BG, L, V)     # π_ref  (frozen reference)

    per_token_logps = get_per_token_logps(policy_logits, completions)        # (B*G, L)
    old_per_token_logps = get_per_token_logps(old_logits, completions)       # (B*G, L)
    ref_per_token_logps = get_per_token_logps(ref_logits, completions)       # (B*G, L)

    for name, t in [
        ("per_token_logps", per_token_logps),
        ("old_per_token_logps", old_per_token_logps),
        ("ref_per_token_logps", ref_per_token_logps),
    ]:
        print(f"{name:22s}: shape={tuple(t.shape)}  mean={t.mean().item():.3f}")

    # ------------------------------------------------------------------
    # Step 3: Group-relative advantages. Whiten rewards within each prompt's group.
    # ------------------------------------------------------------------
    banner("Step 3: compute_group_advantages → A_i within each group")

    advantages = compute_group_advantages(rewards, group_size=G)  # (B*G, 1)
    print(f"rewards (grouped)   :\n{rewards.view(B, G)}")
    print(f"advantages          : shape={tuple(advantages.shape)}")
    print(f"advantages (grouped):\n{advantages.view(B, G)}")

    # ------------------------------------------------------------------
    # Step 4: GRPO loss = clipped surrogate − β·KL, masked-averaged over tokens.
    # ------------------------------------------------------------------
    banner("Step 4: grpo_loss (clipped surrogate + KL penalty)")

    loss, kl = grpo_loss(
        per_token_logps,
        old_per_token_logps,
        ref_per_token_logps,
        advantages,
        completion_mask,
        kl_beta=kl_beta,
        eps=eps,
    )

    # Report the masked-mean KL for monitoring.
    mean_kl = (kl * completion_mask).sum() / completion_mask.sum()
    print(f"per-token KL    : shape={tuple(kl.shape)}  masked-mean={mean_kl.item():.4f}")
    print(f"loss            : {loss.item():.4f}")


if __name__ == "__main__":
    main()
