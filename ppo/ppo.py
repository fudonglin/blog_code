import torch
import torch.nn.functional as F
from typing import Tuple

from torch import Tensor


# ---------------------------------------------------------------------------
# PPO functions
# ---------------------------------------------------------------------------
def get_per_token_logps(logits: Tensor, completions: Tensor) -> Tensor:
    """Gather the log-probability of each generated token under the model.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch, seq_length, vocab_size)
        completions: Token ids that were actually generated. Shape: (batch, seq_length)

    Returns:
        Per-token log-probabilities log π(o_t | q, o_<t). Shape: (batch, seq_length - 1)

    Note:
        A real setup predicts the next token, so you would use logits[:, :-1]
        against completions[:, 1:] (the same shift as in DPO and GRPO).
    """
    assert logits.shape[:-1] == completions.shape

    # Shift by one so logits[:, t] aligns with completions[:, t+1] (next-token prediction).
    completions = completions[:, 1:]   # (batch, seq_length - 1)
    logits = logits[:, :-1, :]         # (batch, seq_length - 1, vocab_size)

    # Pick the log-prob of the realized token at each position. Shape: (batch, seq_length - 1).
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=completions.unsqueeze(2)
    ).squeeze(2)
    return per_token_logps


def compute_kl_divergence(per_token_logps: Tensor, ref_per_token_logps: Tensor) -> Tensor:
    """Ordinary (k1) per-token estimator of KL(π_θ || π_ref).

    This is the plain log-ratio log(π_θ / π_ref). Its expectation over tokens
    sampled from π_θ is exactly the KL.

    Args:
        per_token_logps: Policy log-probs log π_θ(o_t). Shape: (batch, seq_length)
        ref_per_token_logps: Reference log-probs log π_ref(o_t). Shape: (batch, seq_length)

    Returns:
        Per-token KL estimate. Shape: (batch, seq_length)
    """
    # log(π_θ / π_ref); positive when the policy favors the token more than the reference.
    return per_token_logps - ref_per_token_logps


def compute_token_rewards(scores: Tensor,
                          per_token_logps: Tensor,
                          ref_per_token_logps: Tensor,
                          completion_mask: Tensor,
                          kl_beta: float) -> Tensor:
    """Turn a sparse outcome score into a dense per-token reward.

    RLHF-style PPO folds the KL-to-reference term into the reward rather than the
    loss: every token pays a small KL penalty, and the reward-model score is added
    once, at the last real token of the completion.

    Args:
        scores: Scalar outcome reward per completion (from a reward model). Shape: (B,)
        per_token_logps: Policy log-probs log π_θ(o_t). Shape: (B, L)
        ref_per_token_logps: Reference log-probs log π_ref(o_t). Shape: (B, L)
        completion_mask: 1 for real tokens, 0 for padding. Shape: (B, L)
        kl_beta: Weight of the per-token KL-to-reference penalty.

    Returns:
        Per-token rewards r_t. Shape: (B, L)
    """
    # Per-token KL penalty keeps the policy close to the reference. Shape: (B, L).
    kl = compute_kl_divergence(per_token_logps, ref_per_token_logps)
    token_rewards = -kl_beta * kl

    # Index of the last real (unmasked) token in each completion. Shape: (B,).
    last_idx = completion_mask.sum(-1).long() - 1  # (B,)
    rows = torch.arange(token_rewards.size(0))

    # Add the outcome score only at the final token; intermediate tokens get KL alone.
    token_rewards[rows, last_idx] += scores
    return token_rewards * completion_mask


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE).

    Computes advantages using the recursive form:
        A_t = delta_t + (gamma * lam) * A_{t+1}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).

    Args:
        rewards: Tensor of shape [batch, seq_len]. Typically zero
            everywhere except the final step (outcome reward).
        values: Tensor of shape [batch, seq_len]. Value estimates
            V(s_t) from the critic at each step.
        gamma: Discount factor in [0, 1].
        lam: GAE smoothing parameter in [0, 1]. Controls bias-variance
            tradeoff: lam=0 gives one-step TD, lam=1 gives Monte Carlo.

    Returns:
        advantages: Tensor of shape [batch, seq_len].
    """
    gen_length = rewards.shape[1]
    advantages_reversed = []
    A_t = 0.0

    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        A_t = delta + gamma * lam * A_t
        advantages_reversed.append(A_t)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    return advantages


def masked_whiten(advantages: Tensor, completion_mask: Tensor) -> Tensor:
    """Whiten advantages over all real tokens in the batch for stable updates.

    Args:
        advantages: Raw GAE advantages. Shape: (B, L)
        completion_mask: 1 for real tokens, 0 for padding. Shape: (B, L)

    Returns:
        Mean-zero, unit-variance advantages over valid tokens. Shape: (B, L)
    """
    n = completion_mask.sum()
    mean = (advantages * completion_mask).sum() / n
    var = ((advantages - mean) ** 2 * completion_mask).sum() / n
    return (advantages - mean) / (var.sqrt() + 1e-8)


def ppo_loss(per_token_logps: Tensor,
             old_per_token_logps: Tensor,
             advantages: Tensor,
             returns: Tensor,
             values: Tensor,
             old_values: Tensor,
             completion_mask: Tensor,
             eps: float,
             value_eps: float,
             vf_coef: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the PPO loss for a batch of completions.

    Combines the clipped policy surrogate with a clipped value-function loss.

    Args:
        per_token_logps: Current policy log-probs log π_θ(o_t). Shape: (B, L)
        old_per_token_logps: Generation-time policy log-probs log π_θ_old(o_t). Shape: (B, L)
        advantages: Whitened GAE advantages. Shape: (B, L)
        returns: Value targets from GAE. Shape: (B, L)
        values: Current critic estimates V_θ(s_t). Shape: (B, L)
        old_values: Generation-time critic estimates. Shape: (B, L)
        completion_mask: 1 for real tokens, 0 for padding. Shape: (B, L)
        eps: Clipping range for the policy importance ratio (e.g. 0.2 → [0.8, 1.2]).
        value_eps: Clipping range for the value update.
        vf_coef: Weight of the value-function loss.

    Returns:
        A tuple of three tensors: (loss, policy_loss, value_loss).
        loss is the scalar PPO objective to minimize; the other two are for logging.
    """
    # --- Policy loss: PPO clipped surrogate -------------------------------
    # Importance ratio π_θ / π_θ_old, per token. Shape: (B, L).
    policy_ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(policy_ratio, min=1.0 - eps, max=1.0 + eps)

    # Pessimistic (max of the negatives) branch of the surrogate. Shape: (B, L).
    per_token_policy_loss = -torch.min(advantages * policy_ratio, advantages * clipped_ratio)

    # --- Value loss: clipped regression toward the GAE returns -------------
    # Keep the new value within value_eps of the old estimate, then MSE both. Shape: (B, L).
    values_clipped = old_values + torch.clamp(values - old_values, -value_eps, value_eps)
    per_token_value_loss = torch.max((values - returns) ** 2, (values_clipped - returns) ** 2)

    # --- Combine and masked-average over valid tokens, then the batch ------
    def masked_mean(x: Tensor) -> Tensor:
        return ((x * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()

    policy_loss = masked_mean(per_token_policy_loss)
    value_loss = masked_mean(per_token_value_loss)
    loss = policy_loss + vf_coef * value_loss

    return loss, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)

    B = 8            # number of completions in the batch
    L = 10           # completion length (tokens)
    V = 50257        # vocab size (GPT-2 tokenizer)
    kl_beta = 0.1    # weight of the per-token KL-to-reference penalty
    eps = 0.2        # clipping range for the policy ratio → [0.8, 1.2]
    value_eps = 0.2  # clipping range for the value update
    vf_coef = 0.5    # weight of the value-function loss
    gamma = 1.0      # discount factor
    lam = 0.95       # GAE smoothing parameter

    # ------------------------------------------------------------------
    # Step 1: Simulate rollouts. Generate completions and score them.
    # ------------------------------------------------------------------
    banner("Step 1: simulated rollouts and rewards")

    # Token ids that were "generated" for each completion.
    completions = torch.randint(0, V, (B, L))  # (B, L)

    # Outcome score per completion (e.g. from a reward model or verifier).
    scores = torch.randn(B)  # (B,)

    # Variable-length completions: mask out padding past each completion's length.
    lengths = torch.randint(L // 2, L + 1, (B,))                  # (B,)
    positions = torch.arange(L).unsqueeze(0)                      # (1, L)
    completion_mask = (positions < lengths.unsqueeze(1)).float()  # (B, L)

    # The next-token shift drops the first position, so align the mask the same way.
    completion_mask = completion_mask[:, 1:]                      # (B, L - 1)

    print(f"completions     : {tuple(completions.shape)}  # (B, L)")
    print(f"scores          : {tuple(scores.shape)}        # (B,)")
    print(f"tokens per completion : {completion_mask.sum(-1).int().tolist()}")

    # ------------------------------------------------------------------
    # Step 2: Forward passes → per-token log-probabilities.
    # Policy π_θ, generation-time policy π_θ_old, and reference π_ref.
    # ------------------------------------------------------------------
    banner("Step 2: get_per_token_logps → log π(o_t) per token")

    policy_logits = torch.randn(B, L, V)  # π_θ   (being optimized)
    old_logits = torch.randn(B, L, V)     # π_θ_old (frozen snapshot from generation)
    ref_logits = torch.randn(B, L, V)     # π_ref  (frozen reference)

    per_token_logps = get_per_token_logps(policy_logits, completions)        # (B, L-1)
    old_per_token_logps = get_per_token_logps(old_logits, completions)       # (B, L-1)
    ref_per_token_logps = get_per_token_logps(ref_logits, completions)       # (B, L-1)

    for name, t in [
        ("per_token_logps", per_token_logps),
        ("old_per_token_logps", old_per_token_logps),
        ("ref_per_token_logps", ref_per_token_logps),
    ]:
        print(f"{name:22s}: shape={tuple(t.shape)}  mean={t.mean().item():.3f}")

    # ------------------------------------------------------------------
    # Step 3: Critic estimates and dense per-token rewards.
    # The value network is exactly what GRPO removes.
    # ------------------------------------------------------------------
    banner("Step 3: critic values + per-token rewards (score − β·KL)")

    # Value-network output V(s_t) at each token; old_values are the generation-time snapshot.
    values = torch.randn(B, L - 1)      # V_θ(s_t)
    old_values = torch.randn(B, L - 1)  # V_θ_old(s_t)

    token_rewards = compute_token_rewards(
        scores, per_token_logps, ref_per_token_logps, completion_mask, kl_beta
    )
    print(f"values          : shape={tuple(values.shape)}")
    print(f"token_rewards   : shape={tuple(token_rewards.shape)}  "
          f"sum-per-row={[f'{v:.2f}' for v in token_rewards.sum(-1).tolist()]}")

    # ------------------------------------------------------------------
    # Step 4: GAE → advantages and value targets, then whiten advantages.
    # ------------------------------------------------------------------
    banner("Step 4: gae → advantages + returns")

    advantages = gae(token_rewards, values, gamma=gamma, lam=lam)
    returns = advantages + values  # value-network targets
    # Normalized advantages into the policy loss; variance-reduction trick
    advantages = masked_whiten(advantages, completion_mask)
    print(f"advantages      : shape={tuple(advantages.shape)}  "
          f"mean={(advantages * completion_mask).sum().item() / completion_mask.sum().item():.4f}")
    print(f"returns         : shape={tuple(returns.shape)}")

    # ------------------------------------------------------------------
    # Step 5: PPO loss = clipped policy surrogate + clipped value loss.
    # ------------------------------------------------------------------
    banner("Step 5: ppo_loss (clipped policy + clipped value)")

    loss, policy_loss, value_loss = ppo_loss(
        per_token_logps,
        old_per_token_logps,
        advantages,
        returns,
        values,
        old_values,
        completion_mask,
        eps=eps,
        value_eps=value_eps,
        vf_coef=vf_coef,
    )

    print(f"policy_loss     : {policy_loss.item():.4f}")
    print(f"value_loss      : {value_loss.item():.4f}")
    print(f"loss            : {loss.item():.4f}")


if __name__ == "__main__":
    main()
