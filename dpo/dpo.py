import torch
import torch.nn.functional as F
from typing import Tuple

from torch import Tensor


# ---------------------------------------------------------------------------
# DPO functions
# ---------------------------------------------------------------------------
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
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # Log-ratio (chosen vs rejected) under the policy and reference models.
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # How much more the policy prefers chosen-over-rejected than the reference does.
    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    # DPO loss: -log σ(β · logits). Eq. 7 of the DPO paper. Shape: (B,).
    losses = -F.logsigmoid(beta * logits)

    # Implicit DPO rewards r(x,y) = β · log(π_policy / π_ref). Detached: logging only.
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)

    B = 4            # batch size: 4 prompt-response pairs
    T = 16           # sequence length: prompt + response concatenated
    V = 50257        # vocab size (GPT-2 tokenizer)
    prompt_len = 6   # first 6 tokens are the prompt → masked with -1
    beta = 0.1

    # ------------------------------------------------------------------
    # Step 1: Simulate forward passes through policy and reference models.
    # ------------------------------------------------------------------
    banner("Step 1: simulated model outputs")

    # Logits from 4 forward passes: policy×{chosen,rejected}, ref×{chosen,rejected}
    policy_chosen_logits = torch.randn(B, T, V)      # π_θ on chosen responses
    policy_rejected_logits = torch.randn(B, T, V)    # π_θ on rejected responses
    reference_chosen_logits = torch.randn(B, T, V)   # π_ref on chosen responses
    reference_rejected_logits = torch.randn(B, T, V) # π_ref on rejected responses

    # Labels: full [prompt | response] sequence. Mark prompt positions with -1.
    chosen_labels = torch.randint(0, V, (B, T))      # (B, T)
    rejected_labels = torch.randint(0, V, (B, T))    # (B, T)
    chosen_labels[:, :prompt_len] = -1               # mask prompt
    rejected_labels[:, :prompt_len] = -1             # mask prompt

    print(f"policy_chosen_logits   : {tuple(policy_chosen_logits.shape)}  # (B, T, V)")
    print(f"chosen_labels          : {tuple(chosen_labels.shape)}         # (B, T), -1 = prompt/ignored")
    print(f"#response tokens per row (chosen)  : {(chosen_labels != -1).sum(-1).tolist()}")
    print(f"#response tokens per row (rejected): {(rejected_labels != -1).sum(-1).tolist()}")

    # ------------------------------------------------------------------
    # Step 2: Reduce (B, T, V) logits → (B,) log-probabilities.
    # Each scalar is log π(response | prompt) for one example.
    # ------------------------------------------------------------------
    banner("Step 2: get_batch_logps → log π(y|x) per example")

    policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)        # (B,)
    policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)  # (B,)
    reference_chosen_logps = get_batch_logps(reference_chosen_logits, chosen_labels)  # (B,)
    reference_rejected_logps = get_batch_logps(reference_rejected_logits, rejected_labels)  # (B,)

    for name, t in [
        ("policy_chosen_logps", policy_chosen_logps),
        ("policy_rejected_logps", policy_rejected_logps),
        ("reference_chosen_logps", reference_chosen_logps),
        ("reference_rejected_logps", reference_rejected_logps),
    ]:
        vals = [f"{v:.3f}" for v in t.tolist()]
        print(f"{name:26s}: shape={tuple(t.shape)}  values=[{', '.join(vals)}]")

    # ------------------------------------------------------------------
    # Step 3: Compute DPO loss from the 4 log-prob scalars per example.
    # ------------------------------------------------------------------
    banner("Step 3: preference_loss (DPO)")

    losses, chosen_rewards, rejected_rewards = preference_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
    )
    print(f"losses                 : shape={tuple(losses.shape)}  "
          f"values=[{', '.join(f'{v:.3f}' for v in losses.tolist())}]")
    print(f"chosen_rewards         : shape={tuple(chosen_rewards.shape)}  "
          f"values=[{', '.join(f'{v:.3f}' for v in chosen_rewards.tolist())}]")
    print(f"rejected_rewards       : shape={tuple(rejected_rewards.shape)}  "
          f"values=[{', '.join(f'{v:.3f}' for v in rejected_rewards.tolist())}]")
    print(f"mean loss              : {losses.mean().item():.3f}")


if __name__ == "__main__":
    main()