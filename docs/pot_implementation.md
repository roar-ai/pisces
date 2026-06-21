# Partial Optimal Transport in PISCES

PISCES contains two complementary OT mechanisms:

- The **distributional Neural OT map** is trained offline on global
  InternVideo2 text and real-video embeddings. It produces the quality reward.
- **Partial Optimal Transport (POT)** is solved online over lexical text tokens
  and visual patch tokens. It produces the structural prior used by the
  semantic reward.

POT is parameter-free. It does not use the weights in
`pretrained/OT_map_156000.pt`; that checkpoint belongs to the distributional
quality reward.

## Source locations

| File | Responsibility |
|---|---|
| `fastvideo/distill.py` | Enables the fine-grained objective and backpropagates its reward through InternVideo2, the VAE decoder, and the denoiser |
| `fastvideo/reward_fn.py` | Loads InternVideo2, selects lexical prompt tokens, and chooses vanilla or OT-aligned reward evaluation |
| `fastvideo/intern_vid2/demo_utils.py` | Computes the global quality reward and invokes the VTM semantic-reward path |
| `fastvideo/intern_vid2/models/criterions.py` | Runs the multimodal encoder and returns the VTM positive-class probability |
| `fastvideo/intern_vid2/models/backbones/bert/xbert.py` | Integration hook that replaces selected cross-attention rows with POT-refined attention |
| `fastvideo/partial_optimal_transport.py` | Video-token geometry, cost construction, Sinkhorn solver, row normalization, and log-space attention fusion |
| `tests/test_partial_optimal_transport.py` | CPU tests for geometry, numerical stability, shapes, batch validation, and gradient flow |

## Runtime path

The canonical script enables:

```bash
--use_ot_reward \
--use_finegrained_reward_loss \
--train_batch_size 1
```

The call path is:

```text
distill_one_step()
  └─ video_reward_fn(video, caption)
     └─ InternVideo2.reward_OT(...)
        └─ VTC_VTM_Loss.vtm_loss(...)
           └─ InternVideo2 multimodal BERT cross-attention
              └─ apply_partial_ot_attention(...)
```

`--use_finegrained_reward_loss` controls whether the POT token path is invoked.
`--use_ot_reward` selects the OT-aware InternVideo2 reward route. A global-only
OT run uses the learned distributional map but skips POT.

## Token and patch selection

`fastvideo/reward_fn.py` tokenizes each caption to at most 300 tokens for POT.
Padding and tokenizer special tokens are excluded, leaving lexical token
indices `valid_tokens`.

For InternVideo2 Stage 2, visual tokens are interpreted as:

```text
[visual CLS] + frame 0 patches + frame 1 patches + ...
```

Each frame contains a 16 by 16 grid, or 256 patch tokens. The visual CLS token
uses frame index `-1` and spatial coordinate `(-1, -1)`. Patch coordinates are
normalized to `[0, 1]`.

## Cost matrix

For each attention head, let:

- `y_i` be projected lexical text token `i`;
- `x_j` be projected visual token `j`;
- `A_ij` be the original cross-attention probability;
- `t_j` and `s_j` be the frame index and 2D coordinate of visual token `j`.

The attention-induced expected frame and position are:

```text
tau_i = sum_j A_ij t_j
pi_i  = sum_j A_ij s_j
```

The cost is:

```text
C_ij = 1 - cosine(y_i, x_j)
       + gamma * |tau_i - t_j|
       + eta * ||pi_i - s_j||_2
```

The released defaults are:

```text
gamma = 0.2
eta = 0.2
```

Temporal and spatial coordinates and the final cost matrix are normalized for
stable mixed-precision execution.

## Partial transport solver

`solve_partial_ot()` uses a GPU-compatible log-domain unbalanced Sinkhorn
approximation with uniform text and visual marginals:

```text
epsilon = 0.05
transported_mass = 0.9
maximum_iterations = 200
convergence_tolerance = 1e-3
```

The transported-mass setting is converted to a marginal-relaxation exponent.
The resulting plan is row-normalized so every selected text token defines a
distribution over visual tokens.

The plan is built under `torch.no_grad()`. This avoids retaining the Sinkhorn
iteration graph and intentionally treats POT as a structural prior.

## Attention fusion and gradients

For selected lexical-token rows, POT is fused with vanilla cross-attention in
log space:

```text
A_tilde = softmax(log(A + 1e-6) + log(P* + 1e-6))
```

Rows belonging to padding, special tokens, or other unselected text positions
are unchanged. The detached plan `P*` contributes structure, while gradients
continue through `A`, the VTM reward, InternVideo2's visual path, the decoded
video, and finally the trainable denoiser.

The refined features are evaluated by InternVideo2's pretrained Video-Text
Matching classifier. The mean positive-class probability is returned as the
fine-grained semantic reward.

## Current constraints

- Token-level POT supports batch size one. This avoids mixing token-to-patch
  plans across independent examples.
- The visual sequence must contain one CLS token followed by complete 256-patch
  frame grids.
- POT is computed per cross-attention head and per InternVideo2 fusion layer,
  so it adds reward-model compute but no inference-time cost to the generated
  video model.
- No POT checkpoint is required. Only the global quality reward loads the
  learned Neural OT map.
