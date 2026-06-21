from types import SimpleNamespace

import pytest
import torch

from fastvideo.intern_vid2.models.backbones.bert.xbert import BertSelfAttention
from fastvideo.partial_optimal_transport import (
    build_partial_ot_prior,
    build_video_token_geometry,
    compute_spatiotemporal_cost,
    fuse_attention_with_partial_ot,
    normalize_transport_rows,
    solve_partial_ot,
)
from fastvideo.reward_fn import _get_lexical_token_indices


def test_video_token_geometry():
    frames, coordinates = build_video_token_geometry(
        9, patches_per_frame=4
    )
    assert frames.tolist() == [-1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert coordinates.shape == (9, 2)
    assert torch.equal(coordinates[0], torch.tensor([-1.0, -1.0]))
    assert coordinates[1:].min() == 0
    assert coordinates[1:].max() == 1


def test_lexical_token_selection_excludes_padding_and_special_tokens():
    tokenized_text = {
        "input_ids": torch.tensor([[101, 10, 11, 102, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }
    tokenizer = SimpleNamespace(
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
    )
    indices = _get_lexical_token_indices(tokenized_text, tokenizer)
    assert indices.tolist() == [1, 2]


def test_internvideo_single_sequence_tokenizer_metadata_has_matching_length(
    tmp_path,
):
    from fastvideo.intern_vid2.models.backbones.bert.tokenization_bert import (
        BertTokenizer,
    )

    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntoken\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab_path))

    token_ids = [tokenizer.convert_tokens_to_ids("token")] * 299
    model_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    special_mask = tokenizer.get_special_tokens_mask(token_ids)
    token_types = tokenizer.create_token_type_ids_from_sequences(token_ids)

    assert len(model_ids) == 300
    assert len(special_mask) == len(model_ids)
    assert len(token_types) == len(model_ids)


def test_spatiotemporal_cost_and_sinkhorn_are_finite():
    text = torch.randn(3, 4)
    video = torch.randn(5, 4)
    attention = torch.softmax(torch.randn(3, 5), dim=-1)
    frames, coordinates = build_video_token_geometry(
        5, patches_per_frame=4
    )

    cost = compute_spatiotemporal_cost(
        text, video, attention, frames, coordinates
    )
    plan = solve_partial_ot(cost)
    normalized_plan = normalize_transport_rows(plan)

    assert cost.shape == (3, 5)
    assert torch.isfinite(cost).all()
    assert cost.min() >= 0
    assert cost.max() <= 1
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()
    assert torch.allclose(
        normalized_plan.sum(dim=-1), torch.ones(3), atol=1e-5
    )


def test_build_prior_and_log_space_fusion_preserve_shapes_and_gradients():
    logits = torch.randn(1, 2, 4, 5, requires_grad=True)
    attention = torch.softmax(logits, dim=-1)
    queries = torch.randn(1, 4, 4)
    keys = torch.randn(1, 2, 5, 2)
    valid_tokens = torch.tensor([1, 2])

    prior = build_partial_ot_prior(
        queries,
        keys,
        attention,
        valid_tokens,
        patches_per_frame=4,
    )
    fused = fuse_attention_with_partial_ot(
        attention, prior, valid_tokens
    )

    assert prior.shape == (2, 2, 5)
    assert not prior.requires_grad
    assert fused.shape == attention.shape
    assert torch.equal(fused[:, :, 0], attention[:, :, 0])
    assert torch.equal(fused[:, :, 3], attention[:, :, 3])
    assert torch.allclose(
        fused[:, :, valid_tokens].sum(dim=-1),
        torch.ones(1, 2, 2),
        atol=1e-5,
    )

    weights = torch.arange(5, dtype=fused.dtype)
    (fused * weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_partial_ot_rejects_multi_item_attention_batches():
    with pytest.raises(ValueError, match="one video-caption pair"):
        build_partial_ot_prior(
            torch.randn(2, 4, 4),
            torch.randn(2, 2, 5, 2),
            torch.softmax(torch.randn(2, 2, 4, 5), dim=-1),
            torch.tensor([1, 2]),
            patches_per_frame=4,
        )


def test_internvideo_cross_attention_pot_hook():
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        encoder_width=4,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="absolute",
        max_position_embeddings=16,
    )
    attention_layer = BertSelfAttention(config, is_cross_attention=True)
    text = torch.randn(1, 4, 4, requires_grad=True)
    video = torch.randn(1, 257, 4, requires_grad=True)

    context, attention, _scores = attention_layer(
        text,
        encoder_hidden_states=video,
        output_attentions=True,
        valid_tokens=torch.tensor([1, 2]),
        use_pot_tokens=True,
    )[:3]

    assert context.shape == (1, 4, 4)
    assert attention.shape == (1, 2, 4, 257)
    assert torch.allclose(
        attention.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-5
    )
    context.square().mean().backward()
    assert text.grad is not None
    assert video.grad is not None
