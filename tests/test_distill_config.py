from argparse import Namespace

import pytest

from fastvideo.pisces_config import (
    get_lora_target_modules,
    validate_training_args,
)


def make_args(**overrides):
    values = {
        "use_global_reward_loss": False,
        "use_finegrained_reward_loss": False,
        "use_consistency_loss": True,
        "use_ot_reward": False,
        "pred_decay_weight": 0.0,
        "vae_decode_checkpointing": False,
        "vae_tiling": False,
        "train_batch_size": 1,
        "consistency_loss_weight": 1.0,
        "global_reward_loss_weight": 1.0,
        "finegrained_reward_loss_weight": 1.0,
        "use_lora": False,
        "lora_rank": 256,
        "lora_alpha": 32,
        "max_train_steps": 256,
        "multi_phased_distill_schedule": "4000-1",
    }
    values.update(overrides)
    return Namespace(**values)


def test_valid_consistency_only_configuration():
    validate_training_args(make_args())


def test_valid_full_pisces_configuration():
    validate_training_args(
        make_args(
            use_global_reward_loss=True,
            use_finegrained_reward_loss=True,
            use_ot_reward=True,
            use_lora=True,
            vae_decode_checkpointing=True,
        )
    )


def test_requires_an_active_objective():
    with pytest.raises(ValueError, match="At least one"):
        validate_training_args(make_args(use_consistency_loss=False))


def test_ot_requires_a_reward():
    with pytest.raises(ValueError, match="requires at least one reward"):
        validate_training_args(
            make_args(use_consistency_loss=True, use_ot_reward=True)
        )


def test_pot_requires_single_item_batches():
    with pytest.raises(ValueError, match="train_batch_size 1"):
        validate_training_args(
            make_args(
                use_global_reward_loss=True,
                use_finegrained_reward_loss=True,
                use_ot_reward=True,
                train_batch_size=2,
            )
        )


def test_global_ot_allows_larger_batches_without_token_level_pot():
    validate_training_args(
        make_args(
            use_global_reward_loss=True,
            use_ot_reward=True,
            train_batch_size=2,
        )
    )


def test_hunyuan_lora_defaults_and_override():
    assert get_lora_target_modules("hunyuan", None) == [
        "img_attn_qkv",
        "img_attn_proj",
        "txt_attn_qkv",
        "txt_attn_proj",
        "linear1",
        "linear2",
    ]
    assert get_lora_target_modules("hunyuan", "foo, bar") == ["foo", "bar"]
