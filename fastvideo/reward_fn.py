import os
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import AutoModel, AutoProcessor
from torchvision.transforms import (
    CenterCrop,
    InterpolationMode,
    Normalize,
    RandomCrop,
    Resize,
)

from fastvideo.optimal_transport import load_optimal_transport_map


# Image processing
CLIP_RESIZE = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
CLIP_NORMALIZE = Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
CENTER_CROP = CenterCrop(224)

ViCLIP_NORMALIZE = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_pick_score_fn(precision="fp32"):
    """
    Loss function for PICK SCORE
    """
    print("Loading PICK SCORE model")

    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval()
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.requires_grad_(False)
    if precision == "fp16":
        model.to(torch.float16)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str, return_logits=False):
        device = image_inputs.device
        model.to(device)

        pixel_values = CLIP_NORMALIZE(CENTER_CROP(CLIP_RESIZE(image_inputs)))

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.no_grad():
            preprocessed = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_embs = model.get_text_features(**preprocessed)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Get predicted scores from model(s)
        score = (text_embs * image_embs).sum(-1)
        if return_logits:
            score = score * model.logit_scale.exp()
        return score

    return score_fn


def get_hpsv2_fn(precision="amp"):
    precision = "amp" if precision == "no" else precision
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        f"{os.environ['HOME']}/.cache/hpsv2/HPS_v2.1_compressed.pt",
        precision=precision,
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    tokenizer = get_tokenizer("ViT-H-14")
    model.eval()
    model.requires_grad_(False)

    # gets vae decode as input
    def score_fn(
        image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False
    ):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        for t in preprocess_val.transforms[2:]:
            image_inputs = torch.stack([t(img) for img in image_inputs])

        if isinstance(text_inputs[0], str):
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)
            repeat_times = image_features.shape[0] // text_features.shape[0]
            text_features = text_features.repeat(repeat_times, 1)

        hps_score = (image_features * text_features).sum(-1)
        if return_logits:
            hps_score = hps_score * model.logit_scale.exp()
        return hps_score

    return score_fn


def get_img_reward_fn(precision="fp32"):
    # pip install image-reward
    import ImageReward as RM
    import torch.nn.functional as F
    from torchvision.transforms import Compose, Resize, CenterCrop
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC

    model = RM.load("ImageReward-v1.0")
    model.eval()
    model.requires_grad_(False)

    rm_preprocess = Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            CLIP_NORMALIZE,
        ]
    )

    # gets vae decode as input
    def score_fn(
        image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False
    ):
        del return_logits
        device = image_inputs.device
        model.to(device)
        if precision == "fp16":
            model.to(torch.float16)

        image = rm_preprocess(image_inputs).to(device)
        text_input = model.blip.tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        rewards = model.score_gard(
            text_input.input_ids, text_input.attention_mask, image
        )
        return -F.relu(-rewards + 2).squeeze(-1)

    return score_fn


class ResizeCropMinSize(nn.Module):
    def __init__(self, min_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(min_size, int):
            raise TypeError(f"Size should be int. Got {type(min_size)}")
        self.min_size = min_size
        self.interpolation = interpolation
        self.fill = fill
        self.random_crop = RandomCrop((min_size, min_size))

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
        return self.random_crop(img)


def get_vi_clip_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    assert n_frames == 8
    from viclip import get_viclip

    model_dict = get_viclip("l", rm_ckpt_dir)
    vi_clip = model_dict["viclip"]
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        # Process pixels and multicrop
        device = image_inputs.device
        vi_clip.to(device)
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with torch.no_grad():
            text_features = vi_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

    return score_fn


def _load_internvideo2(rm_ckpt_dir: str, n_frames: int):
    from fastvideo.intern_vid2.demo_config import Config, eval_dict_leaf
    from fastvideo.intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file(
        "fastvideo/intern_vid2/configs/internvideo2_stage2_config.py"
    )
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames
    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir
    return setup_internvideo2(config)


def _get_lexical_token_indices(tokenized_text, tokenizer):
    """Return non-padding, non-special token positions for one caption."""

    lexical_mask = tokenized_text["attention_mask"][0].bool()
    input_ids = tokenized_text["input_ids"][0]
    for token_id in (
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ):
        if token_id is not None:
            lexical_mask &= input_ids != token_id
    return lexical_mask.nonzero(as_tuple=False).squeeze(1)


def get_intern_vid2_score_fn(
    rm_ckpt_dir: str,
    precision="amp",
    n_frames=8,
    use_ot=False,
    OT_map_ckpt_dir=None,
    ot_map_ckpt_dir=None,
    use_pot_tokens=True,
):
    ot_map = None
    if use_ot:
        checkpoint_path = ot_map_ckpt_dir or OT_map_ckpt_dir
        if checkpoint_path is None:
            raise ValueError(
                "An OT map checkpoint is required when use_ot=True. Pass "
                "ot_map_ckpt_dir (or the legacy OT_map_ckpt_dir argument)."
            )
        ot_map = load_optimal_transport_map(checkpoint_path)

    vi_clip, tokenizer = _load_internvideo2(rm_ckpt_dir, n_frames)
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)
        if ot_map is not None:
            ot_map.to(torch.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        if image_inputs.ndim != 5:
            raise ValueError(
                "InternVideo2 rewards expect videos shaped [batch, time, "
                f"channels, height, width], got {tuple(image_inputs.shape)}."
            )
        device = image_inputs.device
        vi_clip.to(device)
        if ot_map is not None:
            ot_map.to(device)
        b, t = image_inputs.shape[:2]
        if use_ot and use_pot_tokens and b != 1:
            raise ValueError(
                "Token-level POT currently supports one video-caption pair per "
                "reward-model call. Use --train_batch_size 1."
            )
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        if len(text_inputs) != b:
            raise ValueError(
                f"Received {len(text_inputs)} captions for a batch of {b} videos."
            )

        image_inputs = image_inputs.reshape(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])

        with torch.no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=300 if use_pot_tokens else 40,
                return_tensors="pt",
            ).to(device)
            valid_indices = None
            if use_ot and use_pot_tokens:
                valid_indices = _get_lexical_token_indices(text, tokenizer)

        if use_ot:
            r_global, r_finegrained = vi_clip.reward_OT(
                ot_map,
                pixel_values,
                text,
                None,
                valid_tokens=valid_indices,
                use_pot_tokens=use_pot_tokens,
            )
        else:
            r_global, r_finegrained = vi_clip.reward(pixel_values, text, None)
        return r_global, r_finegrained

    return score_fn


def get_intern_vid2_OT_score_fn(
    rm_ckpt_dir: str, OT_map_ckpt_dir: str, precision="amp", n_frames=8
):
    return get_intern_vid2_score_fn(
        rm_ckpt_dir=rm_ckpt_dir,
        precision=precision,
        n_frames=n_frames,
        use_ot=True,
        OT_map_ckpt_dir=OT_map_ckpt_dir,
    )


def get_intern_vid2_dynamic_score_fn(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "The experimental InternVideo2 dynamic reward is not supported in "
        "this release because its RAFT and CLIP dependencies were never "
        "integrated into the repository."
    )


def get_clip_score_fn(precision="amp"):
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision=precision,
        device="cuda",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=None,
        image_mean=None,
        image_std=None,
        image_interpolation=None,
        image_resize_mode=None,  # only effective for inference
        aug_cfg={},
        pretrained_image=False,
        output_dict=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model.eval()
    model.requires_grad_(False)

    # gets vae decode as input
    def score_fn(
        image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False
    ):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        if isinstance(text_inputs[0], str):
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        clip_score = (image_features * text_features).sum(-1)
        if return_logits:
            clip_score = clip_score * model.logit_scale.exp()
        return clip_score

    return score_fn


def get_weighted_hpsv2_clip_fn(precision="amp", weights=None):
    if weights is None:
        weights = [1.0, 5.0]
    hpsv2_score_fn = get_hpsv2_fn(precision)
    clip_score_fn = get_clip_score_fn(precision)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        hpsv2_score = hpsv2_score_fn(image_inputs, text_inputs)
        img_reward_score = clip_score_fn(image_inputs, text_inputs)
        return weights[0] * hpsv2_score + weights[1] * img_reward_score

    return score_fn


def get_reward_fn(reward_fn_name: str, **kwargs):
    reward_functions = {
        "pick": get_pick_score_fn,
        "hpsv2": get_hpsv2_fn,
        "img_reward": get_img_reward_fn,
        "vi_clip": get_vi_clip_score_fn,
        "vi_clip2": get_intern_vid2_score_fn,
        "vi_clip2_dynamic": get_intern_vid2_dynamic_score_fn,
        "vi_clip2_OT": get_intern_vid2_OT_score_fn,
        "clip": get_clip_score_fn,
        "weighted_hpsv2_clip": get_weighted_hpsv2_clip_fn,
    }
    try:
        reward_function = reward_functions[reward_fn_name]
    except KeyError as error:
        available = ", ".join(sorted(reward_functions))
        raise ValueError(
            f"Unknown reward function '{reward_fn_name}'. Available: {available}."
        ) from error
    return reward_function(**kwargs)
