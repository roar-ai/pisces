from typing import List
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
import torchvision.transforms.functional as F
from torchvision.transforms import (
    Normalize,
    Resize,
    InterpolationMode,
    CenterCrop,
    RandomCrop,
)

# from vbench.third_party.RAFT.core.raft import RAFT
from easydict import EasyDict as edict

# from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
# import clip
# from vbench.utils import clip_transform


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
            img = self.random_crop(img)
        return img


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


def get_intern_vid2_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir

    vi_clip, tokenizer = setup_internvideo2(config)
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
        # video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with torch.no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            ).to(device)
            # _, text_features = vi_clip.encode_text(text)
            # text_features = vi_clip.text_proj(text_features)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
        r_global, r_finegrained = vi_clip.reward(pixel_values, text, None)
        return r_global, r_finegrained
        # score = (video_features * text_features).sum(-1)
        # return score

    return score_fn


def get_intern_vid2_OT_score_fn(
    rm_ckpt_dir: str, OT_map_ckpt_dir: str, precision="amp", n_frames=8
):
    from fastvideo.intern_vid2.demo_config import Config, eval_dict_leaf
    from fastvideo.intern_vid2.demo_utils import setup_internvideo2

    class OptimalTransportMap(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(True),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    T = OptimalTransportMap(512, 1024, 512)
    checkpoint = torch.load(OT_map_ckpt_dir)
    T.load_state_dict(checkpoint)
    T.eval()
    T.requires_grad_(False)

    config = Config.from_file(
        "fastvideo/intern_vid2/configs/internvideo2_stage2_config.py"
    )
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir

    vi_clip, tokenizer = setup_internvideo2(config)
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)
        T.to(torch.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        # Process pixels and multicrop
        device = image_inputs.device
        vi_clip.to(device)
        T.to(device)
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))

        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        # video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with torch.no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            ).to(device)
            # _, text_features = vi_clip.encode_text(text)
            # text_features = vi_clip.text_proj(text_features)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
        r_global, r_finegrained = vi_clip.reward_OT(T, pixel_values, text, None)
        return r_global, r_finegrained
        # score = (video_features * text_features).sum(-1)
        # return score

    return score_fn


def get_intern_vid2_dynamic_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir

    vi_clip, tokenizer = setup_internvideo2(config)
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)

    viclip_resize = ResizeCropMinSize(224)
    args_new = edict(
        {
            "model": "./raft/raft-things.pth",
            "small": False,
            "mixed_precision": True,
            "alternate_corr": False,
        }
    )
    dynamic_model = RAFT(args_new)
    ckpt = torch.load(args_new.model, map_location="cpu")
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    dynamic_model.load_state_dict(new_ckpt)
    # dynamic_model.to(self.device)
    dynamic_model.eval()
    dynamic_model.requires_grad_(False)

    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    image_transform = clip_transform(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        # Process pixels and multicrop
        device = image_inputs.device
        dynamic_model.to(device)
        clip_model.to(device)
        b, t = image_inputs.shape[:2]

        scale_inputs = image_inputs * 255.0
        transformed_inputs = image_transform(scale_inputs.squeeze(0))
        image_features = clip_model.encode_image(transformed_inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1, p=2)

        first_image_feature = image_features[0].unsqueeze(0)
        former_image_feature = first_image_feature
        video_sim = 0.0
        for i in range(t - 1):
            frame_t1, frame_t = (
                scale_inputs[:, i, :, :, :].squeeze(1),
                scale_inputs[:, i + 1, :, :, :].squeeze(1),
            )
            padder = InputPadder(frame_t1.shape)
            frame_t1, frame_t = padder.pad(frame_t1, frame_t)
            _, flow_up = dynamic_model(frame_t1, frame_t, iters=20, test_mode=True)
            flow_up = flow_up[0].permute(1, 2, 0)
            u = flow_up[:, :, 0]
            v = flow_up[:, :, 1]
            rad = torch.sqrt(torch.square(u) + torch.square(v))
            r_dynamic = rad.mean().sigmoid()

            # h, w = rad.size()
            # rad_flat = rad.flatten()
            # cut_index = int(h * w * 0.05)
            # sorted_rad = torch.sort(rad_flat).values  # Sort in ascending order
            # r_dynamic = torch.nn.functional.sigmoid(
            #     torch.mean(sorted_rad[-cut_index:])
            # )  # Take the mean of the last 'cut_index' elements

            ######## background consistency

            image_feature = image_features[i + 1].unsqueeze(0)
            sim_pre = torch.nn.functional.cosine_similarity(
                former_image_feature, image_feature
            )
            # sim_fir = torch.nn.functional.cosine_similarity(
            #     first_image_feature, image_feature
            # )
            # cur_sim = (sim_pre + sim_fir) / 2
            cur_sim = sim_pre
            video_sim += cur_sim

            former_image_feature = image_feature

        vi_clip.to(device)
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))

        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        # video_features = vi_clip.get_vid_feat_with_grad(pixel_values)
        r_background = video_sim / (t - 1)
        with torch.no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            ).to(device)
            # _, text_features = vi_clip.encode_text(text)
            # text_features = vi_clip.text_proj(text_features)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
        r_global, r_finegrained = vi_clip.reward(pixel_values, text, None)
        return r_global, r_finegrained, r_dynamic, r_background

        # score = (video_features * text_features).sum(-1)
        # return score

    return score_fn


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


def get_weighted_hpsv2_clip_fn(precision="amp", weights=[1.0, 5.0]):
    hpsv2_score_fn = get_hpsv2_fn(precision)
    clip_score_fn = get_clip_score_fn(precision)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        hpsv2_score = hpsv2_score_fn(image_inputs, text_inputs)
        img_reward_score = clip_score_fn(image_inputs, text_inputs)
        return weights[0] * hpsv2_score + weights[1] * img_reward_score

    return score_fn


def get_reward_fn(reward_fn_name: str, **kwargs):
    if reward_fn_name == "pick":
        return get_pick_score_fn(**kwargs)
    elif reward_fn_name == "hpsv2":
        return get_hpsv2_fn(**kwargs)
    elif reward_fn_name == "img_reward":
        return get_img_reward_fn(**kwargs)
    elif reward_fn_name == "vi_clip":
        return get_vi_clip_score_fn(**kwargs)
    elif reward_fn_name == "vi_clip2":
        return get_intern_vid2_score_fn(**kwargs)
    elif reward_fn_name == "vi_clip2_dynamic":
        return get_intern_vid2_dynamic_score_fn(**kwargs)
    elif reward_fn_name == "vi_clip2_OT":
        return get_intern_vid2_OT_score_fn(**kwargs)
    elif reward_fn_name == "clip":
        return get_clip_score_fn(**kwargs)
    elif reward_fn_name == "weighted_hpsv2_clip":
        return get_weighted_hpsv2_clip_fn(**kwargs)
    else:
        raise ValueError("Invalid reward_fn_name")
