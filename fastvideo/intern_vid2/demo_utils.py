import numpy as np
import cv2
import os
import io
import gc

import torch
from torch import nn

from fastvideo.intern_vid2.models.backbones.internvideo2 import (
    pretrain_internvideo2_1b_patch14_224,
)
from fastvideo.intern_vid2.models.backbones.bert.builder import build_bert
from fastvideo.intern_vid2.models.criterions import (
    MLMLoss,
    VTC_VTM_Loss,
    get_sim,
    new_UTA_Loss,
)
from fastvideo.intern_vid2.models.backbones.internvideo2.pos_embed import (
    interpolate_pos_embed_internvideo2_new,
)
from fastvideo.intern_vid2.models.backbones.bert.tokenization_bert import BertTokenizer

# from intern_vid2.models.internvideo2_stage2 import InternVideo2_Stage2


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(
    vid_list, fnum=8, target_size=(224, 224), device=torch.device("cuda")
):
    assert len(vid_list) >= fnum
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_text_feat_dict(texts, clip, text_feat_d={}):
    for t in texts:
        feat = clip.get_txt_feat(t)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_text(
    frames, texts, model, topk: int = 5, config: dict = {}, device=torch.device("cuda")
):

    vlm = model
    vlm = vlm.to(device)

    fn = config.get("num_frames", 8)
    size_t = config.get("size_t", 224)
    frames_tensor = frames2tensor(
        frames, fnum=fn, target_size=(size_t, size_t), device=device
    )
    vid_feat = vlm.get_vid_feat(frames_tensor)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)

    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]
    return ret_texts, probs.float().numpy()[0]


def setup_internvideo2(config: dict):
    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
        model = InternVideo2_Stage2_Rewards(
            config=config, tokenizer=tokenizer, is_pretrain=False
        )
    else:
        model = InternVideo2_Stage2_Rewards(config=config, is_pretrain=False)
        tokenizer = model.tokenizer

    if config.get("compile_model", False):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    model_without_ddp = model

    if (
        config.pretrained_path.strip()
        and (os.path.isfile(config.pretrained_path))
        or "s3://" in config.pretrained_path
    ):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["module"]
        text_encoder_state_dict = {}
        vision_proj_state_dict = {}
        text_proj_state_dict = {}
        itm_head_state_dict = {}
        temp_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("text_encoder.bert."):
                text_encoder_state_dict[k[len("text_encoder.bert.") :]] = v
            elif k.startswith("vision_proj."):
                vision_proj_state_dict[k[len("vision_proj.") :]] = v
            elif k.startswith("text_proj."):
                text_proj_state_dict[k[len("text_proj.") :]] = v
            elif k.startswith("itm_head."):
                itm_head_state_dict[k[len("itm_head.") :]] = v
            elif k == "temp":
                # print(k, v)
                temp_state_dict[k] = v  # Load the temp parameter

        model_without_ddp.text_encoder.load_state_dict(text_encoder_state_dict)
        model_without_ddp.vision_proj.load_state_dict(vision_proj_state_dict)
        model_without_ddp.text_proj.load_state_dict(text_proj_state_dict)
        model_without_ddp.itm_head.load_state_dict(itm_head_state_dict)
        # print(temp_state_dict)
        # model_without_ddp.temp.load_state_dict(temp_state_dict)

        # model_without_ddp.temp.data = temp_state_dict["temp"]
        # print(model_without_ddp.temp)

        if config.get("origin_num_frames", None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict,
                model_without_ddp.vision_encoder,
                orig_t_size=config.origin_num_frames,
            )
            assert a == len(state_dict), state_dict.keys()

        del checkpoint
        del state_dict
        gc.collect()

    if config.get("use_bf16", False):
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.get("use_half_precision", False):
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)

    return (
        model_without_ddp,
        tokenizer,
    )


class InternVideo2_Stage2(nn.Module):
    """docstring for InternVideo2_Stage2"""

    def __init__(self, config, tokenizer, is_pretrain: bool = True):
        super(InternVideo2_Stage2, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder()
        self.freeze_text()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self, image: torch.Tensor, test: bool = False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(
            self.dtype
        )  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(
                image
            )
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
            ) = self.vision_encoder(image, mask, use_image)
            return (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
                targets_clip_middle_vis,
                targets_clip_final_vis,
            )

    def encode_text(self, text: dict):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name

        if encoder_name == "pretrain_internvideo2_1b_patch14_224":
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (
            num_frames // tublet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            _, vfeat = self.encode_vision(frames, test=True)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_vid_feat_with_grad(self, frames: torch.Tensor):
        """get the video features for the given frames with grad.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        _, vfeat = self.encode_vision(frames, test=True)
        vfeat = self.vision_proj(vfeat)
        vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_txt_feat(self, text: str):
        """get the text features for the given text."""
        with torch.no_grad():
            text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
                return_tensors="pt",
            ).to(self.config.device)
            _, tfeat = self.encode_text(text)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    def predict_label(
        self, vid_feat: torch.Tensor, txt_feat: torch.Tensor, top: int = 5
    ):
        label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels


class InternVideo2_Stage2_Rewards(nn.Module):
    """docstring for InternVideo2_Stage2"""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super(InternVideo2_Stage2_Rewards, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder()
        self.freeze_text()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = 0.0120
        # self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # criterions
        # self.criterion_uta = new_UTA_Loss(
        #     config.criterion.distill_final_features,
        #     config.criterion.clip_loss_ratio,
        # )
        self.criterion_vtc_vtm = VTC_VTM_Loss()
        # self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)
        # self.uta_image_only = config.criterion.get("uta_image_only", False)
        # logger.info(f"uta_image_only={self.uta_image_only}")

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # ret.update(
        #     {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        # )

        return ret

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def reward(self, image, text, idx, media_type="image"):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict)
            idx (torch.Tensor)
            media_type: str
        Returns:

        """

        # self.clip_contrastive_temperature()
        T = image.shape[1]
        use_image = True if T == 1 else False

        (
            vision_embeds,
            pooled_vision_embeds,
            # student_output,
            # student_output_final,
            # targets_clip_middle_vis,
            # targets_clip_final_vis,
        ) = self.encode_vision(image, test=True)

        with torch.no_grad():
            text_embeds, pooled_text_embeds = self.encode_text(text)
            text_proj = self.text_proj(pooled_text_embeds)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        # calculate loss
        ## VTC loss
        loss_vtc = self.criterion_vtc_vtm.vtc_loss(
            vision_proj, text_proj, idx, self.temp, all_gather=False
        )
        ## VTM loss
        loss_vtm = self.criterion_vtc_vtm.vtm_loss(
            self.get_text_encoder(),
            self.itm_head,
            self.temp,
            vision_embeds,
            text_embeds,
            vision_proj,
            text_proj,
            text.attention_mask,
            idx,
        )
        return loss_vtc, loss_vtm

    def reward_OT(
        self,
        OT_map,
        image,
        text,
        idx,
        valid_tokens=None,
        use_pot_tokens=False,
        media_type="image",
    ):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict)
            idx (torch.Tensor)
            media_type: str
        Returns:

        """

        # self.clip_contrastive_temperature()
        T = image.shape[1]
        use_image = True if T == 1 else False

        (
            vision_embeds,
            pooled_vision_embeds,
            # student_output,
            # student_output_final,
            # targets_clip_middle_vis,
            # targets_clip_final_vis,
        ) = self.encode_vision(image, test=True)

        with torch.no_grad():
            text_embeds, pooled_text_embeds = self.encode_text(text)
            text_proj = self.text_proj(pooled_text_embeds)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        # calculate loss
        ## VTC loss
        loss_vtc = self.criterion_vtc_vtm.vtc_loss_OT(
            OT_map, vision_proj, text_proj, idx, self.temp, all_gather=False
        )
        ## VTM loss
        loss_vtm = self.criterion_vtc_vtm.vtm_loss(
            self.get_text_encoder(),
            self.itm_head,
            self.temp,
            vision_embeds,
            text_embeds,
            vision_proj,
            text_proj,
            text.attention_mask,
            idx,
            valid_tokens=valid_tokens,
            use_pot_tokens=use_pot_tokens,
        )
        return loss_vtc, loss_vtm

    def forward(self, image, text, idx, media_type="image"):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict)
            idx (torch.Tensor)
            media_type: str
        Returns:

        """

        self.clip_contrastive_temperature()
        T = image.shape[1]
        use_image = True if T == 1 else False

        (
            vision_embeds,
            pooled_vision_embeds,
            student_output,
            student_output_final,
            targets_clip_middle_vis,
            targets_clip_final_vis,
        ) = self.encode_vision(image)

        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        text_proj = self.text_proj(pooled_text_embeds)

        # calculate loss
        ## UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only and not use_image:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(
                    student_output,
                    student_output_final,
                    targets_clip_middle_vis,
                    targets_clip_final_vis,
                )
        else:
            loss_uta = torch.tensor(0)

        ## VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True
            )
        else:
            loss_vtc = torch.tensor(0)

        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.itm_head,
                self.temp,
                vision_embeds,
                text_embeds,
                vision_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None
            )
        else:
            loss_mlm = torch.tensor(0)

        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(
            self.dtype
        )  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(
                image
            )
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
            ) = self.vision_encoder(image, mask, use_image)
            return (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
                targets_clip_middle_vis,
                targets_clip_final_vis,
            )

    def encode_text(self, text, valid_tokens=None, use_pot_tokens=False):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
            valid_tokens=valid_tokens,
            use_pot_tokens=use_pot_tokens,
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        if encoder_name == "pretrain_internvideo2_1b_patch14_224":
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (
            num_frames // tublet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            _, vfeat = self.encode_vision(frames, test=True)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_vid_feat_with_grad(self, frames: torch.Tensor):
        """get the video features for the given frames with grad.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        _, vfeat = self.encode_vision(frames, test=True)
        vfeat = self.vision_proj(vfeat)
        vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_txt_feat(self, text: str):
        """get the text features for the given text."""
        with torch.no_grad():
            text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
                return_tensors="pt",
            ).to(self.config.device)
            _, tfeat = self.encode_text(text)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    def predict_label(
        self, vid_feat: torch.Tensor, txt_feat: torch.Tensor, top: int = 5
    ):
        label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels
