import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from fastvideo.intern_vid2.models.utils import allgather_wgrad
from fastvideo.intern_vid2.utils.distributed import get_rank, get_world_size
from fastvideo.intern_vid2.utils.easydict import EasyDict

logger = logging.getLogger(__name__)


def get_sim(
    vision_proj: torch.Tensor,
    text_proj: torch.Tensor,
    temp=1.0,
    agg_method="mean",
):
    """calculate pair-wise video-text similarity.

    Args:
        vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
        text_proj (torch.Tensor): The text representation. Shape: [B,C].
        temp (torch.Tensor): The temperature. Shape: [].

    Returns: The similarity between video and text. Shape: [B,B].

    """
    vision_proj = F.normalize(vision_proj, dim=-1)
    text_proj = F.normalize(text_proj, dim=-1)
    if vision_proj.ndim == 3:
        sim_v2t = (
            torch.einsum("mld,nd->mln", vision_proj, text_proj) / temp
        )  # [B, L, B]
        sim_t2v = (
            torch.einsum("nd,mld->nlm", text_proj, vision_proj) / temp
        )  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    elif text_proj.ndim == 3:
        sim_v2t = (
            torch.einsum("nd,mld->nlm", vision_proj, text_proj) / temp
        )  # [B, L, B]
        sim_t2v = (
            torch.einsum("nld,md->nlm", text_proj, vision_proj) / temp
        )  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    else:
        sim_v2t = vision_proj @ text_proj.T / temp
        sim_t2v = sim_v2t.T

    return sim_v2t, sim_t2v


def get_sim_OT(
    OT_map: nn.Module,
    vision_proj: torch.Tensor,
    text_proj: torch.Tensor,
    temp=1.0,
    agg_method="mean",
):
    """calculate pair-wise video-text similarity.

    Args:
        vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
        text_proj (torch.Tensor): The text representation. Shape: [B,C].
        temp (torch.Tensor): The temperature. Shape: [].

    Returns: The similarity between video and text. Shape: [B,B].

    """
    vision_proj = F.normalize(vision_proj, dim=-1)
    text_proj = F.normalize(text_proj, dim=-1)
    text_proj = OT_map(text_proj)
    text_proj = F.normalize(text_proj, dim=-1)

    if vision_proj.ndim == 3:
        sim_v2t = (
            torch.einsum("mld,nd->mln", vision_proj, text_proj) / temp
        )  # [B, L, B]
        sim_t2v = (
            torch.einsum("nd,mld->nlm", text_proj, vision_proj) / temp
        )  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    elif text_proj.ndim == 3:
        sim_v2t = (
            torch.einsum("nd,mld->nlm", vision_proj, text_proj) / temp
        )  # [B, L, B]
        sim_t2v = (
            torch.einsum("nld,md->nlm", text_proj, vision_proj) / temp
        )  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    else:
        sim_v2t = vision_proj @ text_proj.T / temp
        sim_t2v = sim_v2t.T

    return sim_v2t, sim_t2v


class VTC_VTM_Loss(nn.Module):
    """video-text contrastive and matching losses."""

    def __init__(self):
        super().__init__()

    def vtc_loss_OT(
        self,
        T: nn.Module,
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True,
        agg_method="mean",
    ):
        """Return the global quality reward in the OT-aligned embedding space."""
        if all_gather:
            gather_args = self.get_gather_args()
            vision_proj = allgather_wgrad(vision_proj, gather_args)
            text_proj = allgather_wgrad(text_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_v2t, sim_t2v = get_sim_OT(
            T, vision_proj, text_proj, temp, agg_method=agg_method
        )

        with torch.no_grad():
            sim_v2t_targets = self.get_mask(sim_v2t, idx=idx, normalize=True)
            sim_t2v_targets = sim_v2t_targets

        return (
            torch.sum(
                sim_v2t * sim_v2t_targets * 0.5 * temp
                + sim_t2v * sim_t2v_targets * 0.5 * temp
            )
            / sim_v2t.shape[0]
        )

    def vtc_loss(
        self,
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True,
        agg_method="mean",
    ):
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            vision_proj = allgather_wgrad(vision_proj, gather_args)
            text_proj = allgather_wgrad(text_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_v2t, sim_t2v = get_sim(vision_proj, text_proj, temp, agg_method=agg_method)

        with torch.no_grad():
            sim_v2t_targets = self.get_mask(sim_v2t, idx=idx, normalize=True)
            sim_t2v_targets = sim_v2t_targets

        return (
            torch.sum(
                sim_v2t * sim_v2t_targets * 0.5 * temp
                + sim_t2v * sim_t2v_targets * 0.5 * temp
            )
            / sim_v2t.shape[0]
        )

    def vtm_loss(
        self,
        multimodal_encoder,
        vtm_head: nn.Module,
        temp,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        text_atts: torch.Tensor,
        idx: torch.Tensor,
        valid_tokens=None,
        use_pot_tokens=False,
    ):
        """Return the InternVideo2 positive video-text matching probability.

        With ``use_pot_tokens=True``, the multimodal encoder injects a detached
        POT structural prior into cross-attention before applying the existing
        VTM classification head.
        """
        del temp, vision_proj, text_proj, idx
        with torch.no_grad():
            vision_atts = torch.ones(
                vision_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device
            )

        output = multimodal_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            mode="fusion",
            valid_tokens=valid_tokens,
            use_pot_tokens=use_pot_tokens,
        )

        vtm_embeds = output.last_hidden_state[:, 0]
        vtm_logits = vtm_head(vtm_embeds)
        return F.softmax(vtm_logits, dim=1)[:, 1].mean()

    @torch.no_grad()
    def get_mask(self, sim, idx=None, normalize=False):
        """
        Args:
            sim (torch.Tensor): The similarity between videos and texts. shape: (B, B).
            idx (torch.Tensor): The index for each video. Shape: [B].
            normalize (bool): If true, make row sum equal to 1
        """
        if idx is not None:
            idx = idx.view(-1, 1)
            mask = torch.eq(idx, idx.T).to(sim.dtype)
            if normalize:
                mask = mask / mask.sum(1, keepdim=True)
        else:
            mask = torch.zeros_like(sim)
            mask.fill_diagonal_(1)
        return mask  # `1` mark valid/matched location

    @lru_cache(maxsize=16)
    def get_gather_args(self):
        """obtain the args for all_gather
        Returns: dict.

        """
        return EasyDict({"world_size": get_world_size(), "rank": get_rank()})


class MLMLoss(nn.Module):
    """masked language modeling loss."""

    def __init__(self, masking_prob, tokenizer):
        super(MLMLoss, self).__init__()
        self.tokenizer = tokenizer
        self.masking_prob = masking_prob

    def mlm_loss(
        self,
        text_encoder,
        text,
        vision_embeds,
        vision_atts,
    ):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.masking_prob)
        input_ids, labels = self.mask(
            input_ids,
            text_encoder.config.vocab_size,
            input_ids.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        intermediate_mlm_output = text_encoder.bert(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            mode="text",
        )

        text_embeds = intermediate_mlm_output.last_hidden_state

        mlm_output = text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion",
        )
        return mlm_output.loss

    def simple_mlm_loss(
        self, text_encoder, text, text_embeds, vision_embeds, vision_atts, labels
    ):
        mlm_output = text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion",
        )
        return mlm_output.loss

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        """make deepspeed happy!"""
        # _pad_mask = (input_ids == self.tokenizer.pad_token_id).to(masked_indices.device, non_blocking=True) # 0
        # # print(_pad_mask.device)
        # masked_indices[_pad_mask] = False
        # _cls_mask = (input_ids == self.tokenizer.cls_token_id).to(masked_indices.device, non_blocking=True) # 101
        # masked_indices[_cls_mask] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(
            device
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


class UTA_Loss(nn.Module):
    """mask align clip loss."""

    def __init__(self, uta_norm_type="l2", uta_loss_type="l2"):
        super().__init__()
        self.norm_type = uta_norm_type
        self.loss_type = uta_loss_type
        logger.info(f"Norm type: {uta_norm_type}")
        logger.info(f"Loss type: {uta_loss_type}")

        if uta_loss_type == "mse":
            self.loss_func = nn.MSELoss()
        elif uta_loss_type == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss()

    def uta_loss(self, student_output, clip_output):
        """forward to calculate the loss

        Args:
            student_output (torch.Tensor): The student output. Shape: [K,B,N,C].
            clip_output (torch.Tensor): The teacher representation. Shape: [K,B,N,C].

        Returns: loss_uta (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if self.norm_type == "l2":
            student_output = student_output / student_output.norm(dim=-1, keepdim=True)
            clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == "none":
            pass
        else:
            raise NotImplementedError

        if self.loss_type == "l2":
            loss_uta = (2 - 2 * (student_output * clip_output).sum(dim=-1)).mean()
        elif self.loss_type in ["mse", "smooth_l1"]:
            loss_uta = self.loss_func(input=student_output, target=clip_output)
        else:
            raise NotImplementedError

        return loss_uta

    def uta_vision_loss(self, student_v_output, clip_v_output):
        """forward to calculate the loss

        Args:
            student_v_output (torch.Tensor): The student output. Shape: [B,T,C].
            clip_v_output (torch.Tensor): The teacher representation. Shape: [B,T,C].

        Returns: loss_uta (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if student_v_output.shape[1] != clip_v_output.shape[1]:
            student_v_output = student_v_output.mean(1, keepdim=True)
            clip_v_output = clip_v_output.mean(1, keepdim=True)
        if self.norm_type == "l2":
            student_v_output = student_v_output / student_v_output.norm(
                dim=-1, keepdim=True
            )
            clip_v_output = clip_v_output / clip_v_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == "none":
            pass
        else:
            raise NotImplementedError

        if self.loss_type == "l2":
            loss_uta = (2 - 2 * (student_v_output * clip_v_output).sum(dim=-1)).mean()
        elif self.loss_type in ["mse", "smooth_l1"]:
            loss_uta = self.loss_func(input=student_v_output, target=clip_v_output)
        else:
            raise NotImplementedError

        return loss_uta

    def uta_all_loss(
        self,
        student_v_output,
        clip_v_output,
        student_t_output,
        clip_t_output,
    ):
        """forward to calculate the loss

        Args:
            student_v_output (torch.Tensor): The student output. Shape: [B,T,C].
            clip_v_output (torch.Tensor): The teacher representation. Shape: [B,T,C].
            student_t_output (torch.Tensor): The student output. Shape: [B,1,C].
            clip_t_output (torch.Tensor): The teacher representation. Shape: [B,1,C].

        Returns: loss_uta (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if student_v_output.shape[1] != clip_v_output.shape[1]:
            student_v_output = student_v_output.mean(1, keepdim=True)
            clip_v_output = clip_v_output.mean(1, keepdim=True)
        if self.norm_type == "l2":
            student_v_output = student_v_output / student_v_output.norm(
                dim=-1, keepdim=True
            )
            clip_v_output = clip_v_output / clip_v_output.norm(dim=-1, keepdim=True)
            student_t_output = student_t_output / student_t_output.norm(
                dim=-1, keepdim=True
            )
            clip_t_output = clip_t_output / clip_t_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == "none":
            pass
        else:
            raise NotImplementedError

        if self.loss_type == "l2":
            loss_uta_v = (2 - 2 * (student_v_output * clip_v_output).sum(dim=-1)).mean()
            loss_uta_t = (2 - 2 * (student_t_output * clip_t_output).sum(dim=-1)).mean()
        elif self.loss_type in ["mse", "smooth_l1"]:
            loss_uta_v = self.loss_func(input=student_v_output, target=clip_v_output)
            loss_uta_t = self.loss_func(input=student_t_output, target=clip_t_output)
        else:
            raise NotImplementedError

        return (loss_uta_v + loss_uta_t) / 2.0


class new_UTA_Loss(nn.Module):
    """mask align clip loss."""

    def __init__(self, distill_final_features=True, clip_loss_ratio=[1.0, 1.0]):
        super().__init__()
        self.distill_final_features = distill_final_features
        self.clip_loss_ratio = clip_loss_ratio

        logger.info(f"distill_final_features: {distill_final_features}")
        logger.info(f"clip_loss_ratio: {clip_loss_ratio}")

    def uta_loss(
        self,
        student_output,
        student_output_final,
        targets_clip_middle_vis,
        targets_clip_final_vis,
    ):
        """forward to calculate the loss

        Args:
            student_output (torch.Tensor): The student output. Shape: [K,B,N,C].
            clip_output (torch.Tensor): The teacher representation. Shape: [K,B,N,C].

        Returns: loss_uta (torch.Tensor): The mask clip alignment loss. Shape: [].
        """
        loss_clip_middle = (
            2 - 2 * (student_output * targets_clip_middle_vis).sum(dim=-1)
        ).mean()
        if self.distill_final_features and self.clip_loss_ratio[1] > 0:
            loss_clip_final = (
                2 - 2 * (student_output_final * targets_clip_final_vis).sum(dim=-1)
            ).mean()
        else:
            loss_clip_final = (
                torch.zeros(1).type_as(loss_clip_middle).to(loss_clip_middle.device)
            )
        loss_uta = (
            loss_clip_middle * self.clip_loss_ratio[0]
            + loss_clip_final * self.clip_loss_ratio[1]
        )
        return loss_uta
