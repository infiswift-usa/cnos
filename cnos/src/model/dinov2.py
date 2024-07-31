import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from ..utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from ..model.utils import BatchedData
from copy import deepcopy
import os.path as osp

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


descriptor_map = {
    "dinov2_vits14": "vit_small",
    "dinov2_vitb14": "vit_base",
    "dinov2_vitl14": "vit_large",
    "dinov2_vitg14": "vit_giant2",
}

from enum import Enum
from typing import Union


class Weights(Enum):
    LVD142M = "LVD142M"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        checkpoint_dir,
        patch_size=14,
        validpatch_thresh=0.5,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = _make_dinov2_model(arch_name=descriptor_map[model_name], pretrained=False)
        model_path = osp.join(checkpoint_dir, f"{model_name}_pretrain.pth")
        logging.info(f"Loading from {model_path}")
        self.model.load_state_dict(torch.load(osp.join(checkpoint_dir, f"{model_name}_pretrain.pth")))
        self.validpatch_thresh = validpatch_thresh
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        self.patch_kernel = torch.nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                features = self.model(images)
        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)
        return features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)
    
    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)



# class CustomDINOv2(pl.LightningModule):
#     def __init__(
#         self,
#         model_name,
#         model,
#         token_name,
#         image_size,
#         chunk_size,
#         descriptor_width_size,
#         patch_size=14,
#     ):
#         super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.token_name = token_name
#         self.chunk_size = chunk_size
#         self.patch_size = patch_size
#         self.proposal_size = image_size
#         self.descriptor_width_size = descriptor_width_size
#         logging.info(f"Init CustomDINOv2 done!")
#         self.rgb_normalize = T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ]
#         )
#         # use for global feature
#         self.rgb_proposal_processor = CropResizePad(self.proposal_size)
#         self.rgb_resize = CustomResizeLongestSide(
#             descriptor_width_size, dividable_size=self.patch_size
#         )
#         logging.info(
#             f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
#         )

#     def process_rgb_proposals(self, image_np, masks, boxes):
#         """
#         1. Normalize image with DINOv2 transfom
#         2. Mask and crop each proposals
#         3. Resize each proposals to predefined longest image size
#         """
#         num_proposals = len(masks)
#         rgb = self.rgb_normalize(image_np).to(masks.device).float()
#         rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
#         masked_rgbs = rgbs * masks.unsqueeze(1)
#         processed_masked_rgbs = self.rgb_proposal_processor(
#             masked_rgbs, boxes
#         )  # [N, 3, target_size, target_size]
#         return processed_masked_rgbs

#     @torch.no_grad()
#     def compute_features(self, images, token_name):
#         if token_name == "x_norm_clstoken":
#             if images.shape[0] > self.chunk_size:
#                 features = self.forward_by_chunk(images)
#             else:
#                 features = self.model(images)
#         else:  # get both features
#             raise NotImplementedError
#         return features

#     @torch.no_grad()
#     def forward_by_chunk(self, processed_rgbs):
#         batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
#         del processed_rgbs  # free memory
#         features = BatchedData(batch_size=self.chunk_size)
#         for idx_batch in range(len(batch_rgbs)):
#             feats = self.compute_features(
#                 batch_rgbs[idx_batch], token_name="x_norm_clstoken"
#             )
#             features.cat(feats)
#         return features.data


#     @torch.no_grad()
#     def forward_cls_token(self, image_np, proposals):
#         processed_rgbs = self.process_rgb_proposals(
#             image_np, proposals.masks, proposals.boxes
#         )
#         return self.forward_by_chunk(processed_rgbs)

#     @torch.no_grad()
#     def forward(self, image_np, proposals):
#         return self.forward_cls_token(image_np, proposals)

