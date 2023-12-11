# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""Build a VIDT detector for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (nested_tensor_from_tensor_list,
                        inverse_sigmoid, NestedTensor)
from methods.swin_w_ram import swin_nano, swin_tiny, swin_small, swin_base_win7, swin_large_win7
from methods.coat_w_ram import coat_lite_tiny, coat_lite_mini, coat_lite_small
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessor import PostProcess, PostProcessSegm

from .hoi import (DETRHOI, SetCriterionHOI, PostProcessHOI)
from .transformer import build_transformer
from .deformable_transformer import build_deforamble_transformer
from methods.vidt.fpn_fusion import FPNFusionModule
import copy
import math
from .dct import ProcessorDCT

def _get_clones(module, N):
    """ Clone a moudle N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Detector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, with_box_refine=False,
                 # The three additional techniques for ViDT+
                 epff=None,# (1) Efficient Pyramid Feature Fusion Module
                 with_vector=False, processor_dct=None, vector_hidden_dim=256,# (2) UQR Module
                 iou_aware=False, token_label=False, # (3) Additional losses
                 distil=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            epff: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine


        # For UQR module for ViDT+
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)

        ############ Modified for ViDT+
        # For two additional losses for ViDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil

        # For EPFF module for ViDT+
        if epff is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                  nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = epff
        ############

        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(self.backbone.num_channels[-2], hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )

        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        ############

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])
        ############

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])


    def forward(self, samples: NestedTensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors # RGB input
        mask = samples.mask # padding mask

        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        features, det_tgt, det_pos = self.backbone(x, mask)

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # [PATCH] token projection
        shapes = []
        for l, src in enumerate(features):
            shapes.append(src.shape[-2:])

        srcs = []
        if self.fusion is None:
            for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))
        else:
            # EPFF (multi-scale fusion) is used if fusion is activated
            srcs = self.fusion(features)

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = []
        outputs_coords = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = \
          self.transformer(srcs, masks, det_tgt, det_pos)

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ############ Added for UQR
        outputs_vector = None
        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)
        ############

        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        ############ Added for UQR
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        ############

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector)


        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if outputs_vector is None:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]

class HOIDetector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_queries,
                 num_obj_classes, num_verb_classes, #hoi
                 aux_loss=False, with_box_refine=False,
                 # The three additional techniques for ViDT+
                 epff=None,# (1) Efficient Pyramid Feature Fusion Module
                 with_vector=False, processor_dct=None, vector_hidden_dim=256,# (2) UQR Module
                 iou_aware=False, token_label=False, # (3) Additional losses
                 distil=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            epff: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_obj_classes)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes) #hoi
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) 
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) #hoi
        self.backbone = backbone   


        

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine


        # For UQR module for ViDT+
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)

        ############ Modified for ViDT+
        # For two additional losses for ViDT+
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil

        # For EPFF module for ViDT+
        if epff is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                  nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = epff
        ############

        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(self.backbone.num_channels[-2], hidden_dim*2, kernel_size=1), 
              nn.GroupNorm(32, hidden_dim*2),
            )
        '''2021.12.11 det_token을 qahoi같이 hoi embedding,pos embedding(anchor)로 나누기 위해 hidden_dim 
           두배로 늘림'''


        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )
        
        # reduce det_hoi channel to 1
        self.det_hoi_proj4 = nn.Conv2d(4,1,kernel_size=1) #2023.12.11
        self.det_hoi_proj3 = nn.Conv2d(3,1,kernel_size=1) #2023.12.11
        self.det_hoi_proj2 = nn.Conv2d(2,1,kernel_size=1) #2023.12.11

 
        # det_hoi와 det_pos를 합치는 proj #2023.12.11
        #self.hoi_pos_concat_proj = nn.Conv2d(2,1,kernel_size=1)

        self.hoi_pos_proj = nn.Conv2d(1,1,kernel_size=1)

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_obj_classes) * bias_value
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value #hoi
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0) #hoi
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0) #hoi

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        ############

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0) #hoi
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            self.transformer.decoder.sub_bbox_embed = None

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])
        ############

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])


    def forward(self, samples: NestedTensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors # RGB input
        mask = samples.mask # padding mask

        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        features, det_tgt, det_pos = self.backbone(x, mask)

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        #print('feature before epff:',features[0].shape,'num of feature:',len(features)) 
        #torch.Size([4, 96, x, y]) num of feature: 4
        #print('det_tgt:',det_tgt.shape) # torch.Size([4, 150, 256*2])
        #print('det_pos:',det_pos.shape) # torch.Size([1, 150, 256])

        det_hoi,det_tgt = torch.chunk(det_tgt, 2, axis=2) #det_hoi = torch.Size([1, 150, 256])
        #print('det_tgt:',det_tgt.shape)
        #print('det_hoi:',det_hoi.shape) # torch.Size([4, 150, 256])
        
        if det_hoi.shape[0] == 4:
            det_hoi = self.det_hoi_proj4(det_hoi) 
        elif det_hoi.shape[0] == 3:
            det_hoi = self.det_hoi_proj3(det_hoi)
        elif det_hoi.shape[0] == 2:
            det_hoi = self.det_hoi_proj2(det_hoi)
        
        #print('det_hoi2:',det_hoi.shape) # torch.Size([1, 150, 256])
        det_hoi_pos = det_hoi + det_pos
        #det_hoi_pos = torch.concat((det_hoi,det_pos),dim=0) # torch.Size([2, 150, 256])
        #det_hoi_pos = self.hoi_pos_concat_proj(det_hoi_pos) # torch.Size([1, 150, 256])


        # [PATCH] token projection
        shapes = []
        for l, src in enumerate(features):
            shapes.append(src.shape[-2:])

        srcs = []
        if self.fusion is None:
            for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))
        else:
            # EPFF (multi-scale fusion) is used if fusion is activated
            srcs = self.fusion(features)
        #print('feature after epff:',srcs[0].shape,'num of feature:',len(srcs)) 
        #feature after epff: torch.Size([4, 256, x, y]) num of feature: 4

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = []
        outputs_coords = []
        outputs_sub_coords = []
        outputs_verb_classes = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = \
          self.transformer(srcs, masks, det_tgt, det_hoi_pos)

        '''outputs_obj_class = self.obj_class_embed(hs) #QPIC hoi output
        outputs_verb_class = self.verb_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)'''
        
        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_verb_class = self.verb_class_embed[lvl](hs[lvl]) #hoi
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp2 =self.sub_bbox_embed[lvl](hs[lvl]) #hoi
            if reference.shape[-1] == 4:
                tmp += reference
                tmp2 += reference #hoi
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp2[..., :2] += reference #hoi

            outputs_coord = tmp.sigmoid()
            outputs_sub_coord = tmp2.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_verb_classes.append(outputs_verb_class)
            outputs_coords.append(outputs_coord)
            outputs_sub_coords.append(outputs_sub_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_verb_class = torch.stack(outputs_verb_classes)


        ############ Added for UQR
        outputs_vector = None
        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)
        ############

        # final prediction is made the last decoding layer
        #out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out = {'pred_obj_logits': outputs_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_coord[-1]}
        
        ############ Added for UQR
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        ############

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_coord,outputs_vector) #hoi 수정 12/04


        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
            outputs_iou = torch.stack(outputs_ious)
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        return out

    @torch.jit.unused #hoi 수정 12/04
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if outputs_vector is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'pred_vectors':e}
                for a, b, c, d ,e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1],outputs_vector[:-1])]


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
      super().__init__()
      self.num_layers = num_layers
      h = [hidden_dim] * (num_layers - 1)
      self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
      return x

def build(args, is_teacher=False):

    # distillation is deprecated
    if is_teacher:
        print('Token Distillation is deprecated in this version. Please use the previous version of ViDT.')
    assert is_teacher is False
    # a teacher model for distilation
    #if is_teacher:
    #    return build_teacher(args)
    ############################
    
    if args.dataset_file == 'vcoco':
        num_obj_classes = 81
        num_verb_classes = 29


    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    if args.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_base_win7_22k':
        backbone, hidden_dim = swin_base_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_large_win7_22k':
        backbone, hidden_dim = swin_large_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_tiny':
        backbone, hidden_dim = coat_lite_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_mini':
        backbone, hidden_dim = coat_lite_mini(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_small':
        backbone, hidden_dim = coat_lite_small(pretrained=args.pre_trained)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    epff = None
    if args.epff:
        epff = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim)
    
    if args.hoi: #hoi
        deform_transformers = build_deforamble_transformer(args) #build_transformer(args)

        # Added for UQR module
        if args.with_vector:
            processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)

        model = HOIDetector(
            backbone,
            deform_transformers,
            num_queries=args.det_token_num,
            num_obj_classes=args.num_obj_classes, num_verb_classes = args.num_verb_classes,
            # two essential techniques used in ViDT
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            # an epff module for ViDT+
            epff=epff,
            # an UQR module for ViDT+
            with_vector=args.with_vector,
            processor_dct=processor_dct if args.with_vector else None,
            # two additional losses for VIDT+
            iou_aware=args.iou_aware,
            token_label=args.token_label,
            vector_hidden_dim=args.vector_hidden_dim,
            # distil
            distil=False if args.distil_model is None else True,
        )

    
    else:
        deform_transformers = build_deforamble_transformer(args)

        # Added for UQR module
        if args.with_vector:
            processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)

        model = Detector(
            backbone,
            deform_transformers,
            num_classes=num_classes,
            num_queries=args.det_token_num,
            # two essential techniques used in ViDT
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            # an epff module for ViDT+
            epff=epff,
            # an UQR module for ViDT+
            with_vector=args.with_vector,
            processor_dct=processor_dct if args.with_vector else None,
            # two additional losses for VIDT+
            iou_aware=args.iou_aware,
            token_label=args.token_label,
            vector_hidden_dim=args.vector_hidden_dim,
            # distil
            distil=False if args.distil_model is None else True,
        )

    matcher = build_matcher(args)
    weight_dict = {}
    
    if args.hoi: #hoi
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
        weight_dict['loss_verb_ce'] = args.verb_loss_coef
        weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_giou'] = args.giou_loss_coef
    
    else:
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef

    ##
    if args.iou_aware:
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    # For UQR module
    if args.masks:
        weight_dict["loss_vector"] = 1

    if args.distil_model is not None:
        weight_dict['loss_distil'] = args.distil_loss_coef

    # aux decoding loss
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1 + 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    if args.hoi: #hoi
        losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
        criterion = SetCriterionHOI(args.num_obj_classes, args.det_token_num, args.num_verb_classes, matcher=matcher,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                    verb_loss_type=args.verb_loss_type)
    else:

        losses = ['labels', 'boxes', 'cardinality']
        if args.iou_aware:
            losses += ['iouaware']

        # For UQR
        if args.masks:
            losses += ["masks"]

        # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
        criterion = SetCriterion(num_classes, matcher, weight_dict, losses,
                                focal_alpha=args.focal_alpha,
                                # For UQR
                                with_vector=args.with_vector,
                                processor_dct=processor_dct if args.with_vector else None,
                                vector_loss_coef=args.vector_loss_coef,
                                no_vector_loss_norm=args.no_vector_loss_norm,
                                vector_start_stage=args.vector_start_stage)
    
    criterion.to(device)
    
    if args.hoi: #hoi
        postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}
    else:
        # HER -=> post도 고쳐야함.
        postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}
        if args.masks:
            postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)

    return model, criterion, postprocessors

''' deprecated
def build_teacher(args):

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    if args.distil_model == 'vidt_nano':
        backbone, hidden_dim = swin_nano()
    elif args.distil_model == 'vidt_tiny':
        backbone, hidden_dim = swin_tiny()
    elif args.distil_model == 'vidt_small':
        backbone, hidden_dim = swin_small()
    elif args.distil_model == 'vidt_base':
        backbone, hidden_dim = swin_base_win7()
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    if args.cross_scale_fusion:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim, all=args.cross_all_out)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    return model
'''