import cv2
import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import warnings
import torchvision.transforms.functional as F
import torch.nn.functional as Fun
from torchvision.transforms.functional import InterpolationMode
from collections.abc import Sequence
from typing import Tuple, List, Optional
import numbers
from torch import Tensor
import scipy.stats as st

from functools import partial
from detectron2.modeling.postprocessing import detector_postprocess
class netAttention(object):
    def __init__(self, net,cfg_model):
        self.net = net
        self.cfg_model=cfg_model
        self.feature = list()
        self.gradient = list()
        self.handlers = []
    def _get_grads_hook(self, module, input_grad, output_grad,gamma):
        mask = torch.ones_like(input_grad[0]) * gamma
        return (mask * input_grad[0][:], )

    def _register_hook(self):
        self.feature = list()
        self.gradient = list()
        self.handlers = [] 
        for i, block in enumerate(self.net.backbone.net.blocks._modules.items()):
            hook_grad_norm1=block[1].norm1.register_full_backward_hook( 
                partial(self._get_grads_hook,gamma=0.8) )          
            hook_grad_norm2=block[1].norm2.register_full_backward_hook( 
                partial(self._get_grads_hook,gamma=0.8) )      
            self.handlers.append(hook_grad_norm1) 
            self.handlers.append(hook_grad_norm2) 
            pass
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
            torch.cuda.empty_cache()
        
    def __call__(self, inputs, index=0,retain_graph=True):
        
        img_ori=inputs['image']#.clone()
        img_tmp=[img.mul(255).clamp_(0, 255) for img in img_ori ]   
        images = [(x - self.net.pixel_mean) / self.net.pixel_std for x in img_tmp]
        images=torch.stack(images)
        image_sizes=[(img_ori.shape[-2], img_ori.shape[-1])]*img_ori.shape[0]
        features = self.net.backbone(images)     
        in_features=self.net.proposal_generator.in_features#head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            
        #detectron2\modeling\proposal_generator\rpn.py
        features_tmp = [features[f] for f in in_features]
        anchors = self.net.proposal_generator.anchor_generator(features_tmp)
        pred_objectness_logits, pred_anchor_deltas =self.net.proposal_generator.rpn_head(features_tmp)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        box_dim=self.net.proposal_generator.anchor_generator.box_dim
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        proposals = self.net.proposal_generator.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes
        )        
        #---------------------------------------#

        results, _ = self.net.roi_heads(images, features, proposals, None)
  
        processed_results = []
        for results_per_image,  image_size in zip(results,  image_sizes):
            height=img_ori.shape[-2]
            width=img_ori.shape[-1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
    
        pred_score_tensor=torch.Tensor().cuda()
        pred_BB_tensor=torch.Tensor().cuda()
        for i in range(len(processed_results)):
            if len(processed_results[i]['instances'])>0:
                max_score=processed_results[i]['instances'].scores.max()
                if max_score>0.5: 
                    mask=processed_results[i]['instances'].scores>0.5
                    pred_score_greater_than_thresh=processed_results[i]['instances'].scores[mask]
                    pred_score_tensor_tmp=pred_score_greater_than_thresh.mean().unsqueeze(0)
                    pred_BB_greater_than_thresh=processed_results[i]['instances'].pred_boxes[0].tensor
                else: 
                    pred_score_tensor_tmp=processed_results[i]['instances'].scores.max().unsqueeze(0)
                    pred_BB_greater_than_thresh=processed_results[i]['instances'].pred_boxes[0].tensor 

                pred_score_tensor=torch.cat([pred_score_tensor,pred_score_tensor_tmp],0)
                pred_BB_tensor=torch.cat([pred_BB_tensor,pred_BB_greater_than_thresh],0)
     
        scores=pred_score_tensor

        #IOU
        file_name_BS=inputs['clean_img_path']
        mean_ious=0
        for image_i, pred_BB_i in enumerate(pred_BB_tensor):#pred_BB
            image_pred_BB=pred_BB_i
            GT_path="TrainingGT/ground-truth/"+file_name_BS[image_i].split('\\')[-1].split('.npz')[0]+".txt"
            lines = file_lines_to_list(GT_path)
            for line in lines:
                try:
                    tmp_class_name,left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = int(line_split[-1])
                    right = int(line_split[-2])
                    top = int(line_split[-3])
                    left = int(line_split[-4])                
                GT_bbox = [int(left), int(top), int(right), int(bottom)]

            ious = bbox_iou_V2(torch.tensor(GT_bbox).cuda().unsqueeze(0), image_pred_BB.unsqueeze(0))
            mean_iou=torch.mean(ious)
            mean_iou_tmp=mean_iou.unsqueeze(0)
            if image_i==0:
                mean_ious=mean_iou_tmp
            else:
                mean_ious=torch.cat((mean_ious,mean_iou_tmp),0)

        return scores,mean_ious 
        

def bbox_iou_V2(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

"""
Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

class myRandomAffine(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
            If input is PIL Image, the options is only available for ``Pillow>=5.0.0``.
        fillcolor (sequence or number, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``fill`` parameter instead.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self, degrees, translate=None,inv_translate=None,scale=None,inv_scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0,
        fillcolor=None, resample=None
    ):
        super().__init__()

        if fillcolor is not None:
            warnings.warn(
                "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
            )
            fill = fillcolor

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2, ))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        if inv_translate is not None:
            _check_sequence_input(inv_translate, "inv_translate", req_sizes=(2, ))
        self.inv_translate = inv_translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2, ))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        if inv_scale is not None:
            _check_sequence_input(inv_scale, "inv_scale", req_sizes=(2, ))
        self.inv_scale = inv_scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            inv_translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            inv_scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        elif inv_translate is not None:
            translations=inv_translate
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        elif inv_scale_ranges is not None:
            scale=inv_scale_ranges
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate,self.inv_translate, self.scale,self.inv_scale, self.shear, img_size)
        angle, translations, scale, shear=ret 
        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill),translations, scale

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.interpolation != InterpolationMode.NEAREST:
            s += ', interpolation={interpolation}'
        if self.fill != 0:
            s += ', fill={fill}'
        s += ')'
        d = dict(self.__dict__)
        d['interpolation'] = self.interpolation.value
        return s.format(name=self.__class__.__name__, **d)

def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))