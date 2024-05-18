
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

from functools import reduce
import sys
import random
from torchvision.utils import save_image

import warnings
import numbers
import torchvision.transforms.functional as F1
from torchvision.transforms.functional import InterpolationMode
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

#--------myRandomSunFlare-------------#
import random
import math
from albumentations.augmentations.transforms import ImageOnlyTransform
from albumentations.augmentations.functional import  non_rgb_warning, from_float,to_float
import cv2
#-----------------------------------------#
import albumentations as A
from torch.optim.swa_utils import AveragedModel
import itertools
from torch.nn import Module
from copy import deepcopy

 
class AveragingBaseModel(Module):
    def __init__(self, model, cuda=False, avg_fn=None, use_buffers=False):
        super(AveragingBaseModel, self).__init__()
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.module = deepcopy(model)
        self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers
 
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
 
    def update(self, model):
        self_param = itertools.chain(self.module.parameters(), self.module.buffers() if self.use_buffers else [])
        model_param = itertools.chain(model.parameters(), model.buffers() if self.use_buffers else [])
 
        self_param_detached = [p.detach() for p in self_param]
        model_param_detached = [p.detach().to(p_averaged.device) for p, p_averaged in zip(model_param, self_param_detached)]
 
        if self.n_averaged == 0:
            for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                p_averaged.copy_(p_model)
 
        if self.n_averaged > 0:
            for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                n_averaged = self.n_averaged.to(p_averaged.device)
                p_averaged.copy_(self.avg_fn(p_averaged, p_model, n_averaged))
 
        if not self.use_buffers:
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.copy_(b_model.to(b_swa.device).detach())
 
        self.n_averaged += 1

#---------SWA-------------#
def get_swa_avg_fn():
    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)
    return swa_update
 
class SWAModel(AveragingBaseModel):
    def __init__(self, model, cuda = False,use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_swa_avg_fn(), use_buffers=use_buffers)

#---------EMA-------------#
def get_ema_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param
    return ema_update
 
class EMAModel(AveragingBaseModel):
    def __init__(self, model, cuda = False, decay=0.9, use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_ema_avg_fn(decay), use_buffers=use_buffers)

def get_t_adema(alpha=0.9):
    num_averaged = [0]  
    @torch.no_grad()
    def t_adema_update(averaged_param, current_param, num_averageds):
        num_averaged[0] += 1
        decay = alpha * torch.tanh(torch.tensor(num_averaged[0], dtype=torch.float32))
        tadea_update = decay * averaged_param + (1 - decay) * current_param
        return tadea_update
    return t_adema_update
 
class T_ADEMAModel(AveragingBaseModel):
    def __init__(self, model, cuda=False, alpha=0.9, use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_t_adema(alpha), use_buffers=use_buffers)


class myRandomSunFlare(ImageOnlyTransform):
    
    def __init__(
        self,
        flare_roi=(0, 0, 1, 0.5),
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        src_radius=400,
        src_color=(255, 255, 255),
        always_apply=False,
        p=0.5,
    ):
        super(myRandomSunFlare, self).__init__(always_apply, p)

        (
            flare_center_lower_x,
            flare_center_lower_y,
            flare_center_upper_x,
            flare_center_upper_y,
        ) = flare_roi

        if (
            not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
            or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
        ):
            raise ValueError("Invalid flare_roi. Got: {}".format(flare_roi))
        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(
                "Invalid combination of angle_lower nad angle_upper. Got: {}".format((angle_lower, angle_upper))
            )
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            raise ValueError(
                "Invalid combination of num_flare_circles_lower nad num_flare_circles_upper. Got: {}".format(
                    (num_flare_circles_lower, num_flare_circles_upper)
                )
            )

        self.flare_center_lower_x = flare_center_lower_x
        self.flare_center_upper_x = flare_center_upper_x

        self.flare_center_lower_y = flare_center_lower_y
        self.flare_center_upper_y = flare_center_upper_y

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper

        self.src_radius = src_radius
        self.src_color = src_color

    def apply(self, image, flare_center_x=0.5, flare_center_y=0.5, circles=(), **params):
        return my_add_sun_flare(
            image,
            flare_center_x,
            flare_center_y,
            self.src_radius,
            self.src_color,
            circles,
        )
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        angle = 2 * math.pi * random.uniform(self.angle_lower, self.angle_upper)

        flare_center_x = random.uniform(self.flare_center_lower_x, self.flare_center_upper_x)
        flare_center_y = random.uniform(self.flare_center_lower_y, self.flare_center_upper_y)

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        num_circles = random.randint(self.num_flare_circles_lower, self.num_flare_circles_upper)
        num_circles =0
        circles = []

        x = []
        y = []

        for rand_x in range(0, width, 10):
            rand_y = math.tan(angle) * (rand_x - flare_center_x) + flare_center_y
            x.append(rand_x)
            y.append(2 * flare_center_y - rand_y)

        for _i in range(num_circles):
            # alpha = random.uniform(0.05, 0.2)
            alpha = random.uniform(0.6, 0.7)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            g_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            b_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                )
            ]

        return {
            "circles": circles,
            "flare_center_x": flare_center_x,
            "flare_center_y": flare_center_y,
        }

    def get_transform_init_args(self):
        return {
            "flare_roi": (
                self.flare_center_lower_x,
                self.flare_center_lower_y,
                self.flare_center_upper_x,
                self.flare_center_upper_y,
            ),
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }

def my_add_sun_flare(img, flare_center_x, flare_center_y, src_radius, src_color, circles):
    """Add sun flare.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomSunFlareaugmentation".format(input_dtype))

    overlay = img.copy()
    output = img.copy()

    for (alpha, (x, y), rad3, (r_color, g_color, b_color)) in circles:#zy:任务中不需要，circles为[]
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    point = (int(flare_center_x), int(flare_center_y))

    overlay = output.copy()
    num_times = src_radius // 2
    # alpha = np.linspace(0.0, 1, num=num_times)
    alpha = np.linspace(0.0, 0.8, num=num_times)
    rad = np.linspace(5, src_radius, num=num_times)
    for i in range(num_times):
        # cv2.circle(overlay, point, int(rad[i]), src_color, -1)#原始代码
        # # 参数 1.目标图片  2.椭圆圆心  3.长短轴长度  4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色  8.是否填充
        img=cv2.ellipse(overlay, point, (int(rad[i]*1.2),int(rad[i]*0.9)), 60, 0, 360, src_color, -1)#把圆换成椭圆
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)

    image_rgb = output

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb



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

            tx = int(round(torch.empty(1).uniform_(max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(max_dy, max_dy).item()))
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
                fill = [float(fill)] * F1._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F1._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate,self.inv_translate, self.scale,self.inv_scale, self.shear, img_size)
        angle, translations, scale, shear=ret 
        return F1.affine(img, *ret, interpolation=self.interpolation, fill=fill),translations, scale

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

class TotalVariation_3d(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self, mesh, target_face_id):
        super(TotalVariation_3d, self).__init__()
        
        # TODO: deal with different input meshes. the code assume the mesh topology are the same now.
        # TODO: need adv_patch to initialize everything, think a better way to record topology.
        #https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html?highlight=faces_packed_to_edges_packed#
        # Map from packed faces to packed edges. This represents the index of
        # the edge opposite the vertex for each vertex in the face. E.g.
        #
        #         v0
        #         /\
        #        /  \
        #    e1 /    \ e2
        #      /      \
        #     /________\
        #   v2    e0   v1
        #
        # Face (v0, v1, v2) => Edges (e0, e1, e2)
        # 
        # Step 0: get all info of mesh[0]
        FtoE_id = mesh[0].faces_packed_to_edges_packed().cpu()
        '''
        faces_packed_to_edges_packed:
        Get the packed representation of the faces in terms of edges.
        Faces are given by the indices of the three edges in
        the packed representation of the edges.
        Returns:
            tensor of faces of shape (sum(F_n), 3).'''
        EtoV_id = mesh[0].edges_packed().cpu() 
        V = mesh[0].verts_packed()
        num_of_edges = EtoV_id.shape[0]
        num_of_target_faces = len(target_face_id)
        
        # Step 1: Construct (E_n, 2) tensor as opposite face indexing
        EtoF_idx1 = -1 * torch.ones((num_of_edges),dtype=torch.long)
        EtoF_idx2 = -1 * torch.ones((num_of_edges),dtype=torch.long)

        for i in range(num_of_target_faces):
            for each in FtoE_id[target_face_id[i]]:
                if EtoF_idx1[each]==-1:
                    EtoF_idx1[each] = i 
                else:
                    EtoF_idx2[each] = i 
        # remove all edges that does not belong to 
        valid_id = ~((EtoF_idx1 == -1) | (EtoF_idx2 == -1))
        
        EtoF_idx = torch.stack((EtoF_idx1[valid_id],EtoF_idx2[valid_id]), dim=1)
       
        self.face_to_edges_idx = EtoF_idx.cuda()

        # Step 2: Compute edge length
        valid_edge = EtoV_id[valid_id,:]
        self.edge_len = torch.norm(V[valid_edge[:,0],:]-V[valid_edge[:,1],:], dim=1).cuda()

    
    def forward(self, adv_patch):
        

        f1 = adv_patch[self.face_to_edges_idx[:,0],:,:,:]
        f2 = adv_patch[self.face_to_edges_idx[:,1],:,:,:]
        tv = torch.sum(self.edge_len[:,None,None,None] * torch.abs(f1-f2))
        return tv / adv_patch.shape[0]
    
  


