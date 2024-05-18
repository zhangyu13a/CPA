from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

# from configs.common.coco_loader_lsj import dataloader#zy:dataloader comes from coco_loader_lsj
import detectron2.data.transforms as T
from detectron2.config import LazyConfig

# model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model = LazyConfig.load("F:\Papers\Attack_ViT\detectron2-ViTDet\configs\common\mask_rcnn_vitdet.py").model
#zy:将detectron2系统定义的model copy至本地，用LazyConfig.load读取
# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375
#zy:ref build_lr_scheduler to set lr_multiplier：F:\detectron2-New\detectron2\solver\build.py
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}


#zy:dataloader comes from coco_loader_lsj;
# Data using LSJ
image_size = 608#1024#1024 608 zy
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 64
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]
