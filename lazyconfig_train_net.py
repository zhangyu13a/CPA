#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import os
import torch,gc
import time

from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, ImageList
CLASS_NAMES =["car"] 
DATASET_ROOT = r'F:\detectron2-main\carla\VOCdevkit\VOC2007CocoFormatData'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'test')

TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train_CatID0.json')
#VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_test_CatID0.json')


def plain_register_dataset():
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  
                                                    evaluator_type='coco', 
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, 
                                                evaluator_type='coco',
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)



logger = logging.getLogger("detectron2")


class MyTrainer(SimpleTrainer):
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
    ):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
        """
        super().__init__(model, data_loader, optimizer)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)

        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()
        for i in range(5):
            gc.collect()
            torch.cuda.empty_cache()
        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
        # del data,losses,loss_dict,features,proposals, proposal_losses,detector_losses
        torch.cuda.empty_cache()

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)#

    train_loader = instantiate(cfg.dataloader.train)# 

    model = create_ddp_model(model, **cfg.train.ddp)#
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    # trainer = (AMPTrainer if cfg.train.amp.enabled else MyTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def setup(args):
    # cfg = get_cfg()
    args.config_file='configs\COCO\mask_rcnn_vitdet_b_100ep.py'
    cfg = LazyConfig.load(args.config_file)# 
    cfg = LazyConfig.apply_overrides(cfg, args.opts)# 
  
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.dataloader.train.total_batch_size=1#
    cfg.dataloader.train.num_workers=4
    cfg.dataloader.test.num_workers=4
    cfg.dataloader.train.dataset=L(get_detection_dataset_dicts)(names="coco_my_train")  
    cfg.dataloader.test.dataset=L(get_detection_dataset_dicts)(names="coco_my_val", filter_empty=False)

    cfg.dataloader.train.mapper.use_instance_mask=True

    cfg.model.roi_heads.num_classes = 1

    cfg.train.init_checkpoint=r'trained_weights\ViTDet_carla\model_final.pth'#

    ITERS_IN_ONE_EPOCH = int(1080 / cfg.dataloader.train.total_batch_size) 
    cfg.train.eval_period = ITERS_IN_ONE_EPOCH
    cfg.train.log_period=20
    cfg.train.max_iter = (ITERS_IN_ONE_EPOCH * 12) - 1 
    cfg.train.checkpointer.max_to_keep=100
    cfg.train.checkpointer.period = ITERS_IN_ONE_EPOCH
    cfg.train.amp.enabled = False 
    
    # cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    default_setup(cfg, args)
    plain_register_dataset()
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
