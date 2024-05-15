"""
Modified from mask2former.train_net.py
"""

# fmt: off  # Use fmt: off for easier diff with train_net.py
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import sys
import logging
import os
import weakref
from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config  # type: ignore
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from detectron2.engine.defaults import create_ddp_model
from train_net import Trainer

# Hack: Add our 'ext_repos' dir to path so we can use absolute imports (otherwise get import error)
sys.path.insert(0, str(Path(__file__).parent.parent))
from ext_repos.d2_common.events import (
    CustomWandbWriter,
    CustomJSONWriter,
    CustomCommonMetricPrinter,
    setup_wandb,
)
from ext_repos.d2_common.config import (
    add_wandb_config,
    update_wandb_config,
    update_model_config,
    add_mmdet_config,
)
from ext_repos.d2_common.hooks import update_hooks
from ext_repos.d2_common.train_loop import CustomAMPTrainer, CustomSimpleTrainer


class CustomTrainer(Trainer):
    def __init__(self, cfg):
        """Same as train_net.py but uses CustomAMPTrainer instead of AMPTrainer and
        CustomSimpleTrainer instead of SimpleTrainer
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        # Update: Use CustomAMPTrainer and CustomSimpleTrainer to log epoch / epoch_float
        self._trainer = (
            CustomAMPTrainer if cfg.SOLVER.AMP.ENABLED else CustomSimpleTrainer
        )(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_writers(self):
        """Same as OneFormer.train_net.py but uses cfg.LOGGER.INTERVAL and cfg.WANDB.ENABLED"""
        log_interval = self.cfg.get("LOGGER", {}).get("INTERVAL", 20)
        json_file = os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CustomCommonMetricPrinter(max_iter=self.max_iter, window_size=log_interval),
            CustomJSONWriter(json_file=json_file, window_size=log_interval),
            CustomWandbWriter(enabled=self.cfg.WANDB.ENABLED),
        ]

    def build_hooks(self):
        hooks = super().build_hooks()
        log_interval = self.cfg.get("LOGGER", {}).get("INTERVAL", 20)
        hooks = update_hooks(hooks, log_interval=log_interval)
        return hooks


def setup(args):
    """Same as train_net.py but add wandb/mmdet config items"""
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Added items for mmdet/wandb
    add_wandb_config(cfg)
    add_mmdet_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    update_wandb_config(cfg)
    update_model_config(cfg=cfg, world_size=(args.num_machines * args.num_gpus))

    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only:
        setup_wandb(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )
    return cfg


def main(args):
    """Same as train_net.py, but use CustomTrainer instead of Trainer"""
    cfg = setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CustomTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    """Same as train_net.py"""
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
