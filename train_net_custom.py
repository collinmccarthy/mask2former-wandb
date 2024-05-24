"""
Modified from mask2former.train_net.py
"""

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import logging

from torch.distributed.elastic.multiprocessing.errors import record
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config  # type: ignore


from mask2former import add_maskformer2_config
from train_net import Trainer

from detectron2_plugin.train_net import (
    CustomTrainerMixin,
    setup_loggers,
    maybe_restart_run,
)
from detectron2_plugin.config import add_custom_config, update_custom_config
from detectron2_plugin.engine import launch
from detectron2_plugin.wandb import CustomWandbWriter

logger = logging.getLogger(__name__)


class CustomTrainer(CustomTrainerMixin, Trainer):
    pass


def setup(args):
    """Same as train_net.py, but handle custom config, wandb and setup additional loggers"""
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Add our custom config before merging with opts
    add_custom_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Update our custom config after merging with opts
    update_custom_config(args=args, cfg=cfg)
    cfg.freeze()

    # Restart before default_setup / wandb setup / logger init
    maybe_restart_run(cfg=cfg, args=args)

    default_setup(cfg, args)

    if not args.eval_only:
        CustomWandbWriter.setup_wandb(cfg, args)

    # Use our logger setup method to add new loggers
    setup_loggers(cfg)

    return cfg


@record  # Only used if launched via torchrun
def main(args):
    """Same as train_net.py, but use CustomTrainer w/ CustomTrainerMixin instead of Trainer"""
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

    result = trainer.train()
    logger.info(f"Training finished. Exiting.")
    return result


if __name__ == "__main__":
    launch(main_func=main)
