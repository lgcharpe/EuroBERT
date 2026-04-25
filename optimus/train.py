import contextlib
import os
import fire
import random
import torch
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing

from optimus.trainer.configuration.configs import Config
from optimus.trainer.data import Data, patch_spanner
from optimus.trainer.distributed import Distributed
from optimus.trainer.model.load import load_model, load_tokenizer
from optimus.trainer.pretrain import Pretrain


def set_global_seed(base_seed: int, rank: int = 0) -> int:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@torch.distributed.elastic.multiprocessing.errors.record
def main(**kwargs):
    # Pin GPU before anything else can initialise CUDA on the wrong device.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Load configurations
    config = Config(**kwargs)

    # Distributed training setup
    distributed = None
    if config.use_ddp or config.use_fsdp:
        distributed = Distributed(config)

    rank = config.system.rank if (config.use_ddp or config.use_fsdp) else 0
    actual_seed = set_global_seed(config.train.seed, rank=rank)
    config.log_print(f"Global RNG seed set to {actual_seed}.", main_only=False)

    # Load/set model and get tokenizer.
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    if distributed:
        dist.barrier()
        config.log_print("Model and tokenizer loaded on all rank.")

    # Setup model for distributed training
    if config.use_fsdp:
        config.log_print("Shared model training with FSDP.")
        model = distributed.fsdp_setup_model(model)
    elif config.use_ddp:
        config.log_print("Distributed model training with DDP.")
        model = distributed.ddp_setup_model(model)

    patch_spanner()
    if distributed:
        dist.barrier()
    config.log_print("Mosaic ML Streaming spanner patched successfully.")

    wandb_run = None
    if config.train.wandb and config.is_main_process:
        import wandb

        wandb_run = wandb.init(
            entity=config.train.wandb_entity,
            project=config.train.project_name,
            name=config.train.run_name,
            id=config.train.wandb_id,
            config=config.__dict__,
            resume="allow",
            save_code=True,
        )

    # Load data
    data = Data(config, tokenizer)
    if distributed:
        dist.barrier()
    config.log_print("Data loaded.")

    # Train model
    pretrain = Pretrain(model, data, distributed, config, wandb_run)
    pretrain.train()

    # Cleanup distributed training
    if distributed:
        distributed.cleanup()
    config.log_print("Training completed successfully.")

    if wandb_run is not None:
        wandb_run.finish()

    exit(0)


if __name__ == "__main__":
    fire.Fire(main)
