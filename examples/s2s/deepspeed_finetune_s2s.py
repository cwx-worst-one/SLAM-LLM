from slam_llm.pipeline.finetune_deepspeed import main as train
from slam_llm.utils.deepspeed_utils import deepspeed_main_wrapper

import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from s2s_config import ModelConfig, TrainConfig, DataConfig, LogConfig


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    deepspeed_config: str = field(default="examples/s2s/conf/ds_config.json", metadata={"help": "The path to deepspeed config"})


@deepspeed_main_wrapper(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    # kwargs = to_plain_list(cfg)
    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    train(kwargs)


if __name__ == "__main__":
    main_hydra()
