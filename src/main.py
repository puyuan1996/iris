import hydra
from omegaconf import DictConfig

from trainer_v2 import Trainer


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

# python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cpu wandb.mode=disabled
