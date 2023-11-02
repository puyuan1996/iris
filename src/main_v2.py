import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose, initialize
from trainer_v2 import Trainer


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    overrides_dict = {
        # 'env': {'train': {'id': 'BreakoutNoFrameskip-v4'}},
        'env': {'train': {'id': 'PongNoFrameskip-v4'}},
        'common': {'device': 'cpu', 'seed': 0, 'resume': False},
        'wandb': {'mode': 'disabled'},
        # TODO(pu): debug config
        'collection': {'train': {'num_episodes_to_save': 10, 'num_envs': 1, 'config': {"num_steps": 20}}, 'test': {'num_episodes_to_save': 10}},
        'evaluation': {'actor_critic': {'num_episodes_to_save': 64}},
        # TODO(pu): debug config
        'training': {'should': True, 'tokenizer': {"start_after_epochs": 1, "steps_per_epoch": 2, "batch_num_samples": 4}, 'world_model': {"start_after_epochs": 1, "steps_per_epoch": 2, "batch_num_samples": 4}, 'actor_critic': {"start_after_epochs": 1, "steps_per_epoch": 2, "batch_num_samples": 4}, },
    }

    overrides_dict = flatten(overrides_dict)
    overrides_list = [f"{k}={v}" for k, v in overrides_dict.items()]

    GlobalHydra.instance().clear()
    with initialize(config_path="../config"):
        cfg = compose(config_name="trainer", overrides=overrides_list)

        main(cfg)
