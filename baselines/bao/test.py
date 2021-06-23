import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from fewshot import make_challenge
from copy import deepcopy
logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='test')
def main(cfg: DictConfig):
    cfg2 = deepcopy(cfg)
    cfg2.model = None
    model = hydra.utils.instantiate(
        cfg.model,
        args=OmegaConf.create(OmegaConf.to_container(cfg2)),
    )
    evaluator = make_challenge(cfg.challenge, ignore_verification=cfg.ignore_verification)
    evaluator.save_model_predictions(model)


if __name__ == '__main__':
    main()
