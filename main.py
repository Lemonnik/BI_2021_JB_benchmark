import os
import time

import hydra
import torch
from omegaconf import DictConfig

from utils.learning_cycle import run_model
from utils.model_and_preprocess_selection import select_model, preprocess_dataset
from utils.select_dataset import select_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    """ Dataset selection and loading """
    dataset = select_dataset(cfg.base_path, cfg.dataset)
    # atm. dataset contains all information (train and test)
    # we can decide what part should be returned by changing ``mode``

    # wandb.init(project="DTI-prediction")

    """ Path for model checkpoints """
    # TODO: save model checkpoints
    curr_date = time.strftime('%d-%m-%Y')
    curr_time = time.strftime('%H-%M-%S')
    model_save_path = os.path.join(cfg.base_path, curr_date, curr_time)

    """ SEED """
    torch.manual_seed(cfg.seed)

    """ Preprocess selection """
    print('--- Data Preparation ---')
    dataset = preprocess_dataset(cfg.model, dataset)
    # TODO: how about remove protection from method? So we could save our processed dataset after preprocessing
    dataset._save_processed_data()
    print('--- Finished ---\n')

    """ Device selection """
    print('--- Device Preparation ---')
    device = "cpu"
    if cfg.gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Running models on {device}.\n')

    """ Model selection """
    print('--- Model Preparation ---')
    model = select_model(cfg.model, dataset)
    model = model.to(device)
    print(f'--- Using model {type(model).__name__} ---\n')

    """ Run model """
    # TODO: choose default batch_size/n_epoch/etc if parameter is not stated in cfg.model
    run_model(model=model,
              dataset=dataset,
              device=device,
              **cfg.model)
              # lr=cfg.model.lr,
              # lr_decay=cfg.model.lr_decay,
              # decay_interval=cfg.model.decay_interval,
              # print_every=cfg.print_every,
              # batch_size=cfg.model.batch_size,
              # n_epochs=cfg.model.epoch)


if __name__ == '__main__':
    main()
