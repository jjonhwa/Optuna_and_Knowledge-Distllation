import optuna
from src.subset_dataloader import sub_create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, Tuple
import argparse
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info

import random
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_categorical("epochs", [50, 100, 150, 200])
    img_size = trial.suggest_categorical("img_size", [96, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=32, high=128, step=32)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # lr = trial.suggest_float('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    optim = trial.suggest_categorical("optimizer", ["MomentumSGD", "AdamW"])
    scheduler = trial.suggest_categorical("scheduler", ["linear", "cycle"])

    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
        "INIT_LR": lr,
        "optimizer": optim,
        "scheduler": scheduler,
    }


def objective(
    trial: optuna.trial.Trial, device, fp16, args
) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config = read_yaml(cfg=args.model_config_path)
    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    hyperparams = search_hyperparam(trial)
    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = "../data"
    data_config["DATASET"] = "TACO"
    data_config["AUG_TRAIN"] = "simple_augment_train"  # default: randaugment_train
    data_config["AUG_TEST"] = "simple_augment_test"
    if data_config["AUG_TRAIN"] == "simple_augment_train":
        data_config["AUG_TRAIN_PARAMS"] = None
    elif data_config["AUG_TRAIN"] == "randaugment_train":
        data_config["AUG_TRAIN_PARAMS"] = {
            "n_select": hyperparams["n_select"],
        }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    data_config["INIT_LR"] = hyperparams["INIT_LR"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["optimizer"] = hyperparams["optimizer"]
    data_config["scheduler"] = hyperparams["scheduler"]
    """이부분이 config를 저장하는 부분입니다. 위의 lr,fp16,epochs는 원래 함수에는 없지만
    config를 바로 사용할 수 있게 추가했습니다."""
    if not os.path.exists(args.save_config_path):
        os.makedirs(args.save_config_path)

    k = 1
    model_file = f"model_{k}.yaml"
    model_file_name = os.path.join(args.save_config_path, model_file)
    while os.path.exists(model_file_name):
        k += 1
        model_file = f"model_{k}.yaml"
        model_file_name = os.path.join(args.save_config_path, model_file)

    """model config와 data config를 저장"""
    with open(model_file_name, "w") as outfile:
        yaml.dump(model_config, outfile)

    data_file = f"data_{k}.yaml"
    data_file_name = os.path.join(args.save_config_path, data_file)
    with open(data_file_name, "w") as outfile:
        yaml.dump(data_config, outfile)

    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    model_info(model, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(
        data_config, subset_ratio=args.subset_ratio
    )

    # criterion = nn.CrossEntropyLoss()
    if data_config["optimizer"] == "MomentumSGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
        )
    elif data_config["optimizer"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=data_config["INIT_LR"], eps=1e-8)

    if data_config["scheduler"] == "linear":
        t_total = len(train_loader) * data_config["EPOCHS"]
        warmup_steps = int(t_total * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    elif data_config["scheduler"] == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=data_config["INIT_LR"],
            steps_per_epoch=len(train_loader),
            epochs=hyperparams["EPOCHS"],
            pct_start=0.05,
        )
    # optimizer = torch.optim.SGD(model.parameters(), lr=data_config["INIT_LR"])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=data_config["INIT_LR"],
    #     steps_per_epoch=len(train_loader),
    #     epochs=hyperparams["EPOCHS"],
    #     pct_start=0.05,
    # )

    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    # scaler = torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=args.save_path,
        scaler=scaler,
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    model_info(model, verbose=True)
    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(
    optuna_study: optuna.study.Study, args
) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = args.threshold  # default 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(args, storage, fp16: bool = False):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, device, fp16, args), n_trials=args.n_trials
    )
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study, args)
    print(best_trial)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("search_results.csv", index=False)


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser(description="Optuna HyperParameter Tuner.")
    parser.add_argument(
        "--save_config_path",
        default="./search_hyperparameter",
        type=str,
        help="path to save hyper-parameter configuration",
    )
    parser.add_argument("--save_path", default=".", type=str, help="path to save model")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--storage", default="", type=str, help="Optuna database storage path."
    )
    parser.add_argument("--model_config_path", default="./search_model/model_1.yaml")
    parser.add_argument(
        "--subset_ratio",
        default=0.0,
        type=float,
        help="The ratio of datasets to be used for tuning.",
    )
    parser.add_argument(
        "--threshold",
        default=0.7,
        type=float,
        help="The condition of the trial with an acuracy exceeding the threshold is obtained.",
    )
    parser.add_argument(
        "--n_trials", default=30, type=int, help="How many times will you try optuna?"
    )
    args = parser.parse_args()
    tune(args, storage=args.storage if args.storage != "" else None)
