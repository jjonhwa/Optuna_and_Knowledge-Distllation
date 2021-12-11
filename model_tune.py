import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import create_dataloader
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model, ModelParser
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from optuna.integration.wandb import WeightsAndBiasesCallback

import argparse
import os
import yaml

import numpy as np
import random
import wandb


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


def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2  # 5(224 example): 224, 112, 56, 28, 14, 7
    n_layers = trial.suggest_int("n_layers", 8, 12)
    stride = 1
    input_max = 64
    imput_min = 32
    module_info = {}
    ### 몇개의 레이어를 쌓을지도 search하게 했습니다.
    for i in range(n_layers):
        out_channel = trial.suggest_int(f"{i+1}units", imput_min, input_max)
        block = trial.suggest_categorical(
            f"m{i+1}", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3"]
        )
        repeat = trial.suggest_int(f"m{i+1}/repeat", 1, 5)
        m_stride = trial.suggest_int(f"m{i+1}/stride", low=1, high=UPPER_STRIDE)
        if m_stride == 2:
            stride += 1
        if n_stride == 0:
            m_stride = 2

        if block == "Conv":
            activation = trial.suggest_categorical(
                f"m{i+1}/activation", ["ReLU", "Hardswish"]
            )
            # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
            model_args = [out_channel, 3, m_stride, None, 1, activation]
        elif block == "DWConv":
            activation = trial.suggest_categorical(
                f"m{i+1}/activation", ["ReLU", "Hardswish"]
            )
            # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
            model_args = [out_channel, 3, 1, None, activation]
        elif block == "InvertedResidualv2":
            c = trial.suggest_int(
                f"m{i+1}/v2_c", low=imput_min, high=input_max, step=16
            )
            t = trial.suggest_int(f"m{i+1}/v2_t", low=1, high=4)
            model_args = [c, t, m_stride]
        elif block == "InvertedResidualv3":
            kernel = trial.suggest_int(f"m{i+1}/kernel_size", low=3, high=5, step=2)
            t = round(
                trial.suggest_float(f"m{i+1}/v3_t", low=1.0, high=6.0, step=0.1), 1
            )
            c = trial.suggest_int(f"m{i+1}/v3_c", low=imput_min, high=input_max, step=8)
            se = trial.suggest_categorical(f"m{i+1}/v3_se", [0, 1])
            hs = trial.suggest_categorical(f"m{i+1}/v3_hs", [0, 1])
            # k t c SE HS s
            model_args = [kernel, t, c, se, hs, m_stride]

        in_features = out_channel
        model.append([repeat, block, model_args])
        if i % 2:
            input_max *= 2
            input_max = min(input_max, 160)
        module_info[f"block{i+1}"] = {"type": block, "repeat": repeat, "stride": stride}
    # last layer
    last_dim = trial.suggest_int("last_dim", low=128, high=1024, step=128)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [6, 1, 1, None, 1, None]])
    return model, module_info


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
    model_config: Dict[str, Any] = {}
    model_config["input_channel"] = 3
    # img_size = trial.suggest_categorical("img_asize", [32, 64, 128])

    model_config["depth_multiple"] = trial.suggest_categorical(
        "depth_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["width_multiple"] = trial.suggest_categorical(
        "width_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["backbone"], module_info = search_model(trial)
    hyperparams = {
        "EPOCHS": args.epochs,
        "IMG_SIZE": args.img_size,
        "n_select": args.n_select,
        "BATCH_SIZE": args.batch_size,
    }
    model_config["INPUT_SIZE"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]

    model = Model(model_config, verbose=True)

    model_parser = ModelParser(model_config, verbose=True)
    parsed_model = model_parser._parse_model()
    n_param = sum([x.numel() for x in parsed_model.parameters()])
    # n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

    model.to(device)
    model.model.to(device)

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
    data_config["INIT_LR"] = 0.1
    data_config["EPOCHS"] = hyperparams["EPOCHS"]

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

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

    # wandb.init(
    #     project = 'optimization'
    # )

    model_name = f"trial_{trial.number}: {len(list(parsed_model.modules())):,d} layers, {n_param:,d} parameters"
    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        model_name=model_name,
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
    threshold = args.threshold  # default = 0.7
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

    wandb_kwargs = {"project": "optimization", "entity": "jjonhwa"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, device, fp16, args),
        n_trials=args.n_trials,
        callbacks=[wandbc],
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

    parser = argparse.ArgumentParser(description="Optuna Model Tuner.")
    parser.add_argument(
        "--save_config_path",
        default="./search_model",
        type=str,
        help="path to save model configuration",
    )
    parser.add_argument("--save_path", default=".", type=str, help="path to save model")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--storage", default="", type=str, help="Optuna database storage path."
    )

    parser.add_argument("--epochs", default=50, type=int, help="# of epochs to train.")
    parser.add_argument(
        "--img_size", default=96, type=int, help="The size of the image to be resized."
    )
    parser.add_argument(
        "--n_select",
        default=2,
        type=int,
        help="Number of times to apply RandAugmentation",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Size of batch_size to be used for training.",
    )
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

