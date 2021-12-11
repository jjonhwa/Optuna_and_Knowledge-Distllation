import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.kd_trainer import KDTrainer
from src.utils.common import get_label_counts, read_yaml
from transformers import AdamW
import timm


def train(
    args,
    teacher_path: str,
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    student_model_config = read_yaml(cfg=args.student_model_config)

    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(student_model_config, f, default_flow_style=False)

    #### Calling the pre-trained student model ####
    student_model_instance = Model(student_model_config, verbose=True)
    student_state_dict = torch.load(args.student_model, map_location=device)
    student_model_instance.model.load_state_dict(student_state_dict)

    ####Change Model Name for timm Models ####
    teacher_state_dict = torch.load(teacher_path)
    teacher_model = timm.create_model(
        args.teacher_model_name, pretrained=False, num_classes=args.num_classes
    )
    teacher_model.load_state_dict(teacher_state_dict)

    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")

    if os.path.isfile(model_path):
        student_model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    student_model_instance.model.to(device)
    teacher_model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = AdamW(
        student_model_instance.model.parameters(), lr=data_config["INIT_LR"], eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
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

    # Create trainer
    trainer = KDTrainer(
        student=student_model_instance.model,
        teacher=teacher_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    student_model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=student_model_instance.model,
        test_dataloader=val_dl if val_dl else test_dl,
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--student_model_config",
        default="exp/student/model.yml",
        type=str,
        help="student model configuration",
    )
    parser.add_argument(
        "--student_model",
        default="exp/student/best.pt",
        type=str,
        help="student model weight",
    )
    parser.add_argument(
        "--teacher_model",
        default="exp/teacher_timm/best.pt",
        type=str,
        help="teacher model weight",
    )
    parser.add_argument(
        "--teacher_model_name", default="resnet18", type=str, help="teacher model name",
    )
    parser.add_argument(
        "--num_classes", default=6, type=int, help="The total number of labels."
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    data_config = read_yaml(cfg=args.data)
    data_config["DATA_PATH"] = os.environ.get(
        "SM_CHANNEL_TRAIN", data_config["DATA_PATH"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", "kd_timm"))

    if os.path.exists(log_dir):
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + "/best.pt"))
        new_log_dir = (
            os.path.dirname(log_dir) + "/" + modified.strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        args=args,
        teacher_path=args.teacher_model,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
