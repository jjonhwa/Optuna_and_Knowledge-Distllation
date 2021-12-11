# Optuna_and_Knowledge-Distllation
Maximize the performance of lightweight models using optuna and knowledge distllation.

Writing...

## Data
Datasets can be used under CC-BY-4.0 copyrights.
```python
!git clone https://github.com/jjonhwa/Optuna_and_Knowledge-Distllation.git

!wget -cq https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000081/data/data.zip
!unzip ./data.zip -d ./data/
```

## Install Requirements
```python
pip install -r requirements.txt
```

## Quick Start

### NAS with Optuna
```python
python model_tune.py --n_trials 50
```

### HyperParameter Search with Optuna
```python
python hp_tune.py --n_trials 50
```

### Knowledge Distillation
```python
python train_student.py # train student model
python train_teacher_from_timm.py # train teacher model using the timm library.
# or python train_teacher_from_url.py
python kd_for_timm_from_pretrained.py # knowledge distillation
```