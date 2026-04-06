from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import src.training.trainer as trainer_module
from src.config.schema import TrainConfig
from src.models.base import BaseClassifier, PreprocessingConfig
from src.training.trainer import Trainer, _build_metadata_head


class DummyModel(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
        self._head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self._head(features)
        self._validate_output_shape(logits, x.shape[0])
        return logits

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head(self) -> nn.Module:
        return self._head

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.view(x.shape[0], -1)
        return self._backbone(flattened)

    @property
    def feature_dim(self) -> int:
        return 8

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            input_size=2,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            crop_strategy="pad_to_square",
        )


class DummyDataset(Dataset):
    def __init__(self, use_metadata: bool = False) -> None:
        self.use_metadata = use_metadata
        self.labels = np.array([0.0, 1.0], dtype=np.float32)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "image": torch.full((3, 2, 2), fill_value=float(idx), dtype=torch.float32),
            "label": torch.tensor([self.labels[idx]], dtype=torch.float32),
            "patient_id": f"patient_{idx}",
            "image_path": f"image_{idx}.png",
        }
        if self.use_metadata:
            sample["metadata"] = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return sample


class DummyLogger:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.epochs: list[dict] = []
        self.summary: dict | None = None
        self.finished = False

    def log_epoch(self, **kwargs) -> None:
        self.epochs.append(kwargs)

    def log_summary(self, results: dict) -> None:
        self.summary = results

    def finish(self) -> None:
        self.finished = True


def _make_config(tmp_path, **overrides) -> TrainConfig:
    base = {
        "dataset": "cimt",
        "architecture": "dummy",
        "output_dir": str(tmp_path / "outputs"),
        "phase1_epochs": 1,
        "phase2_epochs": 1,
        "phase1_batch_size": 2,
        "phase2_batch_size": 2,
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": False,
        "labels_csv": "labels.csv",
        "splits_file": "splits.json",
        "data_dir": "data",
        "use_wandb": False,
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_build_metadata_head_returns_expected_module_shapes():
    linear = _build_metadata_head(feature_dim=8, metadata_dim=2, hidden_dim=0)
    mlp = _build_metadata_head(feature_dim=8, metadata_dim=2, hidden_dim=4)

    assert isinstance(linear, nn.Linear)
    assert isinstance(mlp, nn.Sequential)
    assert linear(torch.randn(2, 10)).shape == (2, 1)
    assert mlp(torch.randn(2, 10)).shape == (2, 1)


def test_trainer_init_builds_metadata_head_with_tiny_dataloaders(tmp_path, monkeypatch):
    train_loader = DataLoader(DummyDataset(use_metadata=True), batch_size=2)
    val_loader = DataLoader(DummyDataset(use_metadata=True), batch_size=2)
    metadata_encoder = SimpleNamespace(output_dim=2, to_dict=lambda: {"output_dim": 2})

    monkeypatch.setattr(
        trainer_module,
        "create_dataloaders",
        lambda config, model, batch_size: (train_loader, val_loader, metadata_encoder),
    )
    monkeypatch.setattr(trainer_module, "TrainLogger", DummyLogger)

    config = _make_config(
        tmp_path,
        use_metadata=True,
        metadata_columns=["age", "sex"],
        metadata_head_hidden=4,
    )
    trainer = Trainer(config=config, model=DummyModel(), device=torch.device("cpu"))

    assert trainer.metadata_head is not None
    assert trainer.metadata_head(torch.randn(2, trainer.model.feature_dim + 2)).shape == (2, 1)
    assert trainer.config.checkpoint_dir.exists()
    assert trainer.config.predictions_dir.exists()
    assert trainer.config.log_dir.exists()


def test_forward_uses_model_directly_without_metadata_head():
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = DummyModel()
    trainer.metadata_head = None

    images = torch.randn(2, 3, 2, 2)
    batch = {"image": images}

    logits = trainer._forward(batch, images)

    assert logits.shape == (2, 1)
    assert torch.allclose(logits, trainer.model(images))


def test_forward_concatenates_metadata_when_metadata_head_is_enabled():
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = DummyModel()
    trainer.metadata_head = nn.Linear(trainer.model.feature_dim + 2, 1)

    images = torch.randn(2, 3, 2, 2)
    batch = {"metadata": torch.randn(2, 2)}

    logits = trainer._forward(batch, images)
    expected = trainer.metadata_head(torch.cat([trainer.model.get_features(images), batch["metadata"]], dim=1))

    assert logits.shape == (2, 1)
    assert torch.allclose(logits, expected)


def test_save_and_load_checkpoint_restore_training_state(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = DummyModel()
    trainer.metadata_head = None
    trainer.metadata_encoder = None
    trainer.config = _make_config(tmp_path)
    trainer.best_val_auroc = 0.82
    trainer.start_epoch = 0
    trainer.start_phase = "phase1"
    trainer._resume_checkpoint = None

    optimiser = AdamW(trainer.model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimiser, lr_lambda=lambda epoch: 1.0)
    path = tmp_path / "checkpoint.pt"

    trainer._save_checkpoint(
        path=path,
        epoch=4,
        phase="phase2",
        optimiser=optimiser,
        scheduler=scheduler,
        metrics={"auroc": 0.82},
    )
    expected_state = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}

    for parameter in trainer.model.parameters():
        parameter.data.add_(10.0)

    trainer.best_val_auroc = 0.0
    trainer._load_checkpoint(str(path))

    for name, tensor in trainer.model.state_dict().items():
        assert torch.allclose(tensor, expected_state[name])

    assert trainer.start_epoch == 5
    assert trainer.start_phase == "phase2"
    assert trainer.best_val_auroc == pytest.approx(0.82)
    assert trainer._resume_checkpoint is not None


def test_load_checkpoint_honours_resume_weights_only(tmp_path):
    saver = Trainer.__new__(Trainer)
    saver.device = torch.device("cpu")
    saver.model = DummyModel()
    saver.metadata_head = None
    saver.metadata_encoder = None
    saver.config = _make_config(tmp_path)
    saver.best_val_auroc = 0.5

    optimiser = AdamW(saver.model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimiser, lr_lambda=lambda epoch: 1.0)
    path = tmp_path / "weights_only.pt"

    saver._save_checkpoint(
        path=path,
        epoch=2,
        phase="phase2",
        optimiser=optimiser,
        scheduler=scheduler,
        metrics={"auroc": 0.5},
    )
    expected_state = {name: tensor.clone() for name, tensor in saver.model.state_dict().items()}

    loader = Trainer.__new__(Trainer)
    loader.device = torch.device("cpu")
    loader.model = DummyModel()
    loader.metadata_head = None
    loader.metadata_encoder = None
    loader.config = _make_config(tmp_path, resume_weights_only=True)
    loader.best_val_auroc = 0.0
    loader.start_epoch = 0
    loader.start_phase = "phase1"
    loader._resume_checkpoint = None

    for parameter in loader.model.parameters():
        parameter.data.zero_()

    loader._load_checkpoint(str(path))

    for name, tensor in loader.model.state_dict().items():
        assert torch.allclose(tensor, expected_state[name])

    assert loader.start_epoch == 0
    assert loader.start_phase == "phase1"
    assert loader._resume_checkpoint is None


def test_evaluate_final_writes_predictions_csv(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = DummyModel()
    trainer.metadata_head = None
    trainer.metadata_encoder = None
    trainer.config = _make_config(tmp_path)
    trainer.val_loader = DataLoader(DummyDataset(use_metadata=False), batch_size=2)
    trainer.config.predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics = trainer._evaluate_final()

    csv_path = trainer.config.predictions_dir / "val_predictions.csv"
    saved = pd.read_csv(csv_path)

    assert csv_path.exists()
    assert list(saved.columns) == ["patient_id", "y_true", "y_prob", "y_pred"]
    assert len(saved) == 2
    assert metrics["n_samples"] == 2
