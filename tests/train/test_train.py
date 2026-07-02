from pathlib import Path

from train import train


class FakeResult:
    def __init__(self, save_dir):
        self.save_dir = save_dir


class FakeYOLO:
    last_kwargs = None

    def __init__(self, model):
        self.model = model

    def train(self, **kwargs):
        FakeYOLO.last_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"checkpoint")
        return FakeResult(save_dir)


def test_train_fold_writes_checkpoint_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    data = tmp_path / "fold_2.yaml"
    model = tmp_path / "initial.pt"
    data.write_text("names: {0: insect}\n")
    model.write_bytes(b"model")

    checkpoint = train.train_fold(
        data=data,
        initial_model=model,
        models_root=tmp_path / "models",
        mode="sahi",
        epochs=2,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        seed=7,
    )

    assert checkpoint.is_file()
    assert (tmp_path / "models/sahi/fold_2/yolo/manifest.json").is_file()
    assert FakeYOLO.last_kwargs["data"] == str(data.resolve())
