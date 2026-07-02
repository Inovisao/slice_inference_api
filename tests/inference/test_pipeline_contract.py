import numpy as np

from inference.pipeline import InferencePipeline, _apply_nms


class FakeSlicer:
    def generate_tiles(self, image):
        yield image[:2, :2], {"x": 0, "y": 0, "width": 2, "height": 2}


class FakeEngine:
    class_names = {0: "insect"}

    def __init__(self):
        self.full_calls = 0
        self.tile_calls = 0
        self.received_batch_size = None

    def predict_tiles(self, image, tile_generator, conf_thr=0.25, batch_size=32):
        self.tile_calls += 1
        self.received_batch_size = batch_size
        list(tile_generator)
        return [[0.1, 0.1, 0.2, 0.2]], [0.8], [0]

    def predict_full_image(self, image, conf_thr=0.25):
        self.full_calls += 1
        return [[0.6, 0.6, 0.7, 0.7]], [0.9], [0]


def _pipeline(engine, include_full):
    return InferencePipeline(
        engine=engine,
        slicer=FakeSlicer(),
        suppression="nms",
        include_full_inference=include_full,
        batch_size=7,
    )


def test_tile_only_contract_does_not_run_full_image(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.pipeline.draw_detections", lambda *args: None)
    engine = FakeEngine()

    result = _pipeline(engine, include_full=False).run(
        np.zeros((4, 4, 3), dtype=np.uint8), str(tmp_path / "out.jpg")
    )

    assert engine.tile_calls == 1
    assert engine.full_calls == 0
    assert engine.received_batch_size == 7
    assert result["raw_detections"] == 1


def test_full_image_is_added_exactly_once(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.pipeline.draw_detections", lambda *args: None)
    engine = FakeEngine()

    result = _pipeline(engine, include_full=True).run(
        np.zeros((4, 4, 3), dtype=np.uint8), str(tmp_path / "out.jpg")
    )

    assert engine.tile_calls == 1
    assert engine.full_calls == 1
    assert result["raw_detections"] == 2


def test_nms_never_suppresses_a_different_class():
    boxes = np.array([
        [0.1, 0.1, 0.4, 0.4],
        [0.1, 0.1, 0.4, 0.4],
    ])
    scores = np.array([0.9, 0.8])
    labels = np.array([0, 1])

    kept_boxes, _, kept_labels = _apply_nms(boxes, scores, labels, iou_thr=0.5)

    assert len(kept_boxes) == 2
    assert set(kept_labels) == {0, 1}
