import pytest
from fastapi import HTTPException

from api.routers.inference import _select_image


def test_select_image_accepts_explicit_dataset_member():
    assert _select_image("field.jpg", ["field.jpg", "other.jpg"]) == "field.jpg"


def test_select_image_rejects_path_traversal():
    with pytest.raises(HTTPException) as exc:
        _select_image("../field.jpg", ["field.jpg"])
    assert exc.value.status_code == 404
