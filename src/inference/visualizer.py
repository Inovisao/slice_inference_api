import cv2
import numpy as np

_COLORS = [
    (0, 255, 80),
    (0, 180, 255),
    (255, 100, 0),
    (255, 0, 150),
    (180, 255, 0),
]


def draw_detections(
    image_rgb: np.ndarray,
    boxes: list,
    scores: list,
    labels: list,
    class_names: dict,
    out_path: str,
    conf_thr: float,
) -> None:
    img_h, img_w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for box, score, label in zip(boxes, scores, labels):
        x1 = int(box[0] * img_w)
        y1 = int(box[1] * img_h)
        x2 = int(box[2] * img_w)
        y2 = int(box[3] * img_h)
        color = _COLORS[label % len(_COLORS)]
        name = class_names.get(label, f"cls_{label}")

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        txt = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, txt, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    counter = f"Detectados: {len(boxes)}  |  conf>={conf_thr}"
    cv2.putText(result, counter, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(result, counter, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 80), 3, cv2.LINE_AA)

    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 93])
