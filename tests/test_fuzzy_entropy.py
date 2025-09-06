import numpy as np
from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy


def test_histogram_and_entropy():
    # create synthetic bimodal image
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:50] = 10
    img[50:] = 200
    hist = histogram_from_image(img)
    # check histogram sum
    assert hist.sum() == img.size
    # two thresholds near separating values should produce reasonable entropy
    score = compute_fuzzy_entropy(hist, [100])
    assert isinstance(score, float)
    assert score >= 0.0
