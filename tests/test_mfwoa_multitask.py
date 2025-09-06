import numpy as np
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.metrics.fuzzy_entropy import histogram_from_image


def test_mfwoa_multitask_basic():
    # synthetic image: two peaks
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:200 // 2].flat[:] = 20
    img[200 // 2 :].flat[:] = 220
    hist = histogram_from_image(img)
    Ks = [2, 3]
    hists = [hist, hist]
    best_ths, best_scores = mfwoa_multitask(hists, Ks, pop_size=10, iters=20)
    assert len(best_ths) == 2
    assert len(best_scores) == 2
    # thresholds are within [0,255]
    for ths in best_ths:
        for t in ths:
            assert 0 <= t <= 255
    for s in best_scores:
        assert isinstance(s, float)
