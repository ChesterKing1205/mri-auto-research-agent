from __future__ import annotations


def test_train_stdout_metric_keys_are_documented():
    import train

    source = train.main.__code__.co_consts
    text = "\n".join(str(item) for item in source)
    assert "primary_metric: psnr" in text
    assert "psnr" in text
    assert "ssim" in text
    assert "nmse" in text
    assert "val_loss" in text

