"""
下载 Hand-Gesture-19 模型权重 (~355MB)。

来源: hf-mirror.com (HuggingFace 镜像, 国内可访问)
运行方式: python models/sign_language_model/download_hand_gesture.py

下载到: models/sign_language_model/pretrained/hand_gesture_19/
"""
from __future__ import annotations

import requests
from pathlib import Path
from tqdm import tqdm

MODEL = "prithivMLmods/Hand-Gesture-19"
DIR = Path(__file__).parent / "pretrained" / "hand_gesture_19"
MIRROR = "https://hf-mirror.com"

FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    base = f"{MIRROR}/{MODEL}/resolve/main"

    for fname in FILES:
        url = f"{base}/{fname}"
        out = DIR / fname
        if out.exists() and out.stat().st_size > 100:
            print(f"[SKIP] {fname} ({out.stat().st_size/1024**2:.0f}MB)")
            continue
        print(f"Downloading {fname}...")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=fname) as p:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    p.update(len(chunk))

    print(f"\nDone! Model saved to: {DIR}")
    for f in DIR.iterdir():
        print(f"  {f.name} ({f.stat().st_size/1024**2:.1f}MB)")


if __name__ == "__main__":
    main()
