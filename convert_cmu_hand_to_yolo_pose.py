import os
import json
import random
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict

from PIL import Image

# ----------------------------
# Config
# ----------------------------
SEED = 42
KPT_NUM = 21
MARGIN_RATIO = 0.05  # bbox margin
MIN_BOX_AREA_RATIO = 0.0004  # ~0.04% of image area; adjust if too strict/loose

# If you want to downsample synthetic to reduce training time:
MAX_SYNTH_PER_FOLDER = 7000  # e.g. 4000 or None for all

# Split manual train -> val
MANUAL_VAL_RATIO = 0.85

random.seed(SEED)


# ----------------------------
# Utilities: recursive keypoint extractor
# ----------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _flatten_numbers(obj: Any, out: List[float]):
    if _is_number(obj):
        out.append(float(obj))
        return
    if isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)

def _find_candidates(obj: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Return list of (path, value) for list-like candidates."""
    cands = []
    if isinstance(obj, list):
        cands.append((path, obj))
        for i, v in enumerate(obj):
            cands.extend(_find_candidates(v, f"{path}[{i}]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            newp = f"{path}.{k}" if path else k
            cands.extend(_find_candidates(v, newp))
    return cands

def _parse_kpts_from_list(lst: list) -> Optional[List[Tuple[float, float, int]]]:
    """
    Try parse keypoints from:
    1) list of dict: [{'x':..,'y':..,'score':..}, ...] length 21
    2) list of [x,y] or [x,y,score] length 21
    3) flat list length 42 or 63 (x,y,[v/score]) * 21
    """
    # Case A: list length 21 of dict or list
    if len(lst) == KPT_NUM:
        kpts = []
        for item in lst:
            if isinstance(item, dict):
                # common keys
                x = item.get("x", item.get("X", item.get("u")))
                y = item.get("y", item.get("Y", item.get("v")))
                score = item.get("score", item.get("s", item.get("conf", item.get("confidence"))))
                vis = item.get("vis", item.get("visibility", item.get("v")))
                if x is None or y is None or not _is_number(x) or not _is_number(y):
                    return None
                # YOLO visibility: 2 visible, 1 labeled but occluded, 0 not labeled
                vflag = 2
                if vis is not None and _is_number(vis):
                    # normalize common conventions
                    vflag = 2 if float(vis) > 0 else 0
                elif score is not None and _is_number(score):
                    vflag = 2 if float(score) > 0 else 0
                kpts.append((float(x), float(y), int(vflag)))
            elif isinstance(item, list) and len(item) >= 2 and _is_number(item[0]) and _is_number(item[1]):
                x, y = float(item[0]), float(item[1])
                vflag = 2
                if len(item) >= 3 and _is_number(item[2]):
                    vflag = 2 if float(item[2]) > 0 else 0
                kpts.append((x, y, int(vflag)))
            else:
                return None
        return kpts

    # Case B: flat list 42 or 63
    if len(lst) in (KPT_NUM * 2, KPT_NUM * 3) and all(_is_number(x) for x in lst):
        kpts = []
        if len(lst) == KPT_NUM * 2:
            for i in range(KPT_NUM):
                x = float(lst[2*i])
                y = float(lst[2*i + 1])
                kpts.append((x, y, 2))
        else:
            for i in range(KPT_NUM):
                x = float(lst[3*i])
                y = float(lst[3*i + 1])
                s = float(lst[3*i + 2])
                vflag = 2 if s > 0 else 0
                kpts.append((x, y, int(vflag)))
        return kpts

    return None

def extract_kpts(js):
    # Expect: js["hand_pts"] = [[x,y,v], ...] length 21
    pts = js.get("hand_pts", None)
    if not isinstance(pts, list) or len(pts) != 21:
        return None

    kpts = []
    for x, y, v in pts:
        # Manual seems to use 0/1 flag; map to YOLO 0/2
        vflag = 2 if float(v) > 0 else 0
        kpts.append((float(x), float(y), vflag))
    return kpts



# ----------------------------
# YOLO label writer
# ----------------------------
def kpts_to_yolo_line(
    kpts_px: List[Tuple[float, float, int]],
    W: int, H: int,
    class_id: int = 0
) -> Optional[str]:
    # keep only labeled points (v>0) for bbox
    xs = [x for x, y, v in kpts_px if v > 0]
    ys = [y for x, y, v in kpts_px if v > 0]
    if len(xs) < 5:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # margin
    mx = (x_max - x_min) * MARGIN_RATIO
    my = (y_max - y_min) * MARGIN_RATIO
    x_min -= mx
    x_max += mx
    y_min -= my
    y_max += my

    # clamp
    x_min = max(0.0, min(float(W - 1), x_min))
    x_max = max(0.0, min(float(W - 1), x_max))
    y_min = max(0.0, min(float(H - 1), y_min))
    y_max = max(0.0, min(float(H - 1), y_max))

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 1 or bh <= 1:
        return None

    # filter tiny boxes
    if (bw * bh) < (W * H * MIN_BOX_AREA_RATIO):
        return None

    xc = (x_min + x_max) / 2.0 / W
    yc = (y_min + y_max) / 2.0 / H
    ww = bw / W
    hh = bh / H

    parts = [str(class_id), f"{xc:.6f}", f"{yc:.6f}", f"{ww:.6f}", f"{hh:.6f}"]

    for x, y, v in kpts_px:
        xn = x / W
        yn = y / H
        # clamp normalized
        xn = max(0.0, min(1.0, xn))
        yn = max(0.0, min(1.0, yn))
        vv = 2 if v > 0 else 0
        parts += [f"{xn:.6f}", f"{yn:.6f}", str(vv)]

    return " ".join(parts)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_image(src: Path, dst: Path):
    # simple file copy (binary)
    dst.write_bytes(src.read_bytes())


# ----------------------------
# Dataset conversion
# ----------------------------
def convert_pairs(
    pair_paths: List[Tuple[Path, Path]],
    out_img_dir: Path,
    out_lbl_dir: Path,
    verbose_every: int = 500
) -> Dict[str, int]:
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    ok = 0
    skipped_no_kpt = 0
    skipped_bad = 0

    for i, (img_path, json_path) in enumerate(pair_paths, 1):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                js = json.load(f)
            kpts = extract_kpts(js)
            if kpts is None:
                skipped_no_kpt += 1
                continue

            with Image.open(img_path) as im:
                W, H = im.size

            line = kpts_to_yolo_line(kpts, W, H, class_id=0)
            if line is None:
                skipped_bad += 1
                continue

            # copy image + write label
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / (img_path.stem + ".txt")
            copy_image(img_path, out_img)
            out_lbl.write_text(line + "\n", encoding="utf-8")
            ok += 1

        except Exception:
            skipped_bad += 1

        if verbose_every and i % verbose_every == 0:
            print(f"[INFO] Processed {i}/{len(pair_paths)} pairs. OK={ok}, no_kpt={skipped_no_kpt}, bad={skipped_bad}")

    return {"ok": ok, "no_kpt": skipped_no_kpt, "bad": skipped_bad}

def collect_pairs_from_folder(folder: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img_path in sorted(folder.glob("*.jpg")):
        json_path = folder / (img_path.stem + ".json")
        if json_path.exists():
            pairs.append((img_path, json_path))
    return pairs


def main():
    # ---- EDIT THESE PATHS ----
    # Example:
    # root_synth = Path(r"D:\cmu_hand\synth")
    # root_manual = Path(r"D:\cmu_hand\manual")
    root_synth = Path(r"./Synthetic")   # contains synth1..synth4 + output_viz_synth
    root_manual = Path(r"./Manual")     # contains train/ test/ output_viz

    out_root = Path(r"./hand_pose_data")
    out_images = out_root / "images"
    out_labels = out_root / "labels"

    # Output dirs
    train_img = out_images / "train"
    val_img   = out_images / "val"
    test_img  = out_images / "test"
    train_lbl = out_labels / "train"
    val_lbl   = out_labels / "val"
    test_lbl  = out_labels / "test"

    # 1) Collect synthetic pairs (train only)
    synth_folders = [root_synth / f"synth{i}" for i in range(1, 5)]
    synth_pairs_all: List[Tuple[Path, Path]] = []
    for sf in synth_folders:
        if sf.exists():
            synth_pairs_all.extend(collect_pairs_from_folder(sf))

    if MAX_SYNTH_PER_FOLDER is not None:
        # downsample per folder
        synth_pairs_ds = []
        for sf in synth_folders:
            pairs = collect_pairs_from_folder(sf) if sf.exists() else []
            random.shuffle(pairs)
            synth_pairs_ds.extend(pairs[:MAX_SYNTH_PER_FOLDER])
        synth_pairs_all = synth_pairs_ds

    random.shuffle(synth_pairs_all)
    print(f"[INFO] Synthetic pairs: {len(synth_pairs_all)}")

    # 2) Collect manual pairs
    manual_train_pairs = collect_pairs_from_folder(root_manual / "train") if (root_manual / "train").exists() else []
    manual_test_pairs  = collect_pairs_from_folder(root_manual / "test") if (root_manual / "test").exists() else []
    print(f"[INFO] Manual train pairs: {len(manual_train_pairs)}")
    print(f"[INFO] Manual test pairs : {len(manual_test_pairs)}")

    # split manual train -> val
    random.shuffle(manual_train_pairs)
    val_n = int(len(manual_train_pairs) * MANUAL_VAL_RATIO)
    manual_val_pairs = manual_train_pairs[:val_n]
    manual_train_pairs = manual_train_pairs[val_n:]
    print(f"[INFO] Manual split -> train: {len(manual_train_pairs)}, val: {len(manual_val_pairs)}")

    # 3) Convert
    stats = {}
    stats["synth_train"] = convert_pairs(synth_pairs_all, train_img, train_lbl)
    stats["manual_train"] = convert_pairs(manual_train_pairs, train_img, train_lbl)
    stats["manual_val"] = convert_pairs(manual_val_pairs, val_img, val_lbl)
    stats["manual_test"] = convert_pairs(manual_test_pairs, test_img, test_lbl)

    print("[INFO] Done. Stats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    # 4) Write data.yaml
    data_yaml = f"""path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names: [hand]

kpt_shape: [{KPT_NUM}, 3]
"""
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print("[INFO] Wrote data.yaml")


if __name__ == "__main__":
    main()
