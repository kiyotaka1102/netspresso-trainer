# -*- coding: utf-8 -*-
"""
Dataset Generator (WFLW-style + BBOX)
- Creates images/ + labels/
- Creates id_mapping.json (keypoint info)
- Creates data.yaml
- Removes temporary folders
- Supports bbox from annotations or fallback from keypoints
"""

import pandas as pd
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ====================== CONFIG ======================
CSV_DIR = Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/")

IMG_ROOTS = {
    "CRPD": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_CRPD_keypoint/keypoint/images"),
    "CCPD2019": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_CCPD2019_keypoint/keypoint/images"),
    "KLPR": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_KLPR_keypoint/images"),
    "UC3M-LP": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_UC3M-LP_keypoint/images"),
}

ANNOT_FILES = {
    "CRPD": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_CRPD_keypoint/keypoint/annotations.json"),
    "CCPD2019": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_CCPD2019_keypoint/keypoint/annotations.json"),
    "KLPR": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_KLPR_keypoint/annotations.json"),
    "UC3M-LP": Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/datasets_public_UC3M-LP_keypoint/annotations.json"),
}

DATASET_ROOT = Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/huy_aichallenge/CTY/yolo_lp_dataset")
IMG_DIR = DATASET_ROOT / "images"
LBL_DIR = DATASET_ROOT / "labels"
TMP_DIR = DATASET_ROOT / "tmp"

for p in [IMG_DIR / "train", IMG_DIR / "val",
          LBL_DIR / "train", LBL_DIR / "val"]:
    p.mkdir(parents=True, exist_ok=True)

# ====================== KEYPOINT MAPPING ======================
KEYPOINT_INFO = {
    0: dict(name="top-left",     id=0, color=[255, 0, 0], type="corner", swap="bottom-right"),
    1: dict(name="top-right",    id=1, color=[0, 255, 0], type="corner", swap="bottom-left"),
    2: dict(name="bottom-right", id=2, color=[0, 0, 255], type="corner", swap="top-left"),
    3: dict(name="bottom-left",  id=3, color=[255, 255, 0], type="corner", swap="top-right"),
}

# ====================== LOAD ANNOTATIONS ======================
print("Loading COCO-style annotations...")

def load_coco(annot_path):
    """Try loading JSON file with different encodings."""
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']:
        try:
            with open(annot_path, 'r', encoding=enc) as f:
                data = json.load(f)
            print(f"  Loaded {annot_path.name} ({enc})")

            img_dict = {img['id']: img for img in data.get('images', [])}
            ann_dict = {}

            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in ann_dict:
                    ann_dict[img_id] = ann
                else:
                    if 'bbox' in ann and 'bbox' not in ann_dict[img_id]:
                        ann_dict[img_id] = ann
            return img_dict, ann_dict

        except Exception:
            continue
    print(f"  Failed to load {annot_path}")
    return {}, {}

img_dicts, ann_dicts = {}, {}
for ds, path in ANNOT_FILES.items():
    if path.exists():
        img_dict, ann_dict = load_coco(path)
        if img_dict:
            img_dicts[ds] = img_dict
            ann_dicts[ds] = ann_dict
    else:
        print(f"Annotation file missing: {path}")

# ====================== HELPERS ======================
def find_image_path(dataset, file_name):
    """Find image path in dataset folder."""
    root = IMG_ROOTS.get(dataset)
    if not root or not root.exists():
        return None
    for sub in [root, root / "images", root / "img"]:
        p = sub / file_name
        if p.exists():
            return p
    return None

def bbox_from_keypoints(kps):
    """Compute bbox from keypoints (absolute coords)."""
    xs, ys = [], []
    for i in range(0, len(kps), 3):
        x, y, v = kps[i], kps[i+1], kps[i+2]
        if v > 0:
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]  # absolute pixel values

# ====================== PROCESS SPLITS ======================
total_copied = 0
total_labeled = 0

for split in ["train", "val", "test"]:
    csv_path = CSV_DIR / f"{split}_set.csv"
    if not csv_path.exists():
        print(f"Missing split CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    print(f"\nProcessing {split.upper()} set ({len(df):,} images)...")

    img_out_dir = IMG_DIR / split
    lbl_out_dir = LBL_DIR / split
    copied, labeled = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}", unit="img"):
        ds = row['dataset']
        img_id = int(row['image_id'])

        if ds not in img_dicts:
            continue
        img_info = img_dicts[ds].get(img_id)
        if not img_info:
            continue

        file_name = img_info['file_name']
        src_path = find_image_path(ds, file_name)
        if not src_path:
            continue

        # Output names
        new_name = f"{ds}_{img_id}{src_path.suffix}"
        dst_img_path = img_out_dir / new_name
        dst_lbl_path = lbl_out_dir / f"{ds}_{img_id}.txt"

        # Copy image if not already copied
        if not dst_img_path.exists():
            shutil.copy(src_path, dst_img_path)
            copied += 1

        ann = ann_dicts[ds].get(img_id)
        if not ann or 'keypoints' not in ann:
            continue

        kps = ann['keypoints']
        bbox = ann.get('bbox') if ann.get('bbox') else bbox_from_keypoints(kps)
        if not bbox:
            continue

        # Write WFLW-style label (absolute coords)
        with open(dst_lbl_path, 'w', encoding='utf-8') as f:
            f.write("0"+" "+ " ".join(map(str, bbox)) + " "  + " ".join(map(str, kps)) + "\n")
        labeled += 1

    print(f"  Copied: {copied:,} | Labeled: {labeled:,}")
    total_copied += copied
    total_labeled += labeled

# ====================== SAVE MAPPINGS ======================
id_mapping = [
    {"name": v["name"], "skeleton": None, "swap": v["swap"]}
    for v in KEYPOINT_INFO.values()
]
with open(DATASET_ROOT / "id_mapping.json", 'w', encoding='utf-8') as f:
    json.dump(id_mapping, f, indent=2, ensure_ascii=False)

# ====================== SAVE YAML ======================
data_yaml = f"""
path: {DATASET_ROOT.absolute()}
train: images/train
val: images/val
nc: 1
names: ['plate']

kpt_shape: [4, 3]  # 4 keypoints (x, y, v)
"""
(DATASET_ROOT / "data.yaml").write_text(data_yaml.strip(), encoding='utf-8')

# ====================== CLEAN TMP ======================
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)

print("\n✅ DONE (WFLW + BBOX STYLE)")
print(f"  Dataset root: {DATASET_ROOT}")
print(f"  Total images copied: {total_copied:,}")
print(f"  Total labels created: {total_labeled:,}")
print(f"  id_mapping.json & data.yaml saved.\n")
