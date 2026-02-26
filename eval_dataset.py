import os
import re
import argparse
import subprocess
from tqdm import tqdm

# 解析 read.py 输出: "path/to/img.jpg: PRED"
LINE_RE = re.compile(r"^(.+?):\s*(.*)$")
# CRNN同款过滤：只保留a-z0-9
FILTER_RE = re.compile(r"[^a-z0-9]")

def normalize_gt(raw_gt: str) -> str:
    raw_gt = raw_gt.lower()
    return FILTER_RE.sub("", raw_gt)

def normalize_pred(pred: str) -> str:
    pred = pred.lower()
    return FILTER_RE.sub("", pred)

def load_gt_file(gt_path: str):
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_filename = parts[0]
            raw_gt = "".join(parts[1:])  # 跟你CRNN脚本一致：把后面全拼起来
            gt[img_filename] = raw_gt
    return gt

def run_parseq_read(parseq_dir: str, image_paths, pretrained="parseq", batch=64):
    """
    调用本地 read.py，批量拿预测结果，返回 dict: {basename: pred_str}
    """
    preds = {}
    read_py = os.path.join(parseq_dir, "read.py")
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i+batch]
        cmd = ["python", read_py, f"pretrained={pretrained}", "--images", *chunk]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"read.py failed.\nCMD: {' '.join(cmd)}\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}")

        for line in p.stdout.splitlines():
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            path, pred = m.group(1).strip(), m.group(2).strip()
            preds[os.path.basename(path)] = pred
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pretrained", default="parseq")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--print_mis", action="store_true", help="打印错误样本")
    args = ap.parse_args()

    parseq_dir = os.path.dirname(os.path.abspath(__file__))

    gt_raw = load_gt_file(args.gt)

    # === 完全对齐CRNN：先按GT遍历；不存在图片就跳过；GT过滤后空也跳过 ===
    valid_items = []
    skipped_empty = 0
    skipped_missing_img = 0

    for img_name, raw_gt in gt_raw.items():
        gt_text = normalize_gt(raw_gt)
        if not gt_text:
            skipped_empty += 1
            continue
        img_path = os.path.join(args.images_dir, img_name)
        if not os.path.exists(img_path):
            skipped_missing_img += 1
            continue
        valid_items.append((img_name, img_path, raw_gt, gt_text))

    print(f"[Info] labels in gt file: {len(gt_raw)}")
    print(f"[Info] valid samples (after filter & exists): {len(valid_items)}")
    print(f"[Info] skipped (empty after filter): {skipped_empty}")
    print(f"[Info] skipped (missing image): {skipped_missing_img}")

    image_paths = [it[1] for it in valid_items]
    preds_raw = run_parseq_read(parseq_dir, image_paths, pretrained=args.pretrained, batch=args.batch)

    total = 0
    correct = 0

    for img_name, img_path, raw_gt, gt_text in tqdm(valid_items, desc="Scoring"):
        pred = preds_raw.get(img_name, "")
        pred_text = normalize_pred(pred)

        total += 1
        if pred_text == gt_text:
            correct += 1
        else:
            if args.print_mis:
                print(f"ERR -> {img_name} | raw_gt='{raw_gt.lower()}' -> gt='{gt_text}' | pred='{pred_text}' (raw_pred='{pred}')")

    acc = correct / total if total > 0 else 0.0
    print("-" * 40)
    print(f"测试总数: {total}")
    print(f"正确数量: {correct}")
    print(f"整词识别准确率 (Word Accuracy): {acc*100:.2f}%")

if __name__ == "__main__":
    main()