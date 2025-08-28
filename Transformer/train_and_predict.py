# train_videomae_stage3_win.py
import os
import json
import random
from typing import List, Dict, Any, Tuple
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 임시 우회

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    Trainer,
    TrainingArguments,
)
import evaluate
from inspect import signature
import platform
import pandas as pd
from collections import defaultdict

# =========================
# 사용자 설정 (Windows 경로)
# =========================
DATA_ROOT = r"C:\Users\User\Desktop\MinOfficialProject\VideoClassification\dataset"
CSV_PATH = os.path.join(DATA_ROOT, "ground_truth.csv")  # CSV가 해당 폴더에 있다고 가정
MODEL_NAME = "MCG-NJU/videomae-base"
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs_videomae_stage3")
SEED = 42

# 비디오 샘플링/전처리 설정
NUM_FRAMES = 16
FRAME_STRIDE = 2
CENTER_CROP = True

# 데이터 분할 비율 (train/val/test) → 0.7 / 0.2 / 0.1
TRAIN_VAL_TEST = (0.7, 0.2, 0.1)

# 학습 하이퍼파라미터
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 10
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
FP16_DEFAULT = True  # CUDA일 때만 의미 있음

# =========================
# 유틸
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _norm_join(root: str, rel: str) -> str:
    # 슬래시/백슬래시 혼용 정규화
    rel = str(rel).replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(os.path.join(root, rel))

def scan_from_csv(csv_path: str, data_root: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    CSV의 video_path와 stage3_label(결측 시 stage1_label 대체)을 사용하여
    (abs_video_path, label_str) 리스트와 클래스 리스트 반환
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    # 필수 컬럼 존재 확인
    if "video_path" not in df.columns or "stage1_label" not in df.columns:
        raise ValueError("CSV에는 최소한 'video_path', 'stage1_label' 컬럼이 필요합니다.")

    # stage3_label이 없거나 비어있으면 stage1_label로 채움
    if "stage3_label" not in df.columns:
        df["stage3_label"] = df["stage1_label"]
    else:
        df["stage3_label"] = df["stage3_label"].fillna(df["stage1_label"])

    # 필요한 컬럼 정리/클린
    df = df.dropna(subset=["video_path", "stage3_label"]).copy()
    df["video_path"] = df["video_path"].astype(str)
    df["stage3_label"] = df["stage3_label"].astype(str)

    items: List[Tuple[str, str]] = []
    missing = 0
    for _, row in df.iterrows():
        rel = row["video_path"]
        lbl = row["stage3_label"]
        abs_path = _norm_join(data_root, rel)
        if os.path.exists(abs_path):
            items.append((abs_path, lbl))
        else:
            missing += 1
    if missing > 0:
        print(f"[WARN] 존재하지 않는 파일 {missing}개는 제외됨")

    classes = sorted({lbl for _, lbl in items})
    return classes, items

# =========================
# Dataset
# =========================
class VideoFolderDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, str]],
        label2id: Dict[str, int],
        processor: VideoMAEImageProcessor,
        num_frames: int = 16,
        frame_stride: int = 2,
        center_crop: bool = True,
    ):
        self.items = items
        self.label2id = label2id
        self.processor = processor
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.center_crop = center_crop

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, num_total_frames: int):
        needed = self.num_frames * self.frame_stride  # e.g., 32
        if num_total_frames <= 0:
            return [0] * self.num_frames
        if num_total_frames < needed:
            idx = np.linspace(0, max(0, num_total_frames - 1), self.num_frames).astype(int)
            return idx.tolist()

        if self.center_crop:
            start = (num_total_frames - needed) // 2
        else:
            start = random.randint(0, num_total_frames - needed)
        clip_idx = list(range(start, start + needed, self.frame_stride))
        if len(clip_idx) > self.num_frames:
            clip_idx = np.linspace(clip_idx[0], clip_idx[-1], self.num_frames).astype(int).tolist()
        elif len(clip_idx) < self.num_frames:
            clip_idx += [clip_idx[-1]] * (self.num_frames - len(clip_idx))
        return clip_idx

    def __getitem__(self, idx) -> Dict[str, Any]:
        import decord  # pip install decord
        path, label_name = self.items[idx]
        label = self.label2id[label_name]

        vr = decord.VideoReader(path)
        frame_indices = self._sample_indices(len(vr))
        frames = [vr[i].asnumpy() for i in frame_indices]
        encoding = self.processor(frames, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "path": path,
        }

# =========================
# 안전 계층 분할 (sklearn 불필요)
# =========================
def safe_stratified_split(items, label2id, ratios=(0.7, 0.2, 0.1), seed=42):
    rng = np.random.default_rng(seed)
    train_ratio, val_ratio, test_ratio = ratios

    labels = [label2id[lbl] for _, lbl in items]
    by_cls = defaultdict(list)
    for i, y in enumerate(labels):
        by_cls[y].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for y, inds in by_cls.items():
        inds = np.array(inds)
        rng.shuffle(inds)
        n = len(inds)

        if n == 1:
            tr = inds[:1]; va = np.array([], dtype=int); te = np.array([], dtype=int)
        elif n == 2:
            n_tr = 1
            n_va = 1 if val_ratio > 0 else 0
            n_te = 0 if n_va == 1 else (1 if test_ratio > 0 else 0)
            tr = inds[:n_tr]
            va = inds[n_tr:n_tr+n_va]
            te = inds[n_tr+n_va:n_tr+n_va+n_te]
        else:
            n_tr = int(round(n * train_ratio))
            n_va = int(round(n * val_ratio))
            n_te = n - n_tr - n_va

            def need_min(x, ratio):
                return max(x, 1) if ratio > 0 and n >= 3 else x

            n_tr = need_min(n_tr, train_ratio)
            n_va = need_min(n_va, val_ratio)
            n_te = n - n_tr - n_va

            if test_ratio > 0 and n_te == 0:
                if n_va > 1:
                    n_va -= 1
                    n_te = 1

            if n_te < 0:
                n_te = 0
                n_va = min(n_va, n - n_tr)

            if n_tr + n_va + n_te != n:
                n_tr = n - n_va - n_te

            tr = inds[:n_tr]
            va = inds[n_tr:n_tr + n_va]
            te = inds[n_tr + n_va:n_tr + n_va + n_te]

        train_idx.extend(tr.tolist())
        val_idx.extend(va.tolist())
        test_idx.extend(te.tolist())

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

# =========================
# Metric
# =========================
def make_compute_metrics():
    accuracy = evaluate.load("accuracy")
    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}
    return _fn

# =========================
# TrainingArguments 어댑터 (버전 자동 호환)
# =========================
def build_training_args(args_kwargs, has_val):
    supported = set(signature(TrainingArguments.__init__).parameters.keys())

    if "evaluation_strategy" in args_kwargs and "eval_strategy" in supported:
        args_kwargs["eval_strategy"] = args_kwargs.pop("evaluation_strategy")

    if ("eval_strategy" not in supported) and ("evaluation_strategy" not in supported):
        args_kwargs.pop("eval_strategy", None)
        args_kwargs.pop("evaluation_strategy", None)
        if has_val and "do_eval" in supported:
            args_kwargs["do_eval"] = True
        for k in ["load_best_model_at_end", "metric_for_best_model", "greater_is_better", "save_strategy"]:
            if k not in supported:
                args_kwargs.pop(k, None)

    filtered = {k: v for k, v in args_kwargs.items() if k in supported}
    return TrainingArguments(**filtered)

# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    # --- GPU 강제 사용 ---
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA(GPU)를 찾을 수 없습니다. NVIDIA 드라이버 / CUDA / PyTorch CUDA 빌드를 확인하세요."
        )
    device = torch.device("cuda:0")
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1) CSV에서 stage3 라벨(결측 시 stage1)로 데이터 구성
    classes, items = scan_from_csv(CSV_PATH, DATA_ROOT)
    if len(classes) < 2:
        raise ValueError("stage3_label 기준 라벨(클래스)이 2개 이상 필요합니다.")
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "label_map_stage3.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # 2) 분할 (0.7 / 0.2 / 0.1)
    train_idx, val_idx, test_idx = safe_stratified_split(items, label2id, TRAIN_VAL_TEST, SEED)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]
    print(f"[INFO] #train={len(train_items)}, #val={len(val_items)}, #test={len(test_items)}")

    # 분할 결과 CSV 저장
    split_df = pd.DataFrame(
        [(p, lbl, "train") for p, lbl in train_items] +
        [(p, lbl, "val") for p, lbl in val_items] +
        [(p, lbl, "test") for p, lbl in test_items],
        columns=["abs_path", "label", "split"]
    )
    split_df.to_csv(os.path.join(OUTPUT_DIR, "split_stage3.csv"), index=False)

    # 3) 프로세서/모델
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)

    # 4) Dataset
    train_ds = VideoFolderDataset(train_items, label2id, processor, NUM_FRAMES, FRAME_STRIDE, CENTER_CROP)
    val_ds = VideoFolderDataset(val_items, label2id, processor, NUM_FRAMES, FRAME_STRIDE, CENTER_CROP) if len(val_items) > 0 else None
    test_ds = VideoFolderDataset(test_items, label2id, processor, NUM_FRAMES, FRAME_STRIDE, CENTER_CROP) if len(test_items) > 0 else None

    has_val = val_ds is not None

    # 5) TrainingArguments
    WORKERS = 0 if platform.system() == "Windows" else 2
    args_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        fp16=FP16_DEFAULT,
        report_to="none",
        dataloader_num_workers=WORKERS,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
    )

    if has_val:
        args_kwargs.update(
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
    else:
        args_kwargs.update(
            evaluation_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
        )

    args = build_training_args(args_kwargs, has_val)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        tokenizer=processor,
        compute_metrics=make_compute_metrics() if has_val else None,
    )

    # 6) 학습
    trainer.train()

    # 7) 최적 모델 정보 출력 및 테스트 평가 + 예측 저장
    if has_val:
        try:
            print("▶ Best:", trainer.state.best_metric, "at step", trainer.state.best_step)
        except AttributeError:
            print("▶ Best metric is available, but best step is not.")
            if getattr(trainer.state, "best_metric", None) is not None:
                print("▶ Best:", trainer.state.best_metric)
            else:
                print("▶ No best metric found.")

    # 테스트 평가 및 예측
    if test_ds is not None and len(test_ds) > 0:
        print("\n--- 테스트 평가 (stage3) ---")
        test_metrics = trainer.evaluate(test_ds)
        print("▶ Test metrics:", test_metrics)

        preds = trainer.predict(test_ds)
        logits = preds.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred_ids = probs.argmax(axis=1)
        pred_labels = [id2label[i] for i in pred_ids]

        test_paths = [p for p, _ in test_items]
        true_labels = [lbl for _, lbl in test_items]
        true_ids = [label2id[l] for l in true_labels]

        # 정확도(스스로 계산: sklearn 없이)
        acc = float(np.mean(np.array(true_labels) == np.array(pred_labels)))
        print(f"▶ Test Accuracy (stage3): {acc:.4f}")

        out_df = pd.DataFrame({
            "path": test_paths,
            "true_label": true_labels,
            "true_id": true_ids,
            "pred_label": pred_labels,
            "pred_id": pred_ids,
            "pred_conf": probs.max(axis=1),
        })
        for i, cls in id2label.items():
            out_df[f"prob_{cls}"] = probs[:, i]

        out_csv = os.path.join(OUTPUT_DIR, "test_predictions_stage3.csv")
        out_df.to_csv(out_csv, index=False)
        with open(os.path.join(OUTPUT_DIR, "metrics_test_stage3.json"), "w", encoding="utf-8") as f:
            json.dump({"test_accuracy": acc, **{k: float(v) for k, v in test_metrics.items()}}, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 테스트 예측 저장: {out_csv}")

    # 8) 저장
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
