# ★ 환경 변수는 모든 import 전에!
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OpenMP 중복 허용 (임시 우회)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time

# ============== 경로 설정 ==============
DATASET_ROOT = r"C:\Users\User\Desktop\MinOfficialProject\pose_detection_project2\dataset"
CSV_PATH     = os.path.join(DATASET_ROOT, "ground_truth.csv")
OUTPUT_ROOT  = r"C:\Users\User\Desktop\MinOfficialProject\pose_detection_project2\processed\from_csv"

# ============== 처리 옵션 ==============
FRAME_STRIDE = 1
VIDEO_EXTS   = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv")
IMAGE_EXTS   = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ============== YOLO 포즈 모델 ==============
model = YOLO("yolov8m-pose.pt")

# ============== 유틸 ==============
def _to_abs_path(rel_or_abs: str) -> str:
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.normpath(os.path.join(DATASET_ROOT, rel_or_abs))

def _safe_rel_path(abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, DATASET_ROOT)
    except ValueError:
        return os.path.basename(abs_path)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _fmt_elapsed(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# ============== 포즈 추출 ==============
def _extract_pose_from_frame(frame):
    results = model(frame, verbose=False)
    keypoints = []
    preds = getattr(results[0].keypoints, "data", None)
    if preds is not None and preds.numel() > 0:
        for kp in preds.cpu().numpy():      # (17, 3)
            coords = kp[:, :2].flatten()    # (17,2)->(34,)
            keypoints.append(coords.astype(np.float32))
    if not keypoints:
        keypoints = [np.zeros(34, dtype=np.float32)]
    return np.stack(keypoints, axis=0)

def extract_multi_pose_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return None
    sequence, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if FRAME_STRIDE > 1 and (idx % FRAME_STRIDE != 0):
            idx += 1
            continue
        sequence.append(_extract_pose_from_frame(frame))
        idx += 1
    cap.release()
    if not sequence:
        return None
    return np.array(sequence, dtype=object)

def extract_multi_pose_from_images(path_or_list):
    if isinstance(path_or_list, (list, tuple)):
        files = [p for p in path_or_list if os.path.splitext(p)[1].lower() in IMAGE_EXTS and os.path.exists(p)]
        files.sort()
    else:
        p = path_or_list
        if os.path.isdir(p):
            files = sorted(
                os.path.join(p, f)
                for f in os.listdir(p)
                if f.lower().endswith(IMAGE_EXTS)
            )
        else:
            files = [p] if (os.path.splitext(p)[1].lower() in IMAGE_EXTS and os.path.exists(p)) else []
    if not files:
        return None
    sequence = []
    for i, fp in enumerate(files):
        if FRAME_STRIDE > 1 and (i % FRAME_STRIDE != 0):
            continue
        frame = cv2.imread(fp)
        if frame is None:
            continue
        sequence.append(_extract_pose_from_frame(frame))
    if not sequence:
        return None
    return np.array(sequence, dtype=object)

def extract_multi_pose_from_path(any_path: str):
    if os.path.isdir(any_path):
        return extract_multi_pose_from_images(any_path)
    ext = os.path.splitext(any_path)[1].lower()
    if ext in VIDEO_EXTS:
        return extract_multi_pose_from_video(any_path)
    if ext in IMAGE_EXTS:
        return extract_multi_pose_from_images(any_path)
    print(f"⚠ 지원하지 않는 경로 유형: {any_path}")
    return None

# ============== 실행 ==============
if __name__ == "__main__":
    # CSV 로드 및 검증
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    required_cols = {"video_path", "stage1_label", "stage2_label", "stage3_label"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {required_cols} / 현재={set(df.columns)}")

    rows = df.to_dict(orient="records")
    print(f"📄 CSV 로드 완료: {CSV_PATH} / 총 {len(rows)}개 항목")

    processed_ok = skipped_missing = skipped_extract = already_done = 0

    for i, row in enumerate(rows, 1):
        rel_path = str(row["video_path"]).strip()
        if not rel_path:
            print(f"⚠ ({i}) video_path 비어있음 → 스킵")
            skipped_missing += 1
            continue

        # 원본 라벨 문자열 그대로 저장
        s1 = str(row.get("stage1_label")) if pd.notna(row.get("stage1_label")) else ""
        s2 = str(row.get("stage2_label")) if pd.notna(row.get("stage2_label")) else ""
        s3 = str(row.get("stage3_label")) if pd.notna(row.get("stage3_label")) else ""

        abs_path = _to_abs_path(rel_path)
        if not os.path.exists(abs_path):
            print(f"⚠ ({i}) 경로 없음(스킵): {abs_path}")
            skipped_missing += 1
            continue

        rel_safe = _safe_rel_path(abs_path)
        base_no_ext = os.path.splitext(os.path.basename(abs_path))[0]
        out_dir = os.path.join(OUTPUT_ROOT, os.path.dirname(rel_safe))
        _ensure_dir(out_dir)

        feat_path  = os.path.join(out_dir, f"{base_no_ext}_features.npy")
        label_path = os.path.join(out_dir, f"{base_no_ext}_label.npy")

        if os.path.exists(feat_path) and os.path.exists(label_path):
            print(f"⏭ ({i}/{len(rows)}) 이미 처리됨: {rel_safe}")
            already_done += 1
            continue

        print(f"▶▶ ({i}/{len(rows)}) 처리 중: {rel_safe}")
        t0 = time.perf_counter()  # ▶ 아이템 시작

        try:
            feat = extract_multi_pose_from_path(abs_path)
            if feat is None:
                print(f"⚠ 스킵됨(추출 실패/빈 시퀀스): {rel_safe}")
                skipped_extract += 1
                continue

            # 저장
            np.save(feat_path, feat)
            labels_arr = np.array([s1, s2, s3], dtype=np.str_)
            np.save(label_path, labels_arr)

            processed_ok += 1
            frames = len(feat)
            dt = time.perf_counter() - t0    # ▶ 아이템 소요시간
            fps = (frames / dt) if dt > 0 else float("inf")
            print(f"✔ 저장 완료: {feat_path}")
            print(f"   ⏱ 처리시간: {_fmt_elapsed(dt)} | 프레임: {frames} | 처리속도: {fps:.2f} fps")

        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"❌ 에러로 스킵({ _fmt_elapsed(dt) }): {rel_safe} :: {e}")
            skipped_extract += 1

    # (원하면 총괄 통계는 유지)
    print("\n🎉 CSV 기반 멀티 포즈 전처리 완료! (label.npy에 stage1/2/3 원본 라벨 저장)")
    print(f"   ├─ 처리 성공: {processed_ok}")
    print(f"   ├─ 이미 처리됨: {already_done}")
    print(f"   ├─ 누락/경로없음: {skipped_missing}")
    print(f"   └─ 추출 실패: {skipped_extract}")
