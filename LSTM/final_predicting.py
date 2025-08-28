# train_stageX_70_20_10.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OMP 중복 허용(임시 우회)
os.environ.setdefault("OMP_NUM_THREADS", "1") # 스레드 과다 방지
os.environ.setdefault("MKL_NUM_THREADS", "1")
import os, sys, json, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



# ───────────── 사용자 설정 ─────────────
PROCESSED_ROOT = r"C:\Users\User\Desktop\MinOfficialProject\pose_detection_project2\processed\from_csv"
STAGE = 1                 # ★ 1/2/3 중 원하는 스테이지로 변경
EPOCHS = 100
BATCH_SIZE = 32
MAX_SEQ_LEN = 300
SEED = 63065

# 모델 하이퍼파라미터(기존과 동일)
conv_filters = 128
lstm_hidden  = 64
dropout_rate = 0.4
bidirectional = False

# ───────────── 디바이스/시드 ─────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ───────────── 모델 정의 ─────────────
class ConvLSTMBackbone(nn.Module):
    def __init__(self, input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv_filters)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm  = nn.LSTM(conv_filters, lstm_hidden, batch_first=True, bidirectional=bidirectional)
        self.out_dim = lstm_hidden * (2 if bidirectional else 1)

    def forward(self, x):  # x: (B, T, C=34)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return out[:, -1, :]

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feat):
        return self.fc(feat)

class AggressionDetectionModel(nn.Module):
    def __init__(self, input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional, num_classes):
        super().__init__()
        self.backbone   = ConvLSTMBackbone(input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional)
        self.classifier = ClassificationHead(self.backbone.out_dim, num_classes)
    def forward(self, x):
        return self.classifier(self.backbone(x))

# ───────────── 데이터셋 ─────────────
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ───────────── 유틸: 라벨 정규화 ─────────────
def normalize_label_str(s: str) -> str:
    if s is None: return ""
    t = str(s).strip()
    if t.lower() == "nan": return ""
    t = t.replace(" ", "_")
    fixes = {
        "Hiiting_With_Object": "Hitting_With_Object",
        "Hitting_With_Object,": "Hitting_With_Object",
        "Blocking.": "Blocking",
        "Looming.": "Looming",
    }
    return fixes.get(t, t)

def map_stage1(stage1_str: str):
    s = normalize_label_str(stage1_str)
    if s == "": return None
    return 0 if s == "Benign" else 1

def list_feature_files(root_dir):
    feat_files = []
    for parent, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_features.npy"):
                feat_files.append(os.path.join(parent, f))
    feat_files.sort()
    return feat_files

# ───────────── 데이터 로드 (STAGE별) ─────────────
def load_from_processed_stageX(root_dir, stage: int):
    assert stage in (1, 2, 3)
    feat_files = list_feature_files(root_dir)
    print(f"[INFO] 전처리 파일 수: {len(feat_files)}")

    # 1차 스캔: 클래스 수집
    labels_set = set()
    missing = 0
    for feat_path in feat_files:
        base = os.path.splitext(feat_path)[0].replace("_features", "")
        label_path = base + "_label.npy"
        if not os.path.exists(label_path):
            missing += 1; continue
        try:
            arr = np.load(label_path, allow_pickle=True)
            if len(arr) < 3: continue
            if stage == 1:
                y = map_stage1(arr[0])
                if y is not None:
                    labels_set.add(y)  # 0/1
            else:
                s = normalize_label_str(arr[stage-1])
                if s != "":
                    labels_set.add(s)
        except Exception:
            missing += 1

    # 클래스 목록/매핑
    if stage == 1:
        classes = ["Benign", "Anomaly"]
        label2id = {"Benign": 0, "Anomaly": 1}
    else:
        if len(labels_set) < 2:
            raise RuntimeError(f"❌ stage{stage} 클래스가 2종 미만: {labels_set}")
        classes = sorted(list(labels_set))
        label2id = {c: i for i, c in enumerate(classes)}
    id2label = {v: k for k, v in label2id.items()}
    print(f"[INFO] stage{stage} 클래스({len(classes)}): {', '.join(classes)}")
    with open(f"stage{stage}_label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"💾 라벨 매핑 저장: stage{stage}_label_map.json")

    # 2차 스캔: 시퀀스 생성
    X_list, y_list = [], []
    for feat_path in feat_files:
        base = os.path.splitext(feat_path)[0].replace("_features", "")
        label_path = base + "_label.npy"
        if not os.path.exists(label_path): continue
        try:
            labels = np.load(label_path, allow_pickle=True)
            if len(labels) < 3: continue

            if stage == 1:
                y = map_stage1(labels[0])
                if y is None: continue
            else:
                s = normalize_label_str(labels[stage-1])
                if s == "" or s not in label2id: continue
                y = label2id[s]

            raw = np.load(feat_path, allow_pickle=True)  # dtype=object, len=frames
            people = {}
            for frame in raw:
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 1 and frame.shape[0] == 34:
                        people.setdefault(0, []).append(frame)
                    elif frame.ndim == 2 and frame.shape[1] == 34:
                        for pid, pose in enumerate(frame):
                            people.setdefault(pid, []).append(pose)

            for _, poses in people.items():
                seq = np.asarray(poses, dtype=np.float32)
                if seq.ndim != 2 or seq.shape[1] != 34: continue
                if seq.shape[0] == 0: continue
                if seq.shape[0] > MAX_SEQ_LEN:
                    seq = seq[:MAX_SEQ_LEN]
                X_list.append(seq)
                y_list.append(y)
        except Exception as e:
            print(f"[WARN] 로드 스킵: {feat_path} :: {e}")

    if not X_list:
        raise RuntimeError("❌ 유효한 샘플이 없습니다. (라벨/경로/파일 확인)")

    # 패딩
    max_len = max(seq.shape[0] for seq in X_list)
    input_size = X_list[0].shape[1]
    N = len(X_list)
    X_padded = np.zeros((N, max_len, input_size), dtype=np.float32)
    for i, seq in enumerate(X_list):
        T = seq.shape[0]
        X_padded[i, :T] = seq
    y_arr = np.asarray(y_list, dtype=np.int64)

    binc = np.bincount(y_arr, minlength=len(classes))
    dist_str = ", ".join(f"{id2label[i]}:{int(binc[i])}" for i in range(len(binc)))
    print(f"[INFO] 샘플={N}, 최대길이={max_len}, 입력차원={input_size}, 라벨분포=({dist_str})")

    return X_padded, y_arr, max_len, input_size, classes, id2label

# ───────────── 평가(예측 반환) ─────────────
def evaluate_with_preds(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(yb.numpy())
    return np.array(all_true), np.array(all_pred)

# ───────────── 학습/평가 ─────────────
def train_val_test(X, y, input_size, classes, id2label, stage):
    # 7:2:1 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=SEED)
    val_ratio_in_temp = 0.20 / 0.90
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio_in_temp,
                                                      stratify=y_temp, random_state=SEED)
    print(f"[INFO] 분할: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    train_loader = DataLoader(PoseDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PoseDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(PoseDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    model = AggressionDetectionModel(
        input_size=input_size,
        conv_filters=conv_filters,
        lstm_hidden=lstm_hidden,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        num_classes=len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            y_true_val, y_pred_val = evaluate_with_preds(model, val_loader)
            val_acc = accuracy_score(y_true_val, y_pred_val)
            print(f"[Epoch {epoch:03d}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test: 전체 지표 출력 ──
    y_true_test, y_pred_test = evaluate_with_preds(model, test_loader)
    acc = accuracy_score(y_true_test, y_pred_test)
    target_names = [id2label[i] for i in range(len(classes))]
    print("\n================= Test Metrics (stage{}; 7:2:1) =================".format(stage))
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_true_test, y_pred_test, target_names=target_names, digits=4))

    ckpt = f"model_stage{stage}_70_20_10.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"✅ 모델 저장 완료: {ckpt}")

# ───────────── 메인 ─────────────
if __name__ == "__main__":
    if not os.path.isdir(PROCESSED_ROOT):
        print(f"[ERROR] 폴더 없음: {PROCESSED_ROOT}")
        sys.exit(1)

    X, y, seq_len, input_size, classes, id2label = load_from_processed_stageX(PROCESSED_ROOT, STAGE)
    print(f"[INFO] 데이터: X={X.shape}, y={y.shape}, seq_len={seq_len}, input_size={input_size}")
    train_val_test(X, y, input_size, classes, id2label, STAGE)
