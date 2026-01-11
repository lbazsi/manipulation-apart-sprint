"""### Setup Code"""
import json
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

""" ### Loading dataset to merge with multi-judge scoring (next time: including question_id and model_label in judge output) """

df = pd.read_json("deeb_dataset500.jsonl", lines=True)

""" ### Downloading the multi-judge files as an initial list of lists """

with open("multi_judge_scores500.json", "r") as f:
    all_judge_outputs = json.load(f)

"""### Downloading initial mech interp data (delta framing)"""

df_mech = pd.read_json("normal_frame_deltas.jsonl", lines=True)
df_mech.drop(["layers"], axis=1, inplace=True) # Do not include layer information

"""### Define Dataclass to keep track of variables"""

@dataclass
class Args:
    batch_size: int = 32
    criteria: list[str] = field(default_factory=lambda: [
    "quality",
    "sycophancy",
    "evasion",
    "deception",
    "directness"
    ])
    learning_rate: float = 1e-3
    test_size: float = 0.3
    epochs: int = 100

# Instantiate new Args object
args = Args()

"""### Producing total feature vector dataframe (including mech interp results)"""

# Convert to PyTorch tensor object
all_judge_tensor = t.tensor([
    [
        [
            sample[c] if sample[c] is not None else 0.0
            for c in args.criteria
        ]
        for sample in judge_output
    ]
    for judge_output in all_judge_outputs
], dtype=t.float32)

# Inverse variance weighting for judge-assembly
# shape: (J, M)
judge_var = all_judge_tensor.var(dim=1)

inv_var = 1.0 / (judge_var + 1e-8)
weights = inv_var / inv_var.sum(dim=0, keepdim=True)

# weighted sum over judges
weighted_scores = (weights.unsqueeze(1) * all_judge_tensor).sum(dim=0)

# Converting scores to dataframe
scores_df = pd.DataFrame(weighted_scores.numpy(), columns=args.criteria)
# Merging together
full = pd.concat([df, scores_df], axis=1)

# Separating by framing to calculate abs dEN and dON
df_N = full[full["frame"] == "casual"].set_index("question_id", drop=True)
df_E = full[full["frame"] == "evaluation"].set_index("question_id", drop=True)
df_O = full[full["frame"] == "oversight"].set_index("question_id", drop=True)

delta_E_N = abs(df_E[args.criteria] - df_N[args.criteria])
delta_E_N.columns = [f"dEabs_{c[:5]}" for c in args.criteria]
delta_O_N = abs(df_O[args.criteria] - df_N[args.criteria])
delta_O_N.columns = [f"dOabs_{c[:5]}" for c in args.criteria]

# Initial feature dataframe
feature_df = pd.concat([delta_E_N, delta_O_N], axis=1)
feature_df["model_label"] = df_N["model_label"]
feature_df.reset_index(inplace=True)

# Including mech interp part
df_mech_idx = df_mech.set_index(["question_id", "model_label"])
feature_df_idx = feature_df.set_index(["question_id", "model_label"])

# Final full feature dataframe
full_feature_df = pd.concat([df_mech_idx, feature_df_idx], axis=1).reset_index() # Shape is 2000 obs and 14 features

"""### Classification task using simple feed forward MLP structure"""

X = full_feature_df.drop(columns=["model_label", "question_id"]).values.astype(np.float32)
y = full_feature_df["model_label"].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y).astype(np.int64)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=args.test_size, random_state=42, stratify=y_enc
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

class ModelDataset(Dataset):
    def __init__(self, X, y):
        self.X = t.tensor(X, dtype=t.float32)
        self.y = t.tensor(y, dtype=t.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ModelDataset(X_train, y_train)
val_ds = ModelDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=14, num_classes=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = MLPClassifier(input_dim=14, num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for Xb, yb in tqdm(loader, desc="Training", leave=False):
        Xb, yb = Xb.to(device), yb.to(device)

        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item() * Xb.size(0)

    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with t.no_grad():
        for Xb, yb in tqdm(loader, desc="Validation", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / len(loader.dataset)
    return val_loss, val_acc

# Helper function to produce a final confusion matrix on validation data
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with t.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = yb.numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, args):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(args.epochs):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        y_true, y_pred = get_predictions(model, val_loader, device)

        cm = confusion_matrix(y_true, y_pred)

    return history, cm

history, cm = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=100
)

# Write history file to plot accuracies etc over training epochs
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=4)

# Write final confusion matrix of the validation to understand model performance
cm_dict = {
    "labels": le.classes_.tolist(),
    "matrix": cm.tolist()
}

with open("confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)
