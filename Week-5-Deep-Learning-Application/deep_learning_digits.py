import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)

X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

model = torch.nn.Sequential(
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.25),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.20),
    torch.nn.Linear(64, 10),
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_state = None
best_val_loss = float("inf")
wait = 0
history = []

for epoch in range(60):
    model.train()
    indices = torch.randperm(len(X_train))

    for start in range(0, len(X_train), 64):
        idx = indices[start:start + 64]
        optimizer.zero_grad()
        loss = criterion(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        val_loss = criterion(model(X_val), y_val).item()
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

    history.append((epoch + 1, train_loss, val_loss, val_acc))

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if wait >= 8:
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    predictions = model(X_test).argmax(1).numpy()

y_true = y_test.numpy()
print(f"Test accuracy: {accuracy_score(y_true, predictions):.4f}")
print(f"Macro F1: {f1_score(y_true, predictions, average='macro'):.4f}")
print("\nClassification report:\n")
print(classification_report(y_true, predictions))
print("Confusion matrix:\n", confusion_matrix(y_true, predictions))
