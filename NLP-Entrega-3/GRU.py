import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 42
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 512
EPOCHS = 15
HIDDEN_DIM = 128
LR = 1e-3

BASE_PATH = (
    "C:/Users/d.razquin/Desktop/4ºCURSO/PRIMER CUATRIMESTRE/"
    "Fundamentos del Procesamiento de Lenguaje Natural/PROYECTO/Codigo/"
    "NLP-Entrega-3/datos"
)
PROCESSED_DIR = os.path.join(BASE_PATH, "procesado")
MODELS_DIR = os.path.join(BASE_PATH, "modelos")


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, seq_len=1, input_dim)
        x = x.unsqueeze(1)
        _, h_n = self.gru(x)
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_final)


def load_processed_data():
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "target_names.npy"]
    for fname in required:
        if not os.path.exists(os.path.join(PROCESSED_DIR, fname)):
            raise FileNotFoundError(
                f"No se encontró '{fname}' en {PROCESSED_DIR}. "
                "Ejecuta primero 'procesamiento.py'."
            )

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    target_names = np.load(os.path.join(PROCESSED_DIR, "target_names.npy"), allow_pickle=True)
    return X_train, X_test, y_train, y_test, target_names


def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    X_train, X_test, y_train, y_test, target_names = load_processed_data()
    num_classes = len(target_names)
    input_dim = X_train.shape[1]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST, shuffle=False)

    gru_model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        gru_model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = gru_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} - Loss: {total_loss / len(train_ds):.4f}")

    gru_model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            logits = gru_model(batch_x.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds)
    print("=== Resultados GRU ===")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "gru_model.pt")
    torch.save(gru_model.state_dict(), model_path)
    print(f"Pesos del modelo guardados en: {model_path}")


if __name__ == "__main__":
    main()