"""
Clasificador LSTM de tune-type sobre notación ABC tokenizada a nivel de
carácter. Hiperparámetros de entrenamiento fundamentados en el patrón
dominante observado en los 16 papers ICASSP2026 de referencia (no un
solo paper): AdamW, LR 1e-4-1e-3 con warmup+cosine decay, gradient
clipping por norma, early stopping por validation loss con paciencia
~8-10 épocas. El manejo de desbalance de clases (cross-entropy ponderada
por frecuencia inversa) no tiene precedente en ese pool de papers y es
práctica estándar de ML para clasificación multiclase desbalanceada.
"""
import csv
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

MAX_LEN = 600
EMB_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 128
LR = 1e-3
WARMUP_STEPS = 200
MAX_EPOCHS = 60
PATIENCE = 10
GRAD_CLIP_NORM = 5.0


def load_split(name):
    with open(f"{name}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["abc"], r["type"]) for r in rows]


class CharVocab:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.stoi = {c: i + 2 for i, c in enumerate(chars)}  # 0=pad, 1=unk
        self.itos = {i: c for c, i in self.stoi.items()}
        self.pad_idx = 0
        self.unk_idx = 1

    def encode(self, text):
        return [self.stoi.get(c, self.unk_idx) for c in text[:MAX_LEN]]

    def __len__(self):
        return len(self.stoi) + 2


class TuneDataset(Dataset):
    def __init__(self, data, vocab, label2idx):
        self.data = data
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        ids = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return ids, len(ids), self.label2idx[label]


def collate(batch):
    seqs, lengths, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.tensor(lengths), torch.tensor(labels)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_EPOCHS * 300 - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def run_epoch(model, loader, criterion, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_labels, all_preds


def main():
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")

    vocab = CharVocab([t for t, _ in train_data])
    labels = sorted(set(l for _, l in train_data))
    label2idx = {l: i for i, l in enumerate(labels)}

    print("vocab size:", len(vocab), "num classes:", len(labels))
    print("labels:", labels)

    train_ds = TuneDataset(train_data, vocab, label2idx)
    val_ds = TuneDataset(val_data, vocab, label2idx)
    test_ds = TuneDataset(test_data, vocab, label2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # class weights por frecuencia inversa (cross-entropy ponderada)
    counts = np.zeros(len(labels))
    for _, l in train_data:
        counts[label2idx[l]] += 1
    class_weights = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float32).to(DEVICE)

    model = LSTMClassifier(len(vocab), EMB_DIM, HIDDEN_DIM, len(labels)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # sin class weights (ablation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"early stopping en epoch {epoch} (paciencia={PATIENCE})")
                break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, test_labels, test_preds = run_epoch(model, test_loader, criterion)
    print(f"\nTEST: loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")
    report = classification_report(test_labels, test_preds, target_names=labels, digits=3)
    print(report)

    results = {
        "test_loss": test_loss, "test_acc": test_acc, "test_macro_f1": test_f1,
        "labels": labels, "history": history, "classification_report": report,
        "test_labels": test_labels, "test_preds": test_preds,
        "hyperparams": {"emb_dim": EMB_DIM, "hidden_dim": HIDDEN_DIM, "lr": LR,
                         "batch_size": BATCH_SIZE, "grad_clip_norm": GRAD_CLIP_NORM,
                         "patience": PATIENCE, "vocab_size": len(vocab)}
    }
    with open("lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados guardados en lstm_results.json")


if __name__ == "__main__":
    main()
