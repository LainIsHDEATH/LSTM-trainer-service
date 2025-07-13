import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi import HTTPException
from app.models.lstm_model import TemperatureLSTM
from app.database import get_db_connection
from app.config import STORE_API
import joblib
import logging

logger = logging.getLogger("lstm_trainer")

def fetch_simulation(sim_id: int) -> np.ndarray:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT temp_in, temp_out, heater_power, temp_setpoint
                FROM simulations.simulation_events
                WHERE simulation_id = %s
                ORDER BY timestamp
                """,
                (sim_id,),
            )
            rows = cur.fetchall()
    if not rows:
        raise ValueError(f"Дані відсутні для simulation_id = {sim_id}")
    return np.asarray(rows, dtype=np.float32)

def make_sequences(data: np.ndarray, seq_len: int):
    n = data.shape[0] - seq_len
    X = np.stack([data[i : i + seq_len] for i in range(n)])
    y = data[seq_len:, 0]
    return X, y

def train_val_test_split(X, y, train=0.7, val=0.15):
    n = len(X)
    n_train = int(n * train)
    n_val = int(n * val)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_te, y_te = X[n_train + n_val :], y[n_train + n_val :]
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

def normalise_sets(train_X, *other_sets):
    mean = train_X.reshape(-1, train_X.shape[-1]).mean(0)
    std = train_X.reshape(-1, train_X.shape[-1]).std(0)
    std[std == 0] = 1.0
    res = [(train_X - mean) / std]
    for s in other_sets:
        res.append((s - mean) / std)
    return res, mean, std

def evaluate(model, X, y, y_mean, y_std, batch=64):
    model.eval()
    dev = next(model.parameters()).device
    total_mae = 0.0
    total_mse = 0.0
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.tensor(X[i : i + batch], device=dev)
            pred_n = model(xb).cpu().numpy()
            pred   = pred_n * y_std + y_mean
            err    = y[i : i + batch] - pred
            total_mae += np.abs(err).sum()
            total_mse += (err ** 2).sum()
    mae  = total_mae / n
    mse  = total_mse / n
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def train_sync(req):
    try:
        data = fetch_simulation(req.simulationId)
    except Exception as exc:
        raise HTTPException(500, f"Помилка БД: {exc}")

    try:
        X, y = make_sequences(data, req.SEQ_LENGTH)
    except ValueError:
        raise HTTPException(400, "Недостатньо точок для заданої довжини вікна")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train_val_test_split(X, y)
    (X_tr_n, X_val_n, X_te_n), x_mean, x_std = normalise_sets(X_tr, X_val, X_te)

    y_mean = float(x_mean[0])
    y_std = float(x_std[0])
    y_tr_n = (y_tr - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    def loader(X_, y_, shuffle):
        if len(X_) == 0:
            return None
        dset = TensorDataset(torch.tensor(X_), torch.tensor(y_))
        return DataLoader(dset, batch_size=req.batch_size, shuffle=shuffle)

    dl_train = loader(X_tr_n, y_tr_n, True)
    dl_val = loader(X_val_n, y_val_n, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemperatureLSTM(X.shape[-1], req.HIDDEN_SIZE, req.NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state: Optional[dict] = None
    best_val = float("inf")

    for epoch in range(1, req.epochs_number + 1):
        model.train()
        train_losses = []
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = None
        if dl_val:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(model(xb.to(device)), yb.to(device)).item()
                    for xb, yb in dl_val
                ) / len(dl_val)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(f"Epoch {epoch:3}: Train={train_loss:.6f}  Val={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    model.to('cpu')
    torch.cuda.empty_cache()
    device = torch.device('cpu')

    mae, mse, rmse = evaluate(
        model, X_te_n, y_te, y_mean, y_std, batch=int(req.batch_size / 2)
    )

    preds_n = model(torch.tensor(X_te_n, device=device)).detach().cpu().numpy()
    preds = preds_n * y_std + y_mean
    trues = y_te
    r2 = float(1 - ((trues - preds) ** 2).sum() / ((trues - trues.mean()) ** 2).sum())

    base = Path(f"models/room-{req.roomId}/model-unknown")
    base.mkdir(parents=True, exist_ok=True)

    meta = {
        "hidden_size": req.HIDDEN_SIZE,
        "num_layers": req.NUM_LAYERS,
        "seq_length": req.SEQ_LENGTH,
        "state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    joblib.dump(meta, base / "checkpoint.joblib")
    torch.save(model.state_dict(), base / "model.pth")
    with open(base / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"MAE={mae:.4f}\nMSE={mse:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}\n")

    return {
        "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2},
    }
