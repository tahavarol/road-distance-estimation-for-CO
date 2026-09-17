import datetime, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


for grid_size in [(50, 50),(100, 100),(150, 150),(200, 200),(250, 250)]:
    for feature_size in ['_large', '_small']:
        for _loss_ in ['MAPE', 'MSE', 'MAE']:
            print(grid_size, feature_size, _loss_)
# ──────────────────────── 1) Dataset meta ──────────────────────────
            with open(f"dataset/final/meta_{grid_size[0]}_{grid_size[1]}{feature_size}_new.json") as f:
                meta = json.load(f)
            n_train, n_test, n_feats = meta["train_rows"], meta["test_rows"], meta["n_feats"]

 
            train_feats = np.memmap(f"dataset/final/train_feats_{grid_size[0]}_{grid_size[1]}{feature_size}_new.dat", dtype="float32", mode="r",
                                    shape=(n_train, n_feats))
            test_feats  = np.memmap(f"dataset/final/test_feats_{grid_size[0]}_{grid_size[1]}{feature_size}_new.dat", dtype="float32", mode="r",
                                    shape=(n_test,  n_feats))
            train_labels = np.load(f"dataset/final/train_labels_{grid_size[0]}_{grid_size[1]}{feature_size}_new.npy")
            test_labels  = np.load(f"dataset/final/test_labels_{grid_size[0]}_{grid_size[1]}{feature_size}_new.npy")

            print(f"Train features shape: {train_feats.shape}")
            print(f"Test  features shape: {test_feats.shape}")

            # ──────────────────────── 2) Filtering ─────────────────────────────
            mask_tr  = train_labels >= 1
            mask_te  = test_labels  >= 1
            train_feats, train_labels = train_feats[mask_tr], train_labels[mask_tr]
            test_feats,  test_labels  = test_feats [mask_te], test_labels [mask_te]

            print(f"Filtered train features shape: {train_feats.shape}")
            print(f"Filtered test  features shape: {test_feats.shape}")

            # ──────────────────────── 3) Save-dir ──────────────────────────────
            folder_name = 'Final'                                    # hard-coded as in original
            save_dir = Path(f"models/{folder_name}"); save_dir.mkdir(parents=True, exist_ok=True)

            # ─────────────────────── 3.1) Resume config ─────────────────────────
            RESUME        = False
            RESUME_EPOCH  = 1          # ← the saved epoch you want
            RESUME_LOSS   = 'MAPE'      # ← the loss that produced that checkpoint

            # Build expected ckpt path from your naming template
            resume_ckpt = save_dir / (
                f"model_opt_{RESUME_LOSS}_ckpt_{RESUME_LOSS}_"
                f"{feature_size}_epoch{RESUME_EPOCH}_{grid_size[0]}_{grid_size[1]}.pt"
            )

            # Only resume when we're in the matching loss loop and the file exists
            RESUMING = RESUME and (_loss_ == RESUME_LOSS) and resume_ckpt.exists()

            # ──────────────────────── 4) Config ────────────────────────────────
            BATCH_SIZE, EPOCHS          = 1024, 250
            HIDDEN_DIM, HIDDEN_LAYERS   = 1024, 6
            EVAL_EVERY, VAL_SPLIT       = 1, 0.20
            LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
            LOSS_TO_OPTIMISE            = _loss_   # 'MSE' | 'MAPE' | 'MAE' | 'MQE'
            CKPT_METRIC                 = _loss_   # 'MSE' | 'MAPE' | 'MAE' | 'MQE'
            SEED, LOG_TRANSFORM         = 12, False
            EARLY_STOP_PATIENCE         = 25

            # ── New: choose between epoch-based or time-limited training ───────
            TRAIN_MODE = 'time'          # 'epoch' or 'time'
            TIME_LIMIT_SECONDS = 3600        # e.g., 7200; only used if TRAIN_MODE == 'time'
            if TRAIN_MODE == 'time':
                assert TIME_LIMIT_SECONDS and TIME_LIMIT_SECONDS > 0, \
                    "When TRAIN_MODE='time', TIME_LIMIT_SECONDS must be a positive integer."

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            device = 'mps'
            print(f"Using device: {device}")

            # ──────────────────────── 5) Model ────────────────────────────────
            class MLP(nn.Module):
                def __init__(self, input_dim, hidden_dim=1024, num_layers=6):
                    super().__init__()
                    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                    for _ in range(num_layers - 2):
                        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
                    layers.append(nn.Linear(hidden_dim, 1))
                    self.net = nn.Sequential(*layers)
                def forward(self, x): return self.net(x)

            # ──────────────────────── 6) Losses & metrics ─────────────────────
            class MAPELoss(nn.Module):
                def __init__(self, eps=0.): super().__init__(); self.eps = eps
                def forward(self, pred, tgt):
                    return torch.mean(torch.abs((tgt - pred) / (tgt + self.eps))) * 100

            class MAELoss(nn.Module):
                def forward(self, pred, tgt): return torch.mean(torch.abs(tgt - pred))

            class MQELoss(nn.Module):
                """Mean Quartic Error (E[(ŷ- y)^4])."""
                def forward(self, pred, tgt): return torch.mean((tgt - pred) ** 4)

            def calculate_mape(pred, tgt, eps=0.):
                return torch.mean(torch.abs((tgt - pred) / (tgt + eps)) * 100).item()

            def calculate_mae(pred, tgt):
                return torch.mean(torch.abs(tgt - pred)).item()

            def calculate_mqe(pred, tgt):
                return torch.mean((tgt - pred) ** 4).item()

            criterion_dict = {
                'MSE' : nn.MSELoss(),
                'MAPE': MAPELoss(),
                'MAE' : MAELoss(),
                'MQE' : MQELoss()
            }
            metric_fns = {
                'MSE' : lambda p,t: nn.functional.mse_loss(p, t).item(),
                'MAPE': calculate_mape,
                'MAE' : calculate_mae,
                'MQE' : calculate_mqe
            }

            assert LOSS_TO_OPTIMISE in criterion_dict, "Unknown LOSS_TO_OPTIMISE"
            assert CKPT_METRIC      in metric_fns,     "Unknown CKPT_METRIC"

            # ──────────────────────── 7) Data prep ────────────────────────────
            if LOG_TRANSFORM:
                train_labels, test_labels = np.log(train_labels), np.log(test_labels)

            X_train = torch.tensor(train_feats, dtype=torch.float32)
            y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
            X_test  = torch.tensor(test_feats , dtype=torch.float32)
            y_test  = torch.tensor(test_labels , dtype=torch.float32).unsqueeze(1)

            train_ds = TensorDataset(X_train, y_train)
            test_ds  = TensorDataset(X_test, y_test)
            val_size = int(len(train_ds) * VAL_SPLIT)
            train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
            val_dl   = DataLoader(val_ds  , batch_size=BATCH_SIZE)
            test_dl  = DataLoader(test_ds , batch_size=BATCH_SIZE)

            # ──────────────────────── 8) Init model/opt/sched ────────────────
            model = MLP(X_train.shape[1], HIDDEN_DIM, HIDDEN_LAYERS).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                        weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=False)
            # ───────────────────── 8.1) Resume state (if requested) ─────────────
            start_epoch = 0

            # Initialize tracking defaults
            history = {k: [] for k in
                    ('train_loss','val_loss','test_loss',
                        'train_mse','val_mse','test_mse',
                        'train_mape','val_mape','test_mape',
                        'train_mae','val_mae','test_mae',
                        'train_mqe','val_mqe','test_mqe')}
            best_val_metric = float('inf'); best_epoch = 0; patience = 0

            if RESUMING:
                ckpt = torch.load(resume_ckpt, map_location=torch.device(device))
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

                # Restore history if present
                if 'history' in ckpt and isinstance(ckpt['history'], dict):
                    history = ckpt['history']

                # Recompute early-stop state from history for the selected CKPT_METRIC
                metric_key = f"val_{CKPT_METRIC.lower()}"
                if metric_key in history and len(history[metric_key]) > 0:
                    import numpy as _np
                    vals = _np.array(history[metric_key], dtype=float)
                    best_val_metric = float(vals.min())
                    best_epoch = int(vals.argmin())
                    patience = len(vals) - 1 - best_epoch

                # ckpt['epoch'] was saved zero-based; resume at next epoch
                start_epoch = int(ckpt.get('epoch', RESUME_EPOCH - 1)) + 1
                print(f"↩ Resuming from {resume_ckpt} at epoch {start_epoch} "
                    f"(best {CKPT_METRIC}={best_val_metric:.4f} @ epoch {best_epoch+1})")
            # ──────────────────────── 9) Tracking ────────────────────────────
            history = {k: [] for k in
                    ('train_loss','val_loss','test_loss',
                        'train_mse','val_mse','test_mse',
                        'train_mape','val_mape','test_mape',
                        'train_mae','val_mae','test_mae',
                        'train_mqe','val_mqe','test_mqe')}
            log_path = save_dir / f"{LOSS_TO_OPTIMISE}_{CKPT_METRIC}{feature_size}_{grid_size[0]}_{grid_size[1]}_metrics_log.txt"
            with open(log_path, "a" if RESUMING else "w") as f:
                if not RESUMING:
                    hdr = ",".join(["Epoch"] + list(history.keys())) + "\n"; f.write(hdr)

            best_val_metric = float('inf'); best_epoch = 0; patience = 0

            # ── New: print training mode info
            if TRAIN_MODE == 'time':
                print(f"Start training - optimise: {LOSS_TO_OPTIMISE}, ckpt on: {CKPT_METRIC} "
                      f"(mode: time, limit: {TIME_LIMIT_SECONDS}s)")
            else:
                print(f"Start training - optimise: {LOSS_TO_OPTIMISE}, ckpt on: {CKPT_METRIC} "
                      f"(mode: epoch, epochs: {EPOCHS})")

            # ──────────────────────── 10) Training loop ──────────────────────
            # New: support time-based loop
            time_start = time.time()
            epochs_label = EPOCHS if TRAIN_MODE == 'epoch' else '∞'
            epoch_iter = range(start_epoch, EPOCHS) if TRAIN_MODE == 'epoch' else range(start_epoch, 10**9)

            for epoch in epoch_iter:
                model.train(); running = {k:0. for k in history if k.startswith('train')}
                pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs_label}", leave=False)

                for xb, yb in pbar:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion_dict[LOSS_TO_OPTIMISE](pred, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    running['train_loss'] += loss.item() * xb.size(0)

                    with torch.no_grad():
                        if LOG_TRANSFORM:
                            exp_p, exp_y = torch.exp(pred), torch.exp(yb)
                            running['train_mse']  += nn.functional.mse_loss(exp_p, exp_y, reduction='sum').item()
                            running['train_mape'] += calculate_mape(exp_p, exp_y) * xb.size(0)
                            running['train_mae']  += calculate_mae (exp_p, exp_y) * xb.size(0)
                            running['train_mqe']  += calculate_mqe (exp_p, exp_y) * xb.size(0)
                        else:
                            running['train_mse']  += nn.functional.mse_loss(pred, yb, reduction='sum').item()
                            running['train_mape'] += calculate_mape(pred, yb) * xb.size(0)
                            running['train_mae']  += calculate_mae (pred, yb) * xb.size(0)
                            running['train_mqe']  += calculate_mqe (pred, yb) * xb.size(0)

                for k in running: running[k] /= len(train_ds)

                # ── Validation / Test helpers ────────────────────────────────
                def eval_loop(loader, prefix):
                    totals = {f'{prefix}_{m}':0. for m in ('loss','mse','mape','mae','mqe')}
                    with torch.no_grad():
                        for xb, yb in loader:
                            xb, yb = xb.to(device), yb.to(device)
                            pred = model(xb)
                            loss = criterion_dict[LOSS_TO_OPTIMISE](pred, yb).item()
                            if LOG_TRANSFORM:
                                exp_p, exp_y = torch.exp(pred), torch.exp(yb)
                                mse  = nn.functional.mse_loss(exp_p, exp_y).item()
                                mape = calculate_mape(exp_p, exp_y)
                                mae  = calculate_mae (exp_p, exp_y)
                                mqe  = calculate_mqe (exp_p, exp_y)
                            else:
                                mse  = nn.functional.mse_loss(pred, yb).item()
                                mape = calculate_mape(pred, yb)
                                mae  = calculate_mae (pred, yb)
                                mqe  = calculate_mqe (pred, yb)
                            totals[f'{prefix}_loss'] += loss   * xb.size(0)
                            totals[f'{prefix}_mse']  += mse    * xb.size(0)
                            totals[f'{prefix}_mape'] += mape   * xb.size(0)
                            totals[f'{prefix}_mae']  += mae    * xb.size(0)
                            totals[f'{prefix}_mqe']  += mqe    * xb.size(0)
                    for k in totals: totals[k] /= len(loader.dataset)
                    return totals

                val_stats  = eval_loop(val_dl , 'val')
                test_stats = eval_loop(test_dl, 'test') if (epoch+1)%EVAL_EVERY==0 else \
                            {f'test_{m}':None for m in ('loss','mse','mape','mae','mqe')}

                # ── Aggregate history & log ──────────────────────────────────
                epoch_stats = {**running, **val_stats, **test_stats}
                for k in history: history[k].append(epoch_stats[k])
                with open(log_path,"a") as f:
                    line = ",".join([str(epoch+1)] + [str(epoch_stats[k]) for k in history]) + "\n"
                    f.write(line)

                # ── Scheduler & checkpoint ───────────────────────────────────
                scheduler.step(val_stats[f'val_{CKPT_METRIC.lower()}'])
                current_val = val_stats[f'val_{CKPT_METRIC.lower()}']
                if current_val < best_val_metric:
                    best_val_metric, best_epoch, patience = current_val, epoch, 0
                    ckpt = {'epoch':epoch,'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'history': history}
                    torch.save(ckpt, save_dir / f"model_opt_{LOSS_TO_OPTIMISE}_ckpt_{CKPT_METRIC}{feature_size}_{grid_size[0]}_{grid_size[1]}.pt")
                    print(f"✔ Saved checkpoint (epoch {epoch+1})")
                else:
                    patience += 1
                    if patience >= EARLY_STOP_PATIENCE:
                        print("Early stopping."); break

                print(f"Epoch {epoch+1:3}/{epochs_label} ▸ "
                    f"train {running['train_loss']:.4f} | "
                    f"val {val_stats['val_loss']:.4f} | "
                    f"{CKPT_METRIC} {current_val:.4f}")

                # ── New: stop if time budget reached (checked at end of epoch) ──
                if TRAIN_MODE == 'time' and (time.time() - time_start) >= TIME_LIMIT_SECONDS:
                    elapsed = int(time.time() - time_start)
                    print(f"⏱️ Time limit reached ({elapsed}s ≥ {TIME_LIMIT_SECONDS}s). Stopping training.")
                    break

            # ──────────────────────── 11) Plots ───────────────────────────────
            plt.figure(figsize=(14,10))
            def plot_metric(ax, train, val, test, title, ylabel):
                ax.plot(train, label='train'); ax.plot(val, label='val')
                tst_epochs = [i*EVAL_EVERY for i,x in enumerate(test) if x is not None]
                tst_vals   = [x for x in test if x is not None]
                ax.plot(tst_epochs, tst_vals, 'o-', label='test')
                ax.axvline(best_epoch, ls='--', c='r', label='best')
                ax.set_title(title); ax.set_xlabel('epoch'); ax.set_ylabel(ylabel); ax.grid(); ax.legend()

            metrics = [('mse','MSE'), ('mape','MAPE'), ('mae','MAE'), ('mqe','MQE')]
            for i,(key,name) in enumerate(metrics,1):
                ax = plt.subplot(len(metrics),1,i)
                plot_metric(ax, history[f'train_{key}'], history[f'val_{key}'], history[f'test_{key}'],
                            f'{name} history', name)

            plt.tight_layout()
            plt.savefig(save_dir / f'{LOSS_TO_OPTIMISE}_{CKPT_METRIC}{feature_size}_training_history_{grid_size[0]}_{grid_size[1]}.png')
            #plt.show()
            print(f"Training complete. Best epoch: {best_epoch+1}")