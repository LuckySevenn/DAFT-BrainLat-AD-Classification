"""
daft_plots.py  —  Gráficas de análisis para un experimento de DAFT (CN vs AD).

Genera dentro de <experimento>/figures/:
  Curvas de aprendizaje (train vs validación, por época):
    loss.png, accuracy.png, balanced_accuracy.png
  Con el mejor modelo (mejor balanced accuracy en validación):
    confusion_matrix_valid.png, confusion_matrix_test.png
    roc_valid.png, roc_test.png
    predictions_valid.csv, predictions_test.csv
    metrics.csv   (accuracy, balanced accuracy, sensibilidad, especificidad, F1, AUC)

Uso:
  - Automático: run_grid.py lo llama después de cada entrenamiento.
  - Manual, para un experimento ya entrenado (desde la carpeta de DAFT, entorno 'daft'):
        python daft_plots.py experiments_clf/grid/daft_lr0.013_wd0.0_bs16/<fecha>
    o para todos los de una carpeta:
        python daft_plots.py experiments_clf/grid --all

Requiere: matplotlib, scikit-learn (pip install matplotlib)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

CLASS_NAMES = ["CN", "AD"]          # 0 = CN, 1 = Dementia (AD de BrainLat)
DPI = 150

LEARNING_CURVES = [
    ("cross_entropy/mean", "loss", "Pérdida (entropía cruzada)"),
    ("accuracy", "accuracy", "Accuracy"),
    ("balanced_accuracy", "balanced_accuracy", "Balanced accuracy"),
]


# =====================================================================
# 1. CURVAS DE APRENDIZAJE (desde los registros de TensorBoard)
# =====================================================================
def read_scalars(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        ev = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def plot_learning_curves(exp_dir, fig_dir, title):
    train = read_scalars(exp_dir / "tensorboard" / "train")
    valid = read_scalars(exp_dir / "tensorboard" / "eval")

    best_epoch = None
    if "balanced_accuracy" in valid:
        ep, val = valid["balanced_accuracy"]
        best_epoch = int(ep[int(np.argmax(val))])

    for tag, fname, label in LEARNING_CURVES:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for data, name, style in ((train, "Entrenamiento", "-"), (valid, "Validación", "--")):
            if tag in data:
                ax.plot(*data[tag], style, marker="o", markersize=3, label=name)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2,
                       label="Mejor época (val): {}".format(best_epoch))
        ax.set_xlabel("Época")
        ax.set_ylabel(label)
        ax.set_title("{}\n{}".format(label, title), fontsize=10)
        if tag != "cross_entropy/mean":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / (fname + ".png"), dpi=DPI)
        plt.close(fig)
    return best_epoch


# =====================================================================
# 2. MATRIZ DE CONFUSIÓN Y ROC
# =====================================================================
def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "{}\n({:.1f}%)".format(cm[i, j], pct[i, j]), ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Diagnóstico real")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_roc(y_true, prob, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, linewidth=2, label="AUC = {:.3f}".format(auc))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0.5)")
    ax.set_xlabel("1 - Especificidad (tasa de falsos positivos)")
    ax.set_ylabel("Sensibilidad (tasa de verdaderos positivos)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def summary_metrics(y_true, y_pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_AD": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity_CN": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1_AD": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }
    return out


# =====================================================================
# 3. PREDICCIONES DEL MEJOR MODELO (usa el código de DAFT)
# =====================================================================
def get_predictions(args, ckpt_dir, device):
    import torch
    from daft.cli import HeterogeneousModelFactory
    from daft.testing.test_and_save import evaluate_model, load_best_model

    factory = HeterogeneousModelFactory(args)
    _, valid_loader, test_loader = factory.get_data()
    model = load_best_model(factory, ckpt_dir, device=device)

    out = {}
    for split, loader in (("valid", valid_loader), ("test", test_loader)):
        _, preds = evaluate_model(metrics=[], model=model, data=loader, device=device, progressbar=False)
        logits = preds["logits"].float()
        y_true = preds["target"].view(-1).long().numpy()
        if logits.dim() == 2 and logits.shape[1] > 1:          # multiclase (no es tu caso)
            prob = torch.softmax(logits, dim=1)[:, 1].numpy()
            y_pred = logits.argmax(dim=1).numpy()
        else:                                                   # binario: 1 salida + sigmoide
            prob = torch.sigmoid(logits.view(-1)).numpy()
            y_pred = (prob >= 0.5).astype(int)
        out[split] = (y_true, y_pred, prob)
    return out


def args_from_json(exp_dir):
    """Reconstruye los argumentos de entrenamiento desde experiment_args.json."""
    from daft.cli import create_parser
    saved = json.loads((exp_dir / "experiment_args.json").read_text())
    cmd = []
    for k, v in saved.items():
        if v is None or v is False:
            continue
        if v is True:
            cmd.append("--" + k)
        else:
            cmd += ["--" + k, str(v)]
    return create_parser().parse_args(cmd)


# =====================================================================
# 4. TODO JUNTO PARA UN EXPERIMENTO
# =====================================================================
def analyze_experiment(exp_dir, device=None, args=None):
    import torch
    exp_dir = Path(exp_dir)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args is None:
        args = args_from_json(exp_dir)

    title = "{} | lr={} | wd={} | batch={}".format(
        args.discriminator_net, args.learning_rate, args.decay_rate, args.batchsize)

    best_epoch = plot_learning_curves(exp_dir, fig_dir, title)

    rows = []
    preds = get_predictions(args, exp_dir / "checkpoints", device)
    for split, (y_true, y_pred, prob) in preds.items():
        name = "Validación" if split == "valid" else "Test"
        plot_confusion(y_true, y_pred, fig_dir / "confusion_matrix_{}.png".format(split),
                       "Matriz de confusión — {}\n{}".format(name, title))
        plot_roc(y_true, prob, fig_dir / "roc_{}.png".format(split),
                 "Curva ROC — {}\n{}".format(name, title))
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "prob_AD": prob}).to_csv(
            fig_dir / "predictions_{}.csv".format(split), index=False)
        m = summary_metrics(y_true, y_pred, prob)
        m.update({"split": split, "best_epoch": best_epoch})
        rows.append(m)

    metrics = pd.DataFrame(rows).set_index("split")
    metrics.to_csv(fig_dir / "metrics.csv")
    print("[OK] Gráficas en {}".format(fig_dir))
    print(metrics[["n", "accuracy", "balanced_accuracy", "sensitivity_AD",
                   "specificity_CN", "auc"]].round(3).to_string())
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if "--all" in sys.argv:
        for d in sorted(p.parent for p in target.rglob("experiment_args.json")):
            print("\n=== {} ===".format(d))
            analyze_experiment(d)
    else:
        analyze_experiment(target)
