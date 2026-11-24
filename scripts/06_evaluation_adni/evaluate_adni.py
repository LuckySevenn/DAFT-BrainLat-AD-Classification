"""
evaluate_adni.py — Evaluación externa en ADNI de modelos entrenados con BrainLat (sin reentrenar).

PROPÓSITO
    Mide qué tan bien generalizan a otra cohorte los modelos elegidos en la búsqueda de
    hiperparámetros (run_grid.py). Para cada configuración de EXPERIMENTS:
      1. busca la carpeta del experimento y su mejor checkpoint, elegido por balanced
         accuracy en la validación de BrainLat,
      2. lo evalúa en la validación de BrainLat y en el test de ADNI,
      3. genera las figuras (daft_plots.py) en una COPIA del experimento dentro de OUT_DIR,
         para no sobrescribir las figuras originales de BrainLat,
      4. calcula Acc., Bal. Acc., Sens. AD, Esp. CN, AUC, F1 macro y la matriz de confusión
         (TP, FN, TN, FP). AD es la clase positiva; F1 macro = promedio del F1 de AD y de CN.
    La normalización tabular se sigue ajustando con el train de BrainLat, como en el
    entrenamiento: ADNI se trata exactamente como datos nuevos.

REQUISITOS
    - Haber corrido run_grid.py: deben existir las carpetas de los experimentos.
    - train.h5 y valid.h5 de BrainLat generados con build_hdf5.py y test.h5 de ADNI
      generado con build_hdf5_adni.py, con las mismas variables tabulares y en el mismo orden.
    - Este archivo y daft_plots.py en la raíz del repositorio DAFT.

USO
    python evaluate_adni.py

SALIDAS
    OUT_CSV     todas las métricas (incluye el F1 de cada clase)
    TABLE_CSV   tabla con los nombres de columna usados en el documento
    OUT_DIR/    copias de los experimentos con las figuras de ADNI

QUÉ SE PUEDE CAMBIAR (sección CONFIGURACIÓN)
    DATA_DIR, ADNI_TEST          Rutas de los HDF5 de BrainLat y de ADNI.
    OUT_DIR, OUT_CSV, TABLE_CSV  Dónde se guardan las salidas.
    EXPERIMENTS                  Configuraciones a evaluar:
                                 (arquitectura, optimizador, lr, weight decay, batch).
    ARCH_CLI                     Nombre de cada arquitectura en --discriminator_net.
    EXPERIMENT_NAME_PATTERNS     Cómo se llaman las carpetas de los experimentos.
    N_VALID, N_ADNI              Sujetos por clase; solo se usan si no se pueden contar
                                 en el HDF5.
    FIXED / FLAGS                Deben coincidir con los de run_grid.py.
"""
import itertools
import shutil
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from daft.evaluate import load_model_and_evaluate
from daft_plots import analyze_experiment

try:
    from daft.cli import create_parser
except ImportError:
    from train import create_parser

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
DATA_DIR = Path("/ruta/a/BrainLat/HDF5")            # train.h5 y valid.h5 de BrainLat
ADNI_TEST = Path("/ruta/a/ADNI/HDF5/test.h5")
OUT_DIR = Path("adni_external")
OUT_CSV = Path("external_adni_results.csv")
TABLE_CSV = Path("external_adni_tabla.csv")

# (arquitectura, optimizador, learning rate, weight decay, batch size)
EXPERIMENTS = [
    ("DAFT",   "AdamW", 0.013,  0.01,   16),
    ("DAFT",   "SGD",   0.0023, 0,      32),
    ("DAFT",   "Adam",  0.001,  0.01,   8),
    ("ResNet", "AdamW", 0.013,  0.01,   16),
    ("ResNet", "SGD",   0.0023, 0.0001, 16),
    ("ResNet", "Adam",  0.0055, 0.01,   8),
]

ARCH_CLI = {"DAFT": "daft", "ResNet": "resnet"}

# Patrones de nombre de carpeta, en orden de prioridad: (patrón, optimizadores a los que aplica).
# None = cualquier optimizador.
EXPERIMENT_NAME_PATTERNS = [
    ("grid/{opt}_{arch}_lr{lr}_wd{wd}_bs{bs}", None),         # run_grid.py del anexo
    ("grid/{arch}_lr{lr}_wd{wd}_bs{bs}", ["AdamW"]),          # versión anterior (solo AdamW)
    ("grid_optim/{opt}_{arch}_lr{lr}_wd{wd}_bs{bs}", None),   # versión anterior (SGD y Adam)
]

N_VALID = {"AD": 31, "CN": 21}
N_ADNI = {"AD": 50, "CN": 50}

FIXED = {
    "task": "clf",
    "num_classes": "2",
    "epoch": "30",
    "n_basefilters": "4",
    "bottleneck_factor": "7",
    "normalize_image": "minmax",
    "dataset": "baseline",
    "workers": "2",
}
FLAGS = ["--normalize_tabular"]

PARTS = (("val", "valid", "val"), ("adni", "test", "ADNI"))   # (prefijo, partición en DAFT, etiqueta)


# =====================================================================
# LOCALIZAR LOS EXPERIMENTOS
# =====================================================================
def num_variants(x):
    """Formas en que un número puede aparecer en el nombre de una carpeta (0.0001, 1e-04, 0, 0.0...)."""
    x = float(x)
    v = {str(x), "{:g}".format(x), np.format_float_positional(x, trim="-")}
    if x.is_integer():
        v.add(str(int(x)))
    return v


def candidate_names(arch, opt, lr, wd, bs):
    """Grupos de nombres de experimento posibles, uno por patrón, en orden de prioridad."""
    archs = {arch, arch.lower(), ARCH_CLI.get(arch, arch)}
    groups = []
    for pattern, optimizers in EXPERIMENT_NAME_PATTERNS:
        if optimizers is not None and opt not in optimizers:
            continue
        groups.append(sorted({pattern.format(opt=opt, arch=a, lr=l, wd=w, bs=int(bs))
                              for a, l, w in itertools.product(archs, num_variants(lr), num_variants(wd))}))
    return groups


def find_checkpoints_dir(groups):
    """Devuelve (carpeta de checkpoints, nombre del experimento) del primer patrón que coincida.
    Si hay varias coincidencias dentro de un patrón, toma la más reciente.
    Ignora las copias dentro de OUT_DIR."""
    out = OUT_DIR.resolve()
    for names in groups:
        found = [(p, name) for name in names
                 for p in Path(".").glob("**/{}/**/checkpoints".format(name))
                 if p.is_dir() and out not in p.resolve().parents]
        if found:
            return max(found, key=lambda t: t[0].stat().st_mtime)
    tried = [n for g in groups for n in g]
    raise FileNotFoundError("No se encontraron checkpoints. Nombres probados, entre otros:\n  "
                            + "\n  ".join(tried[:6]))


def build_args(arch, opt, lr, wd, bs, exp_name):
    """Reconstruye los argumentos del entrenamiento, con ADNI como conjunto de test."""
    cfg = dict(FIXED)
    cfg.update({
        "train_data": str(DATA_DIR / "train.h5"),   # BrainLat: estadísticas de normalización
        "val_data": str(DATA_DIR / "valid.h5"),     # BrainLat: validación
        "test_data": str(ADNI_TEST),                # ADNI: test externo
        "optimizer": opt,
        "discriminator_net": ARCH_CLI.get(arch, arch),
        "learning_rate": str(lr),
        "decay_rate": str(wd),
        "batchsize": str(int(bs)),
        "experiment_name": exp_name,
    })
    cmd = []
    for k, v in cfg.items():
        cmd += ["--" + k, v]
    return create_parser().parse_args(cmd + FLAGS)


# =====================================================================
# MÉTRICAS
# =====================================================================
def class_counts(path, default):
    """Cuenta sujetos AD y CN en el HDF5; si no se puede, usa los valores de la configuración."""
    counts = {"AD": 0, "CN": 0}
    try:
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k == "stats":
                    continue
                dx = f[k].attrs.get("DX")
                dx = dx.decode() if isinstance(dx, bytes) else dx
                dx = {"Dementia": "AD"}.get(dx, dx)   # DAFT codifica AD como "Dementia"
                if dx in counts:
                    counts[dx] += 1
    except Exception:
        counts = {"AD": 0, "CN": 0}
    if min(counts.values()) == 0:
        print("[AVISO] No se pudieron contar las clases en {}; se usa {}".format(path, default))
        return default
    if counts != default:
        print("[AVISO] {} tiene {}, distinto de la configuración {}; se usa el del HDF5".format(
            path, counts, default))
    return counts


def confusion(sens, spec, acc, bal, n):
    """Reconstruye la matriz de confusión (AD positivo) y calcula las métricas derivadas.
    Usa sensibilidad y especificidad; si faltan (fallaron las gráficas), las despeja de
    accuracy y balanced accuracy, lo que solo es posible si las clases no están balanceadas."""
    n_ad, n_cn = n["AD"], n["CN"]
    tot = n_ad + n_cn
    if not (pd.isna(sens) or pd.isna(spec)):
        tp, tn = sens * n_ad, spec * n_cn
    elif n_ad != n_cn:
        # acc*tot = tp + tn  y  2*bal = tp/n_ad + tn/n_cn  ->  se despeja tp
        tp = (2 * bal - acc * tot / n_cn) / (1 / n_ad - 1 / n_cn)
        tn = acc * tot - tp
    else:
        raise ValueError("Sin sensibilidad/especificidad no se puede reconstruir la matriz "
                         "con clases balanceadas {}".format(n))
    if max(abs(tp - round(tp)), abs(tn - round(tn))) > 0.01:
        print("[AVISO] La reconstrucción no da enteros: revisa los sujetos por clase {}".format(n))
    tp, tn = int(round(tp)), int(round(tn))
    fn, fp = n_ad - tp, n_cn - tn

    f1_ad = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    f1_cn = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return {"acc": (tp + tn) / tot, "sens_AD": tp / n_ad, "spec_CN": tn / n_cn,
            "f1_AD": f1_ad, "f1_CN": f1_cn, "f1_macro": (f1_ad + f1_cn) / 2,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp}


# =====================================================================
# EVALUACIÓN
# =====================================================================
def evaluate_one(arch, opt, lr, wd, bs, device):
    """Evalúa una configuración. Devuelve un diccionario con las métricas en bruto."""
    raw = {}
    ckpt_dir, exp_name = find_checkpoints_dir(candidate_names(arch, opt, lr, wd, bs))
    print("Experimento:", exp_name)
    print("Checkpoints:", ckpt_dir)
    args = build_args(arch, opt, lr, wd, bs, exp_name)

    metrics = load_model_and_evaluate(args, ckpt_dir, device, progressbar=False)
    for tag, part, _ in PARTS:
        raw[tag + "_acc"] = metrics.get(part + "_accuracy")
        raw[tag + "_bal"] = metrics.get(part + "_balanced_accuracy")

    # Las figuras se generan en una copia para no reemplazar las de BrainLat
    copy_dir = OUT_DIR / exp_name
    if not copy_dir.exists():
        shutil.copytree(str(ckpt_dir.parent), str(copy_dir), ignore=shutil.ignore_patterns("figures"))
    try:
        m = analyze_experiment(copy_dir, device=device, args=args)
        for tag, part, _ in PARTS:
            raw[tag + "_auc"] = m.loc[part, "auc"]
            raw[tag + "_sens"] = m.loc[part, "sensitivity_AD"]
            raw[tag + "_spec"] = m.loc[part, "specificity_CN"]
    except Exception:
        print("[AVISO] analyze_experiment falló; este modelo queda sin AUC:")
        traceback.print_exc()
    return raw


def summarize(raw, counts):
    """Convierte las métricas en bruto en las columnas finales (por partición)."""
    out = {}
    for tag, _, _ in PARTS:
        cm = confusion(raw.get(tag + "_sens", np.nan), raw.get(tag + "_spec", np.nan),
                       raw[tag + "_acc"], raw[tag + "_bal"], counts[tag])
        out.update({
            tag + "_acc": cm["acc"],
            tag + "_bal_acc": raw[tag + "_bal"],
            tag + "_sens_AD": cm["sens_AD"],
            tag + "_spec_CN": cm["spec_CN"],
            tag + "_auc": raw.get(tag + "_auc", np.nan),
            tag + "_f1_macro": cm["f1_macro"],
            tag + "_f1_AD": cm["f1_AD"],          # solo en el CSV completo
            tag + "_f1_CN": cm["f1_CN"],          # solo en el CSV completo
        })
        for k in ("TP", "FN", "TN", "FP"):
            out["{}_{}".format(tag, k)] = cm[k]
    return out


def document_table(df):
    """Tabla con los nombres de columna del documento, en ese orden."""
    names = {"arch": "Arquitectura", "optimizer": "Optimizador", "learning_rate": "LR",
             "decay_rate": "Weight decay", "batchsize": "Batch"}
    for tag, _, lbl in PARTS:
        names.update({
            tag + "_acc": "Acc. ({})".format(lbl),
            tag + "_bal_acc": "Bal. Acc. ({})".format(lbl),
            tag + "_sens_AD": "Sens. AD ({})".format(lbl),
            tag + "_spec_CN": "Esp. CN ({})".format(lbl),
            tag + "_auc": "AUC ({})".format(lbl),
            tag + "_f1_macro": "F1 macro ({})".format(lbl),
        })
        for k in ("TP", "FN", "TN", "FP"):
            names["{}_{}".format(tag, k)] = "{} ({})".format(k, lbl)
    table = df[[c for c in names if c in df.columns]].rename(columns=names)
    num = [c for c in table.columns if any(s in c for s in ("Acc.", "Sens.", "Esp.", "AUC", "F1"))]
    table[num] = table[num].round(3)
    return table


def main():
    assert ADNI_TEST.exists(), "No existe {}".format(ADNI_TEST)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts = {"val": class_counts(DATA_DIR / "valid.h5", N_VALID),
              "adni": class_counts(ADNI_TEST, N_ADNI)}
    print("Sujetos validación BrainLat: {}   ADNI: {}".format(counts["val"], counts["adni"]))

    rows = []
    for arch, opt, lr, wd, bs in EXPERIMENTS:
        print("\n===== {}  {}  lr={}  wd={}  bs={} =====".format(arch, opt, lr, wd, bs))
        row = {"optimizer": opt, "arch": arch, "learning_rate": lr, "decay_rate": wd, "batchsize": bs}
        try:
            row.update(summarize(evaluate_one(arch, opt, lr, wd, bs, device), counts))
        except Exception as e:
            # Un modelo que falla queda registrado y no detiene los demás
            traceback.print_exc()
            row["error"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    table = document_table(df)
    table.to_csv(TABLE_CSV, index=False)

    print("\n" + table.to_string(index=False))
    if "error" in df.columns and df["error"].notna().any():
        print("\n[AVISO] Fallaron:")
        print(df.loc[df["error"].notna(), ["arch", "optimizer", "learning_rate",
                                           "decay_rate", "batchsize"]].to_string(index=False))
    print("\nGuardado en {} y {}".format(OUT_CSV.resolve(), TABLE_CSV.resolve()))


if __name__ == "__main__":
    main()
