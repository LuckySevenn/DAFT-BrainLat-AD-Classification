"""
run_grid.py — Búsqueda de hiperparámetros para DAFT (clasificación CN vs AD).

PROPÓSITO
    Entrena y evalúa DAFT (imagen + datos tabulares) y la línea base ResNet
    (solo imagen) para todas las combinaciones de optimizador, learning rate
    y weight decay definidas en la sección CONFIGURACIÓN. Por cada combinación:
      1. entrena el modelo con train.py del repositorio DAFT,
      2. toma el checkpoint con mejor balanced accuracy en VALIDACIÓN y lo
         evalúa en validación y test,
      3. genera las gráficas de análisis en <experimento>/figures/ (daft_plots.py).
    La mejor configuración se elige solo con validación; test se reporta,
    pero no se usa para elegir.

REQUISITOS
    - Repositorio DAFT (https://github.com/ai-med/DAFT) con su entorno instalado.
    - Este archivo y daft_plots.py en la raíz del repositorio, junto a train.py.
    - train.h5, valid.h5 y test.h5 generados con el pipeline de preprocesamiento.

USO
    python run_grid.py

SALIDAS
    - RESULTS_CSV: una fila por combinación con las métricas de validación y
      test (balanced accuracy, AUC, sensibilidad AD, especificidad CN).
      Se guarda después de cada corrida.
    - Una carpeta por experimento, <EXPERIMENTS_PREFIX>/<opt>_<arch>_lr.._wd.._bs..,
      con los checkpoints y las figuras.
    - Al terminar, imprime la mejor configuración por arquitectura y optimizador.
    Si la ejecución se interrumpe, al relanzarla se saltan las combinaciones ya
    terminadas y se reintentan las que fallaron.

QUÉ SE PUEDE CAMBIAR (todo está en la sección CONFIGURACIÓN)
    DATA_DIR            Carpeta con los tres archivos .h5.
    RESULTS_CSV         Nombre del CSV de resultados.
    EXPERIMENTS_PREFIX  Carpeta donde se agrupan los experimentos.
    OPTIMIZERS          "AdamW", "Adam" y/o "SGD" (los que admite daft/cli.py).
    ARCHITECTURES       "daft", "resnet" (también "concat1fc", "film", que DAFT incluye).
    LEARNING_RATES      Valores de learning rate a probar.
    DECAY_RATES         Valores de weight decay a probar.
    BATCH_SIZES         Tamaños de lote a probar.
    FIXED / FLAGS       Parámetros comunes a todas las corridas (épocas, filtros base,
                        normalización, número de clases, etc.). Se pasan tal cual
                        a train.py, así que admiten cualquier argumento de daft/cli.py.
    El número de corridas es el producto de las listas, menos las combinaciones
    redundantes que se describen en build_combos().
"""
import inspect
import itertools
import traceback
from pathlib import Path

import pandas as pd
import torch

import daft.cli
from daft.evaluate import load_model_and_evaluate
from daft_plots import analyze_experiment
from train import main as main_train

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
DATA_DIR = Path("/ruta/a/HDF5")          # debe contener train.h5, valid.h5, test.h5
RESULTS_CSV = Path("grid_results.csv")
EXPERIMENTS_PREFIX = "grid"

OPTIMIZERS = ["AdamW", "Adam", "SGD"]
ARCHITECTURES = ["daft", "resnet"]       # daft: imagen + tabular | resnet: solo imagen
LEARNING_RATES = [0.03, 0.013, 0.0055, 0.0023, 0.001]   # grilla del paper de DAFT
DECAY_RATES = [0.0, 1e-4, 1e-2]
BATCH_SIZES = [16]   # el paper usa 256, inviable con ~257 sujetos de entrenamiento

FIXED = {
    "task": "clf",
    "num_classes": "2",          # CN vs AD
    "epoch": "30",
    "n_basefilters": "4",
    "bottleneck_factor": "7",
    "normalize_image": "minmax",
    "dataset": "baseline",
    "workers": "2",
}
FLAGS = ["--normalize_tabular"]

METRIC_COLS = ["valid_balanced_accuracy", "valid_auc", "test_balanced_accuracy", "test_auc",
               "test_sensitivity_AD", "test_specificity_CN"]
PARAM_COLS = ["optimizer", "arch", "learning_rate", "decay_rate", "batchsize"]


# =====================================================================
# DEFINICIÓN DE LA GRILLA
# =====================================================================
def sgd_uses_weight_decay():
    """Indica si daft/cli.py le pasa weight_decay a SGD (en el repositorio original no lo hace)."""
    for line in inspect.getsource(daft.cli).splitlines():
        if "torch.optim.SGD(" in line:
            return "weight_decay" in line
    return False


def build_combos(sgd_wd):
    """Producto cartesiano de la grilla, sin las combinaciones que darían resultados idénticos."""
    combos = []
    for opt, arch, lr, wd, bs in itertools.product(OPTIMIZERS, ARCHITECTURES, LEARNING_RATES,
                                                   DECAY_RATES, BATCH_SIZES):
        # Sin weight decay, Adam y AdamW hacen la misma actualización: basta con correr AdamW.
        if opt == "Adam" and wd == 0.0 and "AdamW" in OPTIMIZERS:
            continue
        # Si SGD ignora decay_rate, las corridas con wd > 0 repetirían la de wd = 0.
        if opt == "SGD" and wd != 0.0 and not sgd_wd:
            continue
        combos.append((opt, arch, lr, wd, bs))
    return combos


def build_cmd(opt, arch, lr, wd, bs):
    """Arma la lista de argumentos de línea de comandos que recibe train.py."""
    cfg = dict(FIXED)
    cfg.update({
        "train_data": str(DATA_DIR / "train.h5"),
        "val_data": str(DATA_DIR / "valid.h5"),
        "test_data": str(DATA_DIR / "test.h5"),
        "optimizer": opt,
        "discriminator_net": arch,
        "learning_rate": str(lr),
        "decay_rate": str(wd),
        "batchsize": str(bs),
        "experiment_name": "{}/{}_{}_lr{}_wd{}_bs{}".format(EXPERIMENTS_PREFIX, opt, arch, lr, wd, bs),
    })
    cmd = []
    for k, v in cfg.items():
        cmd += ["--" + k, v]
    return cmd + FLAGS


def key(opt, arch, lr, wd, bs):
    """Identificador de una combinación (normaliza tipos para compararla con lo leído del CSV)."""
    return (str(opt), str(arch), float(lr), float(wd), int(bs))


def row_key(r):
    return key(*(r[c] for c in PARAM_COLS))


# =====================================================================
# EJECUCIÓN
# =====================================================================
def run_one(opt, arch, lr, wd, bs, device):
    """Entrena, evalúa y grafica una combinación. Devuelve una fila con parámetros y métricas."""
    row = dict(zip(PARAM_COLS, (opt, arch, lr, wd, bs)))
    try:
        factory = main_train(build_cmd(opt, arch, lr, wd, bs))
        # Métricas del mejor checkpoint (elegido por balanced accuracy en validación)
        row.update(load_model_and_evaluate(factory.args, factory.checkpoints_dir, device, progressbar=False))
        try:
            exp_dir = Path(factory.checkpoints_dir).parent
            m = analyze_experiment(exp_dir, device=device, args=factory.args)
            row["valid_auc"] = m.loc["valid", "auc"]
            row["test_auc"] = m.loc["test", "auc"]
            row["test_sensitivity_AD"] = m.loc["test", "sensitivity_AD"]
            row["test_specificity_CN"] = m.loc["test", "specificity_CN"]
        except Exception:
            # Si fallan las gráficas se conservan las métricas del entrenamiento
            print("[AVISO] No se pudieron generar las gráficas de este experimento:")
            traceback.print_exc()
    except Exception as e:
        # Una combinación fallida queda registrada y no detiene la grilla
        traceback.print_exc()
        row["error"] = str(e)
    return row


def print_summary(results):
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty or "valid_balanced_accuracy" not in df.columns:
        return

    cols = PARAM_COLS + [c for c in METRIC_COLS if c in df.columns]
    df[cols[len(PARAM_COLS):]] = df[cols[len(PARAM_COLS):]].round(3)   # redondear solo métricas
    df = df.sort_values("valid_balanced_accuracy", ascending=False)

    best = (df.groupby(["arch", "optimizer"], as_index=False).head(1)
              .sort_values(["arch", "valid_balanced_accuracy"], ascending=[True, False]))
    print("\nMejor configuración por arquitectura y optimizador (según VALIDACIÓN):")
    print(best[cols].to_string(index=False))
    print("\nTodas las corridas, ordenadas por balanced accuracy de VALIDACIÓN:")
    print(df[cols].to_string(index=False))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo: {}".format(device))

    sgd_wd = sgd_uses_weight_decay()
    if "SGD" in OPTIMIZERS and not sgd_wd:
        print("[AVISO] daft/cli.py no aplica weight_decay a SGD: para SGD solo se corre decay_rate=0.")

    # Retomar una ejecución previa: se saltan las combinaciones terminadas sin error
    results = pd.read_csv(RESULTS_CSV).to_dict("records") if RESULTS_CSV.exists() else []
    done = {row_key(r) for r in results if pd.isna(r.get("error", float("nan")))}
    if done:
        print("Retomando: {} combinaciones ya hechas".format(len(done)))

    combos = build_combos(sgd_wd)
    print("Total de combinaciones: {}".format(len(combos)))
    for i, combo in enumerate(combos, 1):
        if key(*combo) in done:
            continue
        print("\n===== [{}/{}] {}  {}  lr={}  wd={}  batch={} =====".format(i, len(combos), *combo))
        row = run_one(*combo, device=device)
        # Reemplaza una fila previa de la misma combinación (p. ej. un intento fallido)
        results = [r for r in results if row_key(r) != key(*combo)] + [row]
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    print_summary(results)
    print("\nResultados guardados en {}".format(RESULTS_CSV.resolve()))


if __name__ == "__main__":
    main()
