"""
build_hdf5_adni.py
==================

Versión ADNI de build_hdf5.py. Genera el test.h5 del subconjunto de ADNI (test
externo) a partir de master.csv (salida de build_master_csv_adni.py) y de las
imágenes procesadas con FreeSurfer.

Qué hace
--------
1. Lee de BrainLat train.h5 el orden de las variables tabulares, las estadísticas
   de normalización (tabular e imagen) y las etiquetas DX válidas.
2. Lee master.csv de ADNI y comprueba que tenga las mismas variables, sin valores
   vacíos, sin MRI_ID duplicados y con etiquetas DX que existan en BrainLat.
3. Para cada sujeto recorta un cubo de 64x64x64 vóxeles de norm.mgz centrado en
   el hipocampo izquierdo (centroide de la etiqueta 17 en aseg.mgz), igual que
   build_hdf5.py.
4. Escribe un único test.h5 (ADNI no se particiona) y lo verifica.

  IMPORTANTE: las estadísticas de normalización se copian de BrainLat train.h5 y
  nunca se calculan con ADNI. Los modelos se entrenaron normalizando con BrainLat;
  usar estadísticas de ADNI sería una fuga de información del conjunto de prueba.

Estructura del .h5 (igual que BrainLat)
---------------------------------------
  /<MRI_ID>                                  attrs: RID, VISCODE="bl", DX
  /<MRI_ID>/Left-Hippocampus/vol_with_bg     cubo 64x64x64 (uint8)
  /<MRI_ID>/tabular                          variables en el MISMO orden que BrainLat
  /stats/...                                 copiado de BrainLat train.h5

Requisitos
----------
  Python 3 con numpy, pandas, nibabel y h5py.
  BrainLat train.h5 generado con build_hdf5.py.
  Para cada sujeto: <ADNI_ROOT>/<MRI_ID>[_T1w]/mri/{norm.mgz, aseg.mgz}

Cómo replicar
-------------
  1. Ejecutar antes build_master_csv_adni.py y build_hdf5.py (BrainLat).
  2. Ajustar las rutas de la sección CONFIGURACIÓN.
  3. Ejecutar:  python build_hdf5_adni.py
  4. Usar el test.h5 con evaluate_adni.py.

Parámetros que se pueden cambiar (sección CONFIGURACIÓN)
--------------------------------------------------------
  MASTER_CSV, ADNI_ROOTS, BRAINLAT_TRAIN_H5,
  OUT_DIR, OUT_NAME                      Rutas de entrada y salida.
  IMAGE_FILE, SEG_FILE, ROI_LABEL, CROP  Igual que en build_hdf5.py.
  SUBJ_PATTERNS                          Nombres de carpeta de sujeto aceptados.

  ROI_NAME y DATASET_NAME no se deben cambiar: son los nombres que busca DAFT.
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import h5py

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
MASTER_CSV = "ruta/a/ADNI_salida/master.csv"      # generado por build_master_csv_adni.py

# Carpeta(s) con la salida de FreeSurfer (SUBJECTS_DIR). Cada sujeto: <raíz>/<MRI_ID...>/mri/
ADNI_ROOTS = [
    "ruta/a/ADNI_FreeSurfer",
]

BRAINLAT_TRAIN_H5 = "ruta/a/salida/train.h5"     # fuente de las estadísticas
OUT_DIR  = "ruta/a/ADNI_salida"
OUT_NAME = "test.h5"

IMAGE_FILE = "norm.mgz"
SEG_FILE   = "aseg.mgz"
ROI_LABEL  = 17                  # Left-Hippocampus en aseg
CROP       = 64

ROI_NAME     = "Left-Hippocampus"
DATASET_NAME = "vol_with_bg"

# Nombres de carpeta de sujeto que se aceptan dentro de cada raíz
SUBJ_PATTERNS = ["{id}", "{id}_T1w", "*/{id}", "*/{id}_T1w"]


# =====================================================================
# 2. UBICAR LA CARPETA mri/ DE CADA SUJETO
# =====================================================================
def find_mri_dir(mri_id):
    found = []
    for root in ADNI_ROOTS:
        for pat in SUBJ_PATTERNS:
            for d in glob.glob(os.path.join(root, pat.format(id=mri_id), "mri")):
                if all(os.path.exists(os.path.join(d, f)) for f in (IMAGE_FILE, SEG_FILE)):
                    found.append(os.path.normpath(d))
    found = sorted(set(found))
    if len(found) > 1:
        raise RuntimeError("Varias carpetas para {}: {}".format(mri_id, found))
    return found[0] if found else None


# =====================================================================
# 3. RECORTE ALREDEDOR DEL HIPOCAMPO IZQUIERDO (idéntico a BrainLat)
# =====================================================================
def crop_roi(mri_dir):
    img = np.asarray(nib.load(os.path.join(mri_dir, IMAGE_FILE)).dataobj)
    seg = np.asarray(nib.load(os.path.join(mri_dir, SEG_FILE)).dataobj)
    voxels = np.argwhere(seg == ROI_LABEL)
    if voxels.size == 0:
        raise ValueError("etiqueta {} no encontrada en {}".format(ROI_LABEL, mri_dir))
    h = CROP // 2
    center = np.round(voxels.mean(axis=0)).astype(int) + h
    padded = np.pad(img, h, mode="constant")
    cube = padded[center[0] - h:center[0] + h,
                  center[1] - h:center[1] + h,
                  center[2] - h:center[2] + h]
    return np.ascontiguousarray(cube, dtype=np.uint8)


# =====================================================================
# 4. ESTADÍSTICAS Y CLASES DE BRAINLAT TRAIN
# =====================================================================
def load_brainlat_reference(path):
    with h5py.File(path, "r") as hf:
        cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in hf["stats/tabular/columns"][:]]
        tab_mean = hf["stats/tabular/mean"][:]
        tab_std = hf["stats/tabular/stddev"][:]
        img_g = hf["stats/{}/{}".format(ROI_NAME, DATASET_NAME)]
        img_mean = img_g["mean"][:]
        img_std = img_g["stddev"][:]
        dx_set = {str(hf[k].attrs["DX"]) for k in hf.keys() if k != "stats"}
    return cols, (tab_mean, tab_std, img_mean, img_std), dx_set


# =====================================================================
# 5. ESCRITURA
# =====================================================================
def write_h5(path, df, feat_cols, stats):
    tab_mean, tab_std, img_mean, img_std = stats
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as hf:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            g = hf.create_group(row["MRI_ID"])
            g.attrs["RID"] = row["MRI_ID"]
            g.attrs["VISCODE"] = "bl"          # solo baseline
            g.attrs["DX"] = row["DX"]
            g.create_group(ROI_NAME).create_dataset(
                DATASET_NAME, data=crop_roi(row["mri_dir"]), compression="gzip")
            g.create_dataset("tabular", data=row[feat_cols].to_numpy(np.float32))
            if i % 20 == 0:
                print("  {}: {}/{}".format(os.path.basename(path), i, len(df)))

        stats_g = hf.create_group("stats")
        t = stats_g.create_group("tabular")
        t.create_dataset("columns", data=np.array(feat_cols, dtype=object), dtype=str_dt)
        t.create_dataset("mean", data=tab_mean)
        t.create_dataset("stddev", data=tab_std)
        im = stats_g.create_group(ROI_NAME).create_group(DATASET_NAME)
        im.create_dataset("mean", data=img_mean)
        im.create_dataset("stddev", data=img_std)


# =====================================================================
# 6. VERIFICACIÓN
# =====================================================================
def verify(path, expected_n, feat_cols):
    with h5py.File(path, "r") as hf:
        subjects = [k for k in hf.keys() if k != "stats"]
        assert len(subjects) == expected_n, "número de sujetos no coincide"
        cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in hf["stats/tabular/columns"][:]]
        assert cols == feat_cols, "orden de columnas distinto al de BrainLat"
        for k in subjects:
            g = hf[k]
            assert g[ROI_NAME][DATASET_NAME].shape == (CROP,) * 3
            tab = g["tabular"][:]
            assert tab.shape == (len(feat_cols),) and np.isfinite(tab).all(), "tabular inválido en " + k
            for a in ("RID", "VISCODE", "DX"):
                assert a in g.attrs, "falta atributo {} en {}".format(a, k)
        dx = pd.Series([hf[k].attrs["DX"] for k in subjects]).value_counts().to_dict()
    print("  [OK] {}: {} sujetos, {} features, clases {}".format(
        os.path.basename(path), expected_n, len(feat_cols), dx))


# =====================================================================
# 7. PRINCIPAL
# =====================================================================
def main():
    # Referencia BrainLat: orden de features, estadísticas y etiquetas válidas
    feat_cols, stats, dx_set = load_brainlat_reference(BRAINLAT_TRAIN_H5)
    print("[INFO] {} features de BrainLat; clases BrainLat: {}".format(len(feat_cols), sorted(dx_set)))

    df = pd.read_csv(MASTER_CSV, dtype={"MRI_ID": str})
    df["MRI_ID"] = df["MRI_ID"].str.strip()
    print("[INFO] {} sujetos en el master de ADNI".format(len(df)))

    # Si el master trae columna split, se toma todo igual (ADNI es solo test)
    if "split" in df.columns and set(df["split"].dropna().unique()) - {"test"}:
        print("[AVISO] master tiene valores de split distintos a 'test'; se usan TODOS los sujetos.")

    # --- Comprobaciones antes de escribir nada ---
    dup = df["MRI_ID"][df["MRI_ID"].duplicated()].tolist()
    if dup:
        raise ValueError("MRI_ID duplicados: {}".format(dup[:5]))

    missing_cols = [c for c in feat_cols if c not in df.columns]
    if missing_cols:
        raise KeyError("Faltan en el master de ADNI estas features de BrainLat: {}".format(missing_cols))
    extra = [c for c in df.columns if c not in feat_cols + ["MRI_ID", "DX"]]
    if extra:
        print("[INFO] Columnas ignoradas (no entran al modelo): {}".format(extra))

    bad_dx = sorted(set(df["DX"].astype(str)) - dx_set)
    if bad_dx:
        raise ValueError("Etiquetas DX que no existen en BrainLat: {} (deben ser {})".format(bad_dx, sorted(dx_set)))

    nan_rows = df.loc[df[feat_cols].isna().any(axis=1), "MRI_ID"].tolist()
    if nan_rows:
        raise ValueError("{} sujetos con features vacías, p. ej.: {}".format(len(nan_rows), nan_rows[:5]))

    df["mri_dir"] = df["MRI_ID"].map(find_mri_dir)
    missing = df.loc[df["mri_dir"].isna(), "MRI_ID"].tolist()
    if missing:
        raise FileNotFoundError("No se encontró {} y {} para {} sujetos en {}, p. ej.: {}".format(
            IMAGE_FILE, SEG_FILE, len(missing), ADNI_ROOTS, missing[:5]))
    print("[INFO] Imágenes encontradas para los {} sujetos".format(len(df)))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, OUT_NAME)
    print("[INFO] Escribiendo {}...".format(path))
    write_h5(path, df, feat_cols, stats)
    verify(path, len(df), feat_cols)
    print("\n[OK] Archivo listo en {}".format(path))


if __name__ == "__main__":
    main()
