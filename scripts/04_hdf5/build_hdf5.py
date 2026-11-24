"""
build_hdf5.py
=============

Genera los archivos train.h5, valid.h5 y test.h5 que recibe DAFT, a partir de
master.csv (salida de build_master_csv.py) y de las imágenes procesadas con
FreeSurfer.

Qué hace
--------
1. Lee master.csv y toma como variables tabulares todas las columnas que no son
   metadatos (19 en la configuración del trabajo).
2. Para cada sujeto recorta un cubo de 64x64x64 vóxeles de norm.mgz centrado en
   el hipocampo izquierdo (centroide de la etiqueta 17 en aseg.mgz).
3. Calcula las estadísticas de normalización (tabular e imagen) SOLO con train,
   para evitar fuga de información hacia valid y test.
4. Escribe un .h5 por partición con la estructura que espera el loader de DAFT
   y verifica cada archivo al terminar.

Estructura de cada .h5
----------------------
  /<MRI_ID>                                  attrs: RID, VISCODE="bl", DX
  /<MRI_ID>/Left-Hippocampus/vol_with_bg     cubo 64x64x64 (uint8)
  /<MRI_ID>/tabular                          vector de variables (float32, orden de master.csv)
  /stats/tabular/{columns, mean, stddev}               calculadas con train
  /stats/Left-Hippocampus/vol_with_bg/{mean, stddev}   vóxel a vóxel, calculadas con train

  Los tres archivos llevan las mismas estadísticas (las de train).
  VISCODE es siempre "bl" porque BrainLat es transversal (una sola visita).

Requisitos
----------
  Python 3 con numpy, pandas, nibabel y h5py.
  Para cada sujeto: BRAINLAT_T1_DIR/<country>/<MRI_ID>_T1w/mri/{norm.mgz, aseg.mgz}
  master.csv con al menos un sujeto en la partición "train".

Cómo replicar
-------------
  1. Ejecutar antes build_master_csv.py.
  2. Ajustar las rutas de la sección CONFIGURACIÓN.
  3. Ejecutar:  python build_hdf5.py
  4. Usar los .h5 generados con DAFT (p. ej. --num_classes 2 para CN vs AD).

Parámetros que se pueden cambiar (sección CONFIGURACIÓN)
--------------------------------------------------------
  MASTER_CSV, BRAINLAT_T1_DIR, OUT_DIR   Rutas de entrada y salida.
  IMAGE_FILE                             Volumen del que se recorta (por defecto norm.mgz).
  SEG_FILE, ROI_LABEL                    Segmentación y etiqueta que centra el recorte.
  CROP                                   Lado del cubo en vóxeles (debe ser par).
  META_COLS                              Columnas de master.csv que NO entran al modelo.

  ROI_NAME y DATASET_NAME no se deben cambiar: son los nombres que busca DAFT.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import h5py

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
# Rutas (cambiar según la máquina donde se ejecute)
MASTER_CSV      = "ruta/a/salida/master.csv"   # generado por build_master_csv.py
BRAINLAT_T1_DIR = "ruta/a/BrainLatT1"          # <country>/<MRI_ID>_T1w/mri
OUT_DIR         = "ruta/a/salida"

# Columnas de master.csv que no son variables del modelo
META_COLS = ["MRI_ID", "country", "dx_brainlat", "DX", "split"]

# Recorte de la imagen
IMAGE_FILE = "norm.mgz"   # T1 sin cráneo y normalizada en intensidad (256^3, salida de recon-all)
SEG_FILE   = "aseg.mgz"   # segmentación subcortical de FreeSurfer
ROI_LABEL  = 17           # 17 = Left-Hippocampus en aseg
CROP       = 64           # cubo de 64x64x64, como en el artículo de DAFT

# Nombres que el loader de DAFT busca (no cambiar)
ROI_NAME     = "Left-Hippocampus"
DATASET_NAME = "vol_with_bg"


# =====================================================================
# 2. IMAGEN: recorte alrededor de la región de interés
# =====================================================================
def subject_mri_dir(row):
    """Carpeta mri/ de FreeSurfer de un sujeto de master.csv."""
    return os.path.join(BRAINLAT_T1_DIR, row["country"], row["MRI_ID"] + "_T1w", "mri")


def crop_roi(mri_dir):
    """Recorta un cubo CROP^3 de IMAGE_FILE centrado en el centroide de ROI_LABEL.
    Se rellena con ceros el borde para que el cubo nunca se salga del volumen."""
    img = np.asarray(nib.load(os.path.join(mri_dir, IMAGE_FILE)).dataobj)
    seg = np.asarray(nib.load(os.path.join(mri_dir, SEG_FILE)).dataobj)
    voxels = np.argwhere(seg == ROI_LABEL)
    if voxels.size == 0:
        raise ValueError("etiqueta {} no encontrada en {}".format(ROI_LABEL, SEG_FILE))
    h = CROP // 2
    center = np.round(voxels.mean(axis=0)).astype(int) + h   # +h compensa el padding
    padded = np.pad(img, h, mode="constant")
    cube = padded[center[0] - h:center[0] + h,
                  center[1] - h:center[1] + h,
                  center[2] - h:center[2] + h]
    return np.ascontiguousarray(cube, dtype=np.uint8)


# =====================================================================
# 3. ESTADÍSTICAS DE NORMALIZACIÓN (solo con train)
# =====================================================================
def compute_stats(train_df, feat_cols):
    """Media y desviación estándar de las variables tabulares y de la imagen
    (vóxel a vóxel). La imagen se acumula por sumas para no cargar todo en memoria."""
    tab = train_df[feat_cols].to_numpy(np.float64)
    tab_mean = tab.mean(axis=0).astype(np.float32)
    tab_std = tab.std(axis=0, ddof=1)
    tab_std[tab_std == 0] = 1.0          # evita dividir por cero en variables constantes
    tab_std = tab_std.astype(np.float32)

    s = np.zeros((CROP,) * 3, np.float64)
    s2 = np.zeros((CROP,) * 3, np.float64)
    n = 0
    for i, (_, row) in enumerate(train_df.iterrows(), 1):
        x = crop_roi(subject_mri_dir(row)).astype(np.float64)
        s += x
        s2 += x * x
        n += 1
        if i % 50 == 0:
            print("  estadísticas de imagen: {}/{}".format(i, len(train_df)))
    img_mean = s / n
    img_std = np.sqrt(np.maximum(s2 / n - img_mean ** 2, 0)) + 1e-6
    return tab_mean, tab_std, img_mean.astype(np.float32), img_std.astype(np.float32)


# =====================================================================
# 4. ESCRITURA DE UN ARCHIVO .h5
# =====================================================================
def write_h5(path, part, feat_cols, stats):
    """Escribe los sujetos de una partición y el grupo /stats."""
    tab_mean, tab_std, img_mean, img_std = stats
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as hf:
        for i, (_, row) in enumerate(part.iterrows(), 1):
            g = hf.create_group(row["MRI_ID"])
            g.attrs["RID"] = row["MRI_ID"]
            g.attrs["VISCODE"] = "bl"
            g.attrs["DX"] = row["DX"]
            g.create_group(ROI_NAME).create_dataset(
                DATASET_NAME, data=crop_roi(subject_mri_dir(row)), compression="gzip")
            g.create_dataset("tabular", data=row[feat_cols].to_numpy(np.float32))
            if i % 50 == 0:
                print("  {}: {}/{}".format(os.path.basename(path), i, len(part)))

        stats_g = hf.create_group("stats")
        t = stats_g.create_group("tabular")
        t.create_dataset("columns", data=np.array(feat_cols, dtype=object), dtype=str_dt)
        t.create_dataset("mean", data=tab_mean)
        t.create_dataset("stddev", data=tab_std)
        # DAFT siempre lee este grupo, aunque no se use --normalize_image standardize
        im = stats_g.create_group(ROI_NAME).create_group(DATASET_NAME)
        im.create_dataset("mean", data=img_mean)
        im.create_dataset("stddev", data=img_std)


# =====================================================================
# 5. VERIFICACIÓN
# =====================================================================
def verify(path, expected_n, n_feats):
    """Comprueba número de sujetos, forma del cubo, vector tabular finito y atributos."""
    with h5py.File(path, "r") as hf:
        subjects = [k for k in hf.keys() if k != "stats"]
        assert len(subjects) == expected_n, "número de sujetos no coincide"
        for k in subjects:
            g = hf[k]
            assert g[ROI_NAME][DATASET_NAME].shape == (CROP,) * 3
            tab = g["tabular"][:]
            assert tab.shape == (n_feats,) and np.isfinite(tab).all(), "tabular inválido en " + k
            for a in ("RID", "VISCODE", "DX"):
                assert a in g.attrs, "falta atributo {} en {}".format(a, k)
        dx = pd.Series([hf[k].attrs["DX"] for k in subjects]).value_counts().to_dict()
    print("  [OK] {}: {} sujetos, {} variables, clases {}".format(
        os.path.basename(path), expected_n, n_feats, dx))


# =====================================================================
# 6. PRINCIPAL
# =====================================================================
def main():
    df = pd.read_csv(MASTER_CSV, dtype={"MRI_ID": str})
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print("[INFO] {} sujetos, {} variables tabulares:".format(len(df), len(feat_cols)))
    print("       " + ", ".join(feat_cols))

    # Comprobar que existen todas las imágenes antes de empezar
    missing = [r["MRI_ID"] for _, r in df.iterrows()
               if not all(os.path.exists(os.path.join(subject_mri_dir(r), f))
                          for f in (IMAGE_FILE, SEG_FILE))]
    if missing:
        raise FileNotFoundError("Faltan {} o {} para {} sujetos, p. ej.: {}".format(
            IMAGE_FILE, SEG_FILE, len(missing), missing[:5]))

    os.makedirs(OUT_DIR, exist_ok=True)
    train = df[df["split"] == "train"]
    print("[INFO] Calculando estadísticas con {} sujetos de train...".format(len(train)))
    stats = compute_stats(train, feat_cols)

    for name in ("train", "valid", "test"):
        part = df[df["split"] == name]
        path = os.path.join(OUT_DIR, name + ".h5")
        print("[INFO] Escribiendo {} ({} sujetos)...".format(path, len(part)))
        write_h5(path, part, feat_cols, stats)
        verify(path, len(part), len(feat_cols))

    print("\n[OK] Archivos listos en {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
