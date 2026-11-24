"""
build_master_csv.py
===================

Construye la tabla maestra (master.csv) usada para entrenar y evaluar los modelos
del trabajo de grado (DAFT, ResNet solo imagen, FT-Transformer, Random Forest y
XGBoost) sobre el dataset BrainLat.

Qué hace
--------
1. Recorre BRAINLAT_T1_DIR/<PAÍS>/<sub-XXXX_T1w>/stats y extrae 16 medidas de
   FreeSurfer (8 regiones x 2 hemisferios).
2. Las une con los CSV demográfico y cognitivo de BrainLat (edad, sexo,
   escolaridad y diagnóstico).
3. Filtra los sujetos (validez, diagnóstico, datos completos, FreeSurfer correcto).
4. Asigna una partición estratificada train/valid/test por diagnóstico.

Salidas (en OUT_DIR)
--------------------
  master.csv              Una fila por sujeto. Es la entrada del generador del HDF5
                          de DAFT y de los modelos tabulares.
  excluded.csv            Sujetos con carpeta de imagen que se descartaron y el motivo.
  csv_without_folder.csv  Sujetos que aparecen en los CSV pero no tienen carpeta de imagen.

Columnas de master.csv
----------------------
  Metadatos (no entran al modelo):
    MRI_ID, country, dx_brainlat (diagnóstico original), DX (etiqueta para DAFT), split
  Variables tabulares (19, en este orden):
    age, sex, years_education,
    hippocampus_L/R, lat_ventricle_L/R, entorhinal_thk_L/R, inftemporal_thk_L/R,
    parahippo_thk_L/R, ca1_L/R, subiculum_L/R, basolat_amyg_L/R

  - Volúmenes normalizados por eTIV (volumen intracraneal estimado).
  - Grosores corticales en mm, sin normalizar.
  - VISCODE no se incluye: el generador del HDF5 lo escribe como "bl".

Requisitos
----------
  Python 3 con numpy y pandas.
  Sujetos procesados con FreeSurfer recon-all + segmentación de subcampos del
  hipocampo y núcleos de la amígdala (archivos hipposubfields.* y amygdalar-nuclei.*).

Cómo replicar
-------------
  1. Ajustar las rutas de la sección CONFIGURACIÓN.
  2. Ejecutar:  python build_master_csv.py
  3. Revisar el resumen en consola y excluded.csv.

  Con la configuración por defecto (CN vs AD, solo valid == 1, semilla 42) el
  trabajo obtuvo 343 sujetos: 137 CN y 206 AD.

Parámetros que se pueden cambiar (todos en la sección CONFIGURACIÓN)
--------------------------------------------------------------------
  BRAINLAT_T1_DIR, DEMOGRAPHIC_CSV,
  COGNITION_CSV, VALID_CSV, OUT_DIR   Rutas de entrada y salida.
  DX_MAP                              Qué diagnósticos entran y con qué etiqueta.
  SPLIT_FRACTIONS                     Proporciones train / valid / test.
  SEED                                Semilla de la partición (reproducibilidad).
  IGNORE_DIRS                         Carpetas de BRAINLAT_T1_DIR que no son países.
  CLINICAL_FEATURES                   Variables clínicas incluidas.
  FS_FEATURES                         Regiones de FreeSurfer extraídas (sección 2).
"""
import os
import glob
import re
import numpy as np
import pandas as pd

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
# Rutas (cambiar según la máquina donde se ejecute)
BRAINLAT_T1_DIR = "ruta/a/BrainLatT1"                      # carpetas <PAÍS>/<sub-XXXX_T1w>/stats
DEMOGRAPHIC_CSV = "ruta/a/BrainLat_Demographic_MRI.csv"    # MRI_ID, diagnosis, sex, Age, years_education
COGNITION_CSV   = "ruta/a/BrainLat_Cognition_MRI.csv"      # MRI_ID, diagnosis
VALID_CSV       = "ruta/a/BrainLat_Valid.csv"              # MRI_ID, valid (1 = imagen válida)
OUT_DIR         = "ruta/a/salida"

# Diagnósticos incluidos: {etiqueta BrainLat: etiqueta para DAFT}.
# Los diagnósticos que no estén como clave se excluyen.
# Configuración del trabajo (binaria, entrenar DAFT con --num_classes 2):
DX_MAP = {"CN": "CN", "AD": "Dementia"}
# Alternativa multiclase (--num_classes 3); requiere editar
# DIAGNOSIS_CODES_MULTICLASS en daft/data_utils/adni_hdf.py a {"CN": 0, "AD": 1, "bvFTD": 2}:
# DX_MAP = {"CN": "CN", "AD": "AD", "bvFTD": "bvFTD"}

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train, valid, test (estratificado por DX)
SEED = 42

# Carpetas dentro de BRAINLAT_T1_DIR que no son países
IGNORE_DIRS = {"fsaverage", "lh.EC_average", "rh.EC_average"}

# Variables clínicas (nombres tras renombrar 'Age' -> 'age')
CLINICAL_FEATURES = ["age", "sex", "years_education"]

# =====================================================================
# 2. VARIABLES DE FREESURFER
#   name: nombre de la columna ({H} = L/R)
#   file: archivo dentro de stats/ ({hemi} = lh/rh; v* = versión del .stats)
#   rows: estructuras que se suman ({Hemi} = Left/Right)
#   col:  columna que se lee del .stats
#   etiv: True = dividir por eTIV
# =====================================================================
FS_FEATURES = [
    dict(name="hippocampus_{H}",     file="aseg.stats",
         rows=["{Hemi}-Hippocampus"], col="Volume_mm3", etiv=True),
    dict(name="lat_ventricle_{H}",   file="aseg.stats",
         rows=["{Hemi}-Lateral-Ventricle"], col="Volume_mm3", etiv=True),
    dict(name="entorhinal_thk_{H}",  file="{hemi}.aparc.stats",
         rows=["entorhinal"], col="ThickAvg", etiv=False),
    dict(name="inftemporal_thk_{H}", file="{hemi}.aparc.stats",
         rows=["inferiortemporal"], col="ThickAvg", etiv=False),
    dict(name="parahippo_thk_{H}",   file="{hemi}.aparc.stats",
         rows=["parahippocampal"], col="ThickAvg", etiv=False),
    dict(name="ca1_{H}",             file="hipposubfields.{hemi}.T1.v*.stats",
         rows=["CA1-head", "CA1-body"], col="Volume_mm3", etiv=True),
    dict(name="subiculum_{H}",       file="hipposubfields.{hemi}.T1.v*.stats",
         rows=["subiculum-head", "subiculum-body"], col="Volume_mm3", etiv=True),
    dict(name="basolat_amyg_{H}",    file="amygdalar-nuclei.{hemi}.T1.v*.stats",
         rows=["Lateral-nucleus", "Basal-nucleus", "Accessory-Basal-nucleus"],
         col="Volume_mm3", etiv=True),
]
HEMIS = [("lh", "Left", "L"), ("rh", "Right", "R")]

# Encabezado por defecto de los .stats de segmentación que no traen '# ColHeaders'
SEG_DEFAULT_HEADER = ["Index", "SegId", "NVoxels", "Volume_mm3", "StructName"]


# =====================================================================
# 3. LECTURA DE ARCHIVOS .stats
# =====================================================================
def find_stats_file(stats_dir, pattern):
    """Busca el archivo del patrón. En archivos versionados acepta solo '.vNN.stats'
    (ignora variantes .CA / .HBT / .FS60) y exige que haya una única versión."""
    matches = sorted(glob.glob(os.path.join(stats_dir, pattern)))
    if "v*" in pattern:
        matches = [m for m in matches if re.search(r"\.v\d+\.stats$", m)]
    if not matches:
        raise FileNotFoundError("no existe " + pattern)
    if len(matches) > 1:
        raise ValueError("varias versiones de {}: {}".format(
            pattern, [os.path.basename(m) for m in matches]))
    return matches[0]


def read_stats(path):
    """Devuelve (tabla indexada por StructName en minúsculas, eTIV o None)."""
    header, rows, etiv = None, [], None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("# Measure EstimatedTotalIntraCranialVol"):
                etiv = float(line.split(",")[3])
            elif line.startswith("# ColHeaders"):
                header = line.split()[2:]
            elif not line.startswith("#") and line.strip():
                rows.append(line.split())
    if not rows:
        raise ValueError("archivo vacío: " + os.path.basename(path))
    if header is None:
        header = SEG_DEFAULT_HEADER
    n = min(len(header), min(len(r) for r in rows))
    table = pd.DataFrame([r[:n] for r in rows], columns=header[:n])
    table.index = table["StructName"].str.lower()
    return table, etiv


def extract_freesurfer(stats_dir):
    """Extrae las 16 variables de FreeSurfer de un sujeto. Lanza una excepción
    si falta un archivo o estructura, o si algún valor es <= 0."""
    cache = {}

    def get(pattern):
        if pattern not in cache:
            cache[pattern] = read_stats(find_stats_file(stats_dir, pattern))
        return cache[pattern]

    _, etiv = get("aseg.stats")
    if not etiv:
        raise ValueError("aseg.stats sin eTIV")

    out = {}
    for spec in FS_FEATURES:
        for hemi, Hemi, H in HEMIS:
            fname = spec["file"].format(hemi=hemi)
            table, _ = get(fname)
            wanted = [r.format(Hemi=Hemi).lower() for r in spec["rows"]]
            missing = [w for w in wanted if w not in table.index]
            if missing:
                raise KeyError("{}: faltan filas {}".format(fname, missing))
            value = table.loc[wanted, spec["col"]].astype(float).sum()
            if value <= 0:
                raise ValueError("{} = {} (valor no válido)".format(spec["name"].format(H=H), value))
            out[spec["name"].format(H=H)] = value / etiv if spec["etiv"] else value
    return out


# =====================================================================
# 4. DATOS CLÍNICOS Y VALIDEZ
# =====================================================================
def read_table(path):
    """Lee un CSV tolerando el BOM de Excel, separador ',', ';' o tabulador,
    y espacios o caracteres invisibles en los nombres de columna."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    sep = max([",", ";", "\t"], key=header.count)

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype={"MRI_ID": str})
    df.columns = (df.columns.astype(str)
                  .str.replace("﻿", "", regex=False)
                  .str.replace("​", "", regex=False)
                  .str.strip())
    if "MRI_ID" not in df.columns:
        raise KeyError("{}: no se encontró la columna 'MRI_ID'. Separador usado: {!r}. "
                       "Columnas leídas: {}".format(os.path.basename(path), sep, list(df.columns)))
    return df


def load_valid_ids():
    """Devuelve (IDs con valid == 1, todos los IDs listados en VALID_CSV)."""
    v = read_table(VALID_CSV)
    if "valid" not in v.columns:
        raise KeyError("VALID_CSV no tiene columna 'valid'. Columnas: {}".format(list(v.columns)))
    v["MRI_ID"] = v["MRI_ID"].astype(str).str.strip()
    is_valid = pd.to_numeric(v["valid"], errors="coerce") == 1
    print("[INFO] {}: {} válidos de {}".format(os.path.basename(VALID_CSV), int(is_valid.sum()), len(v)))
    return set(v.loc[is_valid, "MRI_ID"]), set(v["MRI_ID"])


def load_clinical():
    """Une los CSV demográfico y cognitivo por MRI_ID. Conserva el diagnóstico de
    ambos (DX y DX_cog) para verificar luego que coincidan."""
    demo = read_table(DEMOGRAPHIC_CSV)
    cog = read_table(COGNITION_CSV)

    for name, df in (("Demographic", demo), ("Cognition", cog)):
        df["MRI_ID"] = df["MRI_ID"].astype(str).str.strip()
        dup_ids = sorted(df.loc[df["MRI_ID"].duplicated(), "MRI_ID"].unique())
        if dup_ids:
            print("[AVISO] MRI_ID duplicados en {} (se conserva la primera fila): {}".format(name, dup_ids))

    demo = demo.drop_duplicates("MRI_ID")[["MRI_ID", "diagnosis", "sex", "Age", "years_education"]]
    cog = cog.drop_duplicates("MRI_ID")[["MRI_ID", "diagnosis"]]

    clin = demo.merge(cog, on="MRI_ID", how="outer", suffixes=("", "_cog"))
    clin = clin.rename(columns={"Age": "age", "diagnosis": "DX", "diagnosis_cog": "DX_cog"})
    return clin.set_index("MRI_ID")


# =====================================================================
# 5. PARTICIÓN ESTRATIFICADA
# =====================================================================
def stratified_split(dx):
    """Asigna train/valid/test dentro de cada diagnóstico según SPLIT_FRACTIONS.
    Con la misma SEED y los mismos sujetos, la partición es siempre la misma."""
    rng = np.random.RandomState(SEED)
    dx = np.asarray(dx)
    split = np.empty(len(dx), dtype=object)
    for label in sorted(set(dx)):
        pos = np.where(dx == label)[0]
        rng.shuffle(pos)
        n = len(pos)
        n_train = int(round(n * SPLIT_FRACTIONS[0]))
        n_valid = int(round(n * SPLIT_FRACTIONS[1]))
        split[pos[:n_train]] = "train"
        split[pos[n_train:n_train + n_valid]] = "valid"
        split[pos[n_train + n_valid:]] = "test"
    return split


# =====================================================================
# 6. PRINCIPAL
# =====================================================================
def main():
    clin = load_clinical()
    valid_ids, listed_ids = load_valid_ids()
    rows, excluded, seen = [], [], {}

    countries = sorted(d for d in os.listdir(BRAINLAT_T1_DIR)
                       if os.path.isdir(os.path.join(BRAINLAT_T1_DIR, d)) and d not in IGNORE_DIRS)
    print("[INFO] Países encontrados: {}".format(countries))

    for country in countries:
        country_dir = os.path.join(BRAINLAT_T1_DIR, country)
        for subject_folder in sorted(os.listdir(country_dir)):
            subject_dir = os.path.join(country_dir, subject_folder)
            if not os.path.isdir(subject_dir):
                continue
            # La carpeta se llama sub-XXXX_T1w; el ID en los CSV es sub-XXXX
            mri_id = subject_folder[:-4] if subject_folder.endswith("_T1w") else subject_folder

            def exclude(reason):
                excluded.append({"MRI_ID": mri_id, "country": country, "reason": reason})

            # --- Filtros, en orden. El primero que falla define el motivo de exclusión ---
            if mri_id in seen:
                exclude("carpeta duplicada: ya se procesó en {}".format(seen[mri_id]))
                continue
            seen[mri_id] = country

            if mri_id not in listed_ids:
                exclude("no aparece en VALID_CSV")
                continue
            if mri_id not in valid_ids:
                exclude("marcado como no válido (valid != 1)")
                continue
            if mri_id not in clin.index:
                exclude("sin datos clínicos en los CSV")
                continue
            c = clin.loc[mri_id]

            # Diagnóstico: debe coincidir entre ambos CSV y estar en DX_MAP
            if pd.notna(c["DX"]) and pd.notna(c["DX_cog"]) and c["DX"] != c["DX_cog"]:
                exclude("diagnóstico distinto entre CSV ({} vs {})".format(c["DX"], c["DX_cog"]))
                continue
            dx = c["DX"] if pd.notna(c["DX"]) else c["DX_cog"]
            if pd.isna(dx):
                exclude("sin diagnóstico")
                continue
            dx = str(dx).strip()
            if dx not in DX_MAP:
                exclude("diagnóstico no incluido ({})".format(dx))
                continue

            missing_clin = [f for f in CLINICAL_FEATURES if pd.isna(c[f])]
            if missing_clin:
                exclude("clínica incompleta: {}".format(missing_clin))
                continue

            stats_dir = os.path.join(subject_dir, "stats")
            if not os.path.isdir(stats_dir):
                exclude("sin carpeta stats")
                continue
            try:
                fs = extract_freesurfer(stats_dir)
            except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
                exclude("FreeSurfer: {}".format(e))
                continue

            # --- Sujeto aceptado ---
            row = {"MRI_ID": mri_id, "country": country,
                   "dx_brainlat": dx, "DX": DX_MAP[dx]}
            row.update({f: float(c[f]) for f in CLINICAL_FEATURES})
            row.update(fs)
            rows.append(row)

    no_folder = sorted(set(clin.index) - set(seen))

    # Tabla final con columnas en orden fijo y partición asignada
    fs_cols = [s["name"].format(H=H) for s in FS_FEATURES for _, _, H in HEMIS]
    columns = ["MRI_ID", "country", "dx_brainlat", "DX", "split"] + CLINICAL_FEATURES + fs_cols
    df = pd.DataFrame(rows)
    if not df.empty:
        assert df["MRI_ID"].is_unique, "Hay MRI_ID repetidos en master.csv"
        df["sex"] = df["sex"].astype(int)
        df["split"] = stratified_split(df["DX"].to_numpy())
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "master.csv"), index=False)
    pd.DataFrame(excluded, columns=["MRI_ID", "country", "reason"]).to_csv(
        os.path.join(OUT_DIR, "excluded.csv"), index=False)
    pd.DataFrame({"MRI_ID": no_folder}).to_csv(
        os.path.join(OUT_DIR, "csv_without_folder.csv"), index=False)

    # Resumen en consola
    print("\n[OK] master.csv: {} sujetos, {} variables tabulares".format(
        len(df), len(CLINICAL_FEATURES) + len(fs_cols)))
    print("[INFO] Excluidos: {}  (ver excluded.csv)".format(len(excluded)))
    print("[INFO] En los CSV sin carpeta de imagen: {}  (ver csv_without_folder.csv)".format(len(no_folder)))
    if excluded:
        print("\nMotivos de exclusión:")
        print(pd.Series([e["reason"].split(":")[0] for e in excluded]).value_counts().to_string())
    if not df.empty:
        print("\nConversión aplicada (BrainLat -> DAFT):")
        print(pd.crosstab(df["dx_brainlat"], df["DX"]))
        print("\nDiagnóstico por partición:")
        print(pd.crosstab(df["DX"], df["split"], margins=True))
        print("\nDiagnóstico por país:")
        print(pd.crosstab(df["country"], df["DX"], margins=True))


if __name__ == "__main__":
    main()
