"""
build_master_csv_adni.py
========================

Versión ADNI de build_master_csv.py. Construye la tabla maestra (master.csv) del
subconjunto de ADNI usado como test externo, con EXACTAMENTE las mismas columnas,
nombres y codificación que el master de BrainLat, para que pase por el mismo
generador de HDF5 de DAFT (build_hdf5.py).

Qué hace
--------
1. Lee el CSV de ADNI con los 100 sujetos (50 CN y 50 AD, visita basal); todos
   se consideran válidos.
2. Busca recursivamente cada carpeta 'stats' de FreeSurfer dentro de ADNI_DIR y la
   asigna al sujeto cuyo PTID o T1_ImageID aparece en su ruta.
3. Extrae las mismas 16 medidas de FreeSurfer que build_master_csv.py
   (8 regiones x 2 hemisferios) y las une con edad, sexo y escolaridad.
4. Marca a todos los sujetos con split = "test": los modelos ya se entrenaron con
   BrainLat y ADNI no se particiona.

Salidas (en OUT_DIR)
--------------------
  master.csv               Una fila por sujeto, mismas columnas que el de BrainLat.
  excluded.csv             Sujetos con carpeta que se descartaron y el motivo.
  csv_without_folder.csv   Sujetos del CSV sin carpeta stats encontrada.
  folders_without_csv.csv  Carpetas stats que no corresponden a ningún sujeto del CSV.

Columnas de master.csv
----------------------
  Metadatos (no entran al modelo):
    MRI_ID, country ("ADNI"), dx_adni (diagnóstico original), DX (etiqueta para DAFT), split
  Variables tabulares (19, en el mismo orden que BrainLat):
    age, sex, years_education,
    hippocampus_L/R, lat_ventricle_L/R, entorhinal_thk_L/R, inftemporal_thk_L/R,
    parahippo_thk_L/R, ca1_L/R, subiculum_L/R, basolat_amyg_L/R

  Correspondencia de columnas del CSV de ADNI:
    PTID -> MRI_ID, Group -> dx_adni y DX (vía DX_MAP), Age_bl -> age,
    Sex -> sex (con SEX_MAP), Education_yrs -> years_education

  - Volúmenes normalizados por eTIV; grosores corticales en mm, sin normalizar
    (igual que BrainLat).
  - MMSE_bl no se usa: el master de BrainLat tampoco lleva puntaje cognitivo y
    MMSE no equivale a MoCA.
  - Amyloid, AV45, ABETA, FBB, FollowUp y T1_Series no se usan.

Requisitos
----------
  Python 3 con pandas.
  Sujetos de ADNI procesados con PreProcessing.py (INPUT_MODE = "folder"):
  recon-all + subcampos del hipocampo y núcleos de la amígdala.

Cómo replicar
-------------
  1. Ajustar las rutas de la sección CONFIGURACIÓN.
  2. Ejecutar:  python build_master_csv_adni.py
  3. Revisar el resumen en consola, excluded.csv y csv_without_folder.csv.

  Con la configuración por defecto el trabajo obtuvo 100 sujetos: 50 CN y 50 AD.

Parámetros que se pueden cambiar (sección CONFIGURACIÓN)
--------------------------------------------------------
  ADNI_CSV, ADNI_DIR, OUT_DIR   Rutas de entrada y salida.
  BRAINLAT_MASTER               master.csv de BrainLat (opcional, solo para comparar
                                columnas y codificación de 'sex').
  DX_MAP                        Igual que en build_master_csv.py.
  SEX_MAP                       Misma codificación que BrainLat (1 = hombre, 0 = mujer).
"""
import os
import glob
import re
import pandas as pd


# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
# Rutas (cambiar según la máquina donde se ejecute)
ADNI_CSV = "ruta/a/ADNI_100_sujetos_CN_AD_baseline.csv"   # PTID, Group, Sex, Age_bl, Education_yrs, T1_ImageID
ADNI_DIR = "ruta/a/ADNI_FreeSurfer"                       # salidas de FreeSurfer (se buscan las carpetas stats)
OUT_DIR  = "ruta/a/ADNI_salida"

# master.csv de BrainLat: solo se usa para comparar la codificación de 'sex'
BRAINLAT_MASTER = "ruta/a/salida/master.csv"

COUNTRY = "ADNI"

# Mismo DX_MAP que en build_master_csv.py (Escenario A, binario)
DX_MAP = {"CN": "CN", "AD": "Dementia"}

# Misma codificación que BrainLat: 1 = hombre, 0 = mujer
SEX_MAP = {"Male": 1, "Female": 0}

# Validación externa: todos los sujetos van a test, sin partición
SPLIT_VALUE = "test"

# Carpetas que se ignoran al buscar 'stats'
IGNORE_DIRS = {"fsaverage", "lh.EC_average", "rh.EC_average"}

# =====================================================================
# 2. VARIABLES DE FREESURFER (idénticas a build_master_csv.py)
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

CLINICAL_FEATURES = ["age", "sex", "years_education"]

SEG_DEFAULT_HEADER = ["Index", "SegId", "NVoxels", "Volume_mm3", "StructName"]


# =====================================================================
# 3. LECTURA DE ARCHIVOS .stats (idéntica a build_master_csv.py)
# =====================================================================
def find_stats_file(stats_dir, pattern):
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
# 4. CSV DE ADNI
# =====================================================================
def read_table(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    sep = max([",", ";", "\t"], key=header.count)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig",
                     dtype={"PTID": str, "T1_ImageID": str})
    df.columns = (df.columns.astype(str)
                  .str.replace("\ufeff", "", regex=False)
                  .str.replace("\u200b", "", regex=False)
                  .str.strip())
    return df


def load_adni():
    a = read_table(ADNI_CSV)
    needed = ["PTID", "Group", "Sex", "Age_bl", "Education_yrs", "T1_ImageID"]
    missing = [c for c in needed if c not in a.columns]
    if missing:
        raise KeyError("Faltan columnas en el CSV de ADNI: {}. Columnas leídas: {}".format(
            missing, list(a.columns)))

    for c in ["PTID", "Group", "Sex", "T1_ImageID"]:
        a[c] = a[c].astype(str).str.strip()

    dup = sorted(a.loc[a["PTID"].duplicated(), "PTID"].unique())
    if dup:
        print("[AVISO] PTID duplicados en el CSV de ADNI (se conserva la primera fila): {}".format(dup))
    a = a.drop_duplicates("PTID")

    unknown_sex = sorted(set(a["Sex"]) - set(SEX_MAP))
    if unknown_sex:
        print("[AVISO] Valores de Sex sin mapeo en SEX_MAP: {}".format(unknown_sex))

    clin = pd.DataFrame({
        "MRI_ID": a["PTID"],
        "DX": a["Group"],
        "age": pd.to_numeric(a["Age_bl"], errors="coerce"),
        "sex": a["Sex"].map(SEX_MAP),
        "years_education": pd.to_numeric(a["Education_yrs"], errors="coerce"),
        "image_id": a["T1_ImageID"],
    })
    print("[INFO] CSV ADNI: {} sujetos (todos válidos)".format(len(clin)))
    return clin.set_index("MRI_ID")


# =====================================================================
# 5. BÚSQUEDA DE CARPETAS stats
# =====================================================================
def find_stats_dirs(root):
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if os.path.basename(dirpath) == "stats":
            found.append(dirpath)
            dirnames[:] = []
    return sorted(found)


def normalize(s):
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def match_folders(clin, stats_dirs):
    """Asigna carpetas stats a sujetos por PTID (con o sin guiones bajos,
    p. ej. 000_S_0000 o sub-ADNI000S0000) o por T1_ImageID (p. ej. I000000).
    Devuelve {MRI_ID: [rutas candidatas]} y la lista de carpetas sin sujeto."""
    rel = {d: os.path.relpath(d, ADNI_DIR) for d in stats_dirs}
    rel_norm = {d: normalize(r) for d, r in rel.items()}

    candidates, used = {}, set()
    for mri_id, row in clin.iterrows():
        ptid_norm = normalize(mri_id)
        img_re = re.compile(r"(?<![A-Za-z0-9]){}(?![0-9])".format(re.escape(row["image_id"])))
        by_ptid = [d for d in stats_dirs if ptid_norm in rel_norm[d]]
        by_img = [d for d in stats_dirs if img_re.search(rel[d])]
        cands = sorted(set(by_ptid) | set(by_img))
        # Si hay varias, se prefieren las que contienen el T1_ImageID del baseline
        if len(cands) > 1 and by_img:
            cands = sorted(by_img)
        if cands:
            candidates[mri_id] = cands
            used.update(cands)

    orphans = [rel[d] for d in stats_dirs if d not in used]
    return candidates, orphans


# =====================================================================
# 6. PRINCIPAL
# =====================================================================
def main():
    clin = load_adni()

    stats_dirs = find_stats_dirs(ADNI_DIR)
    print("[INFO] Carpetas 'stats' encontradas en {}: {}".format(ADNI_DIR, len(stats_dirs)))
    candidates, orphans = match_folders(clin, stats_dirs)

    rows, excluded = [], []
    for mri_id, c in clin.iterrows():
        if mri_id not in candidates:
            continue  # va a csv_without_folder.csv

        def exclude(reason):
            excluded.append({"MRI_ID": mri_id, "country": COUNTRY, "reason": reason})

        cands = candidates[mri_id]
        if len(cands) > 1:
            exclude("varias carpetas stats: {}".format(
                [os.path.relpath(d, ADNI_DIR) for d in cands]))
            continue

        dx = c["DX"]
        if dx not in DX_MAP:
            exclude("diagnóstico no incluido ({})".format(dx))
            continue

        missing_clin = [f for f in CLINICAL_FEATURES if pd.isna(c[f])]
        if missing_clin:
            exclude("clínica incompleta: {}".format(missing_clin))
            continue

        try:
            fs = extract_freesurfer(cands[0])
        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            exclude("FreeSurfer: {}".format(e))
            continue

        row = {"MRI_ID": mri_id, "country": COUNTRY,
               "dx_adni": dx, "DX": DX_MAP[dx]}
        row.update({f: float(c[f]) for f in CLINICAL_FEATURES})
        row.update(fs)
        rows.append(row)

    no_folder = sorted(set(clin.index) - set(candidates))

    fs_cols = [s["name"].format(H=H) for s in FS_FEATURES for _, _, H in HEMIS]
    columns = ["MRI_ID", "country", "dx_adni", "DX", "split"] + CLINICAL_FEATURES + fs_cols
    df = pd.DataFrame(rows)
    if not df.empty:
        assert df["MRI_ID"].is_unique, "Hay MRI_ID repetidos en master.csv"
        df["sex"] = df["sex"].astype(int)
        df["split"] = SPLIT_VALUE
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "master.csv"), index=False)
    pd.DataFrame(excluded, columns=["MRI_ID", "country", "reason"]).to_csv(
        os.path.join(OUT_DIR, "excluded.csv"), index=False)
    pd.DataFrame({"MRI_ID": no_folder}).to_csv(
        os.path.join(OUT_DIR, "csv_without_folder.csv"), index=False)
    pd.DataFrame({"stats_dir": orphans}).to_csv(
        os.path.join(OUT_DIR, "folders_without_csv.csv"), index=False)

    print("\n[OK] master.csv: {} sujetos, {} features".format(
        len(df), len(CLINICAL_FEATURES) + len(fs_cols)))
    print("[INFO] Excluidos: {}  (ver excluded.csv)".format(len(excluded)))
    print("[INFO] En el CSV sin carpeta stats: {}  (ver csv_without_folder.csv)".format(len(no_folder)))
    print("[INFO] Carpetas stats sin sujeto en el CSV: {}  (ver folders_without_csv.csv)".format(len(orphans)))
    if excluded:
        print("\nMotivos de exclusión:")
        print(pd.Series([e["reason"].split(":")[0] for e in excluded]).value_counts().to_string())
    if not df.empty:
        print("\nConversión aplicada (ADNI -> DAFT):")
        print(pd.crosstab(df["dx_adni"], df["DX"]))
        print("\nDiagnóstico por partición:")
        print(pd.crosstab(df["DX"], df["split"], margins=True))
        print("\nSexo (ADNI, ya recodificado):")
        print(df["sex"].value_counts().sort_index().to_string())

    # Comparación con BrainLat para verificar que 'sex' use la misma codificación
    if os.path.isfile(BRAINLAT_MASTER):
        bl = pd.read_csv(BRAINLAT_MASTER)
        bl_cols = ["dx_adni" if c == "dx_brainlat" else c for c in bl.columns]
        if bl_cols != columns:
            print("\n[AVISO] Las columnas no coinciden con el master de BrainLat "
                  "(aparte de dx_brainlat -> dx_adni).")
        print("\nSexo (BrainLat master.csv), para comparar con SEX_MAP:")
        print(bl["sex"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
