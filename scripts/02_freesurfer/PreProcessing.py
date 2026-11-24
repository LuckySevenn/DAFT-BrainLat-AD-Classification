"""
PreProcessing.py
================

Procesa imágenes T1 (NIfTI) con FreeSurfer y deja, para cada sujeto, la carpeta
que usan build_master_csv.py y generate_hdf5.py:

  <PATH_DESTINATION>/<sujeto>/mri/    norm.mgz, aseg.mgz, ...
  <PATH_DESTINATION>/<sujeto>/stats/  aseg.stats, lh/rh.aparc.stats,
                                      hipposubfields.*.stats, amygdalar-nuclei.*.stats, ...

Pasos por sujeto (en orden)
---------------------------
  1. recon-all -all              reconstrucción completa (crítico)
  2. recon-all -autorecon1/2/3   solo si RERUN_AUTORECON = True (ver nota abajo)
  3. segmentHA_T1.sh             subcampos del hipocampo y núcleos de la amígdala (crítico)
  4. segmentThalamicNuclei.sh    núcleos talámicos (crítico)
  5. Tablas resumen (no críticos; si fallan se registra y se sigue):
       asegstats2table (tálamo, hipocampo, aseg), mri_segstats (brainvol.stats),
       aparcstats2table (grosor lh/rh)
  Si un paso crítico falla, se aborta ese sujeto y se continúa con los demás.

Modos de entrada (INPUT_MODE)
-----------------------------
  "csv"     Usado con BrainLat. Lee VALID_CSV y procesa solo los MRI_ID con
            valid == 1, buscando <MRI_ID>_T1w.nii(.gz) en PATH_ORIGIN.
            El sujeto se llama <MRI_ID>_T1w.
            Para obtener la estructura que espera build_master_csv.py
            (BrainLatT1/<PAÍS>/<MRI_ID>_T1w), se ejecuta una vez por país con
            PATH_ORIGIN = imágenes de ese país y PATH_DESTINATION = BrainLatT1/<PAÍS>.

  "folder"  Usado con ADNI. Recorre PATH_ORIGIN y todas sus subcarpetas y procesa
            todos los .nii / .nii.gz. El sujeto se nombra con el ID de ADNI
            (p. ej. 002_S_0295) si aparece en la ruta; si no, con el nombre del
            archivo. Los nombres repetidos reciben sufijo _2, _3...
            Guarda subject_mapping.csv (sujeto -> archivo de origen).

Salidas adicionales
-------------------
  LOG_FILE           Un registro por paso: estado, error, inicio, fin y duración.
  subject_mapping.csv  (solo modo "folder").
  Los sujetos cuya carpeta ya existe en PATH_DESTINATION se saltan, así que el
  script se puede relanzar sin repetir lo ya procesado.

Requisitos
----------
  Linux o WSL con FreeSurfer instalado (FREESURFER_HOME) y licencia configurada,
  incluido el módulo de subcampos (segmentHA_T1.sh necesita el runtime de MATLAB).
  Python 3 con pandas.

Cómo replicar
-------------
  1. Ajustar la sección CONFIGURACIÓN (modo, rutas, paralelismo).
  2. Ejecutar:  python PreProcessing.py
  3. Revisar LOG_FILE: los sujetos con PROCESS_END = FAILED no tendrán stats
     completos y build_master_csv.py los excluirá.

Parámetros que se pueden cambiar (sección CONFIGURACIÓN)
--------------------------------------------------------
  INPUT_MODE            "csv" (BrainLat) o "folder" (ADNI).
  PATH_ORIGIN           Carpeta con las imágenes NIfTI.
  PATH_DESTINATION      SUBJECTS_DIR de FreeSurfer (donde se crean los sujetos).
  VALID_CSV             Solo modo "csv": CSV con columnas MRI_ID y valid.
  LOG_FILE, BATCH_LABEL Nombre del log y etiqueta del lote dentro del log.
  FREESURFER_HOME       Instalación de FreeSurfer.
  BATCH_SIZE            Sujetos en paralelo (cada recon-all usa ~1 núcleo y 8-12 GB de RAM).
  TIMEOUT_SECONDS       Tiempo máximo por paso.
  RERUN_AUTORECON       Repetir autorecon1/2/3 después de recon-all -all.

Nota sobre RERUN_AUTORECON
--------------------------
  recon-all -all ya incluye autorecon1, 2 y 3, así que repetirlos casi duplica
  el tiempo sin aportar pasos nuevos. En el trabajo, BrainLat se procesó con la
  versión del script que los repetía (equivale a True) y ADNI sin repetirlos
  (False). Para reproducir exactamente lo hecho con BrainLat, usar True.
"""
import os
import re
import csv
import shlex
import datetime
import subprocess
from multiprocessing import Pool

import pandas as pd

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
INPUT_MODE = "csv"                       # "csv" (BrainLat) o "folder" (ADNI)

PATH_ORIGIN      = "ruta/a/imagenes_nifti"         # p. ej. BrainLat/<PAÍS>_T1w o ADNI/
PATH_DESTINATION = "ruta/a/BrainLatT1/<PAIS>"      # SUBJECTS_DIR de FreeSurfer
VALID_CSV        = "ruta/a/BrainLat_Valid.csv"     # solo modo "csv"

LOG_FILE    = os.path.join(PATH_DESTINATION, "execution_log.csv")
BATCH_LABEL = "BrainLat"                 # etiqueta del lote en el log

FREESURFER_HOME = "/usr/local/freesurfer"

BATCH_SIZE      = 2                      # 16 GB RAM -> 2; 32 GB RAM -> 3 o 4
TIMEOUT_SECONDS = 11 * 60 * 60           # 11 horas por paso
RERUN_AUTORECON = True                   # True = como se procesó BrainLat (ver nota arriba)

# =====================================================================
# 2. CONSTANTES (no es necesario cambiarlas)
# =====================================================================
LOG_COLUMNS = ["batch", "case", "file", "step", "status", "error", "process_id",
               "start_time", "end_time", "duration_seconds"]
VALID_EXTENSIONS = (".nii", ".nii.gz")
ADNI_ID_REGEX = re.compile(r"\d{3}_S_\d{4}")
MAPPING_FILE = os.path.join(PATH_DESTINATION, "subject_mapping.csv")


# =====================================================================
# 3. SELECCIÓN DE ARCHIVOS
#    Ambas funciones devuelven una lista de (ruta_nifti, nombre_sujeto).
# =====================================================================
def strip_extension(filename):
    for ext in (".nii.gz", ".nii"):
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def files_from_csv(csv_path, root):
    """Modo "csv": MRI_ID con valid == 1 cuyo <MRI_ID>_T1w.nii(.gz) está en root."""
    df = pd.read_csv(csv_path, dtype={"MRI_ID": str})
    if "valid" not in df.columns:
        raise KeyError("{} no tiene columna 'valid'. Columnas: {}".format(csv_path, list(df.columns)))
    valid_ids = df.loc[pd.to_numeric(df["valid"], errors="coerce") == 1, "MRI_ID"].str.strip()
    print("[INFO] Sujetos con valid == 1: {} de {}".format(len(valid_ids), len(df)))

    file_cases, not_found = [], 0
    for mri_id in valid_ids:
        case = mri_id + "_T1w"
        paths = [os.path.join(root, case + ext) for ext in VALID_EXTENSIONS]
        path = next((p for p in paths if os.path.exists(p)), None)
        if path:
            file_cases.append((path, case))
        else:
            not_found += 1     # normal si root solo contiene las imágenes de un país
    print("[INFO] Encontrados en {}: {} | no encontrados: {}".format(root, len(file_cases), not_found))
    return file_cases


def files_from_folder(root):
    """Modo "folder": todos los NIfTI bajo root, nombrados por ID de ADNI."""
    dest_abs = os.path.abspath(PATH_DESTINATION)
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # No entrar en la carpeta de resultados si está dentro de root
        dirnames[:] = [d for d in dirnames
                       if os.path.abspath(os.path.join(dirpath, d)) != dest_abs]
        files += [os.path.join(dirpath, f) for f in filenames if f.endswith(VALID_EXTENSIONS)]
    files.sort()
    print("[INFO] Archivos NIfTI encontrados en '{}' (recursivo): {}".format(root, len(files)))

    used, file_cases = {}, []
    for path in files:
        rel = os.path.relpath(path, root)
        m = ADNI_ID_REGEX.search(rel)
        case = m.group(0) if m else re.sub(r"[^A-Za-z0-9_\-]", "_",
                                           strip_extension(os.path.basename(path)))
        if case in used:
            used[case] += 1
            new_case = "{}_{}".format(case, used[case])
            print("[AVISO] Nombre repetido '{}' -> '{}' para {}".format(case, new_case, rel))
            case = new_case
        else:
            used[case] = 1
        file_cases.append((path, case))

    pd.DataFrame([(c, os.path.relpath(p, root)) for p, c in file_cases],
                 columns=["case", "source_file"]).to_csv(MAPPING_FILE, index=False)
    print("[INFO] Mapeo sujeto -> archivo guardado en {}".format(MAPPING_FILE))
    return file_cases


# =====================================================================
# 4. LOG Y EJECUCIÓN DE PASOS
# =====================================================================
def write_step_log(log_row, step, status, error, start=None, end=None):
    entry = {
        "batch": log_row["batch"], "case": log_row["case"], "file": log_row["file"],
        "step": step, "status": status, "error": error,
        "process_id": log_row["process_id"],
        "start_time": start.isoformat() if start else "",
        "end_time": end.isoformat() if end else "",
        "duration_seconds": (end - start).total_seconds() if start and end else None,
    }
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, LOG_COLUMNS).writerow(entry)


def extract_recon_error(subject, n_lines=20):
    """Últimas líneas de error de recon-all.log, para registrarlas en el log."""
    log_path = os.path.join(PATH_DESTINATION, subject, "scripts", "recon-all.log")
    if not os.path.exists(log_path):
        return "recon-all.log not found"
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
    errors = []
    for line in reversed(lines):
        if any(k in line for k in ("ERROR", "FATAL", "IFLAG", "exited with ERRORS")):
            errors.append(line.strip())
            if len(errors) >= n_lines:
                break
    return " | ".join(reversed(errors)) if errors else "recon-all failed (no explicit ERROR found)"


def run_step(step, command, log_row, allow_failure):
    """Ejecuta un comando de bash y lo registra.
    Devuelve True si el pipeline puede continuar (éxito, o fallo en paso no crítico)."""
    print("[PID {}] [{}] {}".format(log_row["process_id"], log_row["case"], step))
    start = datetime.datetime.now()
    try:
        result = subprocess.run(command, shell=True, executable="/bin/bash",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        write_step_log(log_row, step, "TIMEOUT_SKIPPED" if allow_failure else "TIMEOUT",
                       "Superó el timeout de {} s".format(TIMEOUT_SECONDS),
                       start, datetime.datetime.now())
        return allow_failure
    end = datetime.datetime.now()

    if result.returncode == 0:
        write_step_log(log_row, step, "OK", "-", start, end)
        return True

    if step.startswith("recon-all"):
        error = extract_recon_error(log_row["case"])
    else:
        error = (result.stderr or result.stdout).replace("\n", " ")[:500]
    write_step_log(log_row, step, "FAIL_SKIPPED" if allow_failure else "FAIL", error, start, end)
    return allow_failure


# =====================================================================
# 5. PASOS DE FREESURFER
#    Cada paso: (nombre, comando, allow_failure). allow_failure=False = crítico.
# =====================================================================
def get_freesurfer_steps(case, input_path):
    sd = shlex.quote(PATH_DESTINATION)
    subj = PATH_DESTINATION + "/" + case
    q = lambda rel: shlex.quote(subj + rel)
    fs = ("export FREESURFER_HOME={} && source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
          "export SUBJECTS_DIR={}".format(shlex.quote(FREESURFER_HOME), sd))

    steps = [("recon-all -all",
              "{} && recon-all -s {} -i {} -all".format(fs, case, shlex.quote(input_path)), False)]
    if RERUN_AUTORECON:
        steps += [("recon-all -autorecon{}".format(k),
                   "{} && recon-all -s {} -autorecon{}".format(fs, case, k), False)
                  for k in (1, 2, 3)]

    # Segmentaciones: generan hipposubfields.*, amygdalar-nuclei.* y thalamic-nuclei.*
    steps += [
        ("segmentHA_T1.sh",          "{} && segmentHA_T1.sh {}".format(fs, case), False),
        ("segmentThalamicNuclei.sh", "{} && segmentThalamicNuclei.sh {}".format(fs, case), False),
    ]

    # Tablas resumen (no críticas)
    table = "{} && cd {} && ".format(fs, sd)
    for hemi in ("lh", "rh"):
        steps.append(("asegstats2table thalamic-nuclei " + hemi,
                      table + "asegstats2table --subjects {0} "
                      "--statsfile=thalamic-nuclei.{1}.v12.T1.stats "
                      "--tablefile=thalamic-nuclei.{1}.v12.T1.dat".format(case, hemi), True))
    for hemi in ("lh", "rh"):
        steps.append(("asegstats2table hipposubfields " + hemi,
                      table + "asegstats2table --subjects {0} "
                      "--statsfile=hipposubfields.{1}.T1.v21.stats "
                      "--tablefile=hipposubfields.{1}.T1.v21.dat".format(case, hemi), True))
    steps.append(("asegstats2table aseg volume",
                  table + "asegstats2table --subjects {} --meas volume "
                  "--tablefile aseg_stats.txt".format(case), True))
    steps.append(("mri_segstats brainvol",
                  "{} && mri_segstats --seg {} --sum {} --pv {} --empty --brainmask {} "
                  "--brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray "
                  "--in {} --in-intensity-name norm --in-intensity-units MR "
                  "--subject {} --surf-wm-vol --surf-ctx-vol --totalgray "
                  "--euler --ctab $FREESURFER_HOME/ASegStatsLUT.txt --etiv".format(
                      fs, q("/mri/aseg.mgz"), q("/stats/brainvol.stats"), q("/mri/norm.mgz"),
                      q("/mri/brainmask.mgz"), q("/mri/norm.mgz"), case), True))
    for hemi in ("rh", "lh"):
        steps.append(("aparcstats2table thickness " + hemi,
                      table + "aparcstats2table --subjects {0} --hemi {1} --meas thickness "
                      "--tablefile aparc_{1}_thickness_stats.txt".format(case, hemi), True))
    return steps


# =====================================================================
# 6. PROCESO POR SUJETO
# =====================================================================
def run_process(args):
    input_path, case, process_id = args
    log_row = {"batch": BATCH_LABEL, "case": case,
               "file": os.path.relpath(input_path, PATH_ORIGIN), "process_id": process_id}
    try:
        if os.path.exists(os.path.join(PATH_DESTINATION, case)):
            print("[SKIP] '{}' ya existe en {}".format(case, PATH_DESTINATION))
            now = datetime.datetime.now()
            write_step_log(log_row, "PROCESS_START", "SKIPPED",
                           "Directorio ya existente, proceso omitido", now, now)
            return

        start = datetime.datetime.now()
        write_step_log(log_row, "PROCESS_START", "STARTED", "-", start, start)

        ok = True
        for step, cmd, allow_failure in get_freesurfer_steps(case, input_path):
            if not run_step(step, cmd, log_row, allow_failure):
                write_step_log(log_row, "PIPELINE_ABORTED", "FAIL",
                               "Pipeline abortado en paso: " + step,
                               start, datetime.datetime.now())
                ok = False
                break

        write_step_log(log_row, "PROCESS_END", "COMPLETED" if ok else "FAILED", "-",
                       start, datetime.datetime.now())
    except Exception as e:
        print("[ERROR] Proceso {} ({}) falló con excepción: {}".format(process_id, case, e))


# =====================================================================
# 7. PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    os.makedirs(PATH_DESTINATION, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False)
    print("[INFO] Modo: {} | resultados en {}".format(INPUT_MODE, PATH_DESTINATION))

    if INPUT_MODE == "csv":
        file_cases = files_from_csv(VALID_CSV, PATH_ORIGIN)
    elif INPUT_MODE == "folder":
        file_cases = files_from_folder(PATH_ORIGIN)
    else:
        raise ValueError("INPUT_MODE debe ser 'csv' o 'folder'")

    if not file_cases:
        print("[INFO] No hay archivos para procesar.")
        raise SystemExit(0)

    tasks = [(path, case, i + 1) for i, (path, case) in enumerate(file_cases)]
    print("[INFO] {} sujetos, {} en paralelo".format(len(tasks), BATCH_SIZE))
    with Pool(processes=BATCH_SIZE) as pool:
        pool.map(run_process, tasks)
    print("[INFO] Todos los procesos finalizados.")
