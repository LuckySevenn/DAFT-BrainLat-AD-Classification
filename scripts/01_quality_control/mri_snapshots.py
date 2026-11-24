"""
capturasMRI.py
==============

Genera capturas PNG de cada imagen T1 (NIfTI) de una carpeta para revisarlas
visualmente (orientación, cobertura del cerebro, artefactos) antes de
procesarlas con FreeSurfer.

Qué hace
--------
Por cada archivo .nii / .nii.gz de la carpeta genera una figura por cada corte
definido en CORTES. Cada figura muestra las tres vistas ortogonales (sagital,
coronal y axial). Por defecto se generan 3 figuras por imagen:
  central   corte en el 50 % de cada eje
  pos_2_5   corte en el 40 % de cada eje (2/5)
  pos_4_5   corte en el 80 % de cada eje (4/5)

Salida
------
  <carpeta>/capturas_mri/<nombre>__<etiqueta>.png

Requisitos
----------
  Python 3 con numpy, nibabel y matplotlib.

Uso
---
  python capturasMRI.py /ruta/a/carpeta_nifti
  (sin argumento usa INPUT_DIR de la sección CONFIGURACIÓN)

Parámetros que se pueden cambiar (sección CONFIGURACIÓN)
--------------------------------------------------------
  INPUT_DIR        Carpeta por defecto si no se pasa como argumento.
  OUTPUT_SUBDIR    Nombre de la subcarpeta de salida.
  CORTES           Lista de (etiqueta, fracción_x, fracción_y, fracción_z);
                   0.0 = inicio del eje, 1.0 = final.
  DPI              Resolución de las imágenes.
"""
import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # sin interfaz gráfica: solo guarda archivos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
INPUT_DIR     = "ruta/a/carpeta_nifti"
OUTPUT_SUBDIR = "capturas_mri"
DPI           = 130

CORTES = [
    ("central", 0.50, 0.50, 0.50),
    ("pos_2_5", 0.40, 0.40, 0.40),
    ("pos_4_5", 0.80, 0.80, 0.80),
]


# =====================================================================
# 2. FUNCIONES
# =====================================================================
def cargar_volumen(ruta):
    """Carga un NIfTI como array 3D float32 (si es 4D, toma el primer volumen)."""
    data = np.asarray(nib.load(ruta).dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data


def slice_indices(shape, fx, fy, fz):
    """Convierte fracciones de cada eje en índices de corte válidos."""
    return tuple(int(np.clip(f * n, 0, n - 1)) for f, n in zip((fx, fy, fz), shape[:3]))


def normalizar(arr):
    """Escala a [0, 1] para visualizar (cada corte por separado)."""
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)


def generar_captura(data, etiqueta, ix, iy, iz, ruta_salida):
    """Guarda una figura con las vistas sagital (X fijo), coronal (Y fijo) y axial (Z fijo)."""
    vistas = [("Sagital (X)", data[ix, :, :]),
              ("Coronal (Y)", data[:, iy, :]),
              ("Axial (Z)",   data[:, :, iz])]

    fig = plt.figure(figsize=(13, 4.5), facecolor="black")
    fig.suptitle("{}  |  X={}  Y={}  Z={}".format(etiqueta, ix, iy, iz),
                 color="white", fontsize=12, y=0.97)
    gs = gridspec.GridSpec(1, 3, wspace=0.04)
    for col, (titulo, img) in enumerate(vistas):
        a = fig.add_subplot(gs[0, col])
        a.imshow(normalizar(img).T, cmap="gray", origin="lower", aspect="equal")
        a.set_title(titulo, color="white", fontsize=10, pad=4)
        a.axis("off")

    plt.savefig(ruta_salida, dpi=DPI, bbox_inches="tight", facecolor="black", pad_inches=0.05)
    plt.close(fig)
    print("    guardada: {}".format(os.path.basename(ruta_salida)))


def procesar_carpeta(carpeta):
    carpeta = os.path.abspath(carpeta)
    if not os.path.isdir(carpeta):
        sys.exit("ERROR: '{}' no es una carpeta válida.".format(carpeta))

    archivos = sorted(glob.glob(os.path.join(carpeta, "*.nii")) +
                      glob.glob(os.path.join(carpeta, "*.nii.gz")))
    if not archivos:
        print("No se encontraron archivos .nii / .nii.gz en '{}'.".format(carpeta))
        return

    carpeta_salida = os.path.join(carpeta, OUTPUT_SUBDIR)
    os.makedirs(carpeta_salida, exist_ok=True)
    print("Carpeta de salida: {}\n".format(carpeta_salida))

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        print("Procesando: {}".format(nombre))
        try:
            data = cargar_volumen(ruta)
        except Exception as e:
            print("    error al cargar: {}".format(e))
            continue
        print("    forma del volumen: {}".format(data.shape))

        prefijo = nombre.replace(".nii.gz", "").replace(".nii", "")
        for etiqueta, fx, fy, fz in CORTES:
            ix, iy, iz = slice_indices(data.shape, fx, fy, fz)
            salida = os.path.join(carpeta_salida, "{}__{}.png".format(prefijo, etiqueta))
            generar_captura(data, etiqueta, ix, iy, iz, salida)
        print()

    print("Listo. {} archivo(s) procesado(s). Capturas en: {}".format(len(archivos), carpeta_salida))


# =====================================================================
# 3. PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    procesar_carpeta(sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR)
