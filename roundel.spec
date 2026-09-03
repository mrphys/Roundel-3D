# roundel.spec
# Build:  python -m PyInstaller roundel.spec --noconfirm
from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_submodules, copy_metadata
)
import os

datas = []
binaries = []
hiddenimports = []

# --- imageio: code + metadata ---
_d, _b, _h = collect_all("imageio")
datas += _d; binaries += _b; hiddenimports += _h
datas += copy_metadata("imageio")
datas += [("assets/three.min.js", "assets")]

# --- shipped config (forces production mode, ports, etc.) ---
datas.append((".streamlit/config.toml", ".streamlit"))

# --- metadata for packages that self-check version at import ---
for _meta in ["imageio", "streamlit", "numpy", "scipy", "scikit-image",
              "scikit-learn", "networkx", "plotly", "nibabel", "pydicom",
              "monai", "pandas", "matplotlib", "Pillow", "torch",
              "streamlit-drawable-canvas", "tqdm", "stqdm",
              "pylibjpeg", "opencv-python-headless"]:
    try:
        datas += copy_metadata(_meta)
    except Exception:
        pass

# --- app source files (bundled flat so launcher finds them) ---
for f in ["roundel_app_clinical.py", "roundel_utils.py", "utils.py",
          "dicom_utils.py", "model_utils.py"]:
    datas.append((f, "."))

# --- model weights ---
datas.append(("models", "models"))          # SEG-25.pth, SEG_COR-31.pth, etc.

# --- Streamlit: static frontend + metadata + submodules (CRITICAL) ---
datas += collect_data_files("streamlit", include_py_files=True)
datas += copy_metadata("streamlit")
hiddenimports += collect_submodules("streamlit")
hiddenimports += ["streamlit.runtime.scriptrunner.magic_funcs", "streamlit.web.cli"]

# --- streamlit-drawable-canvas ---
_d, _b, _h = collect_all("streamlit_drawable_canvas")
datas += _d; binaries += _b; hiddenimports += _h

# --- stqdm ---
_d, _b, _h = collect_all("stqdm")
datas += _d; binaries += _b; hiddenimports += _h

# --- torch (huge; collect everything incl. DLLs) ---
_d, _b, _h = collect_all("torch")
datas += _d; binaries += _b; hiddenimports += _h

# --- torch ecosystem / PyG ---
for pkg in ["torch_geometric", "torch_cluster", "torch_scatter",
            "torch_sparse", "torch_spline_conv", "torchvision"]:
    try:
        _d, _b, _h = collect_all(pkg)
        datas += _d; binaries += _b; hiddenimports += _h
    except Exception:
        pass

# --- im2sim (your model definitions) ---
try:
    _d, _b, _h = collect_all("im2sim")
    datas += _d; binaries += _b; hiddenimports += _h
except Exception:
    pass

# --- scientific stack with dynamic imports / data files ---
for pkg in ["skimage", "scipy", "sklearn", "monai", "nibabel", "pydicom",
            "plotly", "networkx", "matplotlib", "PIL", "pylibjpeg",
            "pylibjpeg_libjpeg", "pylibjpeg_openjpeg", "openjpeg", "libjpeg"]:
    try:
        _d, _b, _h = collect_all(pkg)
        datas += _d; binaries += _b; hiddenimports += _h
    except Exception:
        pass

# --- pydicom pixel handlers + assorted hidden imports ---
hiddenimports += collect_submodules("pydicom")
hiddenimports += ["pylibjpeg", "pylibjpeg_libjpeg", "pylibjpeg_openjpeg",
                  "cv2", "imageio", "imageio.v2", "stqdm"]

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "tensorflow", "pyvista", "vtk", "trame"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Roundel_3D",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Roundel_3D",
)