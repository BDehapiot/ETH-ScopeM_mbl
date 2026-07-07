import os
import sys

# 1. Force the target abstraction API to PySide6
os.environ["QT_API"] = "pyside6"

# 2. Force-load the older Pip PySide6 binaries directly into memory
try:
    import PySide6
    from PySide6 import QtCore
    import qtpy
    qtpy.API_NAME = "pyside6"
except ImportError as e:
    print(f"Direct PySide6 import failed: {e}")

# 3. Safe verification printout
import qtpy
print("\n" + "="*40)
print("CONFIRMED BACKEND IN USE:", qtpy.API_NAME)
print("="*40 + "\n")

import napari
# viewer = napari.Viewer()