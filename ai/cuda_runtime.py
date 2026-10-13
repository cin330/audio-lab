# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""Locate CUDA runtime libraries installed by pip without changing shell settings."""

import ctypes
import sys
from importlib.util import find_spec
from pathlib import Path


_LIBRARIES = []


def prepare_cuda_libraries():
    if sys.platform != 'linux' or _LIBRARIES:
        return
    loaded = []
    for package, names in [
        ('nvidia.cublas', ['libcublasLt.so.12', 'libcublas.so.12']),
        ('nvidia.cudnn', ['libcudnn.so.9', 'libcudnn_ops.so.9', 'libcudnn_cnn.so.9']),
    ]:
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            spec = None
        roots = list(spec.submodule_search_locations or []) if spec else []
        for name in names:
            path = next((Path(root) / 'lib' / name for root in roots
                         if (Path(root) / 'lib' / name).is_file()), None)
            if path is not None:
                loaded.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    _LIBRARIES.extend(loaded)
