from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

ImageInput = Union[str, "Path", "np.ndarray", Any]
LabEntry = Dict[str, Any]
OCRResult = Dict[str, Any]
