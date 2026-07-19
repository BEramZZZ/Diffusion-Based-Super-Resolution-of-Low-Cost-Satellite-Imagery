"""
compat.py
=========

Environment compatibility fixes needed before importing basicsr/realesrgan.

Why this exists
----------------
`basicsr` (a Real-ESRGAN dependency) imports `rgb_to_grayscale` from
`torchvision.transforms.functional_tensor`. Recent torchvision releases
removed that module and moved the function to
`torchvision.transforms.functional` instead. `basicsr` has not been updated
to match, so importing it directly raises:

    ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'

`apply_basicsr_compat_shim()` works around this by pre-registering a fake
module under the old name in `sys.modules`, containing just the one function
`basicsr` actually needs. When `basicsr`'s broken import line runs, Python
finds this fake module already "imported" and uses it -- it never attempts
to load a real file from the broken path.

This must be called before any `import basicsr` or `from realesrgan import ...`
statement, in every fresh Python process (it is not persistent across
sessions/restarts).
"""

import sys
import types


def apply_basicsr_compat_shim() -> None:
    """Register a compatibility shim for basicsr's outdated torchvision import.

    Safe to call multiple times; only does work if the shim isn't already
    registered.
    """
    shim_name = "torchvision.transforms.functional_tensor"

    if shim_name in sys.modules:
        return  # already applied in this process

    import torchvision.transforms.functional as F

    shim = types.ModuleType(shim_name)
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules[shim_name] = shim
