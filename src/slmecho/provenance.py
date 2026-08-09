"""What machine and what library versions produced a result file.

This lives in the package rather than in one script because more than one
entry point writes result metadata, and a run whose environment is recorded in
only some of those files leaves the manuscript with nothing to cite. The
version numbers in the paper are read back out of these records; there is no
fallback literal anywhere.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Dict


def environment_record() -> Dict[str, object]:
    """Versions, platform and CPU model, captured at the start of a run."""
    import torch
    import transformers

    try:
        cpu = subprocess.run(
            ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20
        ).stdout.strip().splitlines()
        cpu = [c.strip() for c in cpu if c.strip() and "Name" not in c]
        cpu_name = cpu[0] if cpu else platform.processor()
    except Exception:  # noqa: BLE001
        cpu_name = platform.processor()
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "cpu": cpu_name,
        "threads": torch.get_num_threads(),
    }
