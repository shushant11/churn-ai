import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnlab import config  # noqa: E402


def _bundle_ok() -> bool:
    if not config.MODEL_PATH.exists():
        return False
    try:
        from churnlab.predict import load_bundle
        load_bundle()
        return True
    except Exception:
        return False


if not _bundle_ok():     # missing, or saved with other library versions -> train a quick model so agent/API tests have one
    config.MODEL_PATH.unlink(missing_ok=True)
    from churnlab.train import main
    main(fast=True)
