"""MOFFlow-2 validity predicate."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import signal
import sys
from typing import Any
import warnings

MOFFLOW2_EXPECTED_CHECK_VALUES = {
    "has_carbon": True,
    "has_hydrogen": True,
    "has_atomic_overlaps": False,
    "has_overcoordinated_c": False,
    "has_overcoordinated_n": False,
    "has_overcoordinated_h": False,
    "has_undercoordinated_c": False,
    "has_undercoordinated_n": False,
    "has_undercoordinated_rare_earth": False,
    "has_metal": True,
    "has_lone_molecule": False,
    "has_high_charges": False,
    "is_porous": True,
    "has_suspicicious_terminal_oxo": False,
    "has_undercoordinated_alkali_alkaline": False,
    "has_geometrically_exposed_metal": False,
}

MOFFLOW2_MOFCHECKER_VERSION = "0.9.6"

def _first_failed_mofchecker_condition(descriptors: dict[str, Any]) -> str | None:
    """Apply MOFFlow-2's ordered MOFChecker predicate."""
    for key, expected in MOFFLOW2_EXPECTED_CHECK_VALUES.items():
        if descriptors[key] != expected:
            return key
    return None

def _mofchecker_descriptor_result(descriptors: dict[str, Any]) -> dict[str, Any]:
    """Retain framework connectivity separately from the benchmark predicate."""
    failed = _first_failed_mofchecker_condition(descriptors)
    connected = descriptors.get("has_3d_connected_graph")
    return {
        "mofchecker_valid": int(failed is None),
        "mofchecker_status": "valid" if failed is None else "invalid",
        "mofchecker_info": "" if failed is None else failed,
        "mofchecker_has_3d_connected_graph": (
            None if connected is None else int(bool(connected))
        ),
    }

def _mofchecker_timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("MOFChecker evaluation timed out")

def _ensure_mofchecker_pymatgen_compatibility() -> bool:
    """Restore the pymatgen property name expected by MOFChecker 0.9.6."""
    from pymatgen.core import Element

    if hasattr(Element, "is_rare_earth_metal"):
        return False
    # pymatgen renamed this property to is_rare_earth after the version pinned
    # by MOFFlow-2. The meaning is unchanged, so this preserves the 0.9.6 check
    # while allowing the repository's sfm environment to run it.
    Element.is_rare_earth_metal = property(lambda self: self.is_rare_earth)
    return True

def _find_zeopp_network() -> str | None:
    """Find Zeo++ on PATH or beside the active conda Python executable."""
    executable = shutil.which("network")
    if executable is not None:
        return executable
    conda_executable = Path(sys.prefix) / "bin" / "network"
    if conda_executable.is_file() and os.access(conda_executable, os.X_OK):
        # MOFChecker launches Zeo++ by its bare command name, so make the
        # discovered executable visible to its subprocess as well.
        os.environ["PATH"] = (
            f"{conda_executable.parent}{os.pathsep}"
            f"{os.environ.get('PATH', '')}"
        )
        return str(conda_executable)
    return None

def _evaluate_mofchecker(task: tuple[str, str, int]) -> dict[str, Any]:
    """Evaluate one sampled CIF using MOFFlow-2/evaluation/valid.py semantics."""
    sample_id, sampled_path, timeout_seconds = task
    base = {"sample_id": sample_id, "mofchecker_has_3d_connected_graph": None}
    if not sampled_path or not Path(sampled_path).is_file():
        return {
            **base,
            "mofchecker_valid": 0,
            "mofchecker_status": "missing",
            "mofchecker_info": "sampled CIF is missing",
        }

    previous_handler = None
    use_alarm = timeout_seconds > 0 and hasattr(signal, "SIGALRM")
    try:
        if _find_zeopp_network() is None:
            raise RuntimeError("Zeo++ 'network' executable is unavailable")
        _ensure_mofchecker_pymatgen_compatibility()
        from mofchecker import MOFChecker

        if use_alarm:
            previous_handler = signal.signal(
                signal.SIGALRM, _mofchecker_timeout_handler
            )
            signal.alarm(timeout_seconds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            descriptors = MOFChecker.from_cif(sampled_path).get_mof_descriptors()
        return {
            **base,
            **_mofchecker_descriptor_result(descriptors),
        }
    except TimeoutError:
        return {
            **base,
            "mofchecker_valid": 0,
            "mofchecker_status": "timeout",
            "mofchecker_info": "Timeout",
        }
    except Exception as exc:
        return {
            **base,
            "mofchecker_valid": 0,
            "mofchecker_status": "error",
            "mofchecker_info": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if use_alarm:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
