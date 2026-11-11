"""MOFChecker 2.0 validity rules."""
import hashlib
from pathlib import Path
import warnings

CHECKS = {
    'has_carbon': True, 'has_hydrogen': True, 'has_metal': True,
    'has_atomic_overlaps': False, 'has_overcoordinated_c': False,
    'has_overcoordinated_n': False, 'has_overcoordinated_h': False,
    'has_undercoordinated_c': False, 'has_undercoordinated_n': False,
    'has_undercoordinated_rare_earth': False, 'has_lone_molecule': False,
    'has_high_charges': False, 'has_suspicious_terminal_oxo': False,
    'has_undercoordinated_alkali_alkaline': False,
    'has_geometrically_exposed_metal': False,
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_rules():
    assert len(CHECKS) == 15

def init():
    warnings.filterwarnings('ignore')
    import mofchecker
    if mofchecker.__version__ != '2.0':
        raise RuntimeError(f'MOFChecker 2.0 required, found {mofchecker.__version__}')
