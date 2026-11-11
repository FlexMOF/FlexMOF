"""Named StructureMatcher parameter sets, recorded alongside every result.

``ltol`` alone does not identify a matching protocol. Entry points record the
complete parameter set and pymatgen version in their results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MatchProtocol:
    name: str
    ltol: float
    stol: float
    angle_tol: float
    primitive_cell: bool
    scale: bool
    attempt_supercell: bool
    allow_subset: bool

    def matcher_kwargs(self) -> dict:
        kwargs = asdict(self)
        kwargs.pop("name")
        return kwargs

    def record(self) -> dict:
        """The parameter set as it should appear in a result JSON."""
        from pymatgen.core import __version__ as pymatgen_version

        return {**asdict(self), "pymatgen": pymatgen_version}


# This project's single reporting protocol. Every structure-matching entry
# point uses it unless a caller explicitly asks for a reference protocol.
FLEXMOF_V2 = MatchProtocol(
    name="flexmof_v2",
    ltol=0.3,
    stol=0.5,
    angle_tol=10,
    primitive_cell=True,
    scale=False,
    attempt_supercell=True,
    allow_subset=False,
)

# A reference protocol for MOFFlow-2 comparisons. It shares
# this project's lattice tolerance but scales lattices and does not attempt
# supercells, so numbers produced under FLEXMOF_V2 are not comparable to it.
MOFFLOW2_REFERENCE = MatchProtocol(
    name="mofflow2_reference",
    ltol=0.3,
    stol=0.5,
    angle_tol=10,
    primitive_cell=True,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
)

PROTOCOLS = {protocol.name: protocol for protocol in (FLEXMOF_V2, MOFFLOW2_REFERENCE)}

DEFAULT_PROTOCOL = FLEXMOF_V2.name


def resolve(name: str | MatchProtocol | None = None) -> MatchProtocol:
    if isinstance(name, MatchProtocol):
        return name
    if name is None:
        return PROTOCOLS[DEFAULT_PROTOCOL]
    try:
        return PROTOCOLS[name]
    except KeyError:
        known = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"unknown match protocol {name!r}; known: {known}") from None


def build_matcher(name: str | MatchProtocol | None = None):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(**resolve(name).matcher_kwargs())
