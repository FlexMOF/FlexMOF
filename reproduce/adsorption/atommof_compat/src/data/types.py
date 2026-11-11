import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any


BlockType = Literal['NODE', 'LINKER', 'SOLVENT', 'ADSORBATE']

####################################################################################################
# DECOMPOSE INTO FRAGMENTS
####################################################################################################
@dataclass
class Fragment:
	"""
	Represents a contiguous fragment of a MOF structure.
	"""
	# Identification
	block_type: BlockType
	canonical_smiles: Optional[str]

	# Geometry
	cart_coords_contiguous: np.ndarray # [N, 3] (BFS unwrapped)
	cart_coords_original: np.ndarray   # [N, 3] (directly from CIF)

	# Composition
	atomic_numbers: np.ndarray   # [num_atoms, ]
	original_indices: np.ndarray # [num_atoms, ] # Mapping relative to the source CIF/subset (TODO: Mapping to global indices in MOF)

@dataclass
class DecomposedMOF:
	"""
	Group ExtractBlock by MOF.
	"""
	fragments: List[Fragment]

	cell: np.ndarray
	lattice: np.ndarray

	info: Optional[Any] = None

####################################################################################################
# METAL NODES
####################################################################################################
@dataclass
class MetalNodeVariant:
	atomic_numbers: np.ndarray
	cart_coords: np.ndarray # Zero-centered coordinates

	@classmethod
	def from_fragment(cls, frag: Fragment):
		center = frag.cart_coords_contiguous.mean(axis=0)
		centered_coords = frag.cart_coords_contiguous - center

		return cls(
			atomic_numbers=frag.atomic_numbers,
			cart_coords=centered_coords,
		)

MetalLibrary = Dict[str, List[MetalNodeVariant]]

####################################################################################################
# FINAL STRUCTURE
####################################################################################################
@dataclass
class Block:
	# Identity
	block_type: BlockType

	cart_coords: np.ndarray          # [N, 3] # TODO: rename as local_coords
	atomic_numbers: np.ndarray       # [N,]

	# Chemical features
	bond_indices: np.ndarray         # [2, E]
	bond_types: np.ndarray           # [E,]
	formal_charges: np.ndarray       # [N,]
	chirality_tags: np.ndarray       # [N,]

@dataclass
class MOFStructure:
	# Model input
	blocks: List[Block]

	atomic_numbers: np.ndarray   		# [num_total_atoms, ]

	# Ground-truth (target)
	# Initialized to zero during inference
	cart_coords: np.ndarray         	# [num_total_atoms, 3]
	frac_coords: np.ndarray          	# [num_total_atoms, 3]
	original_cart_coords: np.ndarray  	# [num_total_atoms, 3] # non-contiguous coordinates from CIF

	lattice: np.ndarray             	# [6,]
	cell: np.ndarray                	# [3, 3]

	info: Optional[Any] = None

# TODO: Remove this
@dataclass
class Structure:
	# Model input
	blocks: List[Block]

	# atomic_numbers: np.ndarray   		# [num_total_atoms, ]

	# Ground-truth (target)
	# Initialized to zero during inference
	cart_coords: np.ndarray         	# [num_total_atoms, 3]
	frac_coords: np.ndarray          	# [num_total_atoms, 3]
	# original_cart_coords: np.ndarray  	# [num_total_atoms, 3] # non-contiguous coordinates from CIF

	lattice: np.ndarray             	# [6,]
	cell: np.ndarray                	# [3, 3]

	info: Optional[Any] = None