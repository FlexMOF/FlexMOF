"""Build the CUDA Floyd-Warshall extension used by MOF graph preprocessing."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
SOURCE = ROOT / "src" / "flexmof" / "cuda"

setup(
    name="flexmof-graph-preprocess",
    ext_modules=[
        CUDAExtension(
            "flexmof_preprocess_cuda",
            [
                str(SOURCE / "graphormer_preprocess_cuda.cpp"),
                str(SOURCE / "graphormer_preprocess_cuda_kernel.cu"),
            ],
            include_dirs=[str(SOURCE)],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
