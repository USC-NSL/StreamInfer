# StreamInfer

Barrier-free distributed Mixture-of-Experts serving system.

## Dependencies

### System Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| CUDA Toolkit | 12.x | |
| GCC | 13+ | C++17 required |
| UCX | 1.15+ | For high-performance transport (`ucp`, `ucs`, `uct`), optional |
| GDRCopy | 2.x | GPUDirect RDMA (`libgdrapi`) |
| MLNX_OFED + nvidia_peermem | 24.10+ | GPU-Direct RDMA stack (e.g. `nvidia_peermem` loaded with `PeerMappingOverride=1`) for enabling GPU-Direct RDMA |
| libzmq | 4.x | ZeroMQ C library; also needs cppzmq C++ headers (`zmq.hpp`) |

### Python (installed via pip)

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.6.0 |
| vllm | 0.8.2 |
| pybind11 | latest |
| flask | latest |
| simpy | latest |
| pyarrow | latest |
| pandas | latest |
| openpyxl | latest |
| matplotlib | latest |

> **ABI Note**: Use the standard PyPI `torch` wheel (bundles CUDA 12.4, old C++ ABI).
> Do **not** install from `--index-url https://download.pytorch.org/whl/cu126` — that
> build uses CXX11 ABI, which causes `undefined symbol` errors when importing vLLM.

### NCCL

| Dependency | Version | Notes |
|---|---|---|
| NCCL | 2.29.x (apt) | Install the pre-built package via apt (`libnccl2` + `libnccl-dev`). Building 2.29 from source if apt not available / version issue. |

### Git Submodules (fetched automatically by `--recursive`)

- **CUTLASS** — grouped GEMM kernels, compiled into `disagmoe_c`
- **cereal** — C++ serialization (header-only)
- **NVTX** — NVIDIA profiling (header-only)
- **pybind11** — Python/C++ binding (header-only)

## Install

### 1. Python environment with conda

```bash
conda create -n streaminfer python=3.12.8 -y
conda activate streaminfer
pip install torch==2.6.0 torchvision torchaudio
pip install vllm==0.8.2
```

### 2. NCCL

Install the pre-built NCCL from NVIDIA's CUDA apt repo (recommended):

```bash
sudo apt-get install libnccl2 libnccl-dev
```

This provides `nccl.h` (`/usr/include`) and `libnccl.so` (`/usr/lib/x86_64-linux-gnu`),
used by the build in step 6.

**Fallback — build NCCL 2.29 from source** (only if the apt NCCL causes problems).
Then point `NCCL_HOME` at this build in step 6:

```bash
# CUDA_HOME must be set before building NCCL.
export CUDA_HOME=/usr/local/cuda            # example — set to your actual CUDA install

git clone --depth 1 --branch v2.29.7-1 https://github.com/NVIDIA/nccl.git nccl-2.29
cd nccl-2.29
# Omit NVCC_GENCODE: NCCL builds its default arch set for this CUDA version
# (includes sm_80/sm_90 cubins + PTX, which cover sm_89/L40S and newer GPUs).
make -j$(nproc) src.build CUDA_HOME=$CUDA_HOME
cd ..
export NCCL_HOME=$(pwd)/nccl-2.29/build
```

### 3. ZeroMQ (C library + C++ headers)

With `apt`:

```bash
sudo apt-get install libzmq3-dev cppzmq-dev
```

Without root (e.g., HPC):

```bash
conda install -c conda-forge zeromq -y
wget -q -O $CONDA_PREFIX/include/zmq.hpp \
    https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp
wget -q -O $CONDA_PREFIX/include/zmq_addon.hpp \
    https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq_addon.hpp
```

### 4. Initialize submodules and install Python deps

```bash
cd StreamInfer
git submodule update --init --recursive   # fetches CUTLASS, cereal, NVTX, pybind11
pip install -r requirements.txt
```

### 5. Patch vLLM

DisagMoE requires a small patch to the **installed** vLLM 0.8.2 package
(`vllm/attention/layer.py`, `vllm/envs.py`, `vllm/platforms/cuda.py`).
Run from the StreamInfer repo root (where you are after step 4):

```bash
PATCH="$(pwd)/patches/vllm_0.8.2.patch"
# Locate the site-packages dir that contains vllm (without importing vllm,
# whose import prints a log line that would corrupt the path):
SITE_PACKAGES=$(python -c "import os, site; print(next(p for p in site.getsitepackages() if os.path.isdir(os.path.join(p,'vllm'))))")
git -C "$SITE_PACKAGES" apply "$PATCH"
```

### 6. Build DisagMoE

> **Note:** The paths below are examples only — set each variable to match your
> actual environment (CUDA, NCCL, ZMQ, GDRCopy, UCX), not verbatim.

```bash
export CUDA_HOME=/path/to/cuda
# NCCL from apt (libnccl2/libnccl-dev):
export NCCL_INCLUDE_DIR=/usr/include
export NCCL_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
# (from-source NCCL instead? set NCCL_HOME=/path/to/nccl-2.27/build and
#  NCCL_INCLUDE_DIR=$NCCL_HOME/include, NCCL_LIBRARY_DIR=$NCCL_HOME/lib)
export ZMQ_HOME=/usr              # or wherever zmq.h and libzmq live
export GDRCOPY_HOME=/usr/local/gdrcopy                   # or wherever gdrapi.h lives
export C_INCLUDE_PATH=/path/to/ucx/include
export CPP_INCLUDE_PATH=/path/to/ucx/include
export LIBRARY_PATH=$ZMQ_HOME/lib:$NCCL_LIBRARY_DIR:/path/to/ucx/lib:$LIBRARY_PATH

make pip
```

### 7. Verify

```bash
python -c "import disagmoe_c; print('OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```
