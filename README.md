# mlops-zoomcamp

assert '_distutils' in core.__file__, assert 'setuptools._distutils.log' not in sys.modules
AssertionError: /opt/conda/python3.10/distutils/core.py

usually comes from a conflict between Python’s native distutils and setuptools’ internal _distutils in newer Python environments. It often surfaces when PyTorch (especially with CUDA) and other C/C++ extension packages are installed in an inconsistent Python environment (e.g., mixing conda and pip or messing with distutils manually).

# Solution: Docker-based PyTorch 2.6.0 Setup Cleanly
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Optional: upgrade pip & install system packages
RUN apt-get update && apt-get install -y git curl && \
    pip install --upgrade pip setuptools
