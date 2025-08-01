"""
Setup script for ParamΔ package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="param-delta",
    version="0.1.0",
    author="ParamDelta Implementation",
    description="Zero-cost post-training for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/param-delta",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "azure": ["azure-storage-blob>=12.14.0", "azure-identity>=1.12.0"],
        "transformers": ["transformers>=4.30.0", "safetensors>=0.3.0"],
    },
    entry_points={
        "console_scripts": [
            "param-delta=param_delta.cli:main",
        ],
    },
)