from setuptools import setup, find_packages
import os

# Leer el contenido del archivo README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Leer las dependencias del archivo requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="Orbix",
    version="1.0.0",
    author="Orbix Team",
    author_email="info@orbix.space",
    description="Sistema avanzado de navegación orbital para predicción de trayectorias y prevención de colisiones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orbix-team/orbix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "orbix=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)