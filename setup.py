import os
from setuptools import setup, find_packages


# Read the README file for the long description
def read_file(filename):
    with open(
        os.path.join(os.path.dirname(__file__), filename), encoding="utf-8"
    ) as file:
        return file.read()


setup(
    name="balancr",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "imbalanced-learn>=0.8.0",
        "openpyxl>=3.0.0",
        "colorama>=0.4.4",
        "plotly>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "balancr=balancr.cli.main:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    python_requires=">=3.8",
    author="Conor Doherty",
    author_email="ruaskillz1@gmail.com",
    description="A unified framework for analysing and comparing techniques for handling imbalanced datasets",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Ruaskill/balancr",
    project_urls={
        "Documentation": "https://github.com/Ruaskill/balancr/blob/main/README.md",
        "Source": "https://github.com/Ruaskill/balancr",
        "Issues": "https://github.com/Ruaskill/balancr/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, imbalanced data, data balancing, classification, resampling, oversampling,"
    "undersampling, SMOTE, ADASYN, imbalanced learning",
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
