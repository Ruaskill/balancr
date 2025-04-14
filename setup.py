from setuptools import setup, find_packages

setup(
    name="balancr",
    version="0.1.0",
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
            "balancr=imbalance_framework.cli.main:main",
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
    description="A framework for analysing and comparing balancing data techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
