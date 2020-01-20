from setuptools import setup, find_packages

setup(
    name="ethicml-models",
    version="0.1.0",
    author="T. Kehrenberg",
    packages=find_packages(),
    description="Models for EthicML",
    python_requires=">=3.6",
    package_data={"models": ["py.typed"]},
    install_requires=[
        "numpy >= 1.15",
        "EthicML == 0.1.0a5",
    ],
    extras_require={
        "ci": ["pytest >= 3.3.2"],
    },
)
