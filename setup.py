from setuptools import setup, find_packages

setup(
    name="qsub",
    version="0.2",  # Update the version number as necessary
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "anytree",
        "graphviz",
        "matplotlib",
        "numpy",
        "plotly",
        "sympy",
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    },
)
