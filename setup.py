from setuptools import setup, find_packages

setup(
    name="qsub",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={
        "console_scripts": [
            "qsub_cli=qsub.main:hello",
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="qsub",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # other configurations...
)
