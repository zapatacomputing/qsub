from setuptools import setup, find_packages

# with open("requirements.txt") as f:
#     install_requires = f.read().splitlines()

# print(install_requires)

setup(
    name="qsub",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "anytree==2.12.1",
        "contourpy==1.2.0",
        "cycler==0.12.1",
        "exceptiongroup==1.2.0",
        "fonttools==4.45.1",
        "graphviz==0.20.1",
        "iniconfig==2.0.0",
        "kiwisolver==1.4.5",
        "matplotlib==3.8.2",
        "numpy==1.25.2",
        "packaging==23.2",
        "Pillow==10.1.0",
        "plotly==5.18.0",
        "pluggy==1.3.0",
        "pyparsing==3.1.1",
        "pytest==7.4.3",
        "python-dateutil==2.8.2",
        "six==1.16.0",
        "tenacity==8.2.3",
        "tomli==2.0.1,",
    ],
    entry_points={
        "console_scripts": [
            "qsub_cli=qsub.main:hello",
        ],
    },
)
