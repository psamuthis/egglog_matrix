from setuptools import setup, find_packages

setup(
    name="egglog_matrix",
    version="0.1.0",
    package_dir={"": "src"},  # Tell setuptools packages are under src
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "egglog",  # if this is a real package
        # add other dependencies
    ],
    python_requires=">=3.8",
    author="ROUSSEAU Emile",
    description="Matrix ops rewrites based on egglog",
)