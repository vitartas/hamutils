from setuptools import setup

setup(
    name="hamutils",
    version="0.1.0",
    description="Machine Learning Hamiltonians Utility Methods",
    author="Valdas Vitartas",
    author_email="vitartc@gmail.com",
    url="https://github.com/vitartc/hamutils",
    packages=["hamutils"],
    install_requires=[
        "h5py",
        "numpy",
        "scipy",
        "numba",
        "matplotlib",
        "scikit-learn",
        "ase==3.22.1",
    ],
    license="MIT"
)