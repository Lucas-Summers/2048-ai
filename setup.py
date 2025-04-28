from setuptools import setup, find_packages

setup(
    name="2048-ai",
    version="0.1",
    description="An AI that plays the 2048 game using various algorithms",
    author="The Dropouts",
    author_email="thedropouts@calpoly.edu",
    packages=find_packages(),
    install_requires=[
        "flask",
        "numpy",
    ],
)
