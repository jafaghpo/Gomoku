from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="gomoku",
    version="0.0.1",
    author="John Afaghpour",
    author_email="johnafaghpour@gmail.com",
    description="A simple implementation of the Gomoku game.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jafaghpo/Gomoku",
    package_dir={"gomoku": "gomoku"},
    license=license,
    packages=find_packages("gomoku", exclude=("tests", "docs")),
    python_requires=">=3.10.1",
    install_requires=[
        "pygame>=2.1.2",
        "numpy>=1.22.2",
        "pandas>=1.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.1",
            "black>=22.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
