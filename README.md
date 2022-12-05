# Gomoku
Gomoku AI &amp; GUI written in Python

## Installation

Make sure you have python 3.10 or newer

(Optional) Create a virtual environment to run the program
```bash
conda create -n gomoku_env python=3.10
conda activate gomoku_env
```

To exit the virtual environment, simply run in the terminal
```bash
conda deactivate
```

To build the Gomoku program, run
```bash
make install
```

Alternatively, to build the Gomoku with dev dependencies, run
```bash
make install_dev
```

You can also clean the package using
```bash
make clean
```

## Usage

To run the program and see the full list of flags/options, run
```bash
python3 -m gomoku --help
```