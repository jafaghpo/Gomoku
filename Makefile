NAME := gomoku
WORKDIR ?= .
VENVDIR ?= $(WORKDIR)/venv
VENV = $(VENVDIR)/bin

.PHONY: install clean

install:
	@python3 -m pip install -e .

clean:
	@python3 setup.py clean
	@rm -rf src/$(NAME)/__pycache__/	2> /dev/null || true
	@rm -rf tests/__pycache__/			2> /dev/null || true
	@rm -rf src/$(NAME).egg-info/ 		2> /dev/null || true
	@find . -iname "*.pyc" -delete		2> /dev/null || true

re: clean install