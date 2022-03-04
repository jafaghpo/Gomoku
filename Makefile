NAME := gomoku
WORKDIR ?= .
VENVDIR ?= $(WORKDIR)/venv
VENV = $(VENVDIR)/bin

.PHONY: install clean

install_prod:
	@python3 -m pip install -e .

install_dev:
	@python3 -m pip install -e .[dev]

clean:
	@python3 setup.py clean
	@rm -rf $(NAME)/__pycache__/	2> /dev/null || true
	@rm -rf tests/__pycache__/			2> /dev/null || true
	@rm -rf $(NAME).egg-info/ 		2> /dev/null || true
	@find . -iname "*.pyc" -delete		2> /dev/null || true

re_prod: clean install_prod

re_dev: clean install_dev