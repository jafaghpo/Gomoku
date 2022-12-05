NAME := gomoku
WORKDIR ?= .

.PHONY: install all clean fclean re

$(NAME): install

install:
	@python3 -m pip install -e $(WORKDIR)

all:
	@make $(NAME)

clean:
	@python3 setup.py clean

fclean: clean
	@rm -rf $(NAME)/__pycache__/	2> /dev/null || true
	@rm -rf tests/__pycache__/			2> /dev/null || true
	@rm -rf $(NAME).egg-info/ 		2> /dev/null || true
	@find $(WORKDIR) -iname "*.pyc" -delete		2> /dev/null || true

re: fclean
	@make all
