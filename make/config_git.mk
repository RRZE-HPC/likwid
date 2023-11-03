GITCMD := $(shell which git)
GITCOMMIT := $(shell if [ -e .git -a ! -z "$(GITCMD)" ]; then $(GITCMD) rev-parse --verify HEAD; else echo "0123456789"; fi)

