GITCMD := $(shell which git 2>/dev/null)
GITCOMMIT := $(shell if [ -e .git -a ! -z "$(GITCMD)" ]; then $(GITCMD) rev-parse --verify HEAD; else echo "0123456789"; fi)

