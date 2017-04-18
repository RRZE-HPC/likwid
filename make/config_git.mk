GITCOMMITRAW=$Id$

GITCOMMIT := $(shell echo $(GITCOMMITRAW) | cut -d ' ' -f 2)
