PROJECT := repro_test
WORKDIR := $(CURDIR)

# list below your targets and their recipies
all:

abm_semantics.pdf: semantic_foraging.qmd
	quarto render semantic_foraging.qmd --to pdf
### Wrap Commands ###
# if a command is to be send to another process e.g. a container/scheduler use:
# $(RUN1) mycommand --myflag $(RUN2)
RUN1 = $(QRUN1) $(SRUN) $(DRUN)
RUN2 = $(QRUN2)

### Rmd's ###
include .repro/Makefile_Rmds

### Docker ###
# this is a workaround for windows users
# please set WINDOWS=TRUE and adapt WINPATH if you are a windows user
# note the unusual way to specify the path
WINPATH = //c/Users/someuser/Documents/myproject/
include .repro/Makefile_Docker

