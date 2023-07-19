# Dockerfile for intro to ds python book
FROM jupyter/scipy-notebook

USER root

# install vim and a few packages for PDF builds
RUN apt-get update && apt-get install -y vim texlive-latex-extra texlive-fonts-extra texlive-xetex latexmk xindy imagemagick

USER ${NB_UID}

WORKDIR "${HOME}"

# remove the "work/" directory added in an earlier layer...
RUN rm -rf work


RUN pip install docutils==0.17.1 # Need to pin docutils to an old version for now, due to https://github.com/executablebooks/jupyter-book/issues/2022
RUN pip install referencing
RUN pip install jupyter-book
RUN pip install numpy jinja2 altair_data_server vl-convert-python click ibis-framework ghp-import jupytext nodejs

# disable warnings that pollute build logs; seems to be related to the update to python 3.11
# https://discourse.jupyter.org/t/debugger-warning-it-seems-that-frozen-modules-are-being-used-python-3-11-0/16544/12
ENV PYDEVD_DISABLE_FILE_VALIDATION=1

## Install various python packages
#RUN mamba install --quiet --yes \
#        'numpy' \
#        'jinja2' \
#        'altair_data_server' \
#        'altair_saver' \
#        'click' \
#        'ibis-framework' \
#        'ghp-import' \
#        'jupytext' \
#        'jupyter-book' \
#        'nodejs' \
#    && mamba clean --all -f -y \
#    && fix-permissions "${CONDA_DIR}" \
#    && fix-permissions "/home/${NB_USER}" 

