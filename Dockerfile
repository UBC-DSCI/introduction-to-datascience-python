# Dockerfile for intro to ds python book
FROM jupyter/scipy-notebook

USER root

# install vim and a few packages for PDF builds
RUN apt-get update && apt-get install -y vim texlive-latex-extra texlive-fonts-extra texlive-xetex latexmk xindy imagemagick

USER ${NB_UID}

WORKDIR "${HOME}"

# remove the "work/" directory added in an earlier layer...
RUN rm -rf work

# disable warnings that pollute build logs; seems to be related to the update to python 3.11
# https://discourse.jupyter.org/t/debugger-warning-it-seems-that-frozen-modules-are-being-used-python-3-11-0/16544/12
ENV PYDEVD_DISABLE_FILE_VALIDATION=1

## Install various python packages
RUN mamba install --quiet --yes \
        'numpy' \
        'pandas>=2.1.3' \
        'jinja2' \
        'altair>=5.1.2' \
        'vl-convert-python>=0.14' \
        'vegafusion[embed]' \
        'click' \
        'ibis-framework' \
        'ghp-import' \
        'jupytext' \
        'jupyter-book' \
        'scikit-learn>=1.3.2' \
        'nodejs' \
        'plotly' \
        'lxml' \
        'referencing' \
        'docutils==0.17.1' \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"


