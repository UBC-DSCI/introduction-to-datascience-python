# Dockerfile for intro to ds python book
FROM jupyter/scipy-notebook

USER root

# install vim and a few tex packages for PDF builds
RUN apt-get update && apt-get install -y vim texlive-latex-extra texlive-fonts-extra texlive-xetex latexmk

USER ${NB_UID}

WORKDIR "${HOME}"

# remove the "work/" directory added in an earlier layer...
RUN rm -rf work

# Install various python packages
RUN mamba install --quiet --yes \
        'numpy' \
        'jinja2' \
        'altair_data_server' \
        'altair_saver' \
        'click' \
        'ibis-framework' \
        'ghp-import' \
        'jupytext' \
        'jupyter-book' \
        'nodejs' \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}" 

