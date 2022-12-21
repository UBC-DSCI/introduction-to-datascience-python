# building the book

## fork or clone the repository

* git clone https://github.com/UBC-DSCI/introduction-to-datascience-python.git
    cd introduction-to-datascience-python
    git checkout -b myst

## installing the build environment

1) Download miniforge from https://github.com/conda-forge/miniforge/releases

2) install and activate the base environment in a shell

3) do:

     cd install
     mamba env create --name dsci --file environment.yml
     conda activate dsci
     pip install -r requirements.txt
     npm install -g live-server
     
This should give you an environment with jupyter-book, scikit-learn, and a live reload server which can automatically build the book following https://github.com/eoas-ubc/jb_tools/blob/master/tools_demo/Readme_conda.md

4) If you'd like to produce and use a lock file, do:

     conda-lock --kind explicit --file environment.yml -p ostype
     
   where ostype is one of `linux-64`, `win-64` or `osx-64`.   Then use the lockfile with:
   
     mamba activate base
     mamba install --name dsci --file conda-lock-ostype
     
## building the book

5) do:

     cd source
     jb build .
     
6) view the html in `./_build/html`

## publishing the book

7) do:

     cd introduction-to-datascience-python
     ./push_html.sh
     
   this will update the github.io version at:  https://phaustin.github.io/introduction-to-datascience-python/intro.html
     
     
## working with livereload

To have the book built locally when you change a file:

8) in one terminal, start a source file watcher:

    cd introduction-to-datascience-python 
    ebp-watch jb source
    
9) in another terminal, use live-server to open a local browser tab with yoru book

    live-server source/_build/html/
     
After you do this, changing any file in the source folder should trigger a jb build and browser refresh.





     
