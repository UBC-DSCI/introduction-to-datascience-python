# Script to generate HTML book
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:0a9efha0we9fhawe /bin/bash -c "jupyter-book build source"
