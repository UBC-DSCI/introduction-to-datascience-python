chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:202308311605146eaa0f /bin/bash -c "jupyter-book build source --builder pdflatex"
chmod -R o-w source/
