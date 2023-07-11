chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:2023071118301600888b /bin/bash -c "jupyter-book build source --builder pdflatex"
chmod -R o-w source/
