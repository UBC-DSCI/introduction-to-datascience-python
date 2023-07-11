chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230711035452135533 /bin/bash -c "jupyter-book build source --builder pdflatex"
chmod -R o-w source/
