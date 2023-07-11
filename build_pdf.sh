chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230711033651d28467 /bin/bash -c "jupyter-book build source --builder pdflatex"
chmod -R o-w source/
