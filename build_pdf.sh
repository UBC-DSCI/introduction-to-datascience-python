chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230718202836018157 /bin/bash -c "jupyter-book build source --builder pdflatex"
chmod -R o-w source/
