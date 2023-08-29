chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:202308182146260106a8 /bin/bash -c "export BOOK_BUILD_TYPE='PDF'; jupyter-book build source --builder pdflatex"
chmod -R o-w source/
