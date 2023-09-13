chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230913180510589822 /bin/bash -c "export BOOK_BUILD_TYPE='PDF'; jupyter-book build source --builder pdflatex"
