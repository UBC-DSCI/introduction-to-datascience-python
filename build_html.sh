chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:202212191809333bdc71 /bin/bash -c "jupyter-book build source"
