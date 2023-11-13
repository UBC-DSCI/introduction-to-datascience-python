chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231113214302c3cc10 /bin/bash -c "jupyter-book build source"
