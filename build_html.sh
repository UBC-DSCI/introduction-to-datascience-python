chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231112004031dd2207 /bin/bash -c "jupyter-book build source"
