chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230831171718ddf538 /bin/bash -c "jupyter-book build source"
