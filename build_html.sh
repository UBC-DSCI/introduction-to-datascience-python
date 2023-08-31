chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230831155446f98b29 /bin/bash -c "jupyter-book build source"
chmod -R o-w source/
