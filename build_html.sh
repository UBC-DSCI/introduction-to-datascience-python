chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231110054348fd23c8 /bin/bash -c "jupyter-book build source"

