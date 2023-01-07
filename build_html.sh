chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230104230634037f38 /bin/bash -c "jupyter-book build source"
