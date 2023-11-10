chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231108192908c9b484 /bin/bash -c "jupyter-book build source"
