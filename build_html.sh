chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231108190825ac6d61 /bin/bash -c "jupyter-book build source"
