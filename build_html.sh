chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20230913180510589822 /bin/bash -c "jupyter-book build source"
