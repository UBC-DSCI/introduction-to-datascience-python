SED PROGRAM:
  s/ubcdsci\/py-intro-to-ds:[[:alnum:]]\\+/ubcdsci/py-intro-to-ds:/g
INPUT:   'build_html.sh' line 1
PATTERN: # Script to generate HTML book
COMMAND: s/ubcdsci\/py-intro-to-ds:[[:alnum:]]\\+/ubcdsci/py-intro-to-ds:/g
PATTERN: # Script to generate HTML book
END-OF-CYCLE:
# Script to generate HTML book
INPUT:   'build_html.sh' line 2
PATTERN: docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:0a9efha0we9fhawe /bin/bash -c "jupyter-book build source"
COMMAND: s/ubcdsci\/py-intro-to-ds:[[:alnum:]]\\+/ubcdsci/py-intro-to-ds:/g
MATCHED REGEX REGISTERS
  regex[0] = 39-78 'ubcdsci/py-intro-to-ds:0a9efha0we9fhawe'
PATTERN: docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds: /bin/bash -c "jupyter-book build source"
END-OF-CYCLE:
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds: /bin/bash -c "jupyter-book build source"
