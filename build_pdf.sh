chmod -R o+w source/
# To build the book, altair charts need to be inlined as images,
# so we substitute in that line just for the PDF build and then remove it again at the end
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:202308182146260106a8 /bin/bash -c "\
    sed 's/; alt.data_transformers.enable(\"vegafusion\")/; alt.data_transformers.enable(\"vegafusion\"); alt.renderers.enable(\"png\", scale_factor=0\.7, ppi=300)/' source/*.md -i \
    ;jupyter-book build source --builder pdflatex \
    ;sed 's/; alt.data_transformers.enable(\"vegafusion\"); alt.renderers.enable(\"png\", scale_factor=0\.7, ppi=300)/; alt.data_transformers.enable(\"vegafusion\")/' source/*.md -i \
"

chmod -R o-w source/
