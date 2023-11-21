# Script to generate PDF book

# backup original index.Rmd
cp source/index.md index_backup.md

# PDF book doesn't need the welcome page. I couldn't find a way to stop jupyterbook from including it.
# so this script manually removes the welcome page before building the PDF. This is a bit painful, but it works...
sed -n -i "/graphic/q;p" source/index.md
echo "# Data Science: A First Introduction" >> source/index.md

chmod -R o+w source/
docker run --rm -v $(pwd):/home/jovyan ubcdsci/py-intro-to-ds:20231112004031dd2207 /bin/bash -c "export BOOK_BUILD_TYPE='PDF'; jupyter-book build source --builder pdflatex"

# restore the backed up full index.Rmd
mv index_backup.md source/index.md
