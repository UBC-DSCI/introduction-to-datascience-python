## Data Science: A First Introduction (Python Edition)
This is the source for the python edition of the *Data Science: A First Introduction* textbook.

The book is available online at: https://python.datasciencebook.ca/

For the R version of the textbook, please visit https://datasciencebook.ca or the github repository
at https://github.com/ubc-dsci/introduction-to-datascience.

## License Information

This textbook is offered under 
the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [the license file](LICENSE.md) for more information. 

## Development

### Setup

Building the book requires Docker (instructions here: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)).

### Build locally

You can build the HTML version of the book on your own machine by running
```
./build_html.sh
```
in the root directory of this repository. The book can be viewed in your browser by opening the `source/_build/html/index.html` file.

You can build the PDF version of the book on your own machine by running
```
./build_pdf.sh
```
in the root directory of this repository. The book can be viewed in a PDF reader by opening `source/_build/latex/python.pdf`.

### Contributing 
Primary development in this repository happens on the `main` branch. If you want to contribute to the book,
please branch off of `main` and make a pull request into `main`. You cannot commit directly to `main`.

The `production` branch contains the source material corresponding to the current publicly-viewable version of the book website.

The `gh-pages` branch serves the current book website at https://python.datasciencebook.ca.

### Workflows

#### Book deployment

You can update the live, publicly viewable HTML book by making changes to the `source/` folder in the `production` branch (e.g. by merging `main` into `production`).
GitHub will trigger a rebuild of the public HTML site, and store the built book in the root folder of the `gh-pages` branch.

#### `main` deploy previews

Any commit to `source/**` on the `main` branch (from a merged PR) will trigger a rebuild of the development preview site served at `https://python.datasciencebook.ca/dev`.
The built preview book will be stored in the `dev/` folder on the `gh-pages` branch.

#### PR deploy previews

Any PR to `source/` will trigger a build of a PR preview site at `https://python.datasciencebook.ca/pull###`, where `###` is the number of the pull request.
The built preview book will be stored in the `pull###/` folder on the `gh-pages` branch.

#### Build environment updates

Any PR to `Dockerfile` will trigger a rebuild of the docker image, push it to DockerHub, and update the image tags in the `build_html.sh` and `build_pdf.sh` scripts on the PR automatically.
This new build environment will be used for the PR deploy preview mentioned above.

 
