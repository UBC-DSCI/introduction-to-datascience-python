## Data Science: A First Introduction (Python Edition)
This is the source for the python edition of the *Data Science: A First Introduction* textbook.

For the R version of the textbook, please visit https://datasciencebook.ca or the github repository
at https://github.com/ubc-dsci/introduction-to-datascience.

## License Information

This textbook is offered under 
the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [the license file](LICENSE.md) for more information. 

## Development

### Setup

Building the book requires Docker (instructions here: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)).

### Contributing 
Primary development in this repository happens on the `main` branch. If you want to contribute to the book,
please branch off of `main` and make a pull request into `main`.

The `production` branch contains the source material for the live, publicly viewable HTML book.

### Build locally

You can build the book on your own machine by running
```
./build_html.sh
```
in the root directory of this repository. The book can be viewed in your browser by opening the `source/_build/html/index.html` file.

### Update build environment

You can update the build environment for the book by making changes to `Dockerfile` in the root of the repository in the `main` branch.
If you push any changes to the `Dockerfile` on the `main` branch, GitHub will trigger a rebuild of the docker image,
push it to DockerHub, and update the `build_html.sh` script and book deploy GitHub action with the new image tag.

### Update public html

You can update the live, publicly viewable HTML book by making changes to the `source/` folder in the `production` branch.
If you push any changes to the `source/` folder on the `production` branch, GitHub will trigger a rebuild of the public HTML site.

 
