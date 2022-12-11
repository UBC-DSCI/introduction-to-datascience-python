## Data Science: A First Introduction (Python Edition)
This is the source for the python edition of the *Data Science: A First Introduction* textbook.

For the R version of the textbook, please visit https://datasciencebook.ca or the github repository
at https://github.com/ubc-dsci/introduction-to-datascience .

## License Information

This textbook is offered under 
the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [the license file](LICENSE.md) for more information. 

## Development

### Setup

In order to build the book, you need to first install Docker 
(instructions here: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)).

### Contributing 
Primary development in this repository happens on the `dev` branch. If you want to contribute to the book,
please branch off of `dev` and make a pull request into `dev`.

The `main` branch contains the source material for the live, publicly viewable HTML book.

### Build locally

You can build the book on your own machine by running
```
./build_html.sh
```
in the root directory of this repository. The book can be viewed in your browser by opening the `source/_build/html/index.html` file.

### Update build environment

You can update the build environment for the book by making changes to `Dockerfile` in the root of the repository in the `dev` branch.
If you push any changes to the `Dockerfile` on the `dev` branch, GitHub will trigger a rebuild of the docker image,
push it to DockerHub, and update the `build_html.sh` script and book deploy GitHub action with the new image digest.

### Update public html

You can update the live, publicly viewable HTML book by making changes to the `source/` folder in the `main` branch.
If you push any changes to the `source/` folder on the `main` branch, GitHub will trigger a rebuild of the public HTML site.

 
