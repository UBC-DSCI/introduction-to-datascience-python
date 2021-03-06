---
jupytext:
  cell_metadata_filter: -all
  formats: py:percent,md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}

---
output:
  html_document: default
  word_document: default
---
```

# Setting up your computer {#move-to-your-own-machine}

## Overview

In this chapter, you'll learn how to install all of the software 
needed to do the data science covered in this book on your own computer.

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

- Install the Git version control software.
- Install and launch a local instance of JupyterLab with the R kernel.
- Download the worksheets that accompany the chapters of this book from GitHub.

## Installing software on your own computer

This section will provide instructions for installing the software required by 
this book on your own computer. 
Given that installation instructions can vary widely based on the computer setup,
we have created instructions for multiple operating systems.
In particular, the installation instructions below have been verified to work
on a computer that:

- runs one of the following operating systems: Ubuntu 20.04, macOS Big Sur (version 11.4.x or 11.5.x), Windows 10 Professional, Enterprise or Education (version 2004, 20H2, or 21H1),
- has a connection to the internet,
- uses a 64-bit CPU,
- uses English as the default language.

### Git

As shown in Chapter \@ref(Getting-started-with-version-control), 
Git \index{git!installation} is a very useful tool for version controlling your projects, 
as well as sharing your work with others. Here's how to install Git on 
the following operating systems: 

**Windows:** To install
Git on Windows, go to <https://git-scm.com/download/win> and download the Windows 
version of Git. Once the download has finished, run the installer and accept 
the default configuration for all pages.

**MacOS:** To install Git on Mac OS, 
open the terminal ([how-to video](https://youtu.be/5AJbWEWwnbY)) 
and type the following command:

```
xcode-select --install
```

**Ubuntu:** To install Git on Ubuntu, open the terminal 
and type the following commands:

```
sudo apt update
sudo apt install git
```

+++

### Miniconda

To run Jupyter notebooks on your computer, 
you will need to install the web-based platform JupyterLab. 
But JupyterLab relies on Python, so we need to install Python first.
We can install Python via
the \index{miniconda} [miniconda Python package distribution](https://docs.conda.io/en/latest/miniconda.html).

**Windows:** To install miniconda on Windows, download 
the [latest Python 64-bit version from here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). 
Once the download has finished, run the installer 
and accept the default configuration for all pages. 
After installation, you can open the Anaconda Prompt
by opening the Start Menu and searching for the program called 
"Anaconda Prompt (miniconda3)". 
When this opens, you will see a prompt similar to 
`(base) C:\Users\your_name`. 

**MacOS:** To install miniconda on MacOS, you will need to use a different 
installation method depending on the type of processor chip your computer has.

If your Mac computer has an Intel x86 processor chip you can download 
the [latest Python 64-bit version from here](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg). 
After the download has finished, run the installer and accept the default 
configuration for all pages.

If your Mac computer has an Apple M1 processor chip you can download 
the [latest Python 64-bit version from here](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh).
After the download has finished, you need to run the downloaded script in the terminal using a command
like:

```
bash path/to/Miniconda3-latest-MacOSX-arm64.sh
```

Make sure to replace `path/to/` with the path of the folder
containing the downloaded script. Most computers will save downloaded files to the `Downloads` folder. 
If this is the case for your computer, you can run the script in the terminal by typing:

```
bash Downloads/Miniconda3-latest-MacOSX-arm64.sh
```

The instructions for the installation will then appear. 
Follow the prompts and agree to accepting the license,
the default installation location,
and to running `conda init`, which makes `conda` available from the terminal.

**Ubuntu:** To install miniconda on Ubuntu, first download
the [latest Python 64-bit version from here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh). 
After the download has finished, open the terminal and execute the following 
command:

```
bash path/to/Miniconda3-latest-Linux-x86_64.sh
```

Make sure to replace `path/to/` with the path of the folder containing the downloaded
script. Most often this file will be downloaded to the `Downloads` folder.
If this is the case for your computer, you can run the script in the terminal by typing:

```
bash Downloads/Miniconda3-latest-Linux-x86_64.sh
```

The instructions for the installation will then appear. 
Follow the prompts and agree to accepting the license,
the default installation location,
and to running `conda init`, which makes `conda` available from the terminal.

### JupyterLab

With miniconda set up, we can now install JupyterLab \index{JupyterLab installation} and the Jupyter Git \index{git!Jupyter extension} extension. 
Type the following into the Anaconda Prompt (Windows) or the terminal (MacOS and Ubuntu) and press enter:

```
conda install -c conda-forge -y jupyterlab
conda install -y nodejs
pip install --upgrade jupyterlab-git
```

To test that your JupyterLab installation is functional, you can type 
`jupyter lab` into the Anaconda Prompt (Windows) 
or terminal (MacOS and Ubuntu) and press enter. This should open a new 
tab in your default browser with the JupyterLab interface. To exit out of 
JupyterLab you can click `File -> Shutdown`, or go to the terminal from which 
you launched JupyterLab, hold `Ctrl`, and press `C` twice.

To improve the experience of using R in JupyterLab, you should also add an extension 
that allows you to set up keyboard shortcuts for inserting text.
By default, 
this extension creates shortcuts for inserting two of the most common R 
operators: `<-` and `|>`. Type the following in the Anaconda Prompt (Windows)
or terminal (MacOS and Ubuntu) and press enter:

```
jupyter labextension install @techrah/text-shortcuts
```

### R, R packages, and the IRkernel 

To have the software \index{R installation} used in this book available to you in JupyterLab, 
you will need to install the R programming language,
several R packages,
and the \index{kernel!installation} IRkernel.
To install versions of these that are compatible with the accompanying worksheets, 
type the command shown below into the Anaconda Prompt (Windows) 
or terminal (MacOS and Ubuntu). 

```
conda env update --file https://raw.githubusercontent.com/UBC-DSCI/data-science-a-first-intro-worksheets/main/environment.yml
```

This command installs the specific R and package versions specified in
the `environment.yml` file found in 
[the worksheets repository](https://ubc-dsci.github.io/data-science-a-first-intro-worksheets).
We will always keep the versions in the `environment.yml` file updated
so that they are compatible with the exercise worksheets that accompany the book.

> You can also install the *latest* version of R 
> and the R packages used in this book by typing the commands shown below 
> in the Anaconda Prompt (Windows) 
> or terminal (MacOS and Ubuntu) and pressing enter.
> **Be careful though:** this may install package versions that are
> incompatible with the worksheets that accompany the book; the automated
> exercise feedback might tell you your answers are not correct even though
> they are!
> 
> ```
> conda install -c conda-forge -y \
>   r-base \
>   r-cowplot \
>   r-ggally \
>   r-gridextra \
>   r-irkernel \
>   r-kknn \
>   r-rpostgres \
>   r-rsqlite \
>   r-scales \
>   r-testthat \
>   r-tidymodels \
>   r-tidyverse \
>   r-tinytex \
>   unixodbc
> ```

### LaTeX

To be able to render `.ipynb` files to `.pdf` you need to install a LaTeX 
distribution. These can be quite large, so we will opt to use `tinytex`, a 
light-weight cross-platform, portable, and easy-to-maintain LaTeX distribution 
based on TeX Live. 

**MacOS:** To install `tinytex` 
we need to make sure that `/usr/local/bin` is writable. 
To do this, type the following in the terminal:

```
sudo chown -R $(whoami):admin /usr/local/bin
```

>  **Note:** You might be asked to enter your password during installation.

**All operating systems:** 
To install LaTeX, open JupyterLab by typing `jupyter lab` 
in the Anaconda Prompt (Windows) or terminal (MacOS and Ubuntu) and press Enter.
Then from JupyterLab, open an R console, type the commands listed below, and 
press Shift + Enter to install `tinytex`:

```
tinytex::install_tinytex()
tinytex::tlmgr_install(c("eurosym", 
                         "adjustbox",
                         "caption",
                         "collectbox",
                         "enumitem",
                         "environ",
                         "fp",
                         "jknapltx",
                         "ms",
                         "oberdiek",
                         "parskip",
                         "pgf",
                         "rsfs",
                         "tcolorbox",
                         "titling",
                         "trimspaces",
                         "ucs",
                         "ulem",
                         "upquote"))
```

**Ubuntu:** 
To append the TinyTex executables to our `PATH` we need to edit our `.bashrc file`. 
The TinyTex executables are usually installed in `~/bin`. 
Thus, add the lines below to the bottom of your `.bashrc` file 
(which you can open by `nano ~/.bashrc` and save the file:

```
# Append TinyTex executables to the path
export PATH="$PATH:~/bin"
```

>  **Note:** If you used `nano` to open your `.bashrc` file, 
follow the keyboard shortcuts at the bottom of the nano text editor 
to save and close the file.

## Finishing up installation

It is good practice to restart all the programs you used when installing this
software stack before you proceed to doing your data analysis.
This includes restarting JupyterLab as well as the terminal (MacOS and Ubuntu) 
or the Anaconda Prompt (Windows).
This will ensure all the software and settings you put in place are 
correctly sourced. 

## Downloading the worksheets for this book

The worksheets containing practice exercises for this book 
can be downloaded by visiting 
[https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets),
clicking the green "Code" button, and then selecting "Download ZIP".
The worksheets are contained within the compressed zip folder that will be downloaded.
Once you unzip the downloaded file, you can open the folder and run each worksheet
using Jupyter. See Chapter \@ref(getting-started-with-jupyter) for instructions on how to use Jupyter.
