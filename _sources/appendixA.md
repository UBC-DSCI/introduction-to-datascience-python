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

# (APPENDIX) Appendix {-}

# Downloading files from JupyterHub {#appendixA}

This section will help you
save your work from a JupyterHub web-based platform to your own computer. 
Let's say you want to download everything inside a folder called `your_folder`
in your home directory.
First open a terminal \index{JupyterHub!file download} by clicking "terminal" in the Launcher tab. 
Next, type the following in the terminal to create a 
compressed `.zip` archive for the work you are interested in downloading:

```
zip -r hub_folder.zip your_folder
```

After the compressing process is complete, right-click on `hub_folder.zip`
in the JupyterHub file browser
and click "Download". After the download is complete, you should be 
able to find the `hub_folder.zip` file on your own computer,
and unzip the file (typically by double-clicking on it).
