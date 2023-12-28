---
jupytext:
  formats: py:percent,md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(reading)=
# Reading in data locally and from the web


## Overview

```{index} see: loading; reading
```

```{index} reading; definition
```

In this chapter, you’ll learn to read tabular data of various formats into Python
from your local device (e.g., your laptop) and the web. “Reading” (or “loading”)
is the process of
converting data (stored as plain text, a database, HTML, etc.) into an object
(e.g., a data frame) that Python can easily access and manipulate. Thus reading data
is the gateway to any data analysis; you won’t be able to analyze data unless
you’ve loaded it first. And because there are many ways to store data, there
are similarly many ways to read data into Python. The more time you spend upfront
matching the data reading method to the type of data you have, the less time
you will have to devote to re-formatting, cleaning and wrangling your data (the
second step to all data analyses). It’s like making sure your shoelaces are
tied well before going for a run so that you don’t trip later on!

## Chapter learning objectives
By the end of the chapter, readers will be able to do the following:

- Define the types of path and use them to locate files:
    - absolute file path
    - relative file path
    - Uniform Resource Locator (URL)
- Read data into Python from various types of path using:
    - `read_csv`
    - `read_excel`
- Compare and contrast `read_csv` and `read_excel`.
- Describe when to use the following `read_csv` function arguments:
    - `skiprows`
    - `sep`
    - `header`
    - `names`
- Choose the appropriate `read_csv` function arguments to load a given plain text tabular data set into Python.
- Use the `rename` function to rename columns in a data frame.
- Use `pandas` package's `read_excel` function and arguments to load a sheet from an excel file into Python.
- Work with databases using functions from the `ibis` package:
    - Connect to a database with `connect`.
    - List tables in the database with `list_tables`.
    - Create a reference to a database table with `table`.
    - Bring data from a database into Python with `execute`.
- Use `to_csv` to save a data frame to a `.csv` file.
- (*Optional*) Obtain data from the web using scraping and application programming interfaces (APIs):
    - Read HTML source code from a URL using the `BeautifulSoup` package.
    - Read data from the NASA "Astronomy Picture of the Day" using the `requests` package.
    - Compare downloading tabular data from a plain text file (e.g., `.csv`), accessing data from an API, and scraping the HTML source code from a website.

## Absolute and relative file paths

```{index} see: location; path
```

```{index} path; local, path; remote, path; relative, path; absolute
```

This chapter will discuss the different functions we can use to import data
into Python, but before we can talk about *how* we read the data into Python with these
functions, we first need to talk about *where* the data lives. When you load a
data set into Python, you first need to tell Python where those files live. The file
could live on your computer (*local*) or somewhere on the internet (*remote*).

The place where the file lives on your computer is referred to as its "path". You can
think of the path as directions to the file. There are two kinds of paths:
*relative* paths and *absolute* paths. A relative path indicates where the file is
with respect to your *working directory* (i.e., "where you are currently") on the computer.
On the other hand, an absolute path indicates where the file is
with respect to the computer's filesystem base (or *root*) folder, regardless of where you are working.

Suppose our computer's filesystem looks like the picture in
{numref}`Filesystem`. We are working in a
file titled `project3.ipynb`, and our current working directory is `project3`;
typically, as is the case here, the working directory is the directory containing the file you are currently
working on.

```{figure} img/reading/filesystem.png
---
name: Filesystem
---
Example file system
```

Let's say we wanted to open the `happiness_report.csv` file. We have two options to indicate
where the file is: using a relative path, or using an absolute path.
The absolute path of the file always starts with a slash `/`&mdash;representing the root folder on the computer&mdash;and
proceeds by listing out the sequence of folders you would have to enter to reach the file, each separated by another slash `/`.
So in this case, `happiness_report.csv` would be reached by starting at the root, and entering the `home` folder,
then the `dsci-100` folder, then the `project3` folder, and then finally the `data` folder. So its absolute
path would be `/home/dsci-100/project3/data/happiness_report.csv`. We can load the file using its absolute path
as a string passed to the `read_csv` function from `pandas`.
```{code-cell} ipython3
:tags: ["remove-output"]
happy_data = pd.read_csv("/home/dsci-100/project3/data/happiness_report.csv")
```
If we instead wanted to use a relative path, we would need to list out the sequence of steps needed to get from our current
working directory to the file, with slashes `/` separating each step. Since we are currently in the `project3` folder,
we just need to enter the `data` folder to reach our desired file. Hence the relative path is `data/happiness_report.csv`,
and we can load the file using its relative path as a string passed to `read_csv`.
```{code-cell} ipython3
:tags: ["remove-output"]
happy_data = pd.read_csv("data/happiness_report.csv")
```
Note that there is no forward slash at the beginning of a relative path; if we accidentally typed `"/data/happiness_report.csv"`,
Python would look for a folder named `data` in the root folder of the computer&mdash;but that doesn't exist!

```{index} path; previous, path; current
```

```{index} see: ..; path
```

```{index} see: .; path
```

Aside from specifying places to go in a path using folder names (like `data` and `project3`), we can also specify two additional
special places: the *current directory* and the *previous directory*. We indicate the current working directory with a single dot `.`, and
the previous directory with two dots `..`. So for instance, if we wanted to reach the `bike_share.csv` file from the `project3` folder, we could
use the relative path `../project2/bike_share.csv`. We can even combine these two; for example, we could reach the `bike_share.csv` file using
the (very silly) path `../project2/../project2/./bike_share.csv` with quite a few redundant directions: it says to go back a folder, then open `project2`,
then go back a folder again, then open `project2` again, then stay in the current directory, then finally get to `bike_share.csv`. Whew, what a long trip!

So which kind of path should you use: relative, or absolute? Generally speaking, you should use relative paths.
Using a relative path helps ensure that your code can be run
on a different computer (and as an added bonus, relative paths are often shorter&mdash;easier to type!).
This is because a file's relative path is often the same across different computers, while a
file's absolute path (the names of
all of the folders between the computer's root, represented by `/`, and the file) isn't usually the same
across different computers. For example, suppose Fatima and Jayden are working on a
project together on the `happiness_report.csv` data. Fatima's file is stored at

```text
/home/Fatima/project3/data/happiness_report.csv
```

while Jayden's is stored at

```text
/home/Jayden/project3/data/happiness_report.csv
```

Even though Fatima and Jayden stored their files in the same place on their
computers (in their home folders), the absolute paths are different due to
their different usernames.  If Jayden has code that loads the
`happiness_report.csv` data using an absolute path, the code won't work on
Fatima's computer.  But the relative path from inside the `project3` folder
(`data/happiness_report.csv`) is the same on both computers; any code that uses
relative paths will work on both! In the additional resources section,
we include a link to a short video on the
difference between absolute and relative paths.

```{index} URL
```

Beyond files stored on your computer (i.e., locally), we also need a way to locate resources
stored elsewhere on the internet (i.e., remotely). For this purpose we use a
*Uniform Resource Locator (URL)*, i.e., a web address that looks something
like https://python.datasciencebook.ca/. URLs indicate the location of a resource on the internet, and
start with a web domain, followed by a forward slash `/`, and then a path
to where the resource is located on the remote machine.

## Reading tabular data from a plain text file into Python

(readcsv)=
### `read_csv` to read in comma-separated values files

```{index} csv, reading; separator, read function; read_csv
```

Now that we have learned about *where* data could be, we will learn about *how*
to import data into Python using various functions. Specifically, we will learn how
to *read* tabular data from a plain text file (a document containing only text)
*into* Python and *write* tabular data to a file *out of* Python. The function we use to do this
depends on the file's format. For example, in the last chapter, we learned about using
the `read_csv` function from `pandas` when reading `.csv` (**c**omma-**s**eparated **v**alues)
files. In that case, the *separator* that divided our columns was a
comma (`,`). We only learned the case where the data matched the expected defaults
of the `read_csv` function
(column names are present, and commas are used as the separator between columns).
In this section, we will learn how to read
files that do not satisfy the default expectations of `read_csv`.

```{index} Canadian languages; canlang data
```

Before we jump into the cases where the data aren't in the expected default format
for `pandas` and `read_csv`, let's revisit the more straightforward
case where the defaults hold, and the only argument we need to give to the function
is the path to the file, `data/can_lang.csv`. The `can_lang` data set contains
language data from the 2016 Canadian census.
We put `data/` before the file's
name when we are loading the data set because this data set is located in a
sub-folder, named `data`, relative to where we are running our Python code.
Here is what the text in the file `data/can_lang.csv` looks like.

```text
category,language,mother_tongue,most_at_home,most_at_work,lang_known
Aboriginal languages,"Aboriginal languages, n.o.s.",590,235,30,665
Non-Official & Non-Aboriginal languages,Afrikaans,10260,4785,85,23415
Non-Official & Non-Aboriginal languages,"Afro-Asiatic languages, n.i.e.",1150,44
Non-Official & Non-Aboriginal languages,Akan (Twi),13460,5985,25,22150
Non-Official & Non-Aboriginal languages,Albanian,26895,13135,345,31930
Aboriginal languages,"Algonquian languages, n.i.e.",45,10,0,120
Aboriginal languages,Algonquin,1260,370,40,2480
Non-Official & Non-Aboriginal languages,American Sign Language,2685,3020,1145,21
Non-Official & Non-Aboriginal languages,Amharic,22465,12785,200,33670
```

```{index} pandas
```

And here is a review of how we can use `read_csv` to load it into Python. First we
load the `pandas` package to gain access to useful
functions for reading the data.

```{code-cell} ipython3
import pandas as pd
```

Next we use `read_csv` to load the data into Python, and in that call we specify the
relative path to the file.

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_csv("data/can_lang.csv")
canlang_data
```

### Skipping rows when reading in data

Oftentimes, information about how data was collected, or other relevant
information, is included at the top of the data file. This information is
usually written in sentence and paragraph form, with no separator because it is
not organized into columns. An example of this is shown below. This information
gives the data scientist useful context and information about the data,
however, it is not well formatted or intended to be read into a data frame cell
along with the tabular data that follows later in the file.

```text
Data source: https://ttimbers.github.io/canlang/
Data originally published in: Statistics Canada Census of Population 2016.
Reproduced and distributed on an as-is basis with their permission.
category,language,mother_tongue,most_at_home,most_at_work,lang_known
Aboriginal languages,"Aboriginal languages, n.o.s.",590,235,30,665
Non-Official & Non-Aboriginal languages,Afrikaans,10260,4785,85,23415
Non-Official & Non-Aboriginal languages,"Afro-Asiatic languages, n.i.e.",1150,445,10,2775
Non-Official & Non-Aboriginal languages,Akan (Twi),13460,5985,25,22150
Non-Official & Non-Aboriginal languages,Albanian,26895,13135,345,31930
Aboriginal languages,"Algonquian languages, n.i.e.",45,10,0,120
Aboriginal languages,Algonquin,1260,370,40,2480
Non-Official & Non-Aboriginal languages,American Sign Language,2685,3020,1145,21930
Non-Official & Non-Aboriginal languages,Amharic,22465,12785,200,33670
```

With this extra information being present at the top of the file, using
`read_csv` as we did previously does not allow us to correctly load the data
into Python. In the case of this file, Python just prints a `ParserError`
message, indicating that it wasn't able to read the file.

```{code-cell} ipython3
:tags: ["remove-output"]
canlang_data = pd.read_csv("data/can_lang_meta-data.csv")
```
```{code-cell} ipython3
:tags: ["remove-input"]
print("ParserError: Error tokenizing data. C error: Expected 1 fields in line 4, saw 6")
```

```{index} ParserError
```

```{index} read function; skiprows argument
```

To successfully read data like this into Python, the `skiprows`
argument can be useful to tell Python
how many rows to skip before
it should start reading in the data. In the example above, we would set this
value to 3 to read and load the data correctly.

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_csv("data/can_lang_meta-data.csv", skiprows=3)
canlang_data
```

How did we know to skip three rows? We looked at the data! The first three rows
of the data had information we didn't need to import:

```text
Data source: https://ttimbers.github.io/canlang/
Data originally published in: Statistics Canada Census of Population 2016.
Reproduced and distributed on an as-is basis with their permission.
```

The column names began at row 4, so we skipped the first three rows.

### Using the `sep` argument for different separators

Another common way data is stored is with tabs as the separator. Notice the
data file, `can_lang.tsv`, has tabs in between the columns instead of
commas.

```text
category	language	mother_tongue	most_at_home	most_at_work	lang_known
Aboriginal languages	Aboriginal languages, n.o.s.	590	235	30	665
Non-Official & Non-Aboriginal languages	Afrikaans	10260	4785	85	23415
Non-Official & Non-Aboriginal languages	Afro-Asiatic languages, n.i.e.	1150	445	10	2775
Non-Official & Non-Aboriginal languages	Akan (Twi)	13460	5985	25	22150
Non-Official & Non-Aboriginal languages	Albanian	26895	13135	345	31930
Aboriginal languages	Algonquian languages, n.i.e.	45	10	0	120
Aboriginal languages	Algonquin	1260	370	40	2480
Non-Official & Non-Aboriginal languages	American Sign Language	2685	3020	1145	21930
Non-Official & Non-Aboriginal languages	Amharic	22465	12785	200	33670
```
```{index} read function; sep argument
```

```{index} see: tab-separated values; tsv
```

```{index} tsv
```

To read in `.tsv` (**t**ab **s**eparated **v**alues) files, we can set the `sep` argument
in the `read_csv` function to the *tab character* `\t`.

```{index} escape character
```

```{note}
`\t` is an example of an *escaped character*,
which always starts with a backslash (`\`).
Escaped characters are used to represent non-printing characters
(like the tab) or characters with special meanings (such as quotation marks).
```


```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_csv("data/can_lang.tsv", sep="\t")
canlang_data
```

If you compare the data frame here to the data frame we obtained in
{numref}`readcsv` using `read_csv`, you'll notice that they look identical: they have
the same number of columns and rows, the same column names, and the same entries!
So even though we needed to use different
arguments depending on the file format, our resulting data frame
(`canlang_data`) in both cases was the same.

### Using the `header` argument to handle missing column names

```{index} read function; header argument, reading; separator
```

The `can_lang_no_names.tsv` file contains a slightly different version
of this data set, except with no column names, and tabs for separators.
Here is how the file looks in a text editor:

```text
Aboriginal languages	Aboriginal languages, n.o.s.	590	235	30	665
Non-Official & Non-Aboriginal languages	Afrikaans	10260	4785	85	23415
Non-Official & Non-Aboriginal languages	Afro-Asiatic languages, n.i.e.	1150	445	10	2775
Non-Official & Non-Aboriginal languages	Akan (Twi)	13460	5985	25	22150
Non-Official & Non-Aboriginal languages	Albanian	26895	13135	345	31930
Aboriginal languages	Algonquian languages, n.i.e.	45	10	0	120
Aboriginal languages	Algonquin	1260	370	40	2480
Non-Official & Non-Aboriginal languages	American Sign Language	2685	3020	1145	21930
Non-Official & Non-Aboriginal languages	Amharic	22465	12785	200	33670

```

Data frames in Python need to have column names.  Thus if you read in data
without column names, Python will assign names automatically. In this example,
Python assigns the column names `0, 1, 2, 3, 4, 5`.
To read this data into Python, we specify the first
argument as the path to the file (as done with `read_csv`), and then provide
values to the `sep` argument (here a tab, which we represent by `"\t"`),
and finally set `header = None` to tell `pandas` that the data file does not
contain its own column names.

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_csv(
    "data/can_lang_no_names.tsv",
    sep="\t",
    header=None
)
canlang_data
```

```{index} DataFrame; rename, pandas
```

It is best to rename your columns manually in this scenario. The current column names
(`0, 1`, etc.) are problematic for two reasons: first, because they not very descriptive names, which will make your analysis
confusing; and second, because your column names should generally be *strings*, but are currently *integers*.
To rename your columns, you can use the `rename` function
from the [pandas package](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html#).
The argument of the `rename` function is `columns`, which takes a mapping between the old column names and the new column names.
In this case, we want to rename the old columns (`0, 1, ..., 5`) in the `canlang_data` data frame to more descriptive names.

To specify the mapping, we create a *dictionary*: a Python object that represents
a mapping from *keys* to *values*. We can create a dictionary by using a pair of curly
braces `{ }`, and inside the braces placing pairs of `key : value` separated by commas.
Below, we create a dictionary called `col_map` that maps the old column names in `canlang_data` to new column
names, and then pass it to the `rename` function.

```{code-cell} ipython3
:tags: ["output_scroll"]
col_map = {
    0 : "category",
    1 : "language",
    2 : "mother_tongue",
    3 : "most_at_home",
    4 : "most_at_work",
    5 : "lang_known"
}
canlang_data_renamed = canlang_data.rename(columns=col_map)
canlang_data_renamed
```

```{index} read function; names argument
```

The column names can also be assigned to the data frame immediately upon reading it from the file by passing a
list of column names to the `names` argument in `read_csv`.

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_csv(
    "data/can_lang_no_names.tsv",
    sep="\t",
    header=None,
    names=[
        "category",
        "language",
        "mother_tongue",
        "most_at_home",
        "most_at_work",
        "lang_known",
    ],
)
canlang_data
```

### Reading tabular data directly from a URL

```{index} URL; reading from
```

We can also use `read_csv` to read in data directly from a **U**niform **R**esource **L**ocator (URL) that
contains tabular data. Here, we provide the URL of a remote file
to `read_csv`, instead of a path to a local file on our
computer. We need to surround the URL with quotes similar to when we specify a
path on our local computer. All other arguments that we use are the same as
when using these functions with a local file on our computer.

```{code-cell} ipython3
:tags: ["output_scroll"]
url = "https://raw.githubusercontent.com/UBC-DSCI/introduction-to-datascience-python/reading/source/data/can_lang.csv"
pd.read_csv(url)
canlang_data = pd.read_csv(url)

canlang_data
```

### Previewing a data file before reading it into Python

In many of the examples above, we gave you previews of the data file before we read
it into Python. Previewing data is essential to see whether or not there are column
names, what the separators are, and if there are rows you need to skip. You
should do this yourself when trying to read in data files: open the file in whichever
text editor you prefer to inspect its contents prior to reading it into Python.

## Reading tabular data from a Microsoft Excel file

```{index} Excel spreadsheet
```

```{index} see: Microsoft Excel; Excel spreadsheet
```

```{index} see: xlsx; Excel spreadsheet
```

There are many other ways to store tabular data sets beyond plain text files,
and similarly, many ways to load those data sets into Python. For example, it is
very common to encounter, and need to load into Python, data stored as a Microsoft
Excel spreadsheet (with the file name
extension `.xlsx`).  To be able to do this, a key thing to know is that even
though `.csv` and `.xlsx` files look almost identical when loaded into Excel,
the data themselves are stored completely differently.  While `.csv` files are
plain text files, where the characters you see when you open the file in a text
editor are exactly the data they represent, this is not the case for `.xlsx`
files. Take a look at a snippet of what a `.xlsx` file would look like in a text editor:

+++

```text
,?'O
    _rels/.rels???J1??>E?{7?
<?V????w8?'J???'QrJ???Tf?d??d?o?wZ'???@>?4'?|??hlIo??F
t                                                       8f??3wn
????t??u"/
          %~Ed2??<?w??
                       ?Pd(??J-?E???7?'t(?-GZ?????y???c~N?g[^_r?4
                                                                  yG?O
                                                                      ?K??G?


     ]TUEe??O??c[???????6q??s??d?m???\???H?^????3} ?rZY? ?:L60?^?????XTP+?|?
X?a??4VT?,D?Jq
```

```{index} read function; read_excel
```

```{index} Excel spreadsheet; reading
```


This type of file representation allows Excel files to store additional things
that you cannot store in a `.csv` file, such as fonts, text formatting,
graphics, multiple sheets and more. And despite looking odd in a plain text
editor, we can read Excel spreadsheets into Python using the `pandas` package's `read_excel`
function developed specifically for this
purpose.

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_data = pd.read_excel("data/can_lang.xlsx")
canlang_data
```

If the `.xlsx` file has multiple sheets, you have to use the `sheet_name` argument
to specify the sheet number or name. This functionality is useful when a single sheet contains
multiple tables (a sad thing that happens to many Excel spreadsheets since this
makes reading in data more difficult). You can also specify cell ranges using the
`usecols` argument (e.g., `usecols="A:D"` for including columns from `A` to `D`).

As with plain text files, you should always explore the data file before
importing it into Python. Exploring the data beforehand helps you decide which
arguments you need to load the data into Python successfully. If you do not have
the Excel program on your computer, you can use other programs to preview the
file. Examples include Google Sheets and Libre Office.

In {numref}`read_func` we summarize the `read_csv` and `read_excel` functions we covered
in this chapter. We also include the arguments for data separated by
semicolons `;`, which you may run into with data sets where the decimal is
represented by a comma instead of a period (as with some data sets from
European countries).


```{list-table} Summary of read_csv and read_excel
:header-rows: 1
:name: read_func

* - Data File Type
  - Python Function
  - Arguments
* - Comma (`,`) separated files
  - `read_csv`
  - just the file path
* - Tab (`\t`) separated files
  - `read_csv`
  - `sep="\t"`
* - Missing header
  - `read_csv`
  - `header=None`
* - European-style numbers, semicolon (`;`) separators
  - `read_csv`
  - `sep=";"`, `thousands="."`, `decimal=","`
* - Excel files (`.xlsx`)
  - `read_excel`
  - `sheet_name`, `usecols`


```

## Reading data from a database

```{index} database
```

Another very common form of data storage is the relational database. Databases
are great when you have large data sets or multiple users
working on a project. There are many relational database management systems,
such as SQLite, MySQL, PostgreSQL, Oracle, and many more. These
different relational database management systems each have their own advantages
and limitations. Almost all employ SQL (*structured query language*) to obtain
data from the database. But you don't need to know SQL to analyze data from
a database; several packages have been written that allow you to connect to
relational databases and use the Python programming language
to obtain data. In this book, we will give examples of how to do this
using Python with SQLite and PostgreSQL databases.

### Reading data from a SQLite database

```{index} database; SQLite
```

SQLite is probably the simplest relational database system
that one can use in combination with Python. SQLite databases are self-contained, and are
usually stored and accessed locally on one computer from
a file with a `.db` extension (or sometimes a `.sqlite` extension).
Similar to Excel files, these are not plain text files and cannot be read in a plain text editor.

```{index} database; connection, ibis; connect
```

```{index} see: ibis; database
```

```{index} see: database; ibis
```

The first thing you need to do to read data into Python from a database is to
connect to the database. For an SQLite database, we will do that using
the `connect` function from the
`sqlite` backend in the
`ibis` package. This command does not read
in the data, but simply tells Python where the database is and opens up a
communication channel that Python can use to send SQL commands to the database.

```{note}
There is another database package in python called `sqlalchemy`.
That package is a bit more mature than `ibis`,
so if you want to dig deeper into working with databases in Python, that is a good next
package to learn about. We will work with `ibis` in this book, as it
provides a more modern and friendlier syntax that is more like `pandas` for data analysis code.
```

```{code-cell} ipython3
import ibis

conn = ibis.sqlite.connect("data/can_lang.db")
```

```{index} database; table, ibis; list_tables, ibis; sqlite
```

Often relational databases have many tables; thus, in order to retrieve
data from a database, you need to know the name of the table
in which the data is stored. You can get the names of
all the tables in the database using the `list_tables`
function:

```{code-cell} ipython3
tables = conn.list_tables()
tables
```

```{index} table, ibis; table
```

The `list_tables` function returned only one name---`"can_lang"`---which tells us
that there is only one table in this database. To reference a table in the
database (so that we can perform operations like selecting columns and filtering rows), we
use the `table` function from the `conn` object. The object returned
by the `table` function allows us to work with data
stored in databases as if they were just regular `pandas` data frames; but secretly, behind
the scenes, `ibis` will turn your commands into SQL queries!

```{code-cell} ipython3
canlang_table = conn.table("can_lang")
canlang_table
```

```{index} ibis; count
```

```{index} see: count; ibis
```

Although it looks like we might have obtained the whole data frame from the database, we didn't!
It's a *reference*; the data is still stored only in the SQLite database. The `canlang_table` object
is a `DatabaseTable`, which, when printed, tells
you which columns are available in the table. But unlike a usual `pandas` data frame,
we do not immediately know how many rows are in the table. In order to find out how many
rows there are, we have to send an SQL *query* (i.e., command) to the data base.
In `ibis`, we can do that using the `count` function from the table object.

```{code-cell} ipython3
canlang_table.count()
```

```{index} ibis; execute
```

Wait a second...this isn't the number of rows in the database. In fact, we haven't actually sent our
SQL query to the database yet! We need to explicitly tell `ibis` when we want to send the query.
The reason for this is that databases are often more efficient at working with (i.e., selecting, filtering,
joining, etc.) large data sets than Python. And typically, the database will not even
be stored on your computer, but rather a more powerful machine somewhere on the
web. So `ibis` is lazy and waits to bring this data into memory until you explicitly
tell it to using the `execute` function. The `execute` function actually sends the SQL query
to the database, and gives you the result. Let's look at the number of rows in the table by executing
the `count` command.

```{code-cell} ipython3
canlang_table.count().execute()
```
There we go! There are 214 rows in the `can_lang` table. If you are interested in seeing
the *actual* text of the SQL query that `ibis` sends to the database, you can use the `compile` function
instead of `execute`. But note that you have to pass the result of `compile` to the `str` function to turn it into
a human-readable string first.

```{index} see: compile;ibis
```
```{index} ibis; compile, str
```

```{code-cell} ipython3
str(canlang_table.count().compile())
```

The output above shows the SQL code that is sent to the database. When we
write `canlang_table.count().execute()` in Python, in the background, the `execute` function is
translating the Python code into SQL, sending that SQL to the database, and then translating the
response for us. So `ibis` does all the hard work of translating from Python to SQL and back for us;
we can just stick with Python!

The `ibis` package provides lots of `pandas`-like tools for working with database tables.
For example, we can look at the first few rows of the table by using the `head` function,
followed by `execute` to retrieve the response.

```{index} ibis; head
```

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_table.head(10).execute()
```

You can see that `ibis` actually returned a `pandas` data frame to us after we executed the query,
which is very convenient for working with the data after getting it from the database.
So now that we have the `canlang_table` table reference for the 2016 Canadian Census data in hand, we
can mostly continue onward as if it were a regular data frame. For example, let's do the same exercise
from {numref}`Chapter %s <intro>`: we will obtain only those rows corresponding to Aboriginal languages, and keep only
the `language` and `mother_tongue` columns.
We can use the `[]` operation with a logical statement
to obtain only certain rows. Below we filter the data to include only Aboriginal languages.

```{index} database; filter rows, ibis; []
```

```{code-cell} ipython3
canlang_table_filtered = canlang_table[canlang_table["category"] == "Aboriginal languages"]
canlang_table_filtered
```
Above you can see that we have not yet executed this command; `canlang_table_filtered` is just showing
the first part of our query (the part that starts with `Selection[r0]` above).
We didn't call `execute` because we are not ready to bring the data into Python yet.
We can still use the database to do some work to obtain *only* the small amount of data we want to work with locally
in Python. Let's add the second part of our SQL query: selecting only the `language` and `mother_tongue` columns.

```{index} database; select columns
```

```{code-cell} ipython3
canlang_table_selected = canlang_table_filtered[["language", "mother_tongue"]]
canlang_table_selected
```
Now you can see that the `ibis` query will have two steps: it will first find rows corresponding to
Aboriginal languages, then it will extract only the `language` and `mother_tongue` columns that we are interested in.
Let's actually execute the query now to bring the data into Python as a `pandas` data frame, and print the result.
```{code-cell} ipython3
aboriginal_lang_data = canlang_table_selected.execute()
aboriginal_lang_data
```

`ibis` provides many more functions (not just the `[]` operation)
that you can use to manipulate the data within the database before calling
`execute` to obtain the data in Python. But `ibis` does not provide *every* function
that we need for analysis; we do eventually need to call `execute`.
For example, `ibis` does not provide the `tail` function to look at the last
rows in a database, even though `pandas` does.

```{index} DataFrame; tail
```

```{code-cell} ipython3
:tags: ["output_scroll"]
canlang_table_selected.tail(6)
```

```{code-cell} ipython3
aboriginal_lang_data.tail(6)
```

So once you have finished your data wrangling of the database reference object, it is advisable to
bring it into Python as a `pandas` data frame using the `execute` function.
But be very careful using `execute`: databases are often *very* big,
and reading an entire table into Python might take a long time to run or even possibly
crash your machine. So make sure you select and filter the database table
to reduce the data to a reasonable size before using `execute` to read it into Python!

### Reading data from a PostgreSQL database

```{index} database; PostgreSQL
```

PostgreSQL (also called Postgres) is a very popular
and open-source option for relational database software.
Unlike SQLite,
PostgreSQL uses a client–server database engine, as it was designed to be used
and accessed on a network. This means that you have to provide more information
to Python when connecting to Postgres databases. The additional information that you
need to include when you call the `connect` function is listed below:

- `database`: the name of the database (a single PostgreSQL instance can host more than one database)
- `host`: the URL pointing to where the database is located (`localhost` if it is on your local machine)
- `port`: the communication endpoint between Python and the PostgreSQL database (usually `5432`)
- `user`: the username for accessing the database
- `password`: the password for accessing the database

Below we demonstrate how to connect to a version of
the `can_mov_db` database, which contains information about Canadian movies.
Note that the `host` (`fakeserver.stat.ubc.ca`), `user` (`user0001`), and
`password` (`abc123`) below are *not real*; you will not actually
be able to connect to a database using this information.

```{index} ibis; postgres, ibis; connect
```

```{code-cell} ipython3
:tags: ["remove-output"]
conn = ibis.postgres.connect(
    database="can_mov_db",
    host="fakeserver.stat.ubc.ca",
    port=5432,
    user="user0001",
    password="abc123"
)
```

Aside from needing to provide that additional information, `ibis` makes it so
that connecting to and working with a Postgres database is identical to
connecting to and working with an SQLite database. For example, we can again use
`list_tables` to find out what tables are in the `can_mov_db` database:

```{index} ibis; list_tables
```

```{code-cell} ipython3
:tags: ["remove-output"]
conn.list_tables()
```

```{code-cell} ipython3
:tags: ["remove-input"]
print('["themes", "medium", "titles", "title_aliases", "forms", "episodes", "names", "names_occupations", "occupation", "ratings"]')
```

We see that there are 10 tables in this database. Let's first look at the
`"ratings"` table to find the lowest rating that exists in the `can_mov_db`
database.

```{index} ibis; table
```

```{code-cell} ipython3
:tags: ["remove-output"]
ratings_table = conn.table("ratings")
ratings_table
```

```{code-cell} ipython3
:tags: ["remove-input"]
print("""
AlchemyTable: ratings
  title           string
  average_rating  float64
  num_votes       int64
""")
```

```{index} ibis; []
```

To find the lowest rating that exists in the data base, we first need to
select the `average_rating` column:

```{code-cell} ipython3
:tags: ["remove-output"]
avg_rating = ratings_table[["average_rating"]]
avg_rating
```

```{code-cell} ipython3
:tags: ["remove-input"]
print("""
r0 := AlchemyTable: ratings
  title           string
  average_rating  float64
  num_votes       int64

Selection[r0]
  selections:
    average_rating: r0.average_rating
""")
```

```{index} database; ordering, ibis; order_by, ibis; head
```

Next we use the `order_by` function from `ibis` order the table by `average_rating`,
and then the `head` function to select the first row (i.e., the lowest score).

```{code-cell} ipython3
:tags: ["remove-output"]
lowest = avg_rating.order_by("average_rating").head(1)
lowest.execute()
```

```{code-cell} ipython3
:tags: ["remove-input"]
lowest = pd.DataFrame({"average_rating" : [1.0]})
lowest
```

We see the lowest rating given to a movie is 1, indicating that it must have
been a really bad movie...

### Why should we bother with databases at all?

```{index} database; reasons to use
```

Opening a database involved a lot more effort than just opening a `.csv`, or any of the
other plain text or Excel formats. We had to open a connection to the database,
then use `ibis` to translate `pandas`-like
commands (the `[]` operation, `head`, etc.) into SQL queries that the database
understands, and then finally `execute` them. And not all `pandas` commands can currently be translated
via `ibis` into database queries. So you might be wondering: why should we use
databases at all?

Databases are beneficial in a large-scale setting:

- They enable storing large data sets across multiple computers with backups.
- They provide mechanisms for ensuring data integrity and validating input.
- They provide security and data access control.
- They allow multiple users to access data simultaneously
  and remotely without conflicts and errors.
  For example, there are billions of Google searches conducted daily in 2021 {cite:p}`googlesearches`.
  Can you imagine if Google stored all of the data
  from those searches in a single `.csv` file!? Chaos would ensue!

## Writing data from Python to a `.csv` file

```{index} write function; to_csv, DataFrame; to_csv
```

At the middle and end of a data analysis, we often want to write a data frame
that has changed (through selecting columns, filtering rows, etc.)
to a file to share it with others or use it for another step in the analysis.
The most straightforward way to do this is to use the `to_csv` function
from the `pandas` package.  The default
arguments are to use a comma (`,`) as the separator, and to include column names
in the first row. We also specify `index = False` to tell `pandas` not to print
row numbers in the `.csv` file. Below we demonstrate creating a new version of the Canadian
languages data set without the "Official languages" category according to the
Canadian 2016 Census, and then writing this to a `.csv` file:

```{code-cell} ipython3
no_official_lang_data = canlang_data[canlang_data["category"] != "Official languages"]
no_official_lang_data.to_csv("data/no_official_languages.csv", index=False)
```

## Obtaining data from the web

```{note}
This section is not required reading for the remainder of the textbook. It
is included for those readers interested in learning a little bit more about
how to obtain different types of data from the web.
```

```{index} see: application programming interface; API
```

```{index} API
```

Data doesn't just magically appear on your computer; you need to get it from
somewhere. Earlier in the chapter we showed you how to access data stored in a
plain text, spreadsheet-like format (e.g., comma- or tab-separated) from a web
URL using the `read_csv` function from `pandas`. But as time goes on, it is
increasingly uncommon to find data (especially large amounts of data) in this
format available for download from a URL. Instead, websites now often offer
something known as an **a**pplication **p**rogramming **i**nterface (API),
which provides a programmatic way to ask for subsets of a data set. This allows
the website owner to control *who* has access to the data, *what portion* of
the data they have access to, and *how much* data they can access.  Typically,
the website owner will give you a *token* or *key* (a secret string of characters
somewhat like a password) that you have to provide when accessing the API.

```{index} web scraping, CSS, HTML
```

```{index} see: hypertext markup language; HTML
```

```{index} see: cascading style sheet; CSS
```

Another interesting thought: websites themselves *are* data! When you type a
URL into your browser window, your browser asks the *web server* (another
computer on the internet whose job it is to respond to requests for the
website) to give it the website's data, and then your browser translates that
data into something you can see. If the website shows you some information that
you're interested in, you could *create* a data set for yourself by copying and
pasting that information into a file. This process of taking information
directly from what a website displays is called
*web scraping* (or sometimes *screen scraping*). Now, of course, copying and pasting
information manually is a painstaking and error-prone process, especially when
there is a lot of information to gather. So instead of asking your browser to
translate the information that the web server provides into something you can
see, you can collect that data programmatically&mdash;in the form of
**h**yper**t**ext **m**arkup **l**anguage (HTML) and **c**ascading **s**tyle **s**heet (CSS)
code&mdash;and process it to extract useful information. HTML provides the
basic structure of a site and tells the webpage how to display the content
(e.g., titles, paragraphs, bullet lists etc.), whereas CSS helps style the
content and tells the webpage how the HTML elements should
be presented (e.g., colors, layouts, fonts etc.).

This subsection will show you the basics of both web scraping
with the [`BeautifulSoup` Python package](https://beautiful-soup-4.readthedocs.io/en/latest/) {cite:p}`beautifulsoup`
and accessing the NASA "Astronomy Picture of the Day" API
using the [`requests` Python package](https://requests.readthedocs.io/en/latest/) {cite:p}`requests`.

+++

### Web scraping

#### HTML and CSS selectors

```{index} web scraping, HTML; selector, CSS; selector, Craiglist
```

When you enter a URL into your browser, your browser connects to the
web server at that URL and asks for the *source code* for the website.
This is the data that the browser translates
into something you can see; so if we
are going to create our own data by scraping a website, we have to first understand
what that data looks like! For example, let's say we are interested
in knowing the average rental price (per square foot) of the most recently
available one-bedroom apartments in Vancouver
on [Craiglist](https://vancouver.craigslist.org). When we visit the Vancouver Craigslist
website and search for one-bedroom apartments,
we should see something similar to {numref}`fig:craigslist-human`.

+++

```{figure} img/reading/craigslist_human.png
:name: fig:craigslist-human

Craigslist webpage of advertisements for one-bedroom apartments.
```

+++

Based on what our browser shows us, it's pretty easy to find the size and price
for each apartment listed. But we would like to be able to obtain that information
using Python, without any manual human effort or copying and pasting. We do this by
examining the *source code* that the web server actually sent our browser to
display for us. We show a snippet of it below; the
entire source
is [included with the code for this book](https://github.com/UBC-DSCI/introduction-to-datascience-python/blob/main/source/data/website_source.txt):

```html
<span class="result-meta">
        <span class="result-price">$800</span>
        <span class="housing">
            1br -
        </span>
        <span class="result-hood"> (13768 108th Avenue)</span>
        <span class="result-tags">
            <span class="maptag" data-pid="6786042973">map</span>
        </span>
        <span class="banish icon icon-trash" role="button">
            <span class="screen-reader-text">hide this posting</span>
        </span>
    <span class="unbanish icon icon-trash red" role="button"></span>
    <a href="#" class="restore-link">
        <span class="restore-narrow-text">restore</span>
        <span class="restore-wide-text">restore this posting</span>
    </a>
    <span class="result-price">$2285</span>
</span>
```

Oof...you can tell that the source code for a web page is not really designed
for humans to understand easily. However, if you look through it closely, you
will find that the information we're interested in is hidden among the muck.
For example, near the top of the snippet
above you can see a line that looks like

```html
<span class="result-price">$800</span>
```

That snippet is definitely storing the price of a particular apartment. With some more
investigation, you should be able to find things like the date and time of the
listing, the address of the listing, and more. So this source code most likely
contains all the information we are interested in!

```{index} HTML; tag
```

Let's dig into that line above a bit more. You can see that
that bit of code has an *opening tag* (words between `<` and `>`, like
`<span>`) and a *closing tag* (the same with a slash, like `</span>`). HTML
source code generally stores its data between opening and closing tags like
these. Tags are keywords that tell the web browser how to display or format
the content. Above you can see that the information we want (`$800`) is stored
between an opening and closing tag (`<span>` and `</span>`). In the opening
tag, you can also see a very useful "class" (a special word that is sometimes
included with opening tags): `class="result-price"`. Since we want R to
programmatically sort through all of the source code for the website to find
apartment prices, maybe we can look for all the tags with the `"result-price"`
class, and grab the information between the opening and closing tag. Indeed,
take a look at another line of the source snippet above:

```html
<span class="result-price">$2285</span>
```

It's yet another price for an apartment listing, and the tags surrounding it
have the `"result-price"` class. Wonderful! Now that we know what pattern we
are looking for&mdash;a dollar amount between opening and closing tags that have the
`"result-price"` class&mdash;we should be able to use code to pull out all of the
matching patterns from the source code to obtain our data. This sort of "pattern"
is known as a *CSS selector* (where CSS stands for **c**ascading **s**tyle **s**heet).

The above was a simple example of "finding the pattern to look for"; many
websites are quite a bit larger and more complex, and so is their website
source code. Fortunately, there are tools available to make this process
easier. For example,
[SelectorGadget](https://selectorgadget.com/) is
an open-source tool that simplifies identifying the generating
and finding of CSS selectors.
At the end of the chapter in the additional resources section, we include a link to
a short video on how to install and use the SelectorGadget tool to
obtain CSS selectors for use in web scraping.
After installing and enabling the tool, you can click the
website element for which you want an appropriate selector. For
example, if we click the price of an apartment listing, we
find that SelectorGadget shows us the selector `.result-price`
in its toolbar, and highlights all the other apartment
prices that would be obtained using that selector ({numref}`fig:sg1`).

```{figure} img/reading/sg1.png
:name: fig:sg1

Using the SelectorGadget on a Craigslist webpage to obtain the CCS selector useful for obtaining apartment prices.
```

If we then click the size of an apartment listing, SelectorGadget shows us
the `span` selector, and highlights many of the lines on the page; this indicates that the
`span` selector is not specific enough to capture only apartment sizes ({numref}`fig:sg3`).

```{figure} img/reading/sg3.png
:name: fig:sg3

Using the SelectorGadget on a Craigslist webpage to obtain a CCS selector useful for obtaining apartment sizes.
```

To narrow the selector, we can click one of the highlighted elements that
we *do not* want. For example, we can deselect the "pic/map" links,
resulting in only the data we want highlighted using the `.housing` selector ({numref}`fig:sg2`).

```{figure} img/reading/sg2.png
:name: fig:sg2

Using the SelectorGadget on a Craigslist webpage to refine the CCS selector to one that is most useful for obtaining apartment sizes.
```

So to scrape information about the square footage and rental price
of apartment listings, we need to use
the two CSS selectors `.housing` and `.result-price`, respectively.
The selector gadget returns them to us as a comma-separated list (here
`.housing , .result-price`), which is exactly the format we need to provide to
Python if we are using more than one CSS selector.

**Caution: are you allowed to scrape that website?**

```{index} web scraping; permission
```

+++

*Before* scraping data from the web, you should always check whether or not
you are *allowed* to scrape it! There are two documents that are important
for this: the `robots.txt` file and the Terms of Service
document. If we take a look at [Craigslist's Terms of Service document](https://www.craigslist.org/about/terms.of.use),
we find the following text: *"You agree not to copy/collect CL content
via robots, spiders, scripts, scrapers, crawlers, or any automated or manual equivalent (e.g., by hand)."*
So unfortunately, without explicit permission, we are not allowed to scrape the website.

```{index} Wikipedia
```

What to do now? Well, we *could* ask the owner of Craigslist for permission to scrape.
However, we are not likely to get a response, and even if we did they would not likely give us permission.
The more realistic answer is that we simply cannot scrape Craigslist. If we still want
to find data about rental prices in Vancouver, we must go elsewhere.
To continue learning how to scrape data from the web, let's instead
scrape data on the population of Canadian cities from Wikipedia.
We have checked the [Terms of Service document](https://foundation.wikimedia.org/wiki/Terms_of_Use/en),
and it does not mention that web scraping is disallowed.
We will use the SelectorGadget tool to pick elements that we are interested in
(city names and population counts) and deselect others to indicate that we are not
interested in them (province names), as shown in {numref}`fig:sg4`.

```{figure} img/reading/sg4.png
:name: fig:sg4

Using the SelectorGadget on a Wikipedia webpage.
```

We include a link to a short video tutorial on this process at the end of the chapter
in the additional resources section. SelectorGadget provides in its toolbar
the following list of CSS selectors to use:

```text
td:nth-child(8) ,
td:nth-child(4) ,
.largestCities-cell-background+ td a
```

Now that we have the CSS selectors that describe the properties of the elements
that we want to target, we can use them to find certain elements in web pages and extract data.


#### Scraping with `BeautifulSoup`

```{index} BeautifulSoup, requests
```

We will use the `requests` and `BeautifulSoup` Python packages to scrape data
from the Wikipedia page. After loading those packages, we tell Python which
page we want to scrape by providing its URL in quotations to the `requests.get`
function. This function obtains the raw HTML of the page, which we then
pass to the `BeautifulSoup` function for parsing:

```{code-cell} ipython3
:tags: ["remove-output"]
import requests
import bs4

wiki = requests.get("https://en.wikipedia.org/wiki/Canada")
page = bs4.BeautifulSoup(wiki.content, "html.parser")
```

```{code-cell} ipython3
:tags: [remove-cell]
import bs4

# the above cell doesn't actually run; this one does run
# and loads the html data from a local, static file

with open("data/canada_wiki.html", "r") as f:
    wiki_hidden = f.read()
page = bs4.BeautifulSoup(wiki_hidden, "html.parser")
```

The `requests.get` function downloads the HTML source code for the page at the
URL you specify, just like your browser would if you navigated to this site.
But instead of displaying the website to you, the `requests.get` function just
returns the HTML source code itself&mdash;stored in the `wiki.content`
variable&mdash;which we then parse using `BeautifulSoup` and store in the
`page` variable. Next, we pass the CSS selectors we obtained from
SelectorGadget to the `select` method of the `page` object.  Make sure to
surround the selectors with quotation marks; `select` expects that argument is
a string. We store the result of the `select` function in the `population_nodes`
variable. Note that `select` returns a list; below we slice the list to
print only the first 5 elements for clarity.

```{code-cell} ipython3
population_nodes = page.select(
    "td:nth-child(8) , td:nth-child(4) , .largestCities-cell-background+ td a"
)
population_nodes[:5]
```

Each of the items in the `population_nodes` list is a *node* from the HTML document that matches the CSS
selectors you specified. A *node* is an HTML tag pair (e.g., `<td>` and `</td>`
which defines the cell of a table) combined with the content stored between the
tags. For our CSS selector `td:nth-child(4)`, an example node that would be
selected would be:

```html
<td style="text-align:left;">
<a href="/wiki/London,_Ontario" title="London, Ontario">London</a>
</td>
```

Next, we extract the meaningful data&mdash;in other words, we get rid of the
HTML code syntax and tags&mdash;from the nodes using the `get_text` function.
In the case of the example node above, `get_text` function returns `"London"`.
Once again we show only the first 5 elements for clarity.

```{code-cell} ipython3
[row.get_text() for row in population_nodes[:5]]
```

Fantastic! We seem to have extracted the data of interest from the raw HTML
source code. But we are not quite done; the data is not yet in an optimal
format for data analysis. Both the city names and population are encoded as
characters in a single vector, instead of being in a data frame with one
character column for city and one numeric column for population (like a
spreadsheet).  Additionally, the populations contain commas (not useful for
programmatically dealing with numbers), and some even contain a line break
character at the end (`\n`). In {numref}`Chapter %s <wrangling>`, we will learn
more about how to *wrangle* data such as this into a more useful format for
data analysis using Python.

+++

#### Scraping with `read_html`

Using `requests` and `BeautifulSoup` to extract data based on CSS selectors is
a very general way to scrape data from the web, albeit perhaps a little bit
complicated.  Fortunately, `pandas` provides the
[`read_html`](https://pandas.pydata.org/docs/reference/api/pandas.read_html.html)
function, which is easier method to try when the data
appear on the webpage already in a tabular format.  The `read_html` function takes one
argument&mdash;the URL of the page to scrape&mdash;and will return a list of
data frames corresponding to all the tables it finds at that URL. We can see
below that `read_html` found 17 tables on the Wikipedia page for Canada.

```{index} read function; read_html
```

```{code-cell} ipython3
:tags: ["remove-output"]
canada_wiki_tables = pd.read_html("https://en.wikipedia.org/wiki/Canada")
len(canada_wiki_tables)
```

```{code-cell} ipython3
:tags: [remove-input]
canada_wiki_tables = pd.read_html("data/canada_wiki.html")
len(canada_wiki_tables)
```

After manually searching through these, we find that the table containing the
population counts of the largest metropolitan areas in Canada is contained in
index 1. We use the `droplevel` method to simplify the column names in the resulting
data frame:

```{code-cell} ipython3
canada_wiki_df = canada_wiki_tables[1]
canada_wiki_df.columns = canada_wiki_df.columns.droplevel()
canada_wiki_df
```

Once again, we have managed to extract the data of interest from the raw HTML
source code&mdash;but this time using the convenient `read_html` function,
without needing to explicitly use CSS selectors! However, once again, we still
need to do some cleaning of this result. Referring back to {numref}`fig:sg4`,
we can see that the table is formatted with two sets of columns (e.g., `Name`
and `Name.1`) that we will need to somehow merge. In {numref}`Chapter %s
<wrangling>`, we will learn more about how to *wrangle* data into a useful
format for data analysis.

### Using an API

```{index} API
```

Rather than posting a data file at a URL for you to download, many websites
these days provide an API that can be accessed through a programming language
like Python. The benefit of using an API is that data owners have much more control
over the data they provide to users. However, unlike web scraping, there is no
consistent way to access an API across websites. Every website typically has
its own API designed especially for its own use case. Therefore we will just
provide one example of accessing data through an API in this book, with the
hope that it gives you enough of a basic idea that you can learn how to use
another API if needed. In particular, in this book we will show you the basics
of how to use the `requests` package in Python to access data from the NASA "Astronomy Picture
of the Day" API (a great source of desktop backgrounds, by the way&mdash;take a look at the stunning
picture of the Rho-Ophiuchi cloud complex {cite:p}`rhoophiuchi` in {numref}`fig:NASA-API-Rho-Ophiuchi` from July 13, 2023!).

```{index} requests, NASA, API; token
```

```{figure} img/reading/NASA-API-Rho-Ophiuchi.png
:name: fig:NASA-API-Rho-Ophiuchi
:width: 400px

The James Webb Space Telescope's NIRCam image of the Rho Ophiuchi molecular cloud complex.
```

+++

First, you will need to visit the [NASA APIs page](https://api.nasa.gov/) and generate an API key (i.e., a password used to identify you when accessing the API).
Note that a valid email address is required to
associate with the key. The signup form looks something like {numref}`fig:NASA-API-signup`.
After filling out the basic information, you will receive the token via email.
Make sure to store the key in a safe place, and keep it private.


```{figure} img/reading/NASA-API-signup.png
:name: fig:NASA-API-signup

Generating the API access token for the NASA API.
```

**Caution: think about your API usage carefully!**

When you access an API, you are initiating a transfer of data from a web server
to your computer. Web servers are expensive to run and do not have infinite resources.
If you try to ask for *too much data* at once, you can use up a huge amount of the server's bandwidth.
If you try to ask for data *too frequently*&mdash;e.g., if you
make many requests to the server in quick succession&mdash;you can also bog the server down and make
it unable to talk to anyone else. Most servers have mechanisms to revoke your access if you are not
careful, but you should try to prevent issues from happening in the first place by being extra careful
with how you write and run your code. You should also keep in mind that when a website owner
grants you API access, they also usually specify a limit (or *quota*) of how much data you can ask for.
Be careful not to overrun your quota! So *before* we try to use the API, we will first visit
[the NASA website](https://api.nasa.gov/) to see what limits we should abide by when using the API.
These limits are outlined in {numref}`fig:NASA-API-limits`.

```{figure} img/reading/NASA-API-limits.png
:name: fig:NASA-API-limits

The NASA website specifies an hourly limit of 1,000 requests.
```

After checking the NASA website, it seems like we can send at most 1,000 requests per hour.
That should be more than enough for our purposes in this section.

+++

#### Accessing the NASA API

```{index} API; HTTP, API; query parameters, API; endpoint
```

The NASA API is what is known as an *HTTP API*: this is a particularly common
kind of API, where you can obtain data simply by accessing a
particular URL as if it were a regular website.  To make a query to the NASA
API, we need to specify three things.  First, we specify the URL *endpoint* of
the API, which is simply a URL that helps the remote server understand which
API you are trying to access. NASA offers a variety of APIs, each with its own
endpoint; in the case of the NASA "Astronomy Picture of the Day" API, the URL
endpoint is `https://api.nasa.gov/planetary/apod`. Second, we write `?`, which denotes that a
list of *query parameters* will follow. And finally, we specify a list of
query parameters of the form `parameter=value`, separated by `&` characters.  The NASA
"Astronomy Picture of the Day" API accepts the parameters shown in
{numref}`fig:NASA-API-parameters`.

```{figure} img/reading/NASA-API-parameters.png
:name: fig:NASA-API-parameters

The set of parameters that you can specify when querying the NASA "Astronomy Picture of the Day" API,
along with syntax, default settings, and a description of each.
```

So for example, to obtain the image of the day
from July 13, 2023, the API query would have two parameters: `api_key=YOUR_API_KEY`
and `date=2023-07-13`. Remember to replace `YOUR_API_KEY` with the API key you
received from NASA in your email! Putting it all together, the query will look like the following:
```
https://api.nasa.gov/planetary/apod?api_key=YOUR_API_KEY&date=2023-07-13
```
If you try putting this URL into your web browser, you'll actually find that the server
responds to your request with some text:

```json
{"date":"2023-07-13","explanation":"A mere 390 light-years away, Sun-like stars
and future planetary systems are forming in the Rho Ophiuchi molecular cloud
complex, the closest star-forming region to our fair planet. The James Webb
Space Telescope's NIRCam peered into the nearby natal chaos to capture this
infrared image at an inspiring scale. The spectacular cosmic snapshot was
released to celebrate the successful first year of Webb's exploration of the
Universe. The frame spans less than a light-year across the Rho Ophiuchi region
and contains about 50 young stars. Brighter stars clearly sport Webb's
characteristic pattern of diffraction spikes. Huge jets of shocked molecular
hydrogen blasting from newborn stars are red in the image, with the large,
yellowish dusty cavity carved out by the energetic young star near its center.
Near some stars in the stunning image are shadows cast by their protoplanetary
disks.","hdurl":"https://apod.nasa.gov/apod/image/2307/STScI-01_RhoOph.png",
"media_type":"image","service_version":"v1","title":"Webb's
Rho Ophiuchi","url":"https://apod.nasa.gov/apod/image/2307/STScI-01_RhoOph1024.png"}
```

```{index} see: JavaScript Object Notation; JSON
```

```{index} JSON, requests; get, requests; json
```

Neat! There is definitely some data there, but it's a bit hard to
see what it all is. As it turns out, this is a common format for data called
*JSON* (JavaScript Object Notation). We won't encounter this kind of data much in this book,
but for now you can interpret this data just like
you'd interpret a Python dictionary: these are `key : value` pairs separated by
commas. For example, if you look closely, you'll see that the first entry is
`"date":"2023-07-13"`, which indicates that we indeed successfully received
data corresponding to July 13, 2023.

So now our job is to do all of this programmatically in Python. We will load
the `requests` package, and make the query using the `get` function, which takes a single URL argument;
you will recognize the same query URL that we pasted into the browser earlier.
We will then obtain a JSON representation of the
response using the `json` method.

<!-- we have disabled the below code for reproducibility, with hidden setting
of the nasa_data object. But you can reproduce this using the DEMO_KEY key -->
```{code-cell} ipython3
:tags: ["remove-output"]
import requests

nasa_data_single = requests.get(
    "https://api.nasa.gov/planetary/apod?api_key=YOUR_API_KEY&date=2023-07-13"
).json()
nasa_data_single
```

```{code-cell} ipython3
:tags: [remove-input]
import json
with open("data/nasa.json", "r") as f:
    nasa_data = json.load(f)
# the last entry in the stored data is July 13, 2023, so print that
nasa_data[-1]
```

We can obtain more records at once by using the `start_date` and `end_date` parameters, as
shown in the table of parameters in {numref}`fig:NASA-API-parameters`.
Let's obtain all the records between May 1, 2023, and July 13, 2023, and store the result
in an object called `nasa_data`; now the response
will take the form of a Python list. Each item in the list will correspond to a single day's record (just like the `nasa_data_single` object),
and there will be 74 items total, one for each day between the start and end dates:

```{code-cell} ipython3
:tags: ["remove-output"]
nasa_data = requests.get(
    "https://api.nasa.gov/planetary/apod?api_key=YOUR_API_KEY&start_date=2023-05-01&end_date=2023-07-13"
    ).json()
len(nasa_data)
```

```{code-cell} ipython3
:tags: [remove-input]
# need to secretly re-load the nasa data again because the above running code destroys it
# see PR 341 for why we need to do things this way (essentially due to PDF build)
with open("data/nasa.json", "r") as f:
    nasa_data = json.load(f)
len(nasa_data)
```

For further data processing using the techniques in this book, you'll need to turn this list of dictionaries
into a `pandas` data frame. Here we will extract the `date`, `title`, `copyright`, and `url` variables
from the JSON data, and construct a `pandas` DataFrame using the extracted information.

```{note}
Understanding this code is not required for the remainder of the textbook. It is included for those
readers who would like to parse JSON data into a `pandas` data frame in their own data analyses.
```

```{code-cell} ipython3
data_dict = {
    "date":[],
    "title": [],
    "copyright" : [],
    "url": []
}

for item in nasa_data:
    if "copyright" not in item:
        item["copyright"] = None
    for entry in ["url", "title", "date", "copyright"]:
        data_dict[entry].append(item[entry])

nasa_df = pd.DataFrame(data_dict)
nasa_df
```

Success&mdash;we have created a small data set using the NASA
API! This data is also quite different from what we obtained from web scraping;
the extracted information is readily available in a JSON format, as opposed to raw
HTML code (although not *every* API will provide data in such a nice format).
From this point onward, the `nasa_df` data frame is stored on your
machine, and you can play with it to your heart's content. For example, you can use
`pandas.to_csv` to save it to a file and `pandas.read_csv` to read it into Python again later;
and after reading the next few chapters you will have the skills to
do even more interesting things! If you decide that you want
to ask any of the various NASA APIs for more data
(see [the list of awesome NASA APIS here](https://api.nasa.gov/)
for more examples of what is possible), just be mindful as usual about how much
data you are requesting and how frequently you are making requests.

+++

## Exercises

Practice exercises for the material covered in this chapter
can be found in the accompanying
[worksheets repository](https://worksheets.python.datasciencebook.ca)
in the "Reading in data locally and from the web" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in {numref}`Chapter %s <move-to-your-own-machine>`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

## Additional resources

- The [`pandas` documentation](https://pandas.pydata.org/docs/getting_started/index.html)
  provides the documentation for the functions we cover in this chapter.
  It is where you should look if you want to learn more about these functions, the
  full set of arguments you can use, and other related functions.
- Sometimes you might run into data in such poor shape that the reading
  functions we cover in this chapter do not work. In that case, you can consult the
  [data loading chapter](https://wesmckinney.com/book/accessing-data.html#io_flat_files)
  from [*Python for Data Analysis*](https://wesmckinney.com/book/) {cite:p}`mckinney2012python`, which goes into a lot
  more detail about how Python parses text from files into data frames.
- A [video](https://www.youtube.com/embed/ephId3mYu9o) from the Udacity
  course *Linux Command Line Basics* provides a good explanation of absolute versus relative paths.
- If you read the subsection on obtaining data from the web via scraping and
  APIs, we provide two companion tutorial video links for how to use the
  SelectorGadget tool to obtain desired CSS selectors for:
    - [extracting the data for apartment listings on Craigslist](https://www.youtube.com/embed/YdIWI6K64zo), and
    - [extracting Canadian city names and populations from Wikipedia](https://www.youtube.com/embed/O9HKbdhqYzk).

+++

## References

```{bibliography}
:filter: docname in docnames
```
