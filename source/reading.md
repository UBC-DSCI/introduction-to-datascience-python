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

- Define the following:
    - absolute file path
    - relative file path
    - **U**niform **R**esource **L**ocator (URL)
- Read data into Python using an absolute path, relative path and a URL.
- Compare and contrast the following functions:
    - `read_csv`
    - `read_excel`
- Match the following `pandas` `read_csv` function arguments to their descriptions:
    - `filepath_or_buffer`
    - `sep`
    - `names`
    - `skiprows`
- Choose the appropriate `read_csv` function arguments to load a given plain text tabular data set into Python.
- Use the `rename` function to rename columns in a data frame.
- Use `pandas` package's `read_excel` function and arguments to load a sheet from an excel file into Python.
- Connect to a database using the `ibis` library's `connect` function.
- List the tables in a database using the `ibis` library's `list_tables` function.
- Create a reference to a database table using the `ibis` library's `table` function.
- Execute queries to bring data from a database into Python using the `ibis` library's `execute` function.
- Use `to_csv` to save a data frame to a `.csv` file.
- (*Optional*) Obtain data using **a**pplication **p**rogramming **i**nterfaces (APIs) and web scraping.
    - Read/scrape data from an internet URL using the `BeautifulSoup` package
    - Compare downloading tabular data from a plain text file (e.g. `.csv`) from the web versus scraping data from a `.html` file

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

The place where the file lives on your computer is called the "path". You can
think of the path as directions to the file. There are two kinds of paths:
*relative* paths and *absolute* paths. A relative path is where the file is
with respect to where you currently are on the computer (e.g., where the file
you're working in is). On the other hand, an absolute path is where the file is
in respect to the computer's filesystem base (or root) folder.

```{index} Happiness Report
```

Suppose our computer's filesystem looks like the picture in
{numref}`Filesystem`, and we are working in a
file titled `worksheet_02.ipynb`. If we want to
read the `.csv` file named `happiness_report.csv` into Python, we could do this
using either a relative or an absolute path.  We show both choices
below.

```{figure} img/reading/filesystem.jpeg
---
height: 500px
name: Filesystem
---
Example file system
```


**Reading `happiness_report.csv` using a relative path:**

+++

```python
happy_data = pd.read_csv("data/happiness_report.csv")
```

+++

**Reading `happiness_report.csv` using an absolute path:**

+++

```python
happy_data = pd.read_csv("/home/dsci-100/worksheet_02/data/happiness_report.csv")
```

+++

So which one should you use? Generally speaking, to ensure your code can be run
on a different computer, you should use relative paths. An added bonus is that
it's also less typing! Generally, you should use relative paths because the file's
absolute path (the names of
folders between the computer's root `/` and the file) isn't usually the same
across different computers. For example, suppose Fatima and Jayden are working on a
project together on the `happiness_report.csv` data. Fatima's file is stored at

```
/home/Fatima/project/data/happiness_report.csv
```

while Jayden's is stored at

```
/home/Jayden/project/data/happiness_report.csv
```

Even though Fatima and Jayden stored their files in the same place on their
computers (in their home folders), the absolute paths are different due to
their different usernames.  If Jayden has code that loads the
`happiness_report.csv` data using an absolute path, the code won't work on
Fatima's computer.  But the relative path from inside the `project` folder
(`data/happiness_report.csv`) is the same on both computers; any code that uses
relative paths will work on both!

```{index} URL
```

Your file could be stored locally, as we discussed, or it could also be
somewhere on the internet (remotely). For this purpose we use a
*Uniform Resource Locator (URL)*, i.e., a web address that looks something
like https://google.com/. URLs indicate the location of a resource on the internet and
helps us retrieve that resource.

## Reading tabular data from a plain text file into Python

(readcsv)=
### `read_csv` to read in comma-separated values files

```{index} csv, reading; separator, read function; read\_csv
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

```python
canlang_data = pd.read_csv("data/can_lang_meta-data.csv")
```
```text
ParserError: Error tokenizing data. C error: Expected 1 fields in line 4, saw 6
```

```{index} Error
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

```{index} tsv, read function; read_tsv
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

```{index} read function; header, reading; separator
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
canlang_data =  pd.read_csv(
    "data/can_lang_no_names.tsv",
    sep = "\t",
    header = None
)
canlang_data
```

```{index} pandas.DataFrame; rename, pandas
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
canlang_data_renamed = canlang_data.rename(columns = col_map)
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

```{index} database; connect, ibis, ibis; ibis
```

```{index} see: ibis; database
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

```{index} database; tables; list_tables
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

```{index} database; table, ibis; table
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

```{index} database; count, ibis; count
```

Although it looks like we might have obtained the whole data frame from the database, we didn't!
It's a *reference*; the data is still stored only in the SQLite database. The `canlang_table` object
is an `AlchemyTable` (`ibis` is using `sqlalchemy` under the hood!), which, when printed, tells
you which columns are available in the table. But unlike a usual `pandas` data frame,
we do not immediately know how many rows are in the table. In order to find out how many
rows there are, we have to send an SQL *query* (i.e., command) to the data base.
In `ibis`, we can do that using the `count` function from the table object.

```{code-cell} ipython3
canlang_table.count()
```

```{index} execute, ibis; execute
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

```{index} compile, ibis; compile
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
For example, we can look at the first few rows of the table by using the `head` function---and
we won't forget to `execute` to see the result!

```{index} database; head, ibis;
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

```{index} database; filter, ibis;
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

```{index} database; select, ibis;
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

```{index} pandas.DataFrame; tail
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

```python
conn = ibis.postgres.connect(
    database = "can_mov_db",
    host = "fakeserver.stat.ubc.ca",
    port = 5432,
    user = "user0001",
    password = "abc123"
)
```

Aside from needing to provide that additional information, `ibis` makes it so
that connecting to and working with a Postgres database is identical to
connecting to and working with an SQLite database. For example, we can again use
`list_tables` to find out what tables are in the `can_mov_db` database:

```python
conn.list_tables()
```

```text
["themes", "medium", "titles", "title_aliases", "forms", "episodes", "names", "names_occupations", "occupation", "ratings"]
```

We see that there are 10 tables in this database. Let's first look at the
`"ratings"` table to find the lowest rating that exists in the `can_mov_db`
database.

```python
ratings_table = conn.table("ratings")
ratings_table
```

```text
AlchemyTable: ratings
  title           string
  average_rating  float64
  num_votes       int64
```

```{index} ibis; select
```

To find the lowest rating that exists in the data base, we first need to
select the `average_rating` column:

```python
avg_rating = ratings_table[["average_rating"]]
avg_rating
```

```text
r0 := AlchemyTable: ratings
  title           string
  average_rating  float64
  num_votes       int64

Selection[r0]
  selections:
    average_rating: r0.average_rating
```

```{index} database; order_by, ibis; head, ibis; ibis
```

Next we use the `order_by` function from `ibis` order the table by `average_rating`,
and then the `head` function to select the first row (i.e., the lowest score).

```python
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

```{index} write function; to_csv, pandas.DataFrame; to_csv
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
URL using the `read_csv` function from `pandas`. But as time goes
on, it is increasingly uncommon to find data (especially large amounts of data)
in this format available for download from a URL. Instead, websites now often
offer something known as an **a**pplication **p**rogramming **i**nterface
(API), which
provides a programmatic way to ask for subsets of a data set. This allows the
website owner to control *who* has access to the data, *what portion* of the
data they have access to, and *how much* data they can access.  Sometimes, the
website owner will give you a *token* (a secret string of characters somewhat
like a password) that you have to provide when accessing the API.

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
**h**yper**t**ext **m**arkup **l**anguage
(HTML)
and **c**ascading **s**tyle **s**heet (CSS) code&mdash;and process it
to extract useful information. HTML provides the
basic structure of a site and tells the webpage how to display the content
(e.g., titles, paragraphs, bullet lists etc.), whereas CSS helps style the
content and tells the webpage how the HTML elements should
be presented (e.g., colors, layouts, fonts etc.).

This subsection will show you the basics of both web scraping
with the [`BeautifulSoup` Python package](https://beautiful-soup-4.readthedocs.io/en/latest/) {cite:p}`beautifulsoup`
and accessing the Twitter API
using the [`tweepy` Python package](https://github.com/tweepy/tweepy) {cite:p}`tweepy`.

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

That is definitely storing the price of a particular apartment. With some more
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

**Stop! Are you allowed to scrape that website?**

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

+++

```text
td:nth-child(8) ,
td:nth-child(6) ,
td:nth-child(4) ,
.mw-parser-output div tr+ tr td:nth-child(2)
```

+++

Now that we have the CSS selectors that describe the properties of the elements
that we want to target (e.g., has a tag name `price`), we can use them to find
certain elements in web pages and extract data.

+++

**Using `pandas.read_html`**

+++

The easiest way to read a table from HTML is to use [`pandas.read_html`](https://pandas.pydata.org/docs/reference/api/pandas.read_html.html). We can see that the Wikipedia page of "Canada" has 18 tables.

```{code-cell} ipython3
:tags: [remove-output]

canada_wiki = pd.read_html("https://en.wikipedia.org/wiki/Canada")
len(canada_wiki)
```

```
18
```

+++

With some inspection, we find that the table that shows the population of the most populated provinces is of index 1.

```{code-cell} ipython3
:tags: [remove-output]

df = canada_wiki[1]
df.columns = df.columns.droplevel()
df
```

```{code-cell} ipython3
:tags: [remove-input]

df = pd.read_csv("data/canada-wiki-read_html.csv", index_col=0)
df
```

**Using `BeautifulSoup`**

```{index} BeautifulSoup, requests
```

Now that we have our CSS selectors we can use the `requests` and `BeautifulSoup` Python packages to scrape our desired data from the website. We start by loading the packages:

```{code-cell} ipython3
import requests
import bs4
```

Next, we tell Python what page we want to scrape by providing the webpage's URL in quotations to the function `requests.get` and pass it into the `BeautifulSoup` function for parsing:

```{code-cell} ipython3
wiki = requests.get("https://en.wikipedia.org/wiki/Canada")
page = BeautifulSoup(wiki.content, "html.parser")
```

The `requests.get` function sends a `GET` request to the specified URL and returns the server's response to the HTTP request (*i.e.* a `requests.Response` object). The `BeautifulSoup` function takes the content of the response and returns the HTML source code itself, which we have
stored in the `page` variable. Next, we use the `select` method of the page object along with the CSS selectors we obtained from the SelectorGadget tool. Make sure to surround the selectors with quotation marks; `select` expects that
argument is a string. It selects *nodes* from the HTML document that
match the CSS selectors you specified. A *node* is an HTML tag pair (e.g.,
`<td>` and `</td>` which defines the cell of a table) combined with the content
stored between the tags. For our CSS selector `td:nth-child(6)`, an example
node that would be selected would be:

+++

```
<td style="text-align:left;background:#f0f0f0;">
<a href="/wiki/London,_Ontario" title="London, Ontario">London</a>
</td>
```

+++

We store the result of the `select` function in the `population_nodes` variable. Note that it returns a list, and we slice the list to only print the first 5 elements.

```{code-cell} ipython3
:tags: [remove-output]

population_nodes = page.select(
    "td:nth-child(8) , td:nth-child(6) , td:nth-child(4) , .mw-parser-output div td:nth-child(2)"
)
population_nodes[:5]
```

```
[<td style="text-align:left;"><a href="/wiki/Greater_Toronto_Area" title="Greater Toronto Area">Toronto</a></td>,
 <td style="text-align:right;">6,202,225</td>,
 <td style="text-align:left;"><a href="/wiki/London,_Ontario" title="London, Ontario">London</a></td>,
 <td style="text-align:right;">543,551
 </td>,
 <td style="text-align:left;"><a href="/wiki/Greater_Montreal" title="Greater Montreal">Montreal</a></td>]
```

+++

Next we extract the meaningful data&mdash;in other words, we get rid of the HTML code syntax and tags&mdash;from
the nodes using the `get_text`
function. In the case of the example
node above, `get_text` function returns `"London"`.

```{code-cell} ipython3
:tags: [remove-output]

[row.get_text() for row in population_nodes][:5]
```

```
["Toronto", "6,202,225", "London", "543,551\n", "Montreal"]
```

+++

Fantastic! We seem to have extracted the data of interest from the
raw HTML source code. But we are not quite done; the data
is not yet in an optimal format for data analysis. Both the city names and
population are encoded as characters in a single vector, instead of being in a
data frame with one character column for city and one numeric column for
population (like a spreadsheet).
Additionally, the populations contain commas (not useful for programmatically
dealing with numbers), and some even contain a line break character at the end
(`\n`). In {numref}`Chapter %s <wrangling>`, we will learn more about how to *wrangle* data
such as this into a more useful format for data analysis using Python.

+++

### Using an API

```{index} API
```

Rather than posting a data file at a URL for you to download, many websites these days
provide an API that must be accessed through a programming language like Python. The benefit of this
is that data owners have much more control over the data they provide to users. However, unlike
web scraping, there is no consistent way to access an API across websites. Every website typically
has its own API designed especially for its own use case. Therefore we will just provide one example
of accessing data through an API in this book, with the hope that it gives you enough of a basic
idea that you can learn how to use another API if needed.

```{index} API; tweepy, tweepy, Twitter, API; token
```

+++

In particular, in this book we will show you the basics of how to use
the `tweepy` package in Python to access
data from the Twitter API. `tweepy` requires the [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard) and you will need to get tokens and secrets from that, through which your access to the data will then be authenticated and controlled.

+++

First, we go to the [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard) and sign up an account if you do not have one yet. Note that you will need a valid phone number to associate with your developer account. After filling out the basic information, we will get the *essential access* to the Twitter API. Then we can create an app and hit the "get key" button, and we will get the API key and API key secret of the app (along with the bearer token which will not be used in this demonstration). **We need to store the key and secret at a safe place, and make sure do not show them to anyone else (also do not accidentally push it to the GitHub repository).** If you lose the key, you can always regenerate it. Next, we go to the "Keys and tokens" tab of the app, and generate an access token and an access token secret. **Save the access token and the access token secret at a safe place as well.** Your app will look something like {numref}`fig:twitter-API-keys-tokens`.

+++

```{figure} img/reading/twitter-API-keys-tokens.png
:name: fig:twitter-API-keys-tokens

Generating the API key-secret pair and the access token-secret pair in Twitter API.
```

+++

Once you get the access keys and secrets, you can follow along with the examples that we show here.
To get started, load the `tweepy` package and authenticate our access to the Twitter developer portal account.

```{code-cell} ipython3
:tags: [remove-output]

import tweepy

# replace these with the api key, api key secret, access token and access token secret
# generated on your own
api_key = "8OxHWiIWjy8M39LvnC8OfSXrj"
api_key_secret = "scqjRqX5stoy4pYB5Zu52tCBKzhGLDh5nRqTEM6CMoLRkRLR8F"

access_token = "1556029189484007425-mYwaDCI1WnCxjuMt0jb2UYD2ns8BYB"
access_token_secret = "pDG4Ta7giYLY3mablPhd6y9bB5y2Aer1Cn18rihIJFBB7"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Successful Authentication")
except:
    print("Failed authentication")
```

```
Successful Authentication
```

+++

`tweepy` provides an extensive set of functions to search
Twitter for tweets, users, their followers, and more.
Let's construct a small data set of the last 200 tweets and
retweets from the [@scikit_learn](https://twitter.com/scikit_learn) account. A few of the most recent tweets
are shown in {numref}`fig:01-scikit-learn-twitter`.

+++

```{figure} img/reading/scikit-learn-twitter.png
:name: fig:01-scikit-learn-twitter

The `scikit-learn` account Twitter feed.
```

+++

**Stop! Think about your API usage carefully!**

When you access an API, you are initiating a transfer of data from a web server
to your computer. Web servers are expensive to run and do not have infinite resources.
If you try to ask for *too much data* at once, you can use up a huge amount of the server's bandwidth.
If you try to ask for data *too frequently*&mdash;e.g., if you
make many requests to the server in quick succession&mdash;you can also bog the server down and make
it unable to talk to anyone else. Most servers have mechanisms to revoke your access if you are not
careful, but you should try to prevent issues from happening in the first place by being extra careful
with how you write and run your code. You should also keep in mind that when a website owner
grants you API access, they also usually specify a limit (or *quota*) of how much data you can ask for.
Be careful not to overrun your quota! In this example, we should take a look at
 [the Twitter website](https://developer.twitter.com/en/docs/twitter-api/rate-limits) to see what limits
we should abide by when using the API.

+++

**Using `tweepy`**

After checking the Twitter website, it seems like asking for 200 tweets one time is acceptable.
So we can use the `user_timeline` function to ask for the last 200 tweets from the [@scikit_learn](https://twitter.com/scikit_learn) account.

```{code-cell} ipython3
:tags: [remove-output]

userID = "scikit_learn"

scikit_learn_tweets = api.user_timeline(
    screen_name=userID,
    count=200,
    include_rts=True,
    tweet_mode="extended",
)
```

Let's take a look at the first 3 most recent tweets of [@scikit_learn](https://twitter.com/scikit_learn) through accessing the attributes of tweet data dictionary:

```{code-cell} ipython3
:tags: [remove-output]

for info in scikit_learn_tweets[:3]:
    print("ID: {}".format(info.id))
    print(info.created_at)
    print(info.full_text)
    print("\n")
```

```
ID: 1555686128971403265
2022-08-05 22:44:11+00:00
scikit-learn 1.1.2 is out on https://t.co/lSpi4eDc2t and conda-forge!

This is a small maintenance release that fixes a couple of regressions:
https://t.co/Oa84ES0qpG


ID: 1549321048943988737
2022-07-19 09:11:37+00:00
RT @MarenWestermann: @scikit_learn It is worth highlighting that this scikit-learn sprint is seeing the highest participation of women out…


ID: 1548339716465930244
2022-07-16 16:12:09+00:00
@StefanieMolin @theBodlina @RichardKlima We continue pulling requests here in Dublin. Putting some Made in Ireland code in the scikit-learn codebase 🇮🇪 . Current stats: 18 PRs opened, 12 merged 🚀 https://t.co/ccWy8vh8YI
```

+++

A full list of available attributes provided by Twitter API can be found [here](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet).

+++

For the demonstration purpose, let's only use a
few variables of interest: `created_at`,  `user.screen_name`, `retweeted`,
and `full_text`, and construct a `pandas` DataFrame using the extracted information.

```{code-cell} ipython3
:tags: [remove-output]

columns = ["time", "user", "is_retweet", "text"]
data = []
for tweet in scikit_learn_tweets:
    data.append(
        [tweet.created_at, tweet.user.screen_name, tweet.retweeted, tweet.full_text]
    )

scikit_learn_tweets_df = pd.DataFrame(data, columns=columns)
scikit_learn_tweets_df
```

```{code-cell} ipython3
:tags: [remove-input]

scikit_learn_tweets_df = pd.read_csv("data/reading_api_df.csv", index_col=0)
scikit_learn_tweets_df
```

If you look back up at the image of the [@scikit_learn](https://twitter.com/scikit_learn) Twitter page, you will
recognize the text of the most recent few tweets in the above data frame.  In
other words, we have successfully created a small data set using the Twitter
API&mdash;neat! This data is also quite different from what we obtained from web scraping;
the extracted information can be easily converted into a `pandas` data frame (although not *every* API will provide data in such a nice format).
From this point onward, the `scikit_learn_tweets_df` data frame is stored on your
machine, and you can play with it to your heart's content. For example, you can use
`pandas.to_csv` to save it to a file and `pandas.read_csv` to read it into Python again later;
and after reading the next few chapters you will have the skills to
compute the percentage of retweets versus tweets, find the most oft-retweeted
account, make visualizations of the data, and much more! If you decide that you want
to ask the Twitter API for more data
(see [the `tweepy` page](https://github.com/tweepy/tweepy)
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
    - [extracting Canadian city names and 2016 populations from Wikipedia](https://www.youtube.com/embed/O9HKbdhqYzk).

## References

+++

```{bibliography}
:filter: docname in docnames
```
