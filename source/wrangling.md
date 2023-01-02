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

(wrangling)=
# Cleaning and wrangling data

```{code-cell} ipython3
:tags: [remove-cell]

import altair as alt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 20)

from myst_nb import glue
```

## Overview

This chapter is centered around defining tidy data&mdash;a data format that is
suitable for analysis&mdash;and the tools needed to transform raw data into this
format. This will be presented in the context of a real-world data science
application, providing more practice working through a whole case study.

+++

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

  - Define the term "tidy data".
  - Discuss the advantages of storing data in a tidy data format.
  - Define what series and data frames are in Python, and describe how they relate to
    each other.
  - Describe the common types of data in Python and their uses.
  - Recall and use the following functions for their
    intended data wrangling tasks:
      - `.agg`
      - `.apply`
      - `.assign`
      - `.groupby`
      - `.melt`
      - `.pivot`
      - `.str.split`
  - Recall and use the following operators for their
    intended data wrangling tasks:
      - `==`
      - `in`
      - `and`
      - `or`
      - `[]`
      - `.loc[]`
      - `.iloc[]`

## Data frames and series

In the chapters on {ref}`intro` and {ref}`reading`, *data frames* were the focus:
we learned how to import data into Python as a data frame, and perform basic operations on data frames in Python.
In the remainder of this book, this pattern continues. The vast majority of tools we use will require
that data are represented as a `pandas` **data frame** in Python. Therefore, in this section,
we will dig more deeply into what data frames are and how they are represented in Python.
This knowledge will be helpful in effectively utilizing these objects in our data analyses.

+++

### What is a data frame?

```{index} data frame; definition
```

```{index} pandas.DataFrame
```

A data frame is a table-like structure for storing data in Python. Data frames are
important to learn about because most data that you will encounter in practice
can be naturally stored as a table.  In order to define data frames precisely,
we need to introduce a few technical terms:

```{index} variable, observation, value
```

- **variable:** a characteristic, number, or quantity that can be measured.
- **observation:** all of the measurements for a given entity.
- **value:** a single measurement of a single variable for a given entity.

Given these definitions, a **data frame** is a tabular data structure in Python
that is designed to store observations, variables, and their values.
Most commonly, each column in a data frame corresponds to a variable,
and each row corresponds to an observation. For example,
{numref}`fig:02-obs` displays a data set of city populations. Here, the variables
are "region, year, population"; each of these are properties that can be
collected or measured.  The first observation is "Toronto, 2016, 2235145";
these are the values that the three variables take for the first entity in the
data set. There are 13 entities in the data set in total, corresponding to the
13 rows in {numref}`fig:02-obs`.

+++

```{figure} img/data_frame_slides_cdn/data_frame_slides_cdn.004.jpeg
:name: fig:02-obs
:figclass: caption-hack

A data frame storing data regarding the population of various regions in Canada. In this example data frame, the row that corresponds to the observation for the city of Vancouver is colored yellow, and the column that corresponds to the population variable is colored blue.
```

### What is a series?

```{index} pandas.Series
```

In Python, `pandas` **series** are are objects that can contain one or more elements (like a list).
`pandas` stores each of the columns of a data frame as a `Series` object.
They are a single column, are ordered and can be indexed.
They can contain any data type: integers, strings, floats, etc. `Series` can contain a
mix of data types, but it is good practice to only include a single type in a series
because all observations of one variable should be the same type.
Python
has several different basic data types, as shown in
{numref}`tab:datatype-table`.
You can create a `pandas` series using the
`pd.Series()` function.  For example, to create the series `region` as shown
in {numref}`fig:02-series`, you can write the following.

```{code-cell} ipython3
import pandas as pd

region = pd.Series(["Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa"])
region
```

**(FIGURE 14 NEEDS UPDATING: (a) ZERO-BASED INDEXING, (b) TYPE SHOULD BE STRING (NOT CHARACTER))**

+++ {"tags": []}

```{figure} img/wrangling/pandas_dataframe_series.png
:name: fig:02-series
:figclass: caption-hack

Example of a `pandas` series whose type is string.
```


```{code-cell} ipython3
:tags: [remove-cell]

# The following table was taken from DSCI511 Lecture 1, credit to Arman Seyed-Ahmadi, MDS 2021
```

```{index} data types, string, integer, floating point number, boolean, list, set, dictionary, tuple, none
```

```{index} see: str; string
```

```{index} see: int; integer
```

```{index} see: float; floating point number
```

```{index} see: bool; boolean
```

```{index} see: NoneType; none
```

```{index} see: dict; dictionary
```

```{table} Basic data types in Python
:name: tab:datatype-table
| Data type             | Abbreviation | Description                                   | Example                                    |
| :-------------------- | :----------- | :-------------------------------------------- | :----------------------------------------- |
| integer               | `int`        | positive/negative whole numbers               | `42`                                       |
| floating point number | `float`      | real number in decimal form                   | `3.14159`                                  |
| boolean               | `bool`       | true or false                                 | `True`                                     |
| string                | `str`        | text                                          | `"Hello World"`                            |
| none                  | `NoneType`   | represents no value                           | `None`                                     |
```

+++

It is important in Python to make sure you represent your data with the correct type.
Many of the `pandas` functions we use in this book treat
the various data types differently. You should use integers and float types
to represent numbers and perform arithmetic. Integers have no decimals while floats contain decimals.
Booleans are variables that can only take on one of two values: True or False.
Strings are used to represent data that should
be thought of as "text", such as words, names, paths, URLs, and more.
A `NoneType` is a special type in Python that is used to indicate no-value; this can occur when
you have missing data.
There are other basic data types in Python, such as *set*
and *complex*, but we do not use these in this textbook.


### What does this have to do with data frames?

+++

```{index} data frame; definition
```

A data frame is really just a collection of series that are stuck together; it follow two rules:

1. Each element itself is a series.
2. Each element (series) must have the same length.

Not all columns in a data frame need to be of the same type.
{numref}`fig:02-dataframe` shows a data frame where
the columns are series of different types. But each element in
one column should usually be the same type, since the values for a single variable
are usually all of the same type. For example, if the variable is the name of a city,
that name should be a string, whereas if the variable is a year, that should be an
integer. So even though series let you put different types in them, it is most common
(and good practice!) to have just one type per column.

+++ {"tags": []}

```{figure} img/wrangling/pandas_dataframe_series-3.png
:name: fig:02-dataframe
:figclass: caption-hack

Data frame and vector types.
```


```{index} type
```

>  **Note:** You can use the function `type` on a data object.
> For example we can check the class of the Canadian languages data set,
> `can_lang`, we worked with in the previous chapters and we see it is a `pandas.core.frame.DataFrame`.


```{code-cell} ipython3
can_lang = pd.read_csv("data/can_lang.csv")
type(can_lang)
```

### Data structures in Python

Series and DataFrames are basic types of *data structure* in Python, which
are core to most data analyses.
The functions from `pandas` that we use often give us back a `DataFrame`
or a `Series` depending on the operation.
These include the ability to add useful attributes (such as grouping, which we will
discuss later) and more predictable type preservation when subsetting. Because
`Series` are essentially simple `DataFrames`, we will refer
to both `DataFrames` and `Series` as "data frames" in the text.

There are other types that represent data structures in Python. We summarize the
most common ones in {numref}`tab:datastructure-table`.

```{table} Basic data structures in Python
:name: tab:datastructure-table
| Data Structure | Description |
| ---            | ----------- |
| list | A 1D ordered collection of values that can store multiple data types at once. |
| dict | A labeled data structure where `keys` are paired with `values` |
| Series | A 1D ordered collection of values *with labels* that can store multiple data types at once. |
| DataFrame | A 2D labeled data structure with columns of potentially different types. |
```

A `list` is an ordered collection of values; it a smple data structure in Python.
To create a list, we put a variable name on the left and the contents of the list in between
square brackets `[]`. Each item of the list is separated by a comma. The example below contains 6 entries that are strings.
```{code-cell} ipython3
cities = ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Winnipeg"]
cities
```
A list can directly be converted to a pandas `Series`.
```
cities_series = pd.Series(cities)
cities_series
```

A `dict`, or dictionary, contains pairs of "keys" which you use to look up a "value". Dictionaries are create
using curly brackets `{}`. Each entry starts with the
"key" on the left, this is often a string. This is followed by a `:` and then the value. A dictionary can have multiple key-value pairs, each separted by a comma.
 In the example below,
we create a dictionary that has two keys: `"cities"` and `"population"` and the values associated with each are a list.
```{code-cell} ipython3
population_in_2016 = {
  "cities": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Winnipeg"],
  "population": [2235145, 1027613, 1823281, 544870, 571146, 321484]
}
population_in_2016
```
A dictionary can be converted to a data frame. The "key"
become the column names, and the values are the entries in
those columns. Dictionaries on their own are quite simple objects; it is preferable to work with a data frame
because then we have access to the built-in functionality in
`pandas` (e.g. `loc`, `[]` and many functions that we will discuss in the upcoming sections)!
```
population_in_2016 = pd.DataFrame(population_in_2016)
population_in_2016
```

There are several other data structures in the Python programming
language (*e.g.,* matrices), but these are beyond the scope of this book.

+++

## Tidy data

```{index} tidy data; definition
```

There are many ways a tabular data set can be organized. This chapter will focus
on introducing the **tidy data** format of organization and how to make your raw
(and likely messy) data tidy. A tidy data frame satisfies
the following three criteria {cite:p}`wickham2014tidy`:

  - each row is a single observation,
  - each column is a single variable, and
  - each value is a single cell (i.e., its entry in the data
    frame is not shared with another value).

{numref}`fig:02-tidy-image` demonstrates a tidy data set that satisfies these
three criteria.

+++ {"tags": []}

```{figure} img/tidy_data/tidy_data.001-cropped.jpeg
:name: fig:02-tidy-image
:figclass: caption-hack

Tidy data satisfies three criteria.
```

+++

```{index} tidy data; arguments for
```

There are many good reasons for making sure your data are tidy as a first step in your analysis.
The most important is that it is a single, consistent format that nearly every function
in the `pandas` recognizes. No matter what the variables and observations
in your data represent, as long as the data frame
is tidy, you can manipulate it, plot it, and analyze it using the same tools.
If your data is *not* tidy, you will have to write special bespoke code
in your analysis that will not only be error-prone, but hard for others to understand.
Beyond making your analysis more accessible to others and less error-prone, tidy data
is also typically easy for humans to interpret. Given these benefits,
it is well worth spending the time to get your data into a tidy format
upfront. Fortunately, there are many well-designed `pandas` data
cleaning/wrangling tools to help you easily tidy your data. Let's explore them
below!

> **Note:** Is there only one shape for tidy data for a given data set? Not
> necessarily! It depends on the statistical question you are asking and what
> the variables are for that question. For tidy data, each variable should be
> its own column. So, just as it's essential to match your statistical question
> with the appropriate data analysis tool, it's important to match your
> statistical question with the appropriate variables and ensure they are
> represented as individual columns to make the data tidy.

+++

### Tidying up: going from wide to long using `.melt`

```{index} pandas.DataFrame; melt
```

One task that is commonly performed to get data into a tidy format
is to combine values that are stored in separate columns,
but are really part of the same variable, into one.
Data is often stored this way
because this format is sometimes more intuitive for human readability
and understanding, and humans create data sets.
In {numref}`fig:02-wide-to-long`,
the table on the left is in an untidy, "wide" format because the year values
(2006, 2011, 2016) are stored as column names.
And as a consequence,
the values for population for the various cities
over these years are also split across several columns.

For humans, this table is easy to read, which is why you will often find data
stored in this wide format.  However, this format is difficult to work with
when performing data visualization or statistical analysis using Python.  For
example, if we wanted to find the latest year it would be challenging because
the year values are stored as column names instead of as values in a single
column.  So before we could apply a function to find the latest year (for
example, by using `max`), we would have to first extract the column names
to get them as a list and then apply a function to extract the latest year.
The problem only gets worse if you would like to find the value for the
population for a given region for the latest year.  Both of these tasks are
greatly simplified once the data is tidied.

Another problem with data in this format is that we don't know what the
numbers under each year actually represent. Do those numbers represent
population size? Land area? It's not clear.
To solve both of these problems,
we can reshape this data set to a tidy data format
by creating a column called "year" and a column called
"population." This transformation&mdash;which makes the data
"longer"&mdash;is shown as the right table in
{numref}`fig:02-wide-to-long`. Note that the number of entries in our data frame
can change in this transformation. The "untidy" data has 5 rows and 3 columns for
a total of 15 entries, whereas the "tidy" data on the right has 15 rows and 2 columns
for a total of 30 entries.

+++ {"tags": []}

```{figure} img/pivot_functions/pivot_functions.001.jpeg
:name: fig:02-wide-to-long
:figclass: caption-hack

Melting data from a wide to long data format.
```

+++

```{index} Canadian languages
```

We can achieve this effect in Python using the `.melt` function from the `pandas` package.
The `.melt` function combines columns,
and is usually used during tidying data
when we need to make the data frame longer and narrower.
To learn how to use `.melt`, we will work through an example with the
`region_lang_top5_cities_wide.csv` data set. This data set contains the
counts of how many Canadians cited each language as their mother tongue for five
major Canadian cities (Toronto, Montréal, Vancouver, Calgary and Edmonton) from
the 2016 Canadian census.
To get started,
we will use `pd.read_csv` to load the (untidy) data.

```{code-cell} ipython3
lang_wide = pd.read_csv("data/region_lang_top5_cities_wide.csv")
lang_wide
```

What is wrong with the untidy format above?
The table on the left in {numref}`fig:img-pivot-longer-with-table`
represents the data in the "wide" (messy) format.
From a data analysis perspective, this format is not ideal because the values of
the variable *region* (Toronto, Montréal, Vancouver, Calgary and Edmonton)
are stored as column names. Thus they
are not easily accessible to the data analysis functions we will apply
to our data set. Additionally, the *mother tongue* variable values are
spread across multiple columns, which will prevent us from doing any desired
visualization or statistical tasks until we combine them into one column. For
instance, suppose we want to know the languages with the highest number of
Canadians reporting it as their mother tongue among all five regions. This
question would be tough to answer with the data in its current format.
We *could* find the answer with the data in this format,
though it would be much easier to answer if we tidy our
data first. If mother tongue were instead stored as one column,
as shown in the tidy data on the right in
{numref}`fig:img-pivot-longer-with-table`,
we could simply use one line of code (`df["mother_tongue"].max()`)
to get the maximum value.

+++ {"tags": []}

```{figure} img/wrangling/pandas_melt_wide-long.png
:name: fig:img-pivot-longer-with-table
:figclass: caption-hack

Going from wide to long with the `.melt` function.
```

+++

{numref}`fig:img-pivot-longer` details the arguments that we need to specify
in the `.melt` function to accomplish this data transformation.

+++ {"tags": []}

**(FIGURE UPDATE NEEDED TO MATCH THE CODE BELOW)**

```{figure} img/wrangling/pandas_melt_args_labels.png
:name: fig:img-pivot-longer
:figclass: caption-hack

Syntax for the `melt` function.
```

+++

```{index} column range
```

```{index} see: :; column range
```

We use `.melt` to combine the Toronto, Montréal,
Vancouver, Calgary, and Edmonton columns into a single column called `region`,
and create a column called `mother_tongue` that contains the count of how many
Canadians report each language as their mother tongue for each metropolitan
area

```{code-cell} ipython3
lang_mother_tidy = lang_wide.melt(
    id_vars=["category", "language"],
    var_name="region",
    value_name="mother_tongue",
)

lang_mother_tidy
```

> **Note**: In the code above, the call to the
> `.melt` function is split across several lines. This is allowed in
> certain cases; for example, when calling a function as above, the input
> arguments are between parentheses `()` and Python knows to keep reading on
> the next line. Each line ends with a comma `,` making it easier to read.
> Splitting long lines like this across multiple lines is encouraged
> as it helps significantly with code readability. Generally speaking, you should
> limit each line of code to about 80 characters.

The data above is now tidy because all three criteria for tidy data have now
been met:

1.  All the variables (`category`, `language`, `region` and `mother_tongue`) are
    now their own columns in the data frame.
2.  Each observation, i.e., each `category`, `language`, `region`, and count of
    Canadians where that language is the mother tongue, are in a single row.
3.  Each value is a single cell, i.e., its row, column position in the data
    frame is not shared with another value.

+++

(pivot-wider)=
### Tidying up: going from long to wide using `.pivot`

```{index} pandas.DataFrame; pivot
```

Suppose we have observations spread across multiple rows rather than in a single
row. For example, in {numref}`fig:long-to-wide`, the table on the left is in an
untidy, long format because the `count` column contains three variables
(population, commuter, and incorporated count) and information about each observation
(here, population, commuter, and incorporated counts for a region) is split across three rows.
Remember: one of the criteria for tidy data
is that each observation must be in a single row.

Using data in this format&mdash;where two or more variables are mixed together
in a single column&mdash;makes it harder to apply many usual `pandas` functions.
For example, finding the maximum number of commuters
would require an additional step of filtering for the commuter values
before the maximum can be computed.
In comparison, if the data were tidy,
all we would have to do is compute the maximum value for the commuter column.
To reshape this untidy data set to a tidy (and in this case, wider) format,
we need to create columns called "population", "commuters", and "incorporated."
This is illustrated in the right table of {numref}`fig:long-to-wide`.

+++ {"tags": []}

```{figure} img/pivot_functions/pivot_functions.002.jpeg
:name: fig:long-to-wide
:figclass: caption-hack

Going from long to wide data.
```

+++

To tidy this type of data in Python, we can use the `.pivot` function.
The `.pivot` function generally increases the number of columns (widens)
and decreases the number of rows in a data set.
To learn how to use `.pivot`,
we will work through an example
with the `region_lang_top5_cities_long.csv` data set.
This data set contains the number of Canadians reporting
the primary language at home and work for five
major cities (Toronto, Montréal, Vancouver, Calgary and Edmonton).

```{code-cell} ipython3
lang_long = pd.read_csv("data/region_lang_top5_cities_long.csv")
lang_long
```

What makes the data set shown above untidy?
In this example, each observation is a language in a region.
However, each observation is split across multiple rows:
one where the count for `most_at_home` is recorded,
and the other where the count for `most_at_work` is recorded.
Suppose the goal with this data was to
visualize the relationship between the number of
Canadians reporting their primary language at home and work.
Doing that would be difficult with this data in its current form,
since these two variables are stored in the same column.
{numref}`fig:img-pivot-wider-table` shows how this data
will be tidied using the `.pivot` function.

+++ {"tags": []}

```{figure} img/wrangling/pandas_pivot_long-wide.png
:name: fig:img-pivot-wider-table
:figclass: caption-hack

Going from long to wide with the `.pivot` function.
```

+++

{numref}`fig:img-pivot-wider` details the arguments that we need to specify
in the `.pivot` function.

+++ {"tags": []}

```{figure} img/wrangling/pandas_pivot_args_labels.png
:name: fig:img-pivot-wider
:figclass: caption-hack

Syntax for the `.pivot` function.
```

+++

We will apply the function as detailed in {numref}`fig:img-pivot-wider`.

```{code-cell} ipython3
lang_home_tidy = lang_long.pivot(
    index=["region", "category", "language"],
    columns=["type"],
    values=["count"]
).reset_index()

lang_home_tidy.columns = [
    "region",
    "category",
    "language",
    "most_at_home",
    "most_at_work",
]
lang_home_tidy
```

In the first step, note that we added a call to `reset_index`. When `pivot` is called with
multiple column names passed to the `index`, those entries become the "name" of each row that
would be used when you filter rows with `[]` or `loc` rather than just simple numbers. This
can be confusing... What `reset_index` does is sets us back with the usual expected behaviour
where each row is "named" with an integer. This is a subtle point, but the main take-away is thatu
when you call `pivot`, it is a good idea to call `reset_index` afterwards.

The second operation we applied is to rename the columns. When we perform the `pivot`
operation, it keeps the original column name `"count"` and adds the `"type"` as a second column name.
Having two names for a column can be confusing! So we rename giving each column only one name.

We can print out some useful information about our data frame using the `info` function.
In the first row it tells us the `type` of `lang_home_tidy` (a pandas `DataFrame`). The second
row tells us how many rows there are: 1070, and to index those rows, you can use numbers between
0 and 1069 (remember that Python starts counting at 0!). Next, there is a print out about the data
colums. Here there are 5 columns total. The little table it prints out tells you the name of each
column, the number of non-null values (e.g. the number of entries that are not missing values), and
the type of the entries. Finally the last two rows summarize the types of each column and how much
memory the data frame is using on your computer.
```{code-cell} ipython3
lang_home_tidy.info()
```

The data above is now tidy! We can go through the three criteria again to check
that this data is a tidy data set.

1.  All the statistical variables are their own columns in the data frame (i.e.,
    `most_at_home`, and `most_at_work` have been separated into their own
    columns in the data frame).
2.  Each observation, (i.e., each language in a region) is in a single row.
3.  Each value is a single cell (i.e., its row, column position in the data
    frame is not shared with another value).

You might notice that we have the same number of columns in the tidy data set as
we did in the messy one. Therefore `.pivot` didn't really "widen" the data.
This is just because the original `type` column only had
two categories in it. If it had more than two, `.pivot` would have created
more columns, and we would see the data set "widen."


+++

(str-split)=
### Tidying up: using `.str.split` to deal with multiple delimiters

```{index} pandas.Series; str.split, delimiter
```

Data are also not considered tidy when multiple values are stored in the same
cell. The data set we show below is even messier than the ones we dealt with
above: the `Toronto`, `Montréal`, `Vancouver`, `Calgary` and `Edmonton` columns
contain the number of Canadians reporting their primary language at home and
work in one column separated by the delimiter (`/`). The column names are the
values of a variable, *and* each value does not have its own cell! To turn this
messy data into tidy data, we'll have to fix these issues.

```{code-cell} ipython3
lang_messy = pd.read_csv("data/region_lang_top5_cities_messy.csv")
lang_messy
```

First we’ll use `.melt` to create two columns, `region` and `value`,
similar to what we did previously.
The new `region` columns will contain the region names,
and the new column `value` will be a temporary holding place for the
data that we need to further separate, i.e., the
number of Canadians reporting their primary language at home and work.

```{code-cell} ipython3
lang_messy_longer = lang_messy.melt(
    id_vars=["category", "language"],
    var_name="region",
    value_name="value",
)

lang_messy_longer
```

Next we'll use `.str.split` to split the `value` column into two columns.
For example if we want to split the string `"50/0"` into two numbers `["50", "0"]`
we tell `.str.split` to split our string on the slash character `"/"` as follows
```{code-cell} ipython3
"50/0".split("/")
```

We can use this operation on the columns of our data frame so that
one column will contain only the counts of Canadians
that speak each language most at home,
and the other will contain the counts of Canadians
that speak each language most at work for each region.
{numref}`fig:img-separate`
outlines what we need to specify to use `.str.split`.

+++ {"tags": []}

```{figure} img/wrangling/str-split_args_labels.png
:name: fig:img-separate
:figclass: caption-hack

Syntax for the `.str.split` function.
```

We will do this in multiple steps. First, we create a new object
that contains two columns

```
split_counts = lang_messy_longer["value"].str.split("/", expand=True)
split_counts
```
The `expand=True` tells `pandas` that we want to `expand` the output into
two columns.
Now this doesnt have the rest of the columns including the language, region, etc.
that were in our original dataframe. We don't want to lose this information, so
we will contatenate (combine) the original data frame with `split_counts` using
`pd.concat`.

```{code-cell} ipython3
tidy_lang = (
    pd.concat(
        (lang_messy_longer, split_counts),
        axis=1,
    )
tidy_lang
```

We combine them along `axis=1` as we want to combine them horizontally
(the default of `axis=0` would try to stack them vertically).

Next, we want to rename our newly created columns that are currently called
`0` and `1` to more meaningful names of `"most_at_home"`, and `"most_at_work"`.
Finally we will drop the column `value` using `drop` to remove it from our dataframe.

```{code-cell} ipython3
tidy_lang = (
    tidy_lang.rename(columns={0: "most_at_home", 1: "most_at_work"})
    .drop(columns=["value"])
)

tidy_lang
```
Note that we could have chained these steps together to make our code more compact.
Is this data set now tidy? If we recall the three criteria for tidy data:

  - each row is a single observation,
  - each column is a single variable, and
  - each value is a single cell.

We can see that this data now satisfies all three criteria, making it easier to
analyze. But we aren't done yet! Notice in the table, all of the variables are
"object" data types. Object data types are columns of strings or columns with mixed types. In the previous example in the section on {ref}`pivot-wider`, the
`most_at_home` and `most_at_work` variables were `int64` (integer)&mdash;you can
verify this by calling `df.info()`&mdash;which is a type
of numeric data. This change is due to the delimiter (`/`) when we read in this
messy data set. Python read these columns in as string types, and by default,
`.str.split` will return columns as object data types.

It makes sense for `region`, `category`, and `language` to be stored as a
object type. However, suppose we want to apply any functions that treat the
`most_at_home` and `most_at_work` columns as a number (e.g., finding rows
above a numeric threshold of a column).
In that case,
it won't be possible to do if the variable is stored as a `object`.
Fortunately, the `pandas.to_numeric` function provides a natural way to fix problems
like this: it will convert the columns to the best numeric data types.

```{code-cell} ipython3
tidy_lang["most_at_home"] = pd.to_numeric(tidy_lang["most_at_home"])
tidy_lang["most_at_work"] = pd.to_numeric(tidy_lang["most_at_work"])
tidy_lang
```

```{code-cell} ipython3
tidy_lang.info()
```

Now we see `most_at_home` and `most_at_work` columns are of `int64` data types,
indicating they are integer data types (i.e., numbers)!

+++

## Using `[]` to extract rows or columns

Now that the `tidy_lang` data is indeed *tidy*, we can start manipulating it
using the powerful suite of functions from the `pandas`.
We revisit the `[]` from the chapter on {ref}`intro`,
which lets us create a subset of rows from a data frame.
Recall the argument to the `[]`:
column names or a logical statement evaluated to either `True` or `False`;
`[]` works by returning the rows  where the logical statement evaluates to `True`.
This section will highlight more advanced usage of the `[]` function.
In particular, this section provides an in-depth treatment of the variety of logical statements
one can use in the `[]` to select subsets of rows.

+++

### Extracting rows that have a certain value with `==`
Suppose we are only interested in the subset of rows in `tidy_lang` corresponding to the
official languages of Canada (English and French).
We can extract these rows by using the *equivalency operator* (`==`)
to compare the values of the `category` column
with the value `"Official languages"`.
With these arguments, `[]` returns a data frame with all the columns
of the input data frame
but only the rows we asked for in the logical statement, i.e.,
those where the `category` column holds the value `"Official languages"`.
We name this data frame `official_langs`.

```{code-cell} ipython3
official_langs = tidy_lang[tidy_lang["category"] == "Official languages"]
official_langs
```

### Extracting rows that do not have a certain value with `!=`

What if we want all the other language categories in the data set *except* for
those in the `"Official languages"` category? We can accomplish this with the `!=`
operator, which means "not equal to". So if we want to find all the rows
where the `category` does *not* equal `"Official languages"` we write the code
below.

```{code-cell} ipython3
tidy_lang[tidy_lang["category"] != "Official languages"]
```

(filter-and)=
### Extracting rows satisfying multiple conditions using `&`

Suppose now we want to look at only the rows
for the French language in Montréal.
To do this, we need to filter the data set
to find rows that satisfy multiple conditions simultaneously.
We can do this with the ampersand symbol (`&`), which
is interpreted by Python as "and".
We write the code as shown below to filter the `official_langs` data frame
to subset the rows where `region == "Montréal"`
*and* the `language == "French"`.

```{code-cell} ipython3
tidy_lang[(tidy_lang["region"] == "Montréal") & (tidy_lang["language"] == "French")]
```

+++ {"tags": []}

### Extracting rows satisfying at least one condition using `|`

Suppose we were interested in only those rows corresponding to cities in Alberta
in the `official_langs` data set (Edmonton and Calgary).
We can't use `&` as we did above because `region`
cannot be both Edmonton *and* Calgary simultaneously.
Instead, we can use the vertical pipe (`|`) logical operator,
which gives us the cases where one condition *or*
another condition *or* both are satisfied.
In the code below, we ask Python to return the rows
where the `region` columns are equal to "Calgary" *or* "Edmonton".

```{code-cell} ipython3
official_langs[
    (official_langs["region"] == "Calgary") | (official_langs["region"] == "Edmonton")
]
```

### Extracting rows with values in a list using `.isin()`

Next, suppose we want to see the populations of our five cities.
Let's read in the `region_data.csv` file
that comes from the 2016 Canadian census,
as it contains statistics for number of households, land area, population
and number of dwellings for different regions.

```{code-cell} ipython3
region_data = pd.read_csv("data/region_data.csv")
region_data
```

To get the population of the five cities
we can filter the data set using the `.isin` method.
The `.isin` method is used to see if an element belongs to a list.
Here we are filtering for rows where the value in the `region` column
matches any of the five cities we are intersted in: Toronto, Montréal,
Vancouver, Calgary, and Edmonton.

```{code-cell} ipython3
city_names = ["Toronto", "Montréal", "Vancouver", "Calgary", "Edmonton"]
five_cities = region_data[region_data["region"].isin(city_names)]
five_cities
```

> **Note:** What's the difference between `==` and `.isin`? Suppose we have two
> Series, `seriesA` and `seriesB`. If you type `seriesA == seriesB` into Python it
> will compare the series element by element. Python checks if the first element of
> `seriesA` equals the first element of `seriesB`, the second element of
> `seriesA` equals the second element of `seriesB`, and so on. On the other hand,
> `seriesA.isin(seriesB)` compares the first element of `seriesA` to all the
> elements in `seriesB`. Then the second element of `seriesA` is compared
> to all the elements in `seriesB`, and so on. Notice the difference between `==` and
> `.isin` in the example below.

```{code-cell} ipython3
pd.Series(["Vancouver", "Toronto"]) == pd.Series(["Toronto", "Vancouver"])
```

```{code-cell} ipython3
pd.Series(["Vancouver", "Toronto"]).isin(pd.Series(["Toronto", "Vancouver"]))
```

### Extracting rows above or below a threshold using `>` and `<`

```{code-cell} ipython3
:tags: [remove-cell]

glue("census_popn", "{0:,.0f}".format(35151728))
glue("most_french", "{0:,.0f}".format(2669195))
```

We saw in the section on {ref}`filter-and` that
{glue:text}`most_french` people reported
speaking French in Montréal as their primary language at home.
If we are interested in finding the official languages in regions
with higher numbers of people who speak it as their primary language at home
compared to French in Montréal, then we can use `[]` to obtain rows
where the value of `most_at_home` is greater than
{glue:text}`most_french`.

```{code-cell} ipython3
official_langs[official_langs["most_at_home"] > 2669195]
```

This operation returns a data frame with only one row, indicating that when
considering the official languages,
only English in Toronto is reported by more people
as their primary language at home
than French in Montréal according to the 2016 Canadian census.

### Extracting rows using `.query()`

You can also extract rows above, below, equal or not-equal to a threshold using the
`.query()` method. For example the following gives us the same result as when we used
`official_langs[official_langs["most_at_home"] > 2669195]`.

```{code-cell} ipython3
official_langs.query("most_at_home > 2669195")
```

The query (criteria we are using to select values) is input as a string. This
comes in handy when we chain operations together.

(loc-iloc)=
## Using `.loc[]` to filter rows and select columns.
```{index} pandas.DataFrame; loc[]
```

The `[]` operation is only used when you want to filter rows or select columns;
it cannot be used to do both operations at the same time. This is where `.loc[]`
comes in. For the first example, recall `.loc[]` from Chapter {ref}`intro`,
which lets us create a subset of columns from a data frame.
Suppose we wanted to select only the columns `language`, `region`,
`most_at_home` and `most_at_work` from the `tidy_lang` data set. Using what we
learned in the chapter on {ref}`intro`, we would pass all of these column names into the square brackets.

```{code-cell} ipython3
selected_columns = tidy_lang.loc[:, ["language", "region", "most_at_home", "most_at_work"]]
selected_columns
```
We pass `:` before the comma indicating we want to retrieve all rows, and the list indicates
the columns that we want.

Note that we could obtain the same result by stating that we would like all of the columns
from `language` through `most_at_work`. Instead of passing a list of all of the column
names that we want, we can ask for the range of columns `"language":"most_at_work"`, which
you can read as "The columns from `language` to (`:`) `most_at_work`.

```{code-cell} ipython3
selected_columns = tidy_lang.loc[:, "language":"most_at_work"]
selected_columns
```

Similarly, you can ask for all of the columns including and after `language` by doing the following

```{code-cell} ipython3
selected_columns = tidy_lang.loc[:, "language":]
selected_columns
```

By not putting anything after the `:`, python reads this as "from `language` until the last column".
Although the notation for selecting a range using `:` is convienent because less code is required,
it must be used carefully. If you were to re-order columns or add a column to the data frame, the
output would change. Using a list is more explicit and less prone to potential confusion.

Suppose instead we wanted to extract columns that followed a particular pattern
rather than just selecting a range. For example, let's say we wanted only to select the
columns `most_at_home` and `most_at_work`. There are other functions that allow
us to select variables based on their names. In particular, we can use the `.str.startswith` method
to choose only the columns that start with the word "most":

```{code-cell} ipython3
tidy_lang.loc[:, tidy_lang.columns.str.startswith('most')]
```

```{index} pandas.Series; str.contains
```

We could also have chosen the columns containing an underscore `_` by using the
`.str.contains("_")`, since we notice
the columns we want contain underscores and the others don't.

```{code-cell} ipython3
tidy_lang.loc[:, tidy_lang.columns.str.contains('_')]
```

There are many different functions that help with selecting
variables based on certain criteria.
The additional resources section at the end of this chapter
provides a comprehensive resource on these functions.

```{code-cell} ipython3
:tags: [remove-cell]

# There are many different `select` helpers that select
# variables based on certain criteria.
# The additional resources section at the end of this chapter
# provides a comprehensive resource on `select` helpers.
```

## Using `.iloc[]` to extract a range of columns
```{index} pandas.DataFrame; iloc[], column range
```
Another approach for selecting columns is to use `.iloc[]`
which allows us to index with integers rather than the names of the columns.
For example, the column names of the `tidy_lang` data frame are
`['category', 'language', 'region', 'most_at_home', 'most_at_work']`.

Then using `.iloc[]` you can ask for the `language` column by doing

```{code-cell} ipython3
column = tidy_lang.iloc[:, 1]
column
```

You can also ask for multiple columns as we did with `[]`. We pass `:` before
the comma indicating we want to retrieve all rows, and `1:` after the comma
indicating we want only columns from index 1 (*i.e.* `language`) and afterwords.

```{code-cell} ipython3
column_range = tidy_lang.iloc[:, 1:]
column_range
```

This is less commonly used and needs to be used with care; it is easy
accidentally put in the wrong integer because you didn't remember if `language`
was column number 1 or 2.

Notice that we get the same output as we did

```{index} pandas.Series; str.startswith
```

+++ {"tags": []}

## Aggregating data with `.agg`

+++

### Calculating summary statistics on whole columns

```{index} summarize
```

As a part of many data analyses, we need to calculate a summary value for the
data (a *summary statistic*).
Examples of summary statistics we might want to calculate
are the number of observations, the average/mean value for a column,
the minimum value, etc.
Oftentimes,
this summary statistic is calculated from the values in a data frame column,
or columns, as shown in {numref}`fig:summarize`.

+++ {"tags": []}

```{figure} img/summarize/summarize.001.jpeg
:name: fig:summarize
:figclass: caption-hack

Calculating summary statistics on one or more column(s). In its simplest use case, it creates a new data frame with a single row containing the summary statistic(s) for each column being summarized. The darker, top row of each table represents the column headers.
```

+++



First a reminder of what `region_lang` looks like:

```{code-cell} ipython3
region_lang = pd.read_csv("data/region_lang.csv")
region_lang
```

We use `.min` to calculate the minimum
and `.max` to calculate maximum number of Canadians
reporting a particular language as their primary language at home,
for any region.

```{code-cell} ipython3
region_lang["most_at_home"].min()
```

```{code-cell} ipython3
region_lang["most_at_home"].max()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("lang_most_people", "{0:,.0f}".format(int(lang_summary["max_most_at_home"])))
```

From this we see that there are some languages in the data set that no one speaks
as their primary language at home. We also see that the most commonly spoken
primary language at home is spoken by
{glue:text}`lang_most_people`
people.

If we wanted to know the total number of people in the survey, we can use `.sum`:
```{code-cell} ipython3
region_lang["most_at_home"].sum()
```

Other handy summary statistics include the `.mean`, `.median` and `.std` for
computing the mean, median, and standard deviation of observations, respectively.

We can compute multiple statistics at once using `.agg` to "aggregate" results, say if we wanted to
compute both the `min` and `max`, we could do the following.

```{code-cell} ipython3
region_lang["most_at_home"].agg(['min', 'max'])
```

Another handy function is `describe`; it gives us a summary.

```{code-cell} ipython3
region_lang["most_at_home"].describe()
```

The `count` is th total number of observations (equal to the number of rows in our data frame),
and the numbers with percents indicate the percentiles. A 75th percentile of 30 means that 75% of
the languages are spoken by 30 people surveyed or less.



The table below provides an overview of some of the useful summary statistics that you can
compute with `pandas`.

```{table} Basic data structures in Python
:name: tab:datastructure-table
| Function | Description |
| -------- | ----------- |
| `count` | The number of observations (rows) |
| `mean` | The mean of the observations |
| `median` | The median value of the observations |
| `std` | The standard deviation of the observations |
| `max` | The largest value in a column |
| `min` | The smallest value in a column |
| `sum` | The sum of all observations |
| `agg` | Aggregate multiple statistics together |
| `describe` | a summary |
```

+++
+++


> **Note:** In `pandas` DataFrame, the value `NaN` is often used to denote missing data.
> By default, when `pandas` calculates summary statistics (e.g., `max`, `min`, `sum`, etc),
> it ignores these values. If you look at the documentation for these functions, you will
> see an input variable `skipna`, which by default is set to `skipna=True`. This means that
> `pandas` will skip `NaN` values when computing statistics.


### Calculating summary statistics for groups of rows

+++

```{index} pandas.DataFrame; groupby
```

A common pairing with summary functions is `.groupby`. Pairing these functions
together can let you summarize values for subgroups within a data set,
as illustrated in {numref}`fig:summarize-groupby`.
For example, we can use `.groupby` to group the regions of the `tidy_lang` data frame and then calculate the minimum and maximum number of Canadians
reporting the language as the primary language at home
for each of the regions in the data set.

```{code-cell} ipython3
:tags: [remove-cell]

# A common pairing with `summarize` is `group_by`. Pairing these functions \index{group\_by}
# together can let you summarize values for subgroups within a data set,
# as illustrated in Figure \@ref(fig:summarize-groupby).
# For example, we can use `group_by` to group the regions of the `tidy_lang` data frame and then calculate the minimum and maximum number of Canadians
# reporting the language as the primary language at home
# for each of the regions in the data set.

# (ref:summarize-groupby) `summarize` and `group_by` is useful for calculating summary statistics on one or more column(s) for each group. It creates a new data frame&mdash;with one row for each group&mdash;containing the summary statistic(s) for each column being summarized. It also creates a column listing the value of the grouping variable. The darker, top row of each table represents the column headers. The gray, blue, and green colored rows correspond to the rows that belong to each of the three groups being represented in this cartoon example.
```

+++ {"tags": []}

```{figure} img/summarize/summarize.002.jpeg
:name: fig:summarize-groupby
:figclass: caption-hack

Calculating summary statistics on one or more column(s) for each group. It creates a new data frame&mdash;with one row for each group&mdash;containing the summary statistic(s) for each column being summarized. It also creates a column listing the value of the grouping variable. The darker, top row of each table represents the column headers. The gray, blue, and green colored rows correspond to the rows that belong to each of the three groups being represented in this cartoon example.
```

+++

The `.groupby` function takes at least one argument&mdash;the columns to use in the
grouping. Here we use only one column for grouping (`region`), but more than one
can also be used. To do this, pass a list of column names.

<!-- ```{code-cell} ipython3
region_summary = pd.DataFrame()
region_summary = region_summary.assign(
    min_most_at_home=region_lang.groupby("region")["most_at_home"].min(),
    max_most_at_home=region_lang.groupby("region")["most_at_home"].max()
).reset_index()

region_summary.columns = ["region", "min_most_at_home", "max_most_at_home"]
region_summary
``` -->

```{code-cell} ipython3
region_lang.groupby("region")["most_at_home"].agg(["min", "max"])
```
<!--
`pandas` also has a convenient method `.agg` (shorthand for `.aggregate`) that allows us to apply multiple aggregate functions in one line of code. We just need to pass in a list of function names to `.agg` as shown below.

```{code-cell} ipython3
region_summary = (
    region_lang.groupby("region")["most_at_home"].agg(["min", "max"]).reset_index()
)
region_summary.columns = ["region", "min_most_at_home", "max_most_at_home"]
region_summary
``` -->

Notice that `.groupby` converts a `DataFrame` object to a `DataFrameGroupBy` object, which contains information about the groups of the dataframe. We can then apply aggregating functions to the `DataFrameGroupBy` object.


```{code-cell} ipython3
region_lang.groupby("region")
```

### Calculating summary statistics on many columns

+++

Sometimes we need to summarize statistics across many columns.
An example of this is illustrated in {numref}`fig:summarize-across`.
In such a case, using summary functions alone means that we have to
type out the name of each column we want to summarize.
In this section we will meet two strategies for performing this task.
First we will see how we can do this using `.loc[]` to slice the columns before applying summary functions.
Then we will also explore how we can use a more general iteration function,
`.apply`, to also accomplish this.

+++ {"tags": []}

```{figure} img/summarize/summarize.003.jpeg
:name: fig:summarize-across
:figclass: caption-hack

`.loc[]` or `.apply` is useful for efficiently calculating summary statistics on many columns at once. The darker, top row of each table represents the column headers.
```

+++

<!-- ### Aggregating data with `.agg` -->
<!-- #### Aggregating on a data frame for calculating summary statistics on many columns

+++

```{index} column range
```

Recall that in the Section {ref}`loc-iloc`, we can use `.loc[]` to extract a range of columns. Here we demonstrate finding the maximum value
of each of the numeric
columns of the `region_lang` data set through pairing `.loc[]` and `.max`. This means that the
summary methods (*e.g.* `.min`, `.max`, `.sum` etc.) can be used for data frames as well.

```{code-cell} ipython3
pd.DataFrame(region_lang.loc[:, "mother_tongue":].max()).T
```

Here we use the `.T` to "transpose" or flip the axes so that rows are columns. Otherwise,
the summary statistics would be in the rows, which is untidy! You can think of the summary
statistics representing one observation (which should be a row).

(apply-summary)= -->
#### `.apply` for calculating summary statistics on many columns

+++

```{index} pandas.DataFrame; apply
```

An alternative to aggregating on a dataframe
for applying a function to many columns is the `.apply` method.
Let's again find the maximum value of each column of the
`region_lang` data frame, but using `.apply` with the `max` function this time.
We focus on the two arguments of `.apply`:
the function that you would like to apply to each column, and the `axis` along which the function will be applied (`0` for columns, `1` for rows).
Note that `.apply` does not have an argument
to specify *which* columns to apply the function to.
Therefore, we will use the `.loc[]` before calling `.apply`
to choose the columns for which we want the maximum.

```{code-cell} ipython3
region_lang.loc[:, ["most_at_home", "most_at_work"]].apply(max)
```

<!-- ```{index} missing data
```

> **Note:** Similar to when we use base Python statistical summary functions
> (e.g., `max`, `min`, `sum`, etc.) when there are `NaN`s,
> `.apply` functions paired with base Python statistical summary functions
> also return `NaN` values when we apply them to columns that
> contain `NaN` values.
>
> To avoid this, again we need to use the `pandas` variants of summary functions (*i.e.*
> `.max`, `.min`, `.sum`, etc.) with `skipna=True`.
> When we use this with `.apply`, we do this by constructing a anonymous function that calls
> the `.max` method with `skipna=True`, as illustrated below:

```{code-cell} ipython3
pd.DataFrame(
    region_lang_na[:, ["most_at_home", "most_at_work"]].apply(lambda col: col.max(skipna=True), axis=0)
).T
``` -->

The `.apply` function is generally quite useful for solving many problems
involving repeatedly applying functions in Python.
Additionally, a variant of `.apply` is `.applymap`,
which can be used to apply functions element-wise.
To learn more about these functions, see the additional resources
section at the end of this chapter.

+++ {"jp-MarkdownHeadingCollapsed": true, "tags": ["remove-cell"]}

(pandas-assign)=
## Using `.assign` to modify or add columns

+++

### Using `.assign` to create a new data frame

If you would like to use the summary statistics that you computed in a further analysis, it can be handy
to store them in a data frame that you can use later.
We can use `.assign` as mentioned in along with proper summary functions to create a aggregated column.

For example, say we wanted to keep the minimum and maximum values in a data frame, we can (1) create a new, empty
data frame and (2) use `.assign` to assign values to that data frame.
```
region_lang_min_max = pd.DataFrame()  # empty data frame
region_lang_min_max.assign(
  most_at_home_min = region_lang["most_at_home"].min(),
  most_at_home_max = region_lang["most_at_home"].max()
)
region_lang_min_max
```

### Using `.assign` to modify columns

```{index} pandas.DataFrame; []
```

In the section on {ref}`str-split`,
when we first read in the `"region_lang_top5_cities_messy.csv"` data,
all of the variables were "object" data types.
During the tidying process,
we used the `pandas.to_numeric` function
to convert the `most_at_home` and `most_at_work` columns
to the desired integer (i.e., numeric class) data types and then used `[]` to overwrite columns.
But suppose we didn't use the `[]`,
and needed to modify the columns some other way.
Below we create such a situation
so that we can demonstrate how to use `.assign`
to change the column types of a data frame.
`.assign` is a useful function to modify or create new data frame columns.

```{code-cell} ipython3
lang_messy = pd.read_csv("data/region_lang_top5_cities_messy.csv")
lang_messy_longer = lang_messy.melt(
    id_vars=["category", "language"],
    var_name="region",
    value_name="value",
)
tidy_lang_obj = (
    pd.concat(
        (lang_messy_longer, lang_messy_longer["value"].str.split("/", expand=True)),
        axis=1,
    )
    .rename(columns={0: "most_at_home", 1: "most_at_work"})
    .drop(columns=["value"])
)
official_langs_obj = tidy_lang_obj[tidy_lang_obj["category"] == "Official languages"]

official_langs_obj
```

```{code-cell} ipython3
official_langs_obj.info()
```

To use the `.assign` method, again we first specify the object to be the data set,
and in the following arguments,
we specify the name of the column we want to modify or create
(here `most_at_home` and `most_at_work`), an `=` sign,
and then the function we want to apply (here `pandas.to_numeric`).
In the function we want to apply,
we refer to the column upon which we want it to act
(here `most_at_home` and `most_at_work`).
In our example, we are naming the columns the same
names as columns that already exist in the data frame
("most\_at\_home", "most\_at\_work")
and this will cause `.assign` to *overwrite* those columns
(also referred to as modifying those columns *in-place*).
If we were to give the columns a new name,
then `.assign` would create new columns with the names we specified.
`.assign`'s general syntax is detailed in {numref}`fig:img-assign`.

+++ {"tags": []}

```{figure} img/wrangling/pandas_assign_args_labels.png
:name: fig:img-assign
:figclass: caption-hack

Syntax for the `.assign` function.
```

+++

Below we use `.assign` to convert the columns `most_at_home` and `most_at_work`
to numeric data types in the `official_langs` data set as described in
{numref}`fig:img-assign`.

```{code-cell} ipython3
official_langs_numeric = official_langs_obj.assign(
    most_at_home=pd.to_numeric(official_langs_obj["most_at_home"]),
    most_at_work=pd.to_numeric(official_langs_obj["most_at_work"]),
)

official_langs_numeric
```

```{code-cell} ipython3
official_langs_numeric.info()
```

Now we see that the `most_at_home` and `most_at_work` columns are both `int64` (which is a numeric data type)!

+++

### Using `.assign` to create new columns

```{code-cell} ipython3
:tags: [remove-cell]

number_most_home = int(
    official_langs[
        (official_langs["language"] == "English")
        & (official_langs["region"] == "Toronto")
    ]["most_at_home"]
)

toronto_popn = int(region_data[region_data["region"] == "Toronto"]["population"])

glue("number_most_home", "{0:,.0f}".format(number_most_home))
glue("toronto_popn", "{0:,.0f}".format(toronto_popn))
glue("prop_eng_tor", "{0:.2f}".format(number_most_home / toronto_popn))
```

We can see in the table that
{glue:text}`number_most_home` people reported
speaking English in Toronto as their primary language at home, according to
the 2016 Canadian census. What does this number mean to us? To understand this
number, we need context. In particular, how many people were in Toronto when
this data was collected? From the 2016 Canadian census profile, the population
of Toronto was reported to be
{glue:text}`toronto_popn` people.
The number of people who report that English is their primary language at home
is much more meaningful when we report it in this context.
We can even go a step further and transform this count to a relative frequency
or proportion.
We can do this by dividing the number of people reporting a given language
as their primary language at home by the number of people who live in Toronto.
For example,
the proportion of people who reported that their primary language at home
was English in the 2016 Canadian census was {glue:text}`prop_eng_tor`
in Toronto.

Let's use `.assign` to create a new column in our data frame
that holds the proportion of people who speak English
for our five cities of focus in this chapter.
To accomplish this, we will need to do two tasks
beforehand:

1. Create a list containing the population values for the cities.
2. Filter the `official_langs` data frame
so that we only keep the rows where the language is English.

To create a list containing the population values for the five cities
(Toronto, Montréal, Vancouver, Calgary, Edmonton),
we will use the `[]` (recall that we can also use `list()` to create a list).

```{code-cell} ipython3
city_pops = [5928040, 4098927, 2463431, 1392609, 1321426]
city_pops
```

And next, we will filter the `official_langs` data frame
so that we only keep the rows where the language is English.
We will name the new data frame we get from this `english_langs`.

```{code-cell} ipython3
english_langs = official_langs[official_langs["language"] == "English"]
english_langs
```

Finally, we can use `.assign` to create a new column,
named `most_at_home_proportion`, that will have value that corresponds to
the proportion of people reporting English as their primary
language at home.
We will compute this by dividing the column by our vector of city populations.

```{code-cell} ipython3
english_langs = english_langs.assign(
    most_at_home_proportion=english_langs["most_at_home"] / city_pops
)

english_langs
```

In the computation above, we had to ensure that we ordered the `city_pops` vector in the
same order as the cities were listed in the `english_langs` data frame.
This is because Python will perform the division computation we did by dividing
each element of the `most_at_home` column by each element of the
`city_pops` list, matching them up by position.
Failing to do this would have resulted in the incorrect math being performed.

> **Note:** In more advanced data wrangling,
> one might solve this problem in a less error-prone way though using
> a technique called "joins".
> We link to resources that discuss this in the additional
> resources at the end of this chapter.

+++

## Apply functions across many columns with `.apply`

Sometimes we need to apply a function to many columns in a data frame.
For example, we would need to do this when converting units of measurements across many columns.
We illustrate such a data transformation in {numref}`fig:mutate-across`.

+++ {"tags": []}

```{figure} img/summarize/summarize.005.jpeg
:name: fig:mutate-across
:figclass: caption-hack

`.apply` is useful for applying functions across many columns. The darker, top row of each table represents the column headers.
```

+++

For example,
imagine that we wanted to convert all the numeric columns
in the `region_lang` data frame from `int64` type to `int32` type
using the `.as_type` function.
When we revisit the `region_lang` data frame,
we can see that this would be the columns from `mother_tongue` to `lang_known`.

```{code-cell} ipython3
region_lang
```

```{index} pandas.DataFrame; apply, pandas.DataFrame; iloc[]
```

To accomplish such a task, we can use `.apply`.
This works in a similar way for column selection,
as we saw when we used in the Section on {ref}`apply-summary` earlier.
As we did above,
we again use `.loc[]` to specify the columns
as well as the `.apply` to specify the function we want to apply on these columns.
However, a key difference here is that we are not using aggregating function here,
which means that we get back a data frame with the same number of rows.

```{code-cell} ipython3
region_lang.info()
```

```{code-cell} ipython3
region_lang_int32 = region_lang.loc[:, ["most_at_home", "most_at_work"]].apply(lambda col: col.astype('int32'))
region_lang_int32
```

```{code-cell} ipython3
region_lang_int32.info()
```

We see that we get back a data frame
with the same number of columns and rows.
The only thing that changes is the transformation we applied
to the specified columns (here `mother_tongue` to `lang_known`).

+++

## Apply functions across columns within one row with `.apply`

What if you want to apply a function across columns but within one row?
We illustrate such a data transformation in {numref}`fig:rowwise`.

+++ {"tags": []}

```{figure} img/summarize/summarize.004.jpeg
:name: fig:rowwise
:figclass: caption-hack

`.apply` is useful for applying functions across columns within one row. The darker, top row of each table represents the column headers.
```

+++

For instance, suppose we want to know the maximum value between `mother_tongue`,
`most_at_home`, `most_at_work`
and `lang_known` for each language and region
in the `region_lang` data set.
In other words, we want to apply the `max` function *row-wise.*
Before we use `.apply`, we will again use `loc[]` to select only the count columns
so we can see all the columns in the data frame's output easily in the book.
So for this demonstration, the data set we are operating on looks like this.

```{code-cell} ipython3
region_lang.loc[:, ["most_at_home", "most_at_work"]]
```

Now we use `.apply` with argument `axis=1`, to tell Python that we would like
the `max` function to be applied across, and within, a row,
as opposed to being applied on a column
(which is the default behavior of `.apply`).

```{code-cell} ipython3
region_lang_rowwise = region_lang.assign(
    maximum=region_lang.loc[:, ["most_at_home", "most_at_work"]].apply(max, axis=1)
)

region_lang_rowwise
```

We see that we get an additional column added to the data frame,
named `maximum`, which is the maximum value between `mother_tongue`,
`most_at_home`, `most_at_work` and `lang_known` for each language
and region.

## Summary

Cleaning and wrangling data can be a very time-consuming process. However,
it is a critical step in any data analysis. We have explored many different
functions for cleaning and wrangling data into a tidy format.
{numref}`tab:summary-functions-table` summarizes some of the key wrangling
functions we learned in this chapter. In the following chapters, you will
learn how you can take this tidy data and do so much more with it to answer your
burning data science questions!

+++

```{table} Summary of wrangling functions
:name: tab:summary-functions-table

| Function | Description |
| ---      | ----------- |
| `.agg` | calculates aggregated summaries of inputs |
| `.apply` | allows you to apply function(s) to multiple columns/rows  |
| `.assign` | adds or modifies columns in a data frame  |
| `.groupby` |  allows you to apply function(s) to groups of rows |
| `.iloc` | subsets columns/rows of a data frame using integer indices |
| `.loc` | subsets columns/rows of a data frame using labels |
| `.melt` | generally makes the data frame longer and narrower |
| `.pivot` | generally makes a data frame wider and decreases the number of rows |
| `.str.split` | splits up a string column into multiple columns  |
```

## Exercises

Practice exercises for the material covered in this chapter
can be found in the accompanying
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the "Cleaning and wrangling data" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in the chapter on {ref}`move-to-your-own-machine`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++ {"tags": []}

## Additional resources

- The [`pandas` package documentation](https://pandas.pydata.org/docs/reference/index.html) is
  another resource to learn more about the functions in this
  chapter, the full set of arguments you can use, and other related functions.
  The site also provides a very nice cheat sheet that summarizes many of the
  data wrangling functions from this chapter.
- *Python for Data Analysis* {cite:p}`mckinney2012python` has a few chapters related to
  data wrangling that go into more depth than this book. For example, the
  [data wrangling chapter](https://wesmckinney.com/book/data-wrangling.html) covers tidy data,
  `.melt` and `.pivot`, but also covers missing values
  and additional wrangling functions (like `.stack`). The [data
  aggregation chapter](https://wesmckinney.com/book/data-aggregation.html) covers
  `.groupby`, aggregating functions, `.apply`, etc.
- You will occasionally encounter a case where you need to iterate over items
  in a data frame, but none of the above functions are flexible enough to do
  what you want. In that case, you may consider using [a for
  loop](https://wesmckinney.com/book/python-basics.html#control_for) {cite:p}`mckinney2012python`.


## References

+++

```{bibliography}
:filter: docname in docnames
```
