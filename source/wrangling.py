# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cleaning and wrangling data

# %% tags=["remove-cell"]
import altair as alt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 20)

from myst_nb import glue

# %% [markdown]
# ## Overview
#
# This chapter is centered around defining tidy data&mdash;a data format that is
# suitable for analysis&mdash;and the tools needed to transform raw data into this
# format. This will be presented in the context of a real-world data science
# application, providing more practice working through a whole case study.

# %% [markdown]
# ## Chapter learning objectives
#
# By the end of the chapter, readers will be able to do the following:
#
#   - Define the term "tidy data".
#   - Discuss the advantages of storing data in a tidy data format.
#   - Define what lists, series and data frames are in Python, and describe how they relate to
#     each other.
#   - Describe the common types of data in Python and their uses.
#   - Recall and use the following functions for their
#     intended data wrangling tasks:
#       - `.agg`
#       - `.apply`
#       - `.assign`
#       - `.groupby`
#       - `.melt`
#       - `.pivot`
#       - `.str.split`
#   - Recall and use the following operators for their
#     intended data wrangling tasks:
#       - `==` 
#       - `in`
#       - `and`
#       - `or`
#       - `df[]`
#       - `.iloc[]`
#       - `.loc[]`

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
# By the end of the chapter, readers will be able to do the following:

#   - Define the term "tidy data".
#   - Discuss the advantages of storing data in a tidy data format.
#   - Define what vectors, lists, and data frames are in R, and describe how they relate to
#     each other.
#   - Describe the common types of data in R and their uses.
#   - Recall and use the following functions for their
#     intended data wrangling tasks:
#       - `across`
#       - `c`
#       - `filter`
#       - `group_by`
#       - `select`
#       - `map`
#       - `mutate`
#       - `pull`
#       - `pivot_longer`
#       - `pivot_wider`
#       - `rowwise`
#       - `separate`
#       - `summarize`
#   - Recall and use the following operators for their
#     intended data wrangling tasks:
#       - `==` 
#       - `%in%`
#       - `!`
#       - `&` 
#       - `|`
#       - `|>` and `%>%`

# %% [markdown]
# ## Data frames, series, and lists
#
# In Chapters {ref}`intro` and {ref}`reading`, *data frames* were the focus:
# we learned how to import data into Python as a data frame, and perform basic operations on data frames in Python.
# In the remainder of this book, this pattern continues. The vast majority of tools we use will require 
# that data are represented as a `pandas` **data frame** in Python. Therefore, in this section,
# we will dig more deeply into what data frames are and how they are represented in Python.
# This knowledge will be helpful in effectively utilizing these objects in our data analyses.

# %% [markdown]
# ### What is a data frame?
#
# A data frame \index{data frame!definition} is a table-like structure for storing data in Python.  Data frames are
# important to learn about because most data that you will encounter in practice
# can be naturally stored as a table.  In order to define data frames precisely,
# we need to introduce a few technical terms:
#
# - **variable:** a \index{variable} characteristic, number, or quantity that can be measured.
# - **observation:** all \index{observation} of the measurements for a given entity.
# - **value:** a \index{value} single measurement of a single variable for a given entity.
#
# Given these definitions, a **data frame** is a tabular data structure in Python
# that is designed to store observations, variables, and their values.
# Most commonly, each column in a data frame corresponds to a variable,
# and each row corresponds to an observation. For example,
# {numref}`fig:02-obs` displays a data set of city populations. Here, the variables
# are "region, year, population"; each of these are properties that can be
# collected or measured.  The first observation is "Toronto, 2016, 2235145";
# these are the values that the three variables take for the first entity in the
# data set. There are 13 entities in the data set in total, corresponding to the
# 13 rows in {numref}`fig:02-obs`.

# %% [markdown]
# ```{figure} img/data_frame_slides_cdn/data_frame_slides_cdn.004.jpeg
# :name: fig:02-obs
# :figclass: caption-hack
#
# A data frame storing data regarding the population of various regions in Canada. In this example data frame, the row that corresponds to the observation for the city of Vancouver is colored yellow, and the column that corresponds to the population variable is colored blue.
# ```

# %% tags=["remove-cell"]
# The following cell was removed because there is no "vector" in Python.

# %% [markdown] tags=["remove-cell"]
# Python stores the columns of a data frame as either
# *lists* or *vectors*.  For example, the data frame in Figure
# {numref}`fig:02-vectors` has three vectors whose names are `region`, `year` and
# `population`. The next two sections will explain what lists and vectors are.
#
# ```{figure} img/data_frame_slides_cdn/data_frame_slides_cdn.005.jpeg
# :name: fig:02-vectors
# :figclass: caption-hack
#
# Data frame with three vectors.
# ```

# %% [markdown]
# ### What is a series?
#
# In Python, `pandas` **series** are arrays with labels. They are strictly 1-dimensional and can contain any data type (integers, strings, floats, etc), including a mix of them (objects);
# Python has several different basic data types, as shown in {numref}`tab:datatype-table`.
# You can create a `pandas` series using the `pd.Series()` function.  For 
# example, to create the vector `region` as shown in
# {numref}`fig:02-series`, you can write:

# %%
import pandas as pd
region = pd.Series(["Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa"])
region

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_dataframe_series.png
# :name: fig:02-series
# :figclass: caption-hack
#
# Example of a `pandas` series whose type is string.
# ```

# %% [markdown] tags=["remove-cell"]
# ### What is a vector?
#
# In R, **vectors** \index{vector}\index{atomic vector|see{vector}} are objects that can contain one or more elements. The vector
# elements are ordered, and they must all be of the same **data type**;
# R has several different basic data types, as shown in {numref}`tab:datatype-table`.
# Figure \@ref(fig:02-vector) provides an example of a vector where all of the elements are
# of character type.
# You can create vectors in R using the `c` function  \index{c function} (`c` stands for "concatenate").  For 
# example, to create the vector `region` as shown in Figure
# \@ref(fig:02-vector), you would write:
#
# ``` {r}
# year <- c("Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa")
# year
# ```
#
# > **Note:** Technically, these objects are called "atomic vectors." In this book
# > we have chosen to call them "vectors," which is how they are most commonly
# > referred to in the R community. To be totally precise, "vector" is an umbrella term that
# > encompasses both atomic vector and list objects in R. But this creates a 
# > confusing situation where the term "vector" could 
# > mean "atomic vector" *or* "the umbrella term for atomic vector and list," 
# > depending on context. Very confusing indeed! So to keep things simple, in 
# > this book we *always* use the term "vector" to refer to "atomic vector." 
# > We encourage readers who are enthusiastic to learn more to read the
# > Vectors chapter of *Advanced R* [@wickham2019advanced].
#
# ``` {r 02-vector, echo = FALSE, message = FALSE, warning = FALSE, fig.cap = "Example of a vector whose type is character.", fig.retina = 2, out.width = "100%"}
# image_read("img/data_frame_slides_cdn/data_frame_slides_cdn.007.jpeg") %>%
#   image_crop("3632x590")
# ```

# %% [markdown]
# \newpage

# %% tags=["remove-cell"]
# The following table was taken from DSCI511 Lecture 1, credit to Arman Seyed-Ahmadi, MDS 2021

# %% [markdown]
# ```{table} Basic data types in Python
# :name: tab:datatype-table
# | English name          | Type name  | Type Category  | Description                                   | Example                                    |
# | :-------------------- | :--------- | :------------- | :-------------------------------------------- | :----------------------------------------- |
# | integer               | `int`      | Numeric Type   | positive/negative whole numbers               | `42`                                       |
# | floating point number | `float`    | Numeric Type   | real number in decimal form                   | `3.14159`                                  |
# | boolean               | `bool`     | Boolean Values | true or false                                 | `True`                                     |
# | string                | `str`      | Sequence Type  | text                                          | `"Can I have a cheezburger?"`              |
# | list                  | `list`     | Sequence Type  | a collection of objects - mutable & ordered   | `['Ali', 'Xinyi', 'Miriam']`               |
# | tuple                 | `tuple`    | Sequence Type  | a collection of objects - immutable & ordered | `('Thursday', 6, 9, 2018)`                 |
# | dictionary            | `dict`     | Mapping Type   | mapping of key-value pairs                    | `{'name':'DSCI', 'code':100, 'credits':2}` |
# | none                  | `NoneType` | Null Object    | represents no value                           | `None`                                     |
# ```

# %% [markdown]
# \index{data types}
# \index{character}\index{chr|see{character}}
# \index{integer}\index{int|see{integer}}
# \index{double}\index{dbl|see{double}}
# \index{logical}\index{lgl|see{logical}}
# \index{factor}\index{fct|see{factor}}
# It is important in Python to make sure you represent your data with the correct type. 
# Many of the `pandas` functions we use in this book treat 
# the various data types differently. You should use integers and float types
# (which both fall under the "numeric" umbrella type) to represent numbers and perform
# arithmetic. Strings are used to represent data that should
# be thought of as "text", such as words, names, paths, URLs, and more. 
# There are other basic data types in Python, such as *set*
# and *complex*, but we do not use these in this textbook.

# %% tags=["remove-cell"]
# It is important in R to make sure you represent your data with the correct type. 
# Many of the `tidyverse` functions we use in this book treat 
# the various data types differently. You should use integers and double types
# (which both fall under the "numeric" umbrella type) to represent numbers and perform
# arithmetic. Doubles are more common than integers in R, though; for instance, a double data type is the
# default when you create a vector of numbers using `c()`, and when you read in
# whole numbers via `read_csv`. Characters are used to represent data that should
# be thought of as "text", such as words, names, paths, URLs, and more. Factors help us
# encode variables that represent *categories*; a factor variable takes one of a discrete
# set of values known as *levels* (one for each category). The levels can be ordered or unordered.  Even though
# factors can sometimes *look* like characters, they are not used to represent
# text, words, names, and paths in the way that characters are; in fact, R
# internally stores factors using integers! There are other basic data types in R, such as *raw*
# and *complex*, but we do not use these in this textbook.

# %% [markdown]
# ### What is a list?
#
# Lists \index{list} are built-in objects in Python that have multiple, ordered elements.
# `pandas` series can be treated as lists with labels (indices).

# %% tags=["remove-cell"]
# Lists \index{list} are also objects in R that have multiple, ordered elements.
# Vectors and lists differ by the requirement of element type
# consistency. All elements within a single vector must be of the same type (e.g.,
# all elements are characters), whereas elements within a single list can be of
# different types (e.g., characters, integers, logicals, and even other lists). 

# %% [markdown] tags=["remove-cell"]
# ```{figure} img/data_frame_slides_cdn/data_frame_slides_cdn.008.jpeg
# :name: fig:02-vec-vs-list
# :figclass: caption-hack
#
# A vector versus a list.
# ```

# %% [markdown]
# ### What does this have to do with data frames?

# %% [markdown]
# A data frame \index{data frame!definition} is really just series stuck together that follows two rules:
#
# 1. Each element itself is a series. 
# 2. Each element (series) must have the same length.
#
# Not all columns in a data frame need to be of the same type. 
# {numref}`fig:02-dataframe` shows a data frame where
# the columns are series of different types.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_dataframe_series-3.png
# :name: fig:02-dataframe
# :figclass: caption-hack
#
# Data frame and vector types.
# ```

# %% tags=["remove-cell"]
# A data frame \index{data frame!definition} is really a special kind of list that follows two rules:

# 1. Each element itself must either be a vector or a list. 
# 2. Each element (vector or list) must have the same length.

# Not all columns in a data frame need to be of the same type. 
# Figure \@ref(fig:02-dataframe) shows a data frame where
# the columns are vectors of different types.
# But remember: because the columns in this example are *vectors*, 
# the elements must be the same data type *within each column.* 
# On the other hand, if our data frame had *list* columns, there would be no such requirement.
# It is generally much more common to use *vector* columns, though, 
# as the values for a single variable are usually all of the same type.

# %% [markdown]
# >  **Note:** You can use the function `type` \index{class} on a data object.
# > For example we can check the class of the Canadian languages data set,
# > `can_lang`, we worked with in the previous chapters and we see it is a `pandas.core.frame.DataFrame`.

# %% tags=["remove-cell"]
# The functions from the `tidyverse` package that we use often give us a
# special class of data frame called a *tibble*. Tibbles have some additional \index{tibble}
# features and benefits over the built-in data frame object. These include the
# ability to add useful attributes (such as grouping, which we will discuss later)
# and more predictable type preservation when subsetting. 
# Because a tibble is just a data frame with some added features, 
# we will collectively refer to both built-in R data frames and
# tibbles as data frames in this book.

# >  **Note:** You can use the function `class` \index{class} on a data object to assess whether a data
# > frame is a built-in R data frame or a tibble. If the data object is a data
# > frame, `class` will return `"data.frame"`. If the data object is a
# > tibble it will return `"tbl_df" "tbl" "data.frame"`. You can easily convert
# > built-in R data frames to tibbles using the `tidyverse` `as_tibble` function.
# > For example we can check the class of the Canadian languages data set,
# > `can_lang`, we worked with in the previous chapters and we see it is a tibble.

# %%
can_lang = pd.read_csv("data/can_lang.csv")
type(can_lang)

# %% [markdown]
# Lists, Series and DataFrames are basic types of *data structure* in Python, which
# are core to most data analyses. We summarize them in 
# {numref}`tab:datastructure-table`. There are several other data structures in the Python programming 
# language (*e.g.,* matrices), but these are beyond the scope of this book.

# %% [markdown]
# ```{table} Basic data structures in Python
# :name: tab:datastructure-table
# | Data Structure | Description |
# | ---            |------------ |
# | list | An 1D ordered collection of values that can store multiple data types at once. |
# | Series | An 1D ordered collection of values *with labels* that can store multiple data types at once. |
# | DataFrame | A 2D labeled data structure with columns of potentially different types. |
# ```

# %% [markdown]
# ## Tidy data
#
# There are many ways a tabular data set can be organized. This chapter will focus
# on introducing the **tidy data** \index{tidy data!definition} format of organization and how to make your raw
# (and likely messy) data tidy. A tidy data frame satisfies 
# the following three criteria {cite:p}`wickham2014tidy`:
#
#   - each row is a single observation,
#   - each column is a single variable, and
#   - each value is a single cell (i.e., its entry in the data
#     frame is not shared with another value).
#
# {numref}`fig:02-tidy-image` demonstrates a tidy data set that satisfies these 
# three criteria.

# %% [markdown] tags=[]
# ```{figure} img/tidy_data/tidy_data.001-cropped.jpeg
# :name: fig:02-tidy-image
# :figclass: caption-hack
#
# Tidy data satisfies three criteria.
# ```

# %% [markdown]
# There are many good reasons for making sure your data are tidy as a first step in your analysis.
# The most important is that it is a single, consistent format that nearly every function
# in the `pandas` recognizes. No matter what the variables and observations 
# in your data represent, as long as the data frame \index{tidy data!arguments for}
# is tidy, you can manipulate it, plot it, and analyze it using the same tools.
# If your data is *not* tidy, you will have to write special bespoke code
# in your analysis that will not only be error-prone, but hard for others to understand.
# Beyond making your analysis more accessible to others and less error-prone, tidy data
# is also typically easy for humans to interpret. Given these benefits,
# it is well worth spending the time to get your data into a tidy format
# upfront. Fortunately, there are many well-designed `pandas` data
# cleaning/wrangling tools to help you easily tidy your data. Let's explore them
# below!
#
# > **Note:** Is there only one shape for tidy data for a given data set? Not
# > necessarily! It depends on the statistical question you are asking and what
# > the variables are for that question. For tidy data, each variable should be
# > its own column. So, just as it's essential to match your statistical question
# > with the appropriate data analysis tool, it's important to match your
# > statistical question with the appropriate variables and ensure they are
# > represented as individual columns to make the data tidy.

# %% [markdown]
# ### Tidying up: going from wide to long using `.melt`
#
# One task that is commonly performed to get data into a tidy format \index{pivot\_longer}
# is to combine values that are stored in separate columns, 
# but are really part of the same variable, into one.
# Data is often stored this way 
# because this format is sometimes more intuitive for human readability 
# and understanding, and humans create data sets.
# In {numref}`fig:02-wide-to-long`, 
# the table on the left is in an untidy, "wide" format because the year values 
# (2006, 2011, 2016) are stored as column names. 
# And as a consequence, 
# the values for population for the various cities 
# over these years are also split across several columns. 
#
# For humans, this table is easy to read, which is why you will often find data
# stored in this wide format.  However, this format is difficult to work with
# when performing data visualization or statistical analysis using Python.  For
# example, if we wanted to find the latest year it would be challenging because
# the year values are stored as column names instead of as values in a single
# column.  So before we could apply a function to find the latest year (for
# example, by using `max`), we would have to first extract the column names
# to get them as a list and then apply a function to extract the latest year.
# The problem only gets worse if you would like to find the value for the
# population for a given region for the latest year.  Both of these tasks are
# greatly simplified once the data is tidied.
#
# Another problem with data in this format is that we don't know what the
# numbers under each year actually represent. Do those numbers represent
# population size? Land area? It's not clear. 
# To solve both of these problems, 
# we can reshape this data set to a tidy data format 
# by creating a column called "year" and a column called
# "population." This transformation&mdash;which makes the data
# "longer"&mdash;is shown as the right table in
# {numref}`fig:02-wide-to-long`.

# %% [markdown] tags=[]
# ```{figure} img/pivot_functions/pivot_functions.001.jpeg
# :name: fig:02-wide-to-long
# :figclass: caption-hack
#
# Melting data from a wide to long data format.
# ```

# %% [markdown]
# We can achieve this effect in Python using the `.melt` function from the `pandas` package.
# The `.melt` function combines columns, 
# and is usually used during tidying data 
# when we need to make the data frame longer and narrower. 
# To learn how to use `.melt`, we will work through an example with the
# `region_lang_top5_cities_wide.csv` data set. This data set contains the
# counts of how many Canadians cited each language as their mother tongue for five 
# major Canadian cities (Toronto, Montréal, Vancouver, Calgary and Edmonton) from
# the 2016 Canadian census.  \index{Canadian languages}
# To get started, 
# we will use `pd.read_csv` to load the (untidy) data.

# %%
lang_wide = pd.read_csv("data/region_lang_top5_cities_wide.csv")
lang_wide

# %% [markdown]
# What is wrong with the untidy format above? 
# The table on the left in {numref}`fig:img-pivot-longer-with-table` 
# represents the data in the "wide" (messy) format.
# From a data analysis perspective, this format is not ideal because the values of 
# the variable *region* (Toronto, Montréal, Vancouver, Calgary and Edmonton) 
# are stored as column names. Thus they
# are not easily accessible to the data analysis functions we will apply
# to our data set. Additionally, the *mother tongue* variable values are
# spread across multiple columns, which will prevent us from doing any desired
# visualization or statistical tasks until we combine them into one column. For
# instance, suppose we want to know the languages with the highest number of 
# Canadians reporting it as their mother tongue among all five regions. This
# question would be tough to answer with the data in its current format. 
# We *could* find the answer with the data in this format, 
# though it would be much easier to answer if we tidy our
# data first. If mother tongue were instead stored as one column, 
# as shown in the tidy data on the right in 
# {numref}`fig:img-pivot-longer-with-table`,
# we could simply use one line of code (`df["mother_tongue"].max()`) 
# to get the maximum value.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_melt_wide-long.png
# :name: fig:img-pivot-longer-with-table
# :figclass: caption-hack
#
# Going from wide to long with the `.melt` function.
# ```

# %% [markdown]
# {numref}`fig:img-pivot-longer` details the arguments that we need to specify 
# in the `.melt` function to accomplish this data transformation.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_melt_args_labels.png
# :name: fig:img-pivot-longer
# :figclass: caption-hack
#
# Syntax for the `melt` function.
# ```

# %% [markdown]
# We use `.melt` to combine the Toronto, Montréal,
# Vancouver, Calgary, and Edmonton columns into a single column called `region`,
# and create a column called `mother_tongue` that contains the count of how many
# Canadians report each language as their mother tongue for each metropolitan
# area. We specify `value_vars` to be all
# the columns between Toronto and Edmonton:  \index{column range}\index{aaacolonsymb@\texttt{:}|see{column range}}

# %%
lang_mother_tidy = lang_wide.melt(
    id_vars=["category", "language"],
    value_vars=["Toronto", "Montréal", "Vancouver", "Calgary", "Edmonton"],
    var_name="region",
    value_name="mother_tongue",
)

lang_mother_tidy

# %% [markdown]
# > **Note**: In the code above, the call to the
# > `.melt` function is split across several lines. This is allowed in
# > certain cases; for example, when calling a function as above, as long as the 
# > line ends with a comma `,` Python knows to keep reading on the next line.
# > Splitting long lines like this across multiple lines is encouraged 
# > as it helps significantly with code readability. Generally speaking, you should
# > limit each line of code to about 80 characters.
#
# The data above is now tidy because all three criteria for tidy data have now
# been met:
#
# 1.  All the variables (`category`, `language`, `region` and `mother_tongue`) are
#     now their own columns in the data frame.
# 2.  Each observation, i.e., each `category`, `language`, `region`, and count of
#     Canadians where that language is the mother tongue, are in a single row.
# 3.  Each value is a single cell, i.e., its row, column position in the data
#     frame is not shared with another value.

# %% [markdown]
# (pivot-wider)=
# ### Tidying up: going from long to wide using `.pivot`
#
# Suppose we have observations spread across multiple rows rather than in a single \index{pivot\_wider}
# row. For example, in {numref}`fig:long-to-wide`, the table on the left is in an
# untidy, long format because the `count` column contains three variables
# (population, commuter, and incorporated count) and information about each observation 
# (here, population, commuter, and incorporated counts for a region) is split across three rows. 
# Remember: one of the criteria for tidy data 
# is that each observation must be in a single row.
#
# Using data in this format&mdash;where two or more variables are mixed together
# in a single column&mdash;makes it harder to apply many usual `pandas` functions.
# For example, finding the maximum number of commuters 
# would require an additional step of filtering for the commuter values
# before the maximum can be computed.
# In comparison, if the data were tidy, 
# all we would have to do is compute the maximum value for the commuter column.
# To reshape this untidy data set to a tidy (and in this case, wider) format,
# we need to create columns called "population", "commuters", and "incorporated."
# This is illustrated in the right table of {numref}`fig:long-to-wide`.

# %% [markdown] tags=[]
# ```{figure} img/pivot_functions/pivot_functions.002.jpeg
# :name: fig:long-to-wide
# :figclass: caption-hack
#
# Going from long to wide data.
# ```

# %% [markdown]
# To tidy this type of data in Python, we can use the `.pivot` function.
# The `.pivot` function generally increases the number of columns (widens) 
# and decreases the number of rows in a data set. 
# To learn how to use `.pivot`, 
# we will work through an example 
# with the `region_lang_top5_cities_long.csv` data set. 
# This data set contains the number of Canadians reporting 
# the primary language at home and work for five
# major cities (Toronto, Montréal, Vancouver, Calgary and Edmonton).

# %%
lang_long = pd.read_csv("data/region_lang_top5_cities_long.csv")
lang_long

# %% [markdown]
# What makes the data set shown above untidy? 
# In this example, each observation is a language in a region. 
# However, each observation is split across multiple rows: 
# one where the count for `most_at_home` is recorded, 
# and the other where the count for `most_at_work` is recorded. 
# Suppose the goal with this data was to 
# visualize the relationship between the number of
# Canadians reporting their primary language at home and work. 
# Doing that would be difficult with this data in its current form,
# since these two variables are stored in the same column.
# {numref}`fig:img-pivot-wider-table` shows how this data
# will be tidied using the `.pivot` function.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_pivot_long-wide.png
# :name: fig:img-pivot-wider-table
# :figclass: caption-hack
#
# Going from long to wide with the `.pivot` function.
# ```

# %% [markdown]
# {numref}`fig:img-pivot-wider` details the arguments that we need to specify 
# in the `.pivot` function.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_pivot_args_labels.png
# :name: fig:img-pivot-wider
# :figclass: caption-hack
#
# Syntax for the `.pivot` function.
# ```

# %% [markdown]
# We will apply the function as detailed in {numref}`fig:img-pivot-wider`.

# %%
lang_home_tidy = lang_long.pivot(
    index=["region", "category", "language"], columns=["type"], values=["count"]
).reset_index()

lang_home_tidy.columns = [
    "region",
    "category",
    "language",
    "most_at_home",
    "most_at_work",
]
lang_home_tidy

# %%
lang_home_tidy.dtypes

# %% [markdown]
# The data above is now tidy! We can go through the three criteria again to check
# that this data is a tidy data set.
#
# 1.  All the statistical variables are their own columns in the data frame (i.e.,
#     `most_at_home`, and `most_at_work` have been separated into their own
#     columns in the data frame).
# 2.  Each observation, (i.e., each language in a region) is in a single row.
# 3.  Each value is a single cell (i.e., its row, column position in the data
#     frame is not shared with another value).
#
# You might notice that we have the same number of columns in the tidy data set as
# we did in the messy one. Therefore `.pivot` didn't really "widen" the data.
# This is just because the original `type` column only had
# two categories in it. If it had more than two, `.pivot` would have created
# more columns, and we would see the data set "widen."

# %% [markdown]
# (str-split)=
# ### Tidying up: using `.str.split` to deal with multiple delimiters
#
# Data are also not considered tidy when multiple values are stored in the same \index{separate}
# cell. The data set we show below is even messier than the ones we dealt with
# above: the `Toronto`, `Montréal`, `Vancouver`, `Calgary` and `Edmonton` columns
# contain the number of Canadians reporting their primary language at home and
# work in one column separated by the delimiter (`/`). The column names are the \index{delimiter}
# values of a variable, *and* each value does not have its own cell! To turn this
# messy data into tidy data, we'll have to fix these issues.

# %%
lang_messy = pd.read_csv("data/region_lang_top5_cities_messy.csv")
lang_messy

# %% [markdown]
# First we’ll use `.melt` to create two columns, `region` and `value`,
# similar to what we did previously. 
# The new `region` columns will contain the region names,
# and the new column `value` will be a temporary holding place for the 
# data that we need to further separate, i.e., the 
# number of Canadians reporting their primary language at home and work.

# %%
lang_messy_longer = lang_messy.melt(
    id_vars=["category", "language"],
    value_vars=["Toronto", "Montréal", "Vancouver", "Calgary", "Edmonton"],
    var_name="region",
    value_name="value",
)

lang_messy_longer

# %% [markdown]
# Next we'll use `.str.split` to split the `value` column into two columns. 
# One column will contain only the counts of Canadians 
# that speak each language most at home, 
# and the other will contain the counts of Canadians 
# that speak each language most at work for each region. 
# {numref}`fig:img-separate`
# outlines what we need to specify to use `.str.split`.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/str-split_args_labels.png
# :name: fig:img-separate
# :figclass: caption-hack
#
# Syntax for the `.str.split` function.
# ```

# %%
tidy_lang = (
    pd.concat(
        (lang_messy_longer, lang_messy_longer["value"].str.split("/", expand=True)),
        axis=1,
    )
    .rename(columns={0: "most_at_home", 1: "most_at_work"})
    .drop(columns=["value"])
)

tidy_lang

# %%
tidy_lang.dtypes

# %% [markdown]
# Is this data set now tidy? If we recall the three criteria for tidy data:
#
#   - each row is a single observation,
#   - each column is a single variable, and
#   - each value is a single cell.
#
# We can see that this data now satisfies all three criteria, making it easier to
# analyze. But we aren't done yet! Notice in the table, all of the variables are
# "object" data types. Object data types are columns of strings or columns with mixed types. In the previous example in Section {ref}`pivot-wider`, the
# `most_at_home` and `most_at_work` variables were `int64` (integer)&mdash;you can
# verify this by calling `df.dtypes`&mdash;which is a type
# of numeric data. This change is due to the delimiter (`/`) when we read in this
# messy data set. Python read these columns in as string types, and by default,
# `.str.split` will return columns as object data types.
#
# It makes sense for `region`, `category`, and `language` to be stored as a
# object type. However, suppose we want to apply any functions that treat the
# `most_at_home` and `most_at_work` columns as a number (e.g., finding rows 
# above a numeric threshold of a column). 
# In that case, 
# it won't be possible to do if the variable is stored as a `object`. 
# Fortunately, the `pandas.to_numeric` function provides a natural way to fix problems
# like this: it will convert the columns to the best numeric data types.

# %% tags=["remove-cell"]
# We can see that this data now satisfies all three criteria, making it easier to
# analyze. But we aren't done yet! Notice in the table above that the word
# `<chr>` appears beneath each of the column names. The word under the column name
# indicates the data type of each column. Here all of the variables are
# "character" data types. Recall, character data types are letter(s) or digits(s)
# surrounded by quotes. In the previous example in Section \@ref(pivot-wider), the
# `most_at_home` and `most_at_work` variables were `<dbl>` (double)&mdash;you can
# verify this by looking at the tables in the previous sections&mdash;which is a type
# of numeric data. This change is due to the delimiter (`/`) when we read in this
# messy data set. R read these columns in as character types, and by default,
# `separate` will return columns as character data types.

# It makes sense for `region`, `category`, and `language` to be stored as a
# character (or perhaps factor) type. However, suppose we want to apply any functions that treat the
# `most_at_home` and `most_at_work` columns as a number (e.g., finding rows 
# above a numeric threshold of a column). 
# In that case, 
# it won't be possible to do if the variable is stored as a `character`. 
# Fortunately, the `separate` function provides a natural way to fix problems
# like this: we can set `convert = TRUE` to convert the `most_at_home` 
# and `most_at_work` columns to the correct data type.

# %%
tidy_lang["most_at_home"] = pd.to_numeric(tidy_lang["most_at_home"])
tidy_lang["most_at_work"] = pd.to_numeric(tidy_lang["most_at_work"])
tidy_lang

# %%
tidy_lang.dtypes

# %% [markdown]
# Now we see `most_at_home` and `most_at_work` columns are of `int64` data types,
# indicating they are integer data types (i.e., numbers)!

# %% [markdown]
# (loc-iloc)=
# ## Using `.loc[]` and `.iloc[]` to extract a range of columns
#
# Now that the `tidy_lang` data is indeed *tidy*, we can start manipulating it \index{select!helpers}
# using the powerful suite of functions from the `pandas`. 
# For the first example, recall `.loc[]` from Chapter {ref}`intro`, 
# which lets us create a subset of columns from a data frame. 
# Suppose we wanted to select only the columns `language`, `region`,
# `most_at_home` and `most_at_work` from the `tidy_lang` data set. Using what we
# learned in Chapter {ref}`intro`, we would pass all of these column names into the square brackets:

# %%
selected_columns = tidy_lang.loc[:, ["language", "region", "most_at_home", "most_at_work"]]
selected_columns

# %% [markdown]
# Here we wrote out the names of each of the columns. However, this method is
# time-consuming, especially if you have a lot of columns! Another approach is to
# index with integers. `.iloc[]` make it easier for
# us to select columns. For instance, we can use `.iloc[]` to choose a
# range of columns rather than typing each column name out. To do this, we use the
# colon (`:`) operator to denote the range. For example, to get all the columns in \index{column range}
# the `tidy_lang` data frame from `language` to `most_at_work`, we pass `:` before the comma indicating we want to retrieve all rows, and `1:` after the comma indicating we want only columns from index 1 (*i.e.* `language`) and afterwords.

# %% tags=["remove-cell"]
# Here we wrote out the names of each of the columns. However, this method is
# time-consuming, especially if you have a lot of columns! Another approach is to
# use a "select helper". Select helpers are operators that make it easier for
# us to select columns. For instance, we can use a select helper to choose a
# range of columns rather than typing each column name out. To do this, we use the
# colon (`:`) operator to denote the range. For example, to get all the columns in \index{column range}
# the `tidy_lang` data frame from `language` to `most_at_work` we pass
# `language:most_at_work` as the second argument to the `select` function.

# %%
column_range = tidy_lang.iloc[:, 1:]
column_range

# %% [markdown]
# Notice that we get the same output as we did above, 
# but with less (and clearer!) code. This type of operator 
# is especially handy for large data sets.
#
# Suppose instead we wanted to extract columns that followed a particular pattern
# rather than just selecting a range. For example, let's say we wanted only to select the
# columns `most_at_home` and `most_at_work`. There are other functions that allow
# us to select variables based on their names. In particular, we can use the `.str.startswith` method \index{select!starts\_with}
# to choose only the columns that start with the word "most":

# %%
tidy_lang.loc[:, tidy_lang.columns.str.startswith('most')]

# %% [markdown]
# We could also have chosen the columns containing an underscore `_` by using the 
# `.str.contains("_")`, since we notice
# the columns we want contain underscores and the others don't. \index{select!contains}

# %%
tidy_lang.loc[:, tidy_lang.columns.str.contains('_')]

# %% [markdown]
# There are many different functions that help with selecting
# variables based on certain criteria. 
# The additional resources section at the end of this chapter 
# provides a comprehensive resource on these functions.

# %% tags=["remove-cell"]
# There are many different `select` helpers that select
# variables based on certain criteria. 
# The additional resources section at the end of this chapter 
# provides a comprehensive resource on `select` helpers.

# %% [markdown]
# ## Using `df[]` to extract rows
#
# Next, we revisit the `df[]` from Chapter {ref}`intro`, 
# which lets us create a subset of rows from a data frame. 
# Recall the argument to the `df[]`:
# column names or a logical statement evaluated to either `True` or `False`;
# `df[]` works by returning the rows  where the logical statement evaluates to `True`.
# This section will highlight more advanced usage of the `df[]` function.
# In particular, this section provides an in-depth treatment of the variety of logical statements
# one can use in the `df[]` to select subsets of rows.

# %% [markdown]
# ### Extracting rows that have a certain value with `==`
# Suppose we are only interested in the subset of rows in `tidy_lang` corresponding to the
# official languages of Canada (English and French).
# We can extract these rows by using the *equivalency operator* (`==`) 
# to compare the values of the `category` column 
# with the value `"Official languages"`. 
# With these arguments, `df[]` returns a data frame with all the columns 
# of the input data frame 
# but only the rows we asked for in the logical statement, i.e., 
# those where the `category` column holds the value `"Official languages"`.
# We name this data frame `official_langs`.

# %%
official_langs = tidy_lang[tidy_lang["category"] == "Official languages"]
official_langs

# %% [markdown]
# ### Extracting rows that do not have a certain value with `!=`
#
# What if we want all the other language categories in the data set *except* for
# those in the `"Official languages"` category? We can accomplish this with the `!=` 
# operator, which means "not equal to". So if we want to find all the rows
# where the `category` does *not* equal `"Official languages"` we write the code
# below.

# %%
tidy_lang[tidy_lang["category"] != "Official languages"]

# %% [markdown]
# (filter-and)=
# ### Extracting rows satisfying multiple conditions using `&`
#
# Suppose now we want to look at only the rows 
# for the French language in Montréal. 
# To do this, we need to filter the data set 
# to find rows that satisfy multiple conditions simultaneously. 
# We can do this with the ampersand symbol (`&`), which
# is interpreted by Python as "and". 
# We write the code as shown below to filter the `official_langs` data frame 
# to subset the rows where `region == "Montréal"` 
# *and* the `language == "French"`.

# %%
tidy_lang[(tidy_lang["region"] == "Montréal") & (tidy_lang["language"] == "French")]

# %% [markdown] tags=[]
# ### Extracting rows satisfying at least one condition using `|`
#
# Suppose we were interested in only those rows corresponding to cities in Alberta
# in the `official_langs` data set (Edmonton and Calgary). 
# We can't use `&` as we did above because `region`
# cannot be both Edmonton *and* Calgary simultaneously. 
# Instead, we can use the vertical pipe (`|`) logical operator, 
# which gives us the cases where one condition *or* 
# another condition *or* both are satisfied. 
# In the code below, we ask Python to return the rows
# where the `region` columns are equal to "Calgary" *or* "Edmonton".

# %%
official_langs[
    (official_langs["region"] == "Calgary") | (official_langs["region"] == "Edmonton")
]

# %% [markdown]
# ### Extracting rows with values in a list using `.isin()`
#
# Next, suppose we want to see the populations of our five cities. 
# Let's read in the `region_data.csv` file 
# that comes from the 2016 Canadian census, 
# as it contains statistics for number of households, land area, population 
# and number of dwellings for different regions.

# %%
region_data = pd.read_csv("data/region_data.csv")
region_data

# %% [markdown]
# To get the population of the five cities 
# we can filter the data set using the `.isin` method. 
# The `.isin` method is used to see if an element belongs to a list. 
# Here we are filtering for rows where the value in the `region` column
# matches any of the five cities we are intersted in: Toronto, Montréal,
# Vancouver, Calgary, and Edmonton.

# %%
city_names = ["Toronto", "Montréal", "Vancouver", "Calgary", "Edmonton"]
five_cities = region_data[region_data["region"].isin(city_names)]
five_cities

# %% [markdown]
# > **Note:** What's the difference between `==` and `.isin`? Suppose we have two
# > Series, `seriesA` and `seriesB`. If you type `seriesA == seriesB` into Python it
# > will compare the series element by element. Python checks if the first element of
# > `seriesA` equals the first element of `seriesB`, the second element of
# > `seriesA` equals the second element of `seriesB`, and so on. On the other hand,
# > `seriesA.isin(seriesB)` compares the first element of `seriesA` to all the
# > elements in `seriesB`. Then the second element of `seriesA` is compared
# > to all the elements in `seriesB`, and so on. Notice the difference between `==` and
# > `.isin` in the example below.

# %%
pd.Series(["Vancouver", "Toronto"]) == pd.Series(["Toronto", "Vancouver"])

# %%
pd.Series(["Vancouver", "Toronto"]).isin(pd.Series(["Toronto", "Vancouver"]))

# %% tags=["remove-cell"]
# > **Note:** What's the difference between `==` and `%in%`? Suppose we have two
# > vectors, `vectorA` and `vectorB`. If you type `vectorA == vectorB` into R it
# > will compare the vectors element by element. R checks if the first element of
# > `vectorA` equals the first element of `vectorB`, the second element of
# > `vectorA` equals the second element of `vectorB`, and so on. On the other hand,
# > `vectorA %in% vectorB` compares the first element of `vectorA` to all the
# > elements in `vectorB`. Then the second element of `vectorA` is compared
# > to all the elements in `vectorB`, and so on. Notice the difference between `==` and
# > `%in%` in the example below.
# > 
# >``` {r}
# >c("Vancouver", "Toronto") == c("Toronto", "Vancouver")
# >c("Vancouver", "Toronto") %in% c("Toronto", "Vancouver")
# >```

# %% [markdown]
# ### Extracting rows above or below a threshold using `>` and `<`

# %% tags=["remove-cell"]
glue("census_popn", "{0:,.0f}".format(35151728))
glue("most_french", "{0:,.0f}".format(2669195))

# %% [markdown]
# We saw in Section {ref}`filter-and` that 
# {glue:text}`most_french` people reported 
# speaking French in Montréal as their primary language at home. 
# If we are interested in finding the official languages in regions 
# with higher numbers of people who speak it as their primary language at home 
# compared to French in Montréal, then we can use `df[]` to obtain rows 
# where the value of `most_at_home` is greater than 
# {glue:text}`most_french`.

# %%
official_langs[official_langs["most_at_home"] > 2669195]

# %% [markdown]
# This operation returns a data frame with only one row, indicating that when 
# considering the official languages, 
# only English in Toronto is reported by more people 
# as their primary language at home 
# than French in Montréal according to the 2016 Canadian census.

# %% [markdown] tags=[]
# (pandas-assign)=
# ## Using `.assign` to modify or add columns

# %% [markdown]
# ### Using `.assign` to modify columns
# In Section {ref}`str-split`, 
# when we first read in the `"region_lang_top5_cities_messy.csv"` data,
# all of the variables were "object" data types. \index{mutate}
# During the tidying process, 
# we used the `pandas.to_numeric` function 
# to convert the `most_at_home` and `most_at_work` columns 
# to the desired integer (i.e., numeric class) data types and then used `df[]` to overwrite columns. 
# But suppose we didn't use the `df[]`,
# and needed to modify the columns some other way.
# Below we create such a situation 
# so that we can demonstrate how to use `.assign`
# to change the column types of a data frame. 
# `.assign` is a useful function to modify or create new data frame columns.

# %%
lang_messy = pd.read_csv("data/region_lang_top5_cities_messy.csv")
lang_messy_longer = lang_messy.melt(
    id_vars=["category", "language"],
    value_vars=["Toronto", "Montréal", "Vancouver", "Calgary", "Edmonton"],
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

# %%
official_langs_obj.dtypes

# %% [markdown]
# To use the `.assign` method, again we first specify the object to be the data set, 
# and in the following arguments, 
# we specify the name of the column we want to modify or create 
# (here `most_at_home` and `most_at_work`), an `=` sign,
# and then the function we want to apply (here `pandas.to_numeric`).
# In the function we want to apply, 
# we refer to the column upon which we want it to act 
# (here `most_at_home` and `most_at_work`).
# In our example, we are naming the columns the same
# names as columns that already exist in the data frame 
# ("most\_at\_home", "most\_at\_work") 
# and this will cause `.assign` to *overwrite* those columns 
# (also referred to as modifying those columns *in-place*).
# If we were to give the columns a new name, 
# then `.assign` would create new columns with the names we specified.
# `.assign`'s general syntax is detailed in {numref}`fig:img-assign`.

# %% [markdown] tags=[]
# ```{figure} img/wrangling/pandas_assign_args_labels.png
# :name: fig:img-assign
# :figclass: caption-hack
#
# Syntax for the `.assign` function.
# ```

# %% [markdown]
# Below we use `.assign` to convert the columns `most_at_home` and `most_at_work`
# to numeric data types in the `official_langs` data set as described in 
# {numref}`fig:img-assign`:

# %%
official_langs_numeric = official_langs_obj.assign(
    most_at_home=pd.to_numeric(official_langs_obj["most_at_home"]),
    most_at_work=pd.to_numeric(official_langs_obj["most_at_work"]),
)

official_langs_numeric

# %%
official_langs_numeric.dtypes

# %% [markdown]
# Now we see that the `most_at_home` and `most_at_work` columns are both `int64` (which is a numeric data type)!

# %% [markdown]
# ### Using `.assign` to create new columns

# %% tags=["remove-cell"]
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

# %% [markdown]
# We can see in the table that
# {glue:text}`number_most_home` people reported
# speaking English in Toronto as their primary language at home, according to
# the 2016 Canadian census. What does this number mean to us? To understand this
# number, we need context. In particular, how many people were in Toronto when
# this data was collected? From the 2016 Canadian census profile, the population
# of Toronto was reported to be
# {glue:text}`toronto_popn` people. 
# The number of people who report that English is their primary language at home 
# is much more meaningful when we report it in this context. 
# We can even go a step further and transform this count to a relative frequency 
# or proportion.
# We can do this by dividing the number of people reporting a given language 
# as their primary language at home by the number of people who live in Toronto. 
# For example, 
# the proportion of people who reported that their primary language at home 
# was English in the 2016 Canadian census was {glue:text}`prop_eng_tor`
# in Toronto.
#
# Let's use `.assign` to create a new column in our data frame 
# that holds the proportion of people who speak English 
# for our five cities of focus in this chapter. 
# To accomplish this, we will need to do two tasks 
# beforehand:
#
# 1. Create a list containing the population values for the cities.
# 2. Filter the `official_langs` data frame 
# so that we only keep the rows where the language is English.
#
# To create a list containing the population values for the five cities
# (Toronto, Montréal, Vancouver, Calgary, Edmonton),
# we will use the `[]` (recall that we can also use `list()` to create a list):

# %%
city_pops = [5928040, 4098927, 2463431, 1392609, 1321426]
city_pops

# %% [markdown]
# And next, we will filter the `official_langs` data frame 
# so that we only keep the rows where the language is English.
# We will name the new data frame we get from this `english_langs`:

# %%
english_langs = official_langs[official_langs["language"] == "English"]
english_langs

# %% [markdown]
# Finally, we can use `.assign` to create a new column, 
# named `most_at_home_proportion`, that will have value that corresponds to 
# the proportion of people reporting English as their primary
# language at home.
# We will compute this by dividing the column by our vector of city populations.

# %%
english_langs = english_langs.assign(
    most_at_home_proportion=english_langs["most_at_home"] / city_pops
)

english_langs

# %% [markdown]
# In the computation above, we had to ensure that we ordered the `city_pops` vector in the
# same order as the cities were listed in the `english_langs` data frame.
# This is because Python will perform the division computation we did by dividing 
# each element of the `most_at_home` column by each element of the 
# `city_pops` list, matching them up by position.
# Failing to do this would have resulted in the incorrect math being performed.
#
# > **Note:** In more advanced data wrangling, 
# > one might solve this problem in a less error-prone way though using 
# > a technique called "joins". 
# > We link to resources that discuss this in the additional
# > resources at the end of this chapter.

# %% [markdown]
# <!--
# #### Creating a visualization with tidy data {-}
#
# Now that we have cleaned and wrangled the data, we can make visualizations or do 
# statistical analyses to answer questions about it! Let's suppose we want to
# answer the question "what proportion of people in each city speak English 
# as their primary language at home in these five cities?" Since the data is
# cleaned already, in a few short lines of code, we can use `ggplot` to create a
# data visualization to answer this question! Here we create a bar plot to represent the proportions for
# each region and color the proportions by language.
#
# > Don't worry too much about the code to make this plot for now. We will cover 
# > visualizations in detail in Chapter \@ref(viz).
#
# ```{r 02-plot, out.width = "100%", fig.cap = "Bar plot of proportions of Canadians reporting English as the most often spoken language at home."}
# ggplot(english_langs,
#   aes(
#     x = region, 
#     y = most_at_home_proportion
#   )
#  ) +
#   geom_bar(stat = "identity") +
#   xlab("Region") +
#   ylab("Proportion of Canadians that speak English most often at home")
# ```
#
# From this visualization, we can see that in Calgary, Edmonton, Toronto, and
# Vancouver, English was reported as the most common primary language used at
# home.  However, in Montréal, this does not seem to be the case!
# -->

# %% [markdown]
# ## Combining functions by chaining the methods
#
# In Python, we often have to call multiple methods in a sequence to process a data
# frame. The basic ways of doing this can become quickly unreadable if there are
# many steps. For example, suppose we need to perform three operations on a data
# frame called `data`:
#
# 1)  add a new column `new_col` that is double another `old_col`,
# 2)  filter for rows where another column, `other_col`, is more than 5, and
# 3)  select only the new column `new_col` for those rows.
#
# One way of performing these three steps is to just write 
# multiple lines of code, storing temporary objects as you go:

# %% tags=["remove-cell"]
# ## Combining functions using the pipe operator, `|>`

# In R, we often have to call multiple functions in a sequence to process a data
# frame. The basic ways of doing this can become quickly unreadable if there are
# many steps. For example, suppose we need to perform three operations on a data
# frame called `data`:  \index{pipe}\index{aaapipesymb@\vert{}>|see{pipe}}

# %% tags=["remove-cell"]
data = pd.DataFrame({"old_col": [1, 2, 5, 0], "other_col": [1, 10, 3, 6]})

# %% tags=["remove-output"]
output_1 = data.assign(new_col=data["old_col"] * 2)
output_2 = output_1[output_1["other_col"] > 5]
output = output_2.loc[:, "new_col"]

# %% [markdown]
# This is difficult to understand for multiple reasons. The reader may be tricked
# into thinking the named `output_1` and `output_2` objects are important for some
# reason, while they are just temporary intermediate computations. Further, the
# reader has to look through and find where `output_1` and `output_2` are used in
# each subsequent line.

# %% [markdown]
# Chaining the sequential functions solves this problem, resulting in cleaner and
# easier-to-follow code. 
# The code below accomplishes the same thing as the previous
# two code blocks:

# %% tags=["remove-output"]
output = (
    data.assign(new_col=data["old_col"] * 2)
    .query("other_col > 5")
    .loc[:, "new_col"]
)

# %% tags=["remove-cell"]
# ``` {r eval = F}
# output <- select(filter(mutate(data, new_col = old_col * 2), 
#                         other_col > 5), 
#                  new_col)
# ```
# Code like this can also be difficult to understand. Functions compose (reading
# from left to right) in the *opposite order* in which they are computed by R
# (above, `mutate` happens first, then `filter`, then `select`). It is also just a
# really long line of code to read in one go.

# The *pipe operator* (`|>`) solves this problem, resulting in cleaner and
# easier-to-follow code. `|>` is built into R so you don't need to load any
# packages to use it. 
# You can think of the pipe as a physical pipe. It takes the output from the
# function on the left-hand side of the pipe, and passes it as the first argument
# to the function on the right-hand side of the pipe. 
# The code below accomplishes the same thing as the previous
# two code blocks:

# %% [markdown]
# > **Note:** You might also have noticed that we split the function calls across
# > lines, similar to when we did this earlier in the chapter
# > for long function calls. Again, this is allowed and recommended, especially when
# > the chained function calls create a long line of code. Doing this makes
# > your code more readable. When you do this, it is important to use parentheses 
# > to tell Python that your code is continuing onto the next line.

# %% tags=["remove-cell"]
# > **Note:** You might also have noticed that we split the function calls across
# > lines after the pipe, similar to when we did this earlier in the chapter
# > for long function calls. Again, this is allowed and recommended, especially when
# > the piped function calls create a long line of code. Doing this makes
# > your code more readable. When you do this, it is important to end each line
# > with the pipe operator `|>` to tell R that your code is continuing onto the
# > next line.

# > **Note:** In this textbook, we will be using the base R pipe operator syntax, `|>`.
# > This base R `|>` pipe operator was inspired by a previous version of the pipe
# > operator, `%>%`. The `%>%` pipe operator is not built into R 
# > and is from the `magrittr` R package. 
# > The `tidyverse` metapackage imports the `%>%` pipe operator via `dplyr` 
# > (which in turn imports the `magrittr` R package).
# > There are some other differences between `%>%` and `|>` related to 
# > more advanced R uses, such as sharing and distributing code as R packages, 
# > however, these are beyond the scope of this textbook. 
# > We have this note in the book to make the reader aware that `%>%` exists
# > as it is still commonly used in data analysis code and in many data science 
# > books and other resources.
# > In most cases these two pipes are interchangeable and either can be used.

# \index{pipe}\index{aaapipesymbb@\%>\%|see{pipe}}

# %% [markdown]
# ### Chaining `df[]` and `.loc`

# %% [markdown]
# Let's work with the tidy `tidy_lang` data set from Section {ref}`str-split`, 
# which contains the number of Canadians reporting their primary language at home 
# and work for five major cities 
# (Toronto, Montréal, Vancouver, Calgary, and Edmonton):

# %%
tidy_lang

# %% [markdown]
# Suppose we want to create a subset of the data with only the languages and
# counts of each language spoken most at home for the city of Vancouver. To do
# this, we can use the `df[]` and `.loc`. First, we use `df[]` to
# create a data frame called `van_data` that contains only values for Vancouver.

# %%
van_data = tidy_lang[tidy_lang["region"] == "Vancouver"]
van_data

# %% [markdown]
# We then use `.loc` on this data frame to keep only the variables we want:

# %%
van_data_selected = van_data.loc[:, ["language", "most_at_home"]]
van_data_selected

# %% [markdown]
# Although this is valid code, there is a more readable approach we could take by
# chaining the operations. With chaining, we do not need to create an intermediate
# object to store the output from `df[]`. Instead, we can directly call `.loc` upon the
# output of `df[]`:

# %%
van_data_selected = tidy_lang[tidy_lang["region"] == "Vancouver"].loc[
    :, ["language", "most_at_home"]
]

van_data_selected

# %% tags=["remove-cell"]
# But wait...Why do the `select` and `filter` function calls 
# look different in these two examples? 
# Remember: when you use the pipe, 
# the output of the first function is automatically provided 
# as the first argument for the function that comes after it. 
# Therefore you do not specify the first argument in that function call. 
# In the code above,
# the first line is just the `tidy_lang` data frame with a pipe.
# The pipe passes the left-hand side (`tidy_lang`) to the first argument of the function on the right (`filter`),
# so in the `filter` function you only see the second argument (and beyond).
# Then again after `filter` there is a pipe, which passes the result of the `filter` step
# to the first argument of the `select` function.

# %% [markdown]
# As you can see, both of these approaches&mdash;with and without chaining&mdash;give us the same output, but the second
# approach is clearer and more readable.

# %% [markdown]
# ### Chaining more than two functions

# %% [markdown]
# Chaining can be used with any method in Python. 
# Additionally, we can chain together more than two functions. 
# For example, we can chain together three functions to: 
#
# - extract rows (`df[]`) to include only those where the counts of the language most spoken at home are greater than 10,000, 
# - extract only the columns (`.loc`) corresponding to `region`, `language` and `most_at_home`, and
# - sort the data frame rows in order (`.sort_values`) by counts of the language most spoken at home 
# from smallest to largest.
#
# As we saw in Chapter {ref}`intro`, 
# we can use the `.sort_values` function \index{arrange}
# to order the rows in the data frame by the values of one or more columns. 
# Here we pass the column name `most_at_home` to sort the data frame rows by the values in that column, in ascending order.

# %%
large_region_lang = (
    tidy_lang[tidy_lang["most_at_home"] > 10000]
    .loc[:, ["region", "language", "most_at_home"]]
    .sort_values(by="most_at_home")
)

large_region_lang

# %% tags=["remove-cell"]
# You will notice above that we passed `tidy_lang` as the first argument of the `filter` function.
# We can also pipe the data frame into the same sequence of functions rather than
# using it as the first argument of the first function. These two choices are equivalent,
# and we get the same result.
# ``` {r}
# large_region_lang <- tidy_lang |> 
#   filter(most_at_home > 10000) |>
#   select(region, language, most_at_home) |>
#   arrange(most_at_home)

# large_region_lang
# ```

# %% [markdown]
# Now that we've shown you chaining as an alternative to storing
# temporary objects and composing code, does this mean you should *never* store 
# temporary objects or compose code? Not necessarily! 
# There are times when you will still want to do these things. 
# For example, you might store a temporary object before feeding it into a plot function 
# so you can iteratively change the plot without having to
# redo all of your data transformations. 
# Additionally, chaining many functions can be overwhelming and difficult to debug;
# you may want to store a temporary object midway through to inspect your result
# before moving on with further steps.

# %% [markdown]
# ## Aggregating data with `.assign`, `.agg` and `.apply`

# %% [markdown]
# ### Calculating summary statistics on whole columns
#
# As a part of many data analyses, we need to calculate a summary value for the
# data (a *summary statistic*). \index{summarize}
# Examples of summary statistics we might want to calculate 
# are the number of observations, the average/mean value for a column, 
# the minimum value, etc. 
# Oftentimes, 
# this summary statistic is calculated from the values in a data frame column, 
# or columns, as shown in {numref}`fig:summarize`.

# %% [markdown] tags=[]
# ```{figure} img/summarize/summarize.001.jpeg
# :name: fig:summarize
# :figclass: caption-hack
#
# Calculating summary statistics on one or more column(s). In its simplest use case, it creates a new data frame with a single row containing the summary statistic(s) for each column being summarized. The darker, top row of each table represents the column headers.
# ```

# %% [markdown]
# We can use `.assign` as mentioned in Section {ref}`pandas-assign` along with proper summary functions to create a aggregated column.
#
# First a reminder of what `region_lang` looks like:

# %% tags=["remove-cell"]
# A useful `dplyr` function for calculating summary statistics is `summarize`, 
# where the first argument is the data frame and subsequent arguments
# are the summaries we want to perform. 
# Here we show how to use the `summarize` function to calculate the minimum 
# and maximum number of Canadians 
# reporting a particular language as their primary language at home.
# First a reminder of what `region_lang` looks like:

# %%
region_lang = pd.read_csv("data/region_lang.csv")
region_lang

# %% [markdown]
# We apply `min` to calculate the minimum 
# and `max` to calculate maximum number of Canadians 
# reporting a particular language as their primary language at home, 
# for any region, and `.assign` a column name to each:

# %% tags=["remove-cell"]
pd.DataFrame(region_lang["most_at_home"].agg(["min", "max"])).T

# pd.DataFrame(region_lang["most_at_home"].agg(["min", "max"])).T.rename(
#     columns={"min": "min_most_at_home", "max": "max_most_at_home"}
# )

# %% tags=[]
lang_summary = pd.DataFrame()
lang_summary = lang_summary.assign(min_most_at_home=[min(region_lang["most_at_home"])])
lang_summary = lang_summary.assign(max_most_at_home=[max(region_lang["most_at_home"])])
lang_summary

# %% tags=["remove-cell"]
glue("lang_most_people", "{0:,.0f}".format(int(lang_summary["max_most_at_home"])))

# %% [markdown]
# From this we see that there are some languages in the data set that no one speaks
# as their primary language at home. We also see that the most commonly spoken
# primary language at home is spoken by
# {glue:text}`lang_most_people`
# people.

# %% [markdown]
# ### Calculating summary statistics when there are `NaN`s
#
# In `pandas` DataFrame, the value `NaN` is often used to denote missing data. 
# Many of the base python statistical summary functions 
# (e.g., `max`, `min`, `sum`, etc) will return `NaN` 
# when applied to columns containing `NaN` values. \index{missing data}\index{NA|see{missing data}}
# Usually that is not what we want to happen; 
# instead, we would usually like Python to ignore the missing entries
# and calculate the summary statistic using all of the other non-`NaN` values
# in the column.
# Fortunately `pandas` provides many equivalent methods (e.g., `.max`, `.min`, `.sum`, etc) to 
# these summary functions while providing an extra argument `skipna` that lets
# us tell the function what to do when it encounters `NaN` values.
# In particular, if we specify `skipna=True` (default), the function will ignore
# missing values and return a summary of all the non-missing entries.
# We show an example of this below.
#
# First we create a new version of the `region_lang` data frame,
# named `region_lang_na`, that has a seemingly innocuous `NaN` 
# in the first row of the `most_at_home` column:

# %% tags=["remove-cell"]
# In data frames in R, the value `NA` is often used to denote missing data. 
# Many of the base R statistical summary functions 
# (e.g., `max`, `min`, `mean`, `sum`, etc) will return `NA` 
# when applied to columns containing `NA` values. \index{missing data}\index{NA|see{missing data}}
# Usually that is not what we want to happen; 
# instead, we would usually like R to ignore the missing entries
# and calculate the summary statistic using all of the other non-`NA` values
# in the column.
# Fortunately many of these functions provide an argument `na.rm` that lets
# us tell the function what to do when it encounters `NA` values.
# In particular, if we specify `na.rm = TRUE`, the function will ignore
# missing values and return a summary of all the non-missing entries.
# We show an example of this combined with `summarize` below.

# %% tags=["remove-cell"]
region_lang_na = region_lang.copy()
region_lang_na.loc[0, "most_at_home"] = np.nan

# %%
region_lang_na

# %% [markdown]
# Now if we apply the Python built-in summary function as above, 
# we see that we no longer get the minimum and maximum returned, 
# but just an `NaN` instead!

# %%
lang_summary_na = pd.DataFrame()
lang_summary_na = lang_summary_na.assign(
    min_most_at_home=[min(region_lang_na["most_at_home"])]
)
lang_summary_na = lang_summary_na.assign(
    max_most_at_home=[max(region_lang_na["most_at_home"])]
)
lang_summary_na

# %% [markdown]
# We can fix this by using the `pandas` Series methods (*i.e.* `.min` and `.max`) with `skipna=True` as explained above:

# %%
lang_summary_na = pd.DataFrame()
lang_summary_na = lang_summary_na.assign(
    min_most_at_home=[region_lang_na["most_at_home"].min(skipna=True)]
)
lang_summary_na = lang_summary_na.assign(
    max_most_at_home=[region_lang_na["most_at_home"].max(skipna=True)]
)
lang_summary_na

# %% [markdown]
# ### Calculating summary statistics for groups of rows

# %% [markdown]
# A common pairing with summary functions is `.groupby`. Pairing these functions \index{group\_by}
# together can let you summarize values for subgroups within a data set,
# as illustrated in {numref}`fig:summarize-groupby`. 
# For example, we can use `.groupby` to group the regions of the `tidy_lang` data frame and then calculate the minimum and maximum number of Canadians 
# reporting the language as the primary language at home 
# for each of the regions in the data set.

# %% tags=["remove-cell"]
# A common pairing with `summarize` is `group_by`. Pairing these functions \index{group\_by}
# together can let you summarize values for subgroups within a data set,
# as illustrated in Figure \@ref(fig:summarize-groupby). 
# For example, we can use `group_by` to group the regions of the `tidy_lang` data frame and then calculate the minimum and maximum number of Canadians 
# reporting the language as the primary language at home 
# for each of the regions in the data set.

# (ref:summarize-groupby) `summarize` and `group_by` is useful for calculating summary statistics on one or more column(s) for each group. It creates a new data frame&mdash;with one row for each group&mdash;containing the summary statistic(s) for each column being summarized. It also creates a column listing the value of the grouping variable. The darker, top row of each table represents the column headers. The gray, blue, and green colored rows correspond to the rows that belong to each of the three groups being represented in this cartoon example.

# %% [markdown] tags=[]
# ```{figure} img/summarize/summarize.002.jpeg
# :name: fig:summarize-groupby
# :figclass: caption-hack
#
# Calculating summary statistics on one or more column(s) for each group. It creates a new data frame&mdash;with one row for each group&mdash;containing the summary statistic(s) for each column being summarized. It also creates a column listing the value of the grouping variable. The darker, top row of each table represents the column headers. The gray, blue, and green colored rows correspond to the rows that belong to each of the three groups being represented in this cartoon example.
# ```

# %% [markdown]
# The `.groupby` function takes at least one argument&mdash;the columns to use in the
# grouping. Here we use only one column for grouping (`region`), but more than one
# can also be used. To do this, pass a list of column names to the `by` argument.

# %%
region_summary = pd.DataFrame()
region_summary = region_summary.assign(
    min_most_at_home=region_lang.groupby(by="region")["most_at_home"].min(),
    max_most_at_home=region_lang.groupby(by="region")["most_at_home"].max()
).reset_index()

region_summary.columns = ["region", "min_most_at_home", "max_most_at_home"]
region_summary

# %% [markdown]
# `pandas` also has a convenient method `.agg` (shorthand for `.aggregate`) that allows us to apply multiple aggregate functions in one line of code. We just need to pass in a list of function names to `.agg` as shown below.

# %%
region_summary = (
    region_lang.groupby(by="region")["most_at_home"].agg(["min", "max"]).reset_index()
)
region_summary.columns = ["region", "min_most_at_home", "max_most_at_home"]
region_summary

# %% [markdown]
# Notice that `.groupby` converts a `DataFrame` object to a `DataFrameGroupBy` object, which contains information about the groups of the dataframe. We can then apply aggregating functions to the `DataFrameGroupBy` object.

# %% tags=["remove-cell"]
# Notice that `group_by` on its own doesn't change the way the data looks. 
# In the output below, the grouped data set looks the same, 
# and it doesn't *appear* to be grouped by `region`. 
# Instead, `group_by` simply changes how other functions work with the data, 
# as we saw with `summarize` above.  

# %%
region_lang.groupby("region")

# %% [markdown]
# ### Calculating summary statistics on many columns

# %% [markdown]
# Sometimes we need to summarize statistics across many columns.
# An example of this is illustrated in {numref}`fig:summarize-across`.
# In such a case, using summary functions alone means that we have to 
# type out the name of each column we want to summarize.
# In this section we will meet two strategies for performing this task. 
# First we will see how we can do this using `.iloc[]` to slice the columns before applying summary functions.
# Then we will also explore how we can use a more general iteration function, 
# `.apply`, to also accomplish this.

# %% tags=["remove-cell"]
# Sometimes we need to summarize statistics across many columns.
# An example of this is illustrated in Figure \@ref(fig:summarize-across).
# In such a case, using `summarize` alone means that we have to 
# type out the name of each column we want to summarize.
# In this section we will meet two strategies for performing this task. 
# First we will see how we can do this using `summarize` + `across`.
# Then we will also explore how we can use a more general iteration function, 
# `map`, to also accomplish this.

# %% [markdown] tags=[]
# ```{figure} img/summarize/summarize.003.jpeg
# :name: fig:summarize-across
# :figclass: caption-hack
#
# `.iloc[]` or `.apply` is useful for efficiently calculating summary statistics on many columns at once. The darker, top row of each table represents the column headers.
# ```

# %% [markdown]
# #### Aggregating on a data frame for calculating summary statistics on many columns

# %% [markdown]
# Recall that in the Section {ref}`loc-iloc`, we can use `.iloc[]` to extract a range of columns with indices. Here we demonstrate finding the maximum value 
# of each of the numeric
# columns of the `region_lang` data set through pairing `.iloc[]` and `.max`. This means that the 
# summary methods (*e.g.* `.min`, `.max`, `.sum` etc.) can be used for data frames as well.

# %%
pd.DataFrame(region_lang.iloc[:, 3:].max(axis=0)).T

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
# To summarize statistics across many columns, we can use the 
# `summarize` function we have just recently learned about.
# However, in such a case, using `summarize` alone means that we have to 
# type out the name of each column we want to summarize. 
# To do this more efficiently, we can pair `summarize` with `across` \index{across}
# and use a colon `:` to specify a range of columns we would like  \index{column range}
# to perform the statistical summaries on.
# Here we demonstrate finding the maximum value 
# of each of the numeric
# columns of the `region_lang` data set.

# ``` {r 02-across-data}
# region_lang |>
#   summarize(across(mother_tongue:lang_known, max))
# ``` 

# > **Note:** Similar to when we use base R statistical summary functions 
# > (e.g., `max`, `min`, `mean`, `sum`, etc) with `summarize` alone, 
# > the use of the `summarize` + `across` functions paired 
# > with base R statistical summary functions
# > also return `NA`s when we apply them to columns that 
# > contain `NA`s in the data frame.  \index{missing data}
# > 
# > To avoid this, again we need to add the argument `na.rm = TRUE`,
# > but in this case we need to use it a little bit differently.
# > In this case, we need to add a `,` and then `na.rm = TRUE`,
# > after specifying the function we want `summarize` + `across` to apply, 
# > as illustrated below:
# > 
# > ``` {r}
# > region_lang_na |>
# >   summarize(across(mother_tongue:lang_known, max, na.rm = TRUE))
# > ```

# %% [markdown]
# (apply-summary)=
# #### `.apply` for calculating summary statistics on many columns

# %% [markdown]
# An alternative to aggregating on a dataframe
# for applying a function to many columns is the `.apply` method.
# Let's again find the maximum value of each column of the
# `region_lang` data frame, but using `.apply` with the `max` function this time.
# We focus on the two arguments of `.apply`: 
# the function that you would like to apply to each column, and the `axis` along which the function will be applied (`0` for columns, `1` for rows).
# Note that `.apply` does not have an argument 
# to specify *which* columns to apply the function to.
# Therefore, we will use the `.iloc[]` before calling `.apply`
# to choose the columns for which we want the maximum.

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
# An alternative to `summarize` and `across` 
# for applying a function to many columns is the `map` family of functions. \index{map}
# Let's again find the maximum value of each column of the
# `region_lang` data frame, but using `map` with the `max` function this time.
# `map` takes two arguments: 
# an object (a vector, data frame or list) that you want to apply the function to, 
# and the function that you would like to apply to each column.
# Note that `map` does not have an argument 
# to specify *which* columns to apply the function to.
# Therefore, we will use the `select` function before calling `map`
# to choose the columns for which we want the maximum.

# %%
pd.DataFrame(region_lang.iloc[:, 3:].apply(max, axis=0)).T

# %% [markdown]
# > **Note:** Similar to when we use base Python statistical summary functions 
# > (e.g., `max`, `min`, `sum`, etc.) when there are `NaN`s, 
# > `.apply` functions paired with base Python statistical summary functions
# > also return `NaN` values when we apply them to columns that 
# > contain `NaN` values. \index{missing data}
# > 
# > To avoid this, again we need to use the `pandas` variants of summary functions (*i.e.*
# > `.max`, `.min`, `.sum`, etc.) with `skipna=True`.
# > When we use this with `.apply`, we do this by constructing a anonymous function that calls 
# > the `.max` method with `skipna=True`, as illustrated below:

# %%
pd.DataFrame(
    region_lang_na.iloc[:, 3:].apply(lambda col: col.max(skipna=True), axis=0)
).T

# %% [markdown]
# The `.apply` function is generally quite useful for solving many problems 
# involving repeatedly applying functions in Python. 
# Additionally, a variant of `.apply` is `.applymap`, 
# which can be used to apply functions element-wise.
# To learn more about these functions, see the additional resources
# section at the end of this chapter.

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=["remove-cell"]
# <!-- > **Note:** The `map` function comes from the `purrr` package. But since
# > `purrr` is part of the tidyverse, once we call `library(tidyverse)` we 
# > do not need to load the `purrr` package separately.
#
# The output looks a bit weird... we passed in a data frame, but the output
# doesn't look like a data frame. As it so happens, it is *not* a data frame, but
# rather a plain list:
#
# ``` {r 02-type-out-of-map}
# region_lang |>
#   select(mother_tongue:lang_known) |>
#   map(max) |>
#   typeof()
# ```
#
# So what do we do? Should we convert this to a data frame? We could, but a
# simpler alternative is to just use a different `map` function. There 
# are quite a few to choose from, they all work similarly, but
# their name reflects the type of output you want from the mapping operation.
# Table \@ref(tab:map-table) lists the commonly used `map` functions as well
# as their output type. \index{map!map\_\* functions}
#
# Table: (#tab:map-table) The `map` functions in R.
#
# | `map` function | Output |
# | --- | --- |
# | `map` | list |
# | `map_lgl` | logical vector |
# | `map_int` | integer vector |
# | `map_dbl` | double vector |
# | `map_chr` | character vector |
# | `map_dfc` | data frame, combining column-wise |
# | `map_dfr` | data frame, combining row-wise |
#
# Let's get the columns' maximums again, but this time use the `map_dfr` function
# to return the output as a data frame:
#
# ``` {r 02-map-df-example}
# region_lang |>
#   select(mother_tongue:lang_known) |>
#   map_dfr(max)
# ```
#
# > **Note:** Similar to when we use base R statistical summary functions 
# > (e.g., `max`, `min`, `mean`, `sum`, etc.) with `summarize`, 
# > `map` functions paired with base R statistical summary functions
# > also return `NA` values when we apply them to columns that 
# > contain `NA` values. \index{missing data}
# > 
# > To avoid this, again we need to add the argument `na.rm = TRUE`.
# > When we use this with `map`, we do this by adding a `,` 
# > and then `na.rm = TRUE` after specifying the function, as illustrated below:
# > 
# > ``` {r}
# > region_lang_na |>
# >   select(mother_tongue:lang_known) |>
# >   map_dfr(max, na.rm = TRUE)
# > ```
#
# The `map` functions are generally quite useful for solving many problems 
# involving repeatedly applying functions in R. 
# Additionally, their use is not limited to columns of a data frame;
# `map` family functions can be used to apply functions to elements of a vector,
# or a list, and even to lists of (nested!) data frames.
# To learn more about the `map` functions, see the additional resources
# section at the end of this chapter. -->

# %% [markdown] tags=[]
# ## Apply functions across many columns with `.apply`
#
# Sometimes we need to apply a function to many columns in a data frame. 
# For example, we would need to do this when converting units of measurements across many columns. 
# We illustrate such a data transformation in {numref}`fig:mutate-across`.

# %% [markdown] tags=[]
# ```{figure} img/summarize/summarize.005.jpeg
# :name: fig:mutate-across
# :figclass: caption-hack
#
# `.apply` is useful for applying functions across many columns. The darker, top row of each table represents the column headers.
# ```

# %% [markdown]
# For example, 
# imagine that we wanted to convert all the numeric columns 
# in the `region_lang` data frame from `int64` type to `int32` type 
# using the `.as_type` function.
# When we revisit the `region_lang` data frame, 
# we can see that this would be the columns from `mother_tongue` to `lang_known`.

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
# For example, 
# imagine that we wanted to convert all the numeric columns 
# in the `region_lang` data frame from double type to integer type 
# using the `as.integer` function.
# When we revisit the `region_lang` data frame, 
# we can see that this would be the columns from `mother_tongue` to `lang_known`.

# %%
region_lang

# %% [markdown]
# To accomplish such a task, we can use `.apply`.
# This works in a similar way for column selection, 
# as we saw when we used in Section {ref}`apply-summary` earlier.
# As we did above, 
# we again use `.iloc` to specify the columns
# as well as the `.apply` to specify the function we want to apply on these columns.
# However, a key difference here is that we are not using aggregating function here, 
# which means that we get back a data frame with the same number of rows.

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
# To accomplish such a task, we can use `mutate` paired with `across`. \index{across}
# This works in a similar way for column selection, 
# as we saw when we used `summarize` + `across` earlier.
# As we did above, 
# we again use `across` to specify the columns using `select` syntax
# as well as the function we want to apply on the specified columns.
# However, a key difference here is that we are using `mutate`, 
# which means that we get back a data frame with the same number of rows.

# %%
region_lang.dtypes

# %%
region_lang_int32 = region_lang.iloc[:, 3:].apply(lambda col: col.astype('int32'), axis=0)
region_lang_int32 = pd.concat((region_lang.iloc[:, :3], region_lang_int32), axis=1)
region_lang_int32

# %%
region_lang_int32.dtypes

# %% [markdown]
# We see that we get back a data frame
# with the same number of columns and rows.
# The only thing that changes is the transformation we applied 
# to the specified columns (here `mother_tongue` to `lang_known`).

# %% [markdown]
# ## Apply functions across columns within one row with `.apply`
#
# What if you want to apply a function across columns but within one row? 
# We illustrate such a data transformation in {numref}`fig:rowwise`.

# %% [markdown] tags=[]
# ```{figure} img/summarize/summarize.004.jpeg
# :name: fig:rowwise
# :figclass: caption-hack
#
# `.apply` is useful for applying functions across columns within one row. The darker, top row of each table represents the column headers.
# ```

# %% [markdown]
# For instance, suppose we want to know the maximum value between `mother_tongue`,
# `most_at_home`, `most_at_work` 
# and `lang_known` for each language and region
# in the `region_lang` data set.
# In other words, we want to apply the `max` function *row-wise.*
# Before we use `.apply`, we will again use `.iloc` to select only the count columns
# so we can see all the columns in the data frame's output easily in the book. 
# So for this demonstration, the data set we are operating on looks like this:

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
# For instance, suppose we want to know the maximum value between `mother_tongue`,
# `most_at_home`, `most_at_work` 
# and `lang_known` for each language and region
# in the `region_lang` data set.
# In other words, we want to apply the `max` function *row-wise.*
# We will use the (aptly named) `rowwise` function in combination with `mutate` 
# to accomplish this task. 

# Before we apply `rowwise`, we will `select` only the count columns \index{rowwise}
# so we can see all the columns in the data frame's output easily in the book. 
# So for this demonstration, the data set we are operating on looks like this:

# %%
region_lang.iloc[:, 3:]

# %% [markdown]
# Now we use `.apply` with argument `axis=1`, to tell Python that we would like
# the `max` function to be applied across, and within, a row,
# as opposed to being applied on a column 
# (which is the default behavior of `.apply`):

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
# Now we apply `rowwise` before `mutate`, to tell R that we would like
# the mutate function to be applied across, and within, a row,
# as opposed to being applied on a column 
# (which is the default behavior of `mutate`):

# %%
region_lang_rowwise = region_lang.assign(
    maximum=region_lang.iloc[:, 3:].apply(max, axis=1)
)

region_lang_rowwise

# %% [markdown]
# We see that we get an additional column added to the data frame, 
# named `maximum`, which is the maximum value between `mother_tongue`,
# `most_at_home`, `most_at_work` and `lang_known` for each language
# and region.

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
# Similar to `group_by`, 
# `rowwise` doesn't appear to do anything when it is called by itself. 
# However, we can apply `rowwise` in combination 
# with other functions to change how these other functions operate on the data.
# Notice if we used `mutate` without `rowwise`, 
# we would have computed the maximum value across *all* rows 
# rather than the maximum value for *each* row. 
# Below we show what would have happened had we not used
# `rowwise`. In particular, the same maximum value is reported 
# in every single row; this code does not provide the desired result.

# ```{r}
# region_lang |> 
#   select(mother_tongue:lang_known) |>
#   mutate(maximum = max(c(mother_tongue, 
#                          most_at_home, 
#                          most_at_home, 
#                          lang_known)))
# ```

# %% [markdown]
# ## Summary
#
# Cleaning and wrangling data can be a very time-consuming process. However, 
# it is a critical step in any data analysis. We have explored many different
# functions for cleaning and wrangling data into a tidy format. 
# {numref}`tab:summary-functions-table` summarizes some of the key wrangling 
# functions we learned in this chapter. In the following chapters, you will 
# learn how you can take this tidy data and do so much more with it to answer your 
# burning data science questions!

# %% [markdown]
# ```{table} Summary of wrangling functions 
# :name: tab:summary-functions-table
#
# | Function | Description |
# | ---      | ----------- | 
# | `.agg` | calculates aggregated summaries of inputs |
# | `.apply` | allows you to apply function(s) to multiple columns/rows  | 
# | `.assign` | adds or modifies columns in a data frame  | 
# | `.groupby` |  allows you to apply function(s) to groups of rows |
# | `.iloc` | subsets columns/rows of a data frame using integer indices |
# | `.loc` | subsets columns/rows of a data frame using labels | 
# | `.melt` | generally makes the data frame longer and narrower |
# | `.pivot` | generally makes a data frame wider and decreases the number of rows | 
# | `.str.split` | splits up a string column into multiple columns  |
# ```

# %% jupyter={"source_hidden": true} tags=["remove-cell"]
# ## Summary

# Cleaning and wrangling data can be a very time-consuming process. However, 
# it is a critical step in any data analysis. We have explored many different
# functions for cleaning and wrangling data into a tidy format. 
# Table \@ref(tab:summary-functions-table) summarizes some of the key wrangling 
# functions we learned in this chapter. In the following chapters, you will 
# learn how you can take this tidy data and do so much more with it to answer your 
# burning data science questions!

# \newpage

# Table: (#tab:summary-functions-table) Summary of wrangling functions 

# | Function | Description |
# | ---      | ----------- | 
# | `across` | allows you to apply function(s) to multiple columns  | 
# | `filter` | subsets rows of a data frame | 
# | `group_by` |  allows you to apply function(s) to groups of rows |
# | `mutate` | adds or modifies columns in a data frame |
# | `map` | general iteration function |
# | `pivot_longer` | generally makes the data frame longer and narrower |
# | `pivot_wider` | generally makes a data frame wider and decreases the number of rows | 
# | `rowwise` | applies functions across columns within one row | 
# | `separate` | splits up a character column into multiple columns  | 
# | `select` | subsets columns of a data frame |
# | `summarize` | calculates summaries of inputs |

# %% [markdown]
# ## Exercises
#
# Practice exercises for the material covered in this chapter 
# can be found in the accompanying 
# [worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
# in the "Cleaning and wrangling data" row.
# You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
# You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
# If you instead decide to download the worksheet and run it on your own machine,
# make sure to follow the instructions for computer setup
# found in Chapter {ref}`move-to-your-own-machine`. This will ensure that the automated feedback
# and guidance that the worksheets provide will function as intended.

# %% [markdown] tags=[]
# ## Additional resources 
#
# - The [`pandas` package documentation](https://pandas.pydata.org/docs/reference/index.html) is
#   another resource to learn more about the functions in this
#   chapter, the full set of arguments you can use, and other related functions.
#   The site also provides a very nice cheat sheet that summarizes many of the
#   data wrangling functions from this chapter.
# - *Python for Data Analysis* {cite:p}`mckinney2012python` has a few chapters related to
#   data wrangling that go into more depth than this book. For example, the
#   [data wrangling chapter](https://wesmckinney.com/book/data-wrangling.html) covers tidy data,
#   `.melt` and `.pivot`, but also covers missing values
#   and additional wrangling functions (like `.stack`). The [data
#   aggregation chapter](https://wesmckinney.com/book/data-aggregation.html) covers
#   `.groupby`, aggregating functions, `.apply`, etc.
# - You will occasionally encounter a case where you need to iterate over items
#   in a data frame, but none of the above functions are flexible enough to do
#   what you want. In that case, you may consider using [a for
#   loop](https://wesmckinney.com/book/python-basics.html#control_for) {cite:p}`mckinney2012python`.

# %% jp-MarkdownHeadingCollapsed=true tags=["remove-cell"] jupyter={"source_hidden": true}
# ## Additional resources 

# - As we mentioned earlier, `tidyverse` is actually an *R
#   meta package*: it installs and loads a collection of R packages that all
#   follow the tidy data philosophy we discussed above. One of the `tidyverse`
#   packages is `dplyr`&mdash;a data wrangling workhorse. You have already met many
#   of `dplyr`'s functions 
#   (`select`, `filter`, `mutate`, `arrange`, `summarize`, and `group_by`). 
#   To learn more about these functions and meet a few more useful
#   functions, we recommend you check out Chapters 5-9 of the [STAT545 online notes](https://stat545.com/).
#   of the data wrangling, exploration, and analysis with R book.
# - The [`dplyr` R package documentation](https://dplyr.tidyverse.org/) [@dplyr] is
#   another resource to learn more about the functions in this
#   chapter, the full set of arguments you can use, and other related functions.
#   The site also provides a very nice cheat sheet that summarizes many of the
#   data wrangling functions from this chapter.
# - Check out the [`tidyselect` R package page](https://tidyselect.r-lib.org/index.html)
#   [@tidyselect] for a comprehensive list of `select` helpers. 
#   These helpers can be used to choose columns in a data frame when paired with  the `select` function 
#   (and other functions that use the `tidyselect` syntax, such as `pivot_longer`).
#   The [documentation for `select` helpers](https://tidyselect.r-lib.org/reference/select_helpers.html) 
#   is a useful reference to find the helper you need for your particular problem.
# - *R for Data Science* [@wickham2016r] has a few chapters related to
#   data wrangling that go into more depth than this book. For example, the
#   [tidy data chapter](https://r4ds.had.co.nz/tidy-data.html) covers tidy data,
#   `pivot_longer`/`pivot_wider` and `separate`, but also covers missing values
#   and additional wrangling functions (like `unite`). The [data
#   transformation chapter](https://r4ds.had.co.nz/transform.html) covers
#   `select`, `filter`, `arrange`, `mutate`, and `summarize`. And the [`map`
#   functions chapter](https://r4ds.had.co.nz/iteration.html#the-map-functions)
#   provides more about the `map` functions.
# - You will occasionally encounter a case where you need to iterate over items
#   in a data frame, but none of the above functions are flexible enough to do
#   what you want. In that case, you may consider using [a for
#   loop](https://r4ds.had.co.nz/iteration.html#iteration).
