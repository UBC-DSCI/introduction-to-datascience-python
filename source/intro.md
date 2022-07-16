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
# Python and the Pandas

```{code-cell} ipython3
:tags: ["remove-cell"]
from myst_nb import glue
```

## Overview

This chapter provides an introduction to data science and the Python programming language.
The goal here is to get your hands dirty right from the start! We will walk through an entire data analysis,
and along the way introduce different types of data analysis question, some fundamental programming 
concepts in Python, and the basics of loading, cleaning, and visualizing data. In the following chapters, we will
dig into each of these steps in much more detail; but for now, let's jump in to see how much we can do 
with data science!

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

- Identify the different types of data analysis question and categorize a question into the correct type.
- Load the `pandas` package into Python.
- Read tabular data with `read_csv`.
- Use `help()` to access help and documentation tools in Python.
- Create new variables and objects in Python.
- Do indexing and slicing with `.loc[]` and `.iloc[]` properties.
- Select columns of a dataframe using df[] notation.
- Visualize data with an `altair` bar plot.

## Canadian languages data set

In this chapter, \index{Canadian languages} we will walk through a full analysis of a data set relating to
languages spoken at home by Canadian residents. Many Indigenous peoples exist in Canada 
with their own cultures and languages; these languages are often unique to Canada and not spoken
anywhere else in the world [@statcan2018mothertongue]. Sadly, colonization has
led to the loss of many of these languages. For instance, generations of
children were not allowed to speak their mother tongue (the first language an
individual learns in childhood) in Canadian residential schools. Colonizers
also renamed places they had "discovered" [@wilson2018].  Acts such as these
have significantly harmed the continuity of Indigenous languages in Canada, and
some languages are considered "endangered" as few people report speaking them. 
To learn more, please see *Canadian Geographic*'s article, "Mapping Indigenous Languages in 
Canada" [@walker2017], 
*They Came for the Children: Canada, Aboriginal 
peoples, and Residential Schools* [@children2012] 
and the *Truth and Reconciliation Commission of Canada's* 
*Calls to Action* [@calls2015].

The data set we will study in this chapter is taken from 
[the `canlang` R data package](https://ttimbers.github.io/canlang/) 
[@timbers2020canlang], which has
population language data collected during the 2016 Canadian census [@cancensus2016]. 
In this data, there are 214 languages recorded, each having six different properties:

1. `category`: Higher-level language category, describing whether the language is an Official Canadian language, an Aboriginal (i.e., Indigenous) language, or a Non-Official and Non-Aboriginal language.
2. `language`: The name of the language.
3. `mother_tongue`: Number of Canadian residents who reported the language as their mother tongue. Mother tongue is generally defined as the language someone was exposed to since birth.
4. `most_at_home`: Number of Canadian residents who reported the language as being spoken most often at home.
5. `most_at_work`: Number of Canadian residents who reported the language as being used most often at work.
6. `lang_known`: Number of Canadian residents who reported knowledge of the language.

According to the census, more than 60 Aboriginal languages were reported
as being spoken in Canada. Suppose we want to know which are the most common;
then we might ask the following question, which we wish to answer using our data: 

*Which ten Aboriginal languages were most often reported in 2016 as mother
tongues in Canada, and how many people speak each of them?* 

> **Note:** Data science\index{data science!good practices} cannot be done without 
> a deep understanding of the data and
> problem domain. In this book, we have simplified the data sets used in our
> examples to concentrate on methods and fundamental concepts. But in real
> life, you cannot and should not do data science without a domain expert.
> Alternatively, it is common to practice data science in your own domain of
> expertise! Remember that when you work with data, it is essential to think
> about *how* the data were collected, which affects the conclusions you can
> draw. If your data are biased, then your results will be biased!

## Asking a question 

Every good data analysis begins with a *question*&mdash;like the
above&mdash;that you aim to answer using data. As it turns out, there
are actually a number of different *types* of question regarding data:
descriptive, exploratory, inferential, predictive, causal, and mechanistic,
all of which are defined in {numref}`questions-table`. [@leek2015question; @peng2015art]
Carefully formulating a question as early as possible in your analysis&mdash;and 
correctly identifying which type of question it is&mdash;will guide your overall approach to 
the analysis as well as the selection of appropriate tools.\index{question!data analysis}
\index{descriptive question!definition}
\index{exploratory question!definition}
\index{predictive question!definition}
\index{inferential question!definition}
\index{causal question!definition}
\index{mechanistic question!definition}




```{list-table} Types of data analysis question.
:header-rows: 1
:name: questions-table

* - Question type
  - Description
  - Example
* - Descriptive
  - A question that asks about summarized characteristics of a data set without interpretation (i.e., report a fact).
  - How many people live in each province and territory in Canada?
* - Exploratory
  - A question that asks if there are patterns, trends, or relationships within a single data set. Often used to propose hypotheses for future study.
  - Does political party voting change with indicators of wealth in a set of data collected on 2,000 people living in Canada?
* - Predictive
  - A question that asks about predicting measurements or labels for individuals (people or things). The focus is on what things predict some outcome, but not what causes the outcome.
  - What political party will someone vote for in the next Canadian election?
* - Inferential
  - A question that looks for patterns, trends, or relationships in a single data set **and** also asks for quantification of how applicable these findings are to the wider population.
  - Does political party voting change with indicators of wealth for all people living in Canada?
* - Causal
  - A question that asks about whether changing one factor will lead to a change in another factor, on average, in the wider population.
  - Does wealth lead to voting for a certain political party in Canadian elections?
* - Mechanistic
  - A question that asks about the underlying mechanism of the observed patterns, trends, or relationships (i.e., how does it happen?)
  - How does wealth lead to voting for a certain political party in Canadian elections?
  
```


In this book, you will learn techniques to answer the 
first four types of question: descriptive, exploratory, predictive, and inferential; 
causal and mechanistic questions are beyond the scope of this book.
In particular, you will learn how to apply the following analysis tools:

1. **Summarization:** \index{summarization!overview} computing and reporting aggregated values pertaining to a data set. 
Summarization is most often used to answer descriptive questions,
and can occasionally help with answering exploratory questions.
For example, you might use summarization to answer the following question: 
*What is the average race time for runners in this data set?*
Tools for summarization are covered in detail in Chapters \@ref(reading)
and \@ref(wrangling), but appear regularly throughout the text.
2. **Visualization:** \index{visualization!overview} plotting data graphically. 
Visualization is typically used to answer descriptive and exploratory questions,
but plays a critical supporting role in answering all of the types of question in {numref}`questions-table`.
For example, you might use visualization to answer the following question:
*Is there any relationship between race time and age for runners in this data set?* 
This is covered in detail in Chapter \@ref(viz), but again appears regularly throughout the book.
3. **Classification:** \index{classification!overview} predicting a class or category for a new observation.
Classification is used to answer predictive questions.
For example, you might use classification to answer the following question:
*Given measurements of a tumor's average cell area and perimeter, is the tumor benign or malignant?*
Classification is covered in Chapters \@ref(classification) and \@ref(classification2).
4. **Regression:**  \index{regression!overview} predicting a quantitative value for a new observation. 
Regression is also used to answer predictive questions.
For example, you might use regression to answer the following question:
*What will be the race time for a 20-year-old runner who weighs 50kg?*
Regression is covered in Chapters \@ref(regression1) and \@ref(regression2).
5. **Clustering:** \index{clustering!overview} finding previously unknown/unlabeled subgroups in a
data set. Clustering is often used to answer exploratory questions.
For example, you might use clustering to answer the following question:
*What products are commonly bought together on Amazon?*
Clustering is covered in Chapter \@ref(clustering).
6. **Estimation:**  \index{estimation!overview} taking measurements for a small number of items from a large group 
 and making a good guess for the average or proportion for the large group. Estimation 
is used to answer inferential questions.
For example, you might use estimation to answer the following question:
*Given a survey of cellphone ownership of 100 Canadians, what proportion
of the entire Canadian population own Android phones?* 
Estimation is covered in Chapter \@ref(inference).

Referring to {numref}`questions-table`, our question about 
Aboriginal languages is an example of a *descriptive question*: we are
summarizing the characteristics of a data set without further interpretation.
And referring to the list above, it looks like we should use visualization
and perhaps some summarization to answer the question. So in the remainder
of this chapter, we will work towards making a visualization that shows 
us the ten most common Aboriginal languages in Canada and their associated counts,
according to the 2016 census. 

## Loading a tabular data set
A data set is, at its core essence, a structured collection of numbers and characters.
Aside from that, there are really no strict rules; data sets can come in 
many different forms! Perhaps the most common form of data set that you will
find in the wild, however, is *tabular data*\index{tabular data}. Think spreadsheets in Microsoft Excel: tabular data are
rectangular-shaped and spreadsheet-like, as shown in {numref}`img-spreadsheet-vs-dataframe`. In this book, we will focus primarily on tabular data.

Since we are using Python for data analysis in this book, the first step for us is to
load the data into Python. When we load tabular data into
Python, it is represented as a *data frame* object\index{data frame!overview}. {numref}`img-spreadsheet-vs-dataframe` shows that an Python data frame is very similar
to a spreadsheet. We refer to the rows as \index{observation} **observations**; these are the things that we
collect the data on, e.g., voters, cities, etc. We refer to the columns as \index{variable}
**variables**; these are the characteristics of those observations, e.g., voters' political
affiliations, cities' populations, etc. 



```{figure} img/spreadsheet_vs_df.png
---
height: 400px
name: img-spreadsheet-vs-dataframe
---
A spreadsheet versus a data frame in Python
```

The first kind of data file that we will learn how to load into Python as a data
frame is the *comma-separated values* format (`.csv` for short)\index{comma-separated values|see{csv}}\index{csv}.  These files
have names ending in `.csv`, and can be opened and saved using common
spreadsheet programs like Microsoft Excel and Google Sheets.  For example, the
`.csv` file named `can_lang.csv` 
is included with [the code for this book](https://github.com/UBC-DSCI/introduction-to-datascience/tree/master/data).
If we were to open this data in a plain text editor (a program like Notepad that just shows
text with no formatting), we would see each row on its own line, and each entry in the table separated by a comma:

```code
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

To load this data into Python so that we can do things with it (e.g., perform
analyses or create data visualizations), we will need to use a *function.* \index{function} A
function is a special word in Python that takes instructions (we call these
*arguments*) \index{argument} and does something. The function we will use to load a `.csv` file
into Python is called `read_csv`. \index{read function!read\_csv} In its most basic 
use-case, `read_csv` expects that the data file:

- has column names (or *headers*),
- uses a comma (`,`) to separate the columns, and
- does not have row names.

+++

Below you'll see the code used to load the data into Python using the `read_csv`
function. Note that the `read_csv` function is not included in the base
installation of Python, meaning that it is not one of the primary functions ready to
use when you install Python. Therefore, you need to load it from somewhere else
before you can use it. The place from which we will load it is called a Python *package*. 
A Python package \index{package} is a collection of functions that can be used in addition to the
built-in Python package functions once loaded. The `read_csv` function, in
particular, can be made accessible by loading 
[the `pandas` Python package](https://pypi.org/project/pandas/) [@tidyverse; @wickham2019tidverse]
using the `import` command. \index{library} The `pandas` \index{tidyverse} package contains many
functions that we will use throughout this book to load, clean, wrangle, 
and visualize data. 

+++



```{code-cell} ipython3
import pandas as pd
```



After loading the `pandas` package and accessing it using the alias `pd`, we can call the `read_csv` function and
pass it a single argument: the name of the file, `"can_lang.csv"`. We have to
put quotes around file names and other letters and words that we use in our
code to distinguish it from the special words (like functions!) that make up the Python programming
language.  The file's name is the only argument we need to provide because our
file satisfies everything else that the `read_csv` function expects in the default
use-case. {numref}`img-read-csv` describes how we use the `read_csv`
to read data into Python. 

(ref:img-read-csv) Syntax for the `read_csv` function.



```{figure} img/read_csv_function.jpeg
---
height: 200px
name: img-read-csv
---
Syntax for the read_csv function
```


+++
```{code-cell} ipython3
pd.read_csv("data/can_lang.csv")

```



## Naming things in Python

When we loaded the 2016 Canadian census language data
using `read_csv`, we did not give this data frame a name. 
Therefore the data was just printed on the screen, 
and we cannot do anything else with it. That isn't very useful. 
What would be more useful would be to give a name 
to the data frame that `read_csv` outputs, 
so that we can refer to it later for analysis and visualization.

The way to assign a name to a value in Python is via the *assignment symbol* `=`. 
\index{aaaassignsymb@\texttt{<-}|see{assignment symbol}}\index{assignment symbol}
On the left side of the assignment symbol you put the name that you want
to use, and on the right side of the assignment symbol
you put the value that you want the name to refer to.
Names can be used to refer to almost anything in Python, such as numbers,
words (also known as *strings* of characters), and data frames!
Below, we set `my_number` to `3` (the result of `1+2`)
and we set `name` to the string `"Alice"`. \index{string}


```{code-cell} ipython3
my_number = 1 + 2
print(my_number)

name = "Alice"
print(name)
```

Note that when 
we name something in Python using the assignment symbol, `=`, 
we do not need to surround the name we are creating  with quotes. This is 
because we are formally telling Python that this special word denotes
the value of whatever is on the right-hand side.
Only characters and words that act as *values* on the right-hand side of the assignment
symbol&mdash;e.g., the file name `"data/can_lang.csv"` that we specified before, or `"Alice"` above&mdash;need 
to be surrounded by quotes.

After making the assignment, we can use the special name words we have created in
place of their values. For example, if we want to do something with the value `3` later on, 
we can just use `my_number` instead. Let's try adding 2 to `my_number`; you will see that
Python just interprets this as adding 2 and 3:


```{code-cell} ipython3
my_number + 2
```

Object names \index{object} can consist of letters, numbers, periods `.` and underscores `_`.
Other symbols won't work since they have their own meanings in Python. For example,
`+` is the addition symbol(operator); if we try to assign a name with
the `+` symbol, Python will complain and we will get an error!


```
na+me = 1
```

```
SyntaxError: cannot assign to operator
```

There are certain conventions for naming objects in Python. 
When naming \index{object!naming convention} an object we
suggest using only lowercase letters, numbers and underscores `_` to separate
the words in a name.  Python is case sensitive, which means that `Letter` and
`letter` would be two different objects in Python.  You should also try to give your
objects meaningful names.  For instance, you *can* name a data frame `x`.
However, using more meaningful terms, such as `language_data`, will help you
remember what each name in your code represents.  We recommend following the
**PEP 8** naming conventions outlined in the *[PEP 8](https://peps.python.org/pep-0008/)* [@tidyversestyleguide].  Let's
now use the assignment symbol to give the name
`can_lang` to the 2016 Canadian census language data frame that we get from
`read_csv`. 



```{code-cell} ipython3
can_lang = pd.read_csv("data/can_lang.csv")
```

Wait a minute, nothing happened this time! Where's our data?
Actually, something did happen: the data was loaded in 
and now has the name `can_lang` associated with it. 
And we can use that name to access the data frame and do things with it. 
For example, we can type the name of the data frame to print the first few rows 
on the screen. You will also see at the top that the number of observations (i.e., rows) and 
variables (i.e., columns) are printed. Printing the first few rows of a data frame 
like this is a handy way to get a quick sense for what is contained in a data frame.



```{code-cell} ipython3
can_lang
```

## Creating subsets of data frames with `df[]` & `loc[]`

Now that we've loaded our data into Python, we can start wrangling the data to
find the ten Aboriginal languages that were most often reported
in 2016 as mother tongues in Canada. In particular, we will construct 
a table with the ten Aboriginal languages that have the largest 
counts in the `mother_tongue` column. 
The `df[]` and `loc[]` properties of the `pandas` dataframe will help us
here. The `df[]` \index{filter} property allows you to obtain a subset of the
rows with specific values, while the `loc[]` \index{select} property allows you 
to obtain a subset of the columns. Therefore, we can use `df[]` 
to filter the rows to extract the Aboriginal languages in the data set, and 
then use `loc[]` property to obtain only the columns we want to include in our table.

### Using `df[]` to extract rows
Looking at the `can_lang` data above, we see the column `category` contains different
high-level categories of languages, which include "Aboriginal languages",
"Non-Official & Non-Aboriginal languages" and "Official languages".  To answer
our question we want to filter our data set so we restrict our attention 
to only those languages in the "Aboriginal languages" category. 

We can use the `df[]` \index{filter} property to obtain the subset of rows with desired
values from a data frame, where `df[]` is same as the data frame object you are using. 
In our case it would be `can_lang[]`
The argument to `can_lang[]` is selection of column name `category` of the data frame
object, `can_lang`. The second argument is a *logical statement* \index{logical statement} to use when
filtering the rows with the logical statement evaluating to either `TRUE` or `FALSE`;
`df[]` keeps only those rows for which the logical statement evaluates to `TRUE`.
For example, in our analysis, we are interested in keeping only languages in the
"Aboriginal languages" higher-level category. We can use 
the *equivalency operator* `==` \index{logical statement!equivalency operator} to compare the values
of the `category` column with the value `"Aboriginal languages"`; you will learn about
many other kinds of logical statements in Chapter \@ref(wrangling).  Similar to
when we loaded the data file and put quotes around the file name, here we need
to put quotes around `"Aboriginal languages"`. Using quotes tells Python that this
is a string *value* \index{string} and not one of the special words that make up Python
programming language, or one of the names we have given to data frames in the
code we have already written. 



With the logical filter statement inside `df[]`, we get a data frame that has all the columns of
the input data frame, but only those rows we asked for in our logical filter
statement. 



```{code-cell} ipython3
aboriginal_lang = can_lang[can_lang['category'] == 'Aboriginal languages']
aboriginal_lang
```

It's good practice to check the output after using a
function in Python. We can see the original `can_lang` data set contained 214 rows
with multiple kinds of `category`. The data frame
`aboriginal_lang` contains only 67 rows, and looks like it only contains languages in
the "Aboriginal languages" in the `category` column. So it looks like the `df[]` property
gave us the result we wanted!

### Using `.loc[]` to extract columns

Now let's use `loc[]` \index{select} to extract the `language` and `mother_tongue` columns
from this data frame. To extract these columns, we need to provide the `loc[]`
property with list of rows separated by list of columns i.e `df.loc[[row1, row2, ...], [col1, col2, ...]]`
As we want to access all the rows of the dataframe, instead of passing the names of all the rows, 
we can instead just depict them using `:` and then pass a list of columns `language` and `mother_tongue`.
After passing these as arguments, the  `loc[]` property
returns two columns (the `language` and `mother_tongue` columns that we asked
for) as a data frame.

+++

```{code-cell} ipython3
selected_lang = aboriginal_lang.loc[:, ['language', 'mother_tongue']]
selected_lang
```

### Using `sort_values` to order and `iloc[]` to select rows by index number

We have used `df[]` and `loc[]` properties of dataframe to obtain a table with only the Aboriginal
languages in the data set and their associated counts. However, we want to know
the **ten** languages that are spoken most often. As a next step, we could
order the `mother_tongue` column from greatest to least and then extract only
the top ten rows. This is where the `sort_values` function and `.iloc[]` property come to the
rescue! \index{arrange}\index{slice}

The `sort_values` function allows us to order the rows of a data frame by the
values of a particular column.  We need to specify the column name
by which we want to sort the dataframe by passing it to the argument `by`.
Since we want to choose the ten Aboriginal languages most often reported as a mother tongue
language, we will use the `sort_values` function to order the rows in our
`selected_lang` data frame by the `mother_tongue` column. We want to
arrange the rows in descending order (from largest to smallest),
so we specify the argument `ascending` as `False`.


```{code-cell} ipython3
arranged_lang = selected_lang.sort_values(by='mother_tongue', ascending=False)
arranged_lang
```

Next we will use the `iloc[]` property, which selects rows according to their
row number. Since we want to choose the most common ten languages, we will indicate we want the
rows 1 to 10 using the argument `:10`.


```{code-cell} ipython3
ten_lang = arranged_lang.iloc[:10]
ten_lang
```

We have now answered our initial question by generating this table!
Are we done? Well, not quite; tables are almost never the best way to present
the result of your analysis to your audience. Even the simple table above with
only two columns presents some difficulty: for example, you have to scrutinize
the table quite closely to get a sense for the relative numbers of speakers of 
each language. When you move on to more complicated analyses, this issue only 
gets worse. In contrast, a *visualization* would convey this information in a much 
more easily understood format. 
Visualizations are a great tool for summarizing information to help you
effectively communicate with your audience. 

## Exploring data with visualizations
Creating effective data visualizations \index{visualization} is an essential component of any data
analysis. In this section we will develop a visualization of the 
 ten Aboriginal languages that were most often reported in 2016 as mother tongues in
Canada, as well as the number of people that speak each of them.

### Using `altair` to create a bar plot

In our data set, we can see that `language` and `mother_tongue` are in separate
columns (or variables). In addition, there is a single row (or observation) for each language.
The data are, therefore, in what we call a *tidy data* format. Tidy data is a
fundamental concept and will be a significant focus in the remainder of this
book: many of the functions from `pandas` require tidy data, including the
`altair` \index{ggplot} package that we will use shortly for our visualization. We will
formally introduce tidy data in Chapter \@ref(wrangling).

We will make a bar plot to visualize our data. A bar plot \index{plot|see{visualization}}\index{visualization|see{ggplot}}\index{visualization!bar} is a chart where the
heights of the bars represent certain values, like counts or proportions. We
will make a bar plot using the `mother_tongue` and `language` columns from our
`ten_lang` data frame. To create a bar plot of these two variables using the 
`altair` package, we must specify the data frame, which variables
to put on the x and y axes, and what kind of plot to create. 
First, we need to import the `altair` package.






```{code-cell} ipython3
import altair as alt

```
+++

The fundamental object in Altair is the `Chart`, which takes a data frame as a single argument `alt.Chart(ten_lang)`.
With a chart object in hand, we can now specify how we would like the data to be visualized. 
We first indicate what kind of geometric mark we want to use to represent the data. We can set the mark attribute 
of the chart object using the `Chart.mark_*` methods.
Here, as we want to plot the bar chart, so we will use `mark_bar()` method.
Next, we need to encode the fields of the data frame using 
the `x`(represents the x-axis position of the points) and 
`y`(represents the y-axis position of the points) channels. The `encode()`
method builds a key-value mapping between encoding channels (such as x, y) 
to fields in the dataset, accessed by field name(column names)

+++

```{code-cell} ipython3
:tags: []

barplot_mother_tongue = (
    alt.Chart(ten_lang)
    .mark_bar().encode(
        x='language',
        y='mother_tongue'
    ))
    

```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue('barplot-mother-tongue', barplot_mother_tongue, display=True)

```

:::{glue:figure} barplot-mother-tongue
:figwidth: 700px
:name: barplot-mother-tongue

Bar plot of the ten Aboriginal languages most often reported by Canadian residents as their mother tongue
:::

+++




> **Note:** The vast majority of the
> time, a single expression in Python must be contained in a single line of code.
> However, there *are* a small number of situations in which you can have a
> single Python expression span multiple lines. Above is one such case: here, Python knows that a line cannot
> end with a `.` symbol, \index{aaaplussymb@$+$|see{ggplot (add layer)}} and so it keeps reading the next line to figure out
> what the right-hand side of the `.` symbol should be.  We could, of course,
> put all of the added layers on one line of code, but splitting them across
> multiple lines helps a lot with code readability. \index{multi-line expression}

### Formatting altair objects

It is exciting that we can already visualize our data to help answer our
question, but we are not done yet! We can (and should) do more to improve the
interpretability of the data visualization that we created. For example, by
default, Python uses the column names as the axis labels. Usually these
column names do not have enough information about the variable in the column.
We really should replace this default with a more informative label. For the
example above, Python uses the column name `mother_tongue` as the label for the
y axis, but most people will not know what that is. And even if they did, they
will not know how we measured this variable, or the group of people on which the
measurements were taken. An axis label that reads "Mother Tongue (Number of
Canadian Residents)" would be much more informative.

Adding additional labels \index{plot!layers} to our visualizations that we create in `altair` is
one common and easy way to improve and refine our data visualizations. We can add titles for the axes 
in the `altair` objects using `alt.X` and `alt.Y` with the `title` argument to make 
the axes titles more informative.
\index{plot!axis labels} Again, since we are specifying
words (e.g. `"Mother Tongue (Number of Canadian Residents)"`) as arguments to
`alt.X` and `alt.Y`, we surround them with double quotation marks. We can do many other modifications
to format the plot further, and we will explore these in Chapter
\@ref(viz).



```{code-cell} ipython3
barplot_mother_tongue = (
    alt.Chart(ten_lang)
    .mark_bar().encode(
        x=alt.X('language', title='Language'),
        y=alt.Y('mother_tongue', title='Mother Tongue (Number of Canadian Residents)')
    ))

```


```{code-cell} ipython3
:tags: ["remove-cell"]

glue('barplot-mother-tongue-labs', barplot_mother_tongue, display=True)

```


:::{glue:figure} barplot-mother-tongue-labs
:figwidth: 700px
:name: barplot-mother-tongue-labs

Bar plot of the ten Aboriginal languages most often reported by Canadian residents as their mother tongue with x and y labels. Note that this visualization is not done yet; there are still improvements to be made.
:::


The result is shown in {numref}`barplot-mother-tongue-labs`. 
This is already quite an improvement! Let's tackle the next major issue with the visualization
in {numref}`barplot-mother-tongue-labs`: the vertical x axis labels, which are
currently making it difficult to read the different language names.
One solution is to rotate the plot such that the bars are horizontal rather than vertical.
To accomplish this, we will swap the x and y coordinate axes:


```{code-cell} ipython3
barplot_mother_tongue_axis = (
    alt.Chart(ten_lang)
    .mark_bar().encode(
        x=alt.X('mother_tongue', title='Mother Tongue (Number of Canadian Residents)'),
        y=alt.Y('language', title='Language')
    ))

```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue('barplot-mother-tongue-labs-axis', barplot_mother_tongue_axis, display=True)

```


:::{glue:figure} barplot-mother-tongue-labs-axis
:figwidth: 700px
:name: barplot-mother-tongue-labs-axis

Horizontal bar plot of the ten Aboriginal languages most often reported by Canadian residents as their mother tongue. There are no more serious issues with this visualization, but it could be refined further.
:::

Another big step forward, as shown in {numref}`barplot-mother-tongue-labs-axis`! There 
are no more serious issues with the visualization. Now comes time to refine
the visualization to make it even more well-suited to answering the question
we asked earlier in this chapter. For example, the visualization could be made more transparent by
organizing the bars according to the number of Canadian residents reporting
each language, rather than in alphabetical order. We can reorder the bars using
the `sort` \index{reorder} argument, which orders a variable (here `language`) based on the
values of the variable(`mother_tongue`) on the `x-axis`. 

\newpage

```{code-cell} ipython3
ordered_barplot_mother_tongue = (
    alt.Chart(ten_lang)
    .mark_bar().encode(
        x=alt.X('mother_tongue', title='Mother Tongue (Number of Canadian Residents)'),
        y=alt.Y('language', sort='x', title='Language')
    ))

```

+++

```{code-cell} ipython3
:tags: ["remove-cell"]

glue('barplot-mother-tongue-reorder', ordered_barplot_mother_tongue, display=True)

```


:::{glue:figure} barplot-mother-tongue-reorder
:figwidth: 700px
:name: barplot-mother-tongue-reorder

Bar plot of the ten Aboriginal languages most often reported by Canadian residents as their mother tongue with bars reordered.
:::        


{numref}`barplot-mother-tongue-reorder` provides a very clear and well-organized
answer to our original question; we can see what the ten most often reported Aboriginal languages
were, according to the 2016 Canadian census, and how many people speak each of them. For
instance, we can see that the Aboriginal language most often reported was Cree
n.o.s. with over 60,000 Canadian residents reporting it as their mother tongue.

> **Note:** "n.o.s." means "not otherwise specified", so Cree n.o.s. refers to
> individuals who reported Cree as their mother tongue. In this data set, the
> Cree languages include the following categories: Cree n.o.s., Swampy Cree,
> Plains Cree, Woods Cree, and a 'Cree not included elsewhere' category (which
> includes Moose Cree, Northern East Cree and Southern East Cree)
> [@language2016]. 

### Putting it all together

In the block of code below, we put everything from this chapter together, with a few
modifications. In particular, we have actually skipped the
`loc[]` step that we did above; since you specify the variable names to plot
in the `altair` function, you don't actually need to select the columns in advance
when creating a visualization. We have also provided *comments* next to 
many of the lines of code below using the
hash symbol `#`. When Python sees a `#` sign, \index{comment} \index{aaacommentsymb@\#|see{comment}} it 
will ignore all of the text that
comes after the symbol on that line. So you can use comments to explain lines 
of code for others, and perhaps more importantly, your future self!
It's good practice to get in the habit of
commenting your code to improve its readability.

This exercise demonstrates the power of Python. In relatively few lines of code, we
performed an entire data science workflow with a highly effective data
visualization! We asked a question, loaded the data into Python, wrangled the data
(using `df[]`, `sort_values()` and `iloc[]`) and created a data visualization to
help answer our question. In this chapter, you got a quick taste of the data
science workflow; continue on with the next few chapters to learn each of 
these steps in much more detail!



```{code-cell} ipython3
# load the data set
can_lang = pd.read_csv("data/can_lang.csv")

# obtain the 10 most common Aboriginal languages
aboriginal_lang = can_lang[can_lang['category'] == 'Aboriginal languages']
arranged_lang = selected_lang.sort_values(['mother_tongue'], ascending=False)
ten_lang = arranged_lang[:10]

# create the visualization
ten_lang_plot = (alt.Chart(ten_lang)
                 .mark_bar().encode(
                  x = alt.X('mother_tongue', title="Mother Tongue (Number of Canadian Residents)"),
                  y = alt.Y('language', title="Language")))

```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue('final_plot', ten_lang_plot, display=True)

```


:::{glue:figure} final_plot
:figwidth: 700px
:name: final_plot

Bar plot of the ten Aboriginal languages most often reported by Canadian residents as their mother tongue
:::


## Accessing documentation

There are many Python functions in the `pandas` package (and beyond!), and 
nobody can be expected to remember what every one of them does
or all of the arguments we have to give them. Fortunately, Python provides 
the `help` function and `__doc__` attrribute, which 
\index{aaaquestionmark@?|see{documentation}}
\index{help|see{documentation}}
\index{documentation} provides an easy way to pull up the documentation for 
most functions quickly. To use the `help` function to access documentation, you 
just put the name of the function you are curious about as an argument inside the `help` function.
For example, if you had forgotten what the `pd.read_csv` function
did or exactly what arguments to pass in, you could run the following
code: 



```{code-cell} ipython3
:tags: ["remove-output"]
help(pd.read_csv)
```
To use the `__doc__` attribute to access documentation, you 
just put `.__doc__`  after the function you are curious about.
For example, to access the documentation of `read_csv` you could run the following code:
 
 

```{code-cell} ipython3
:tags: ["remove-output"]
print(pd.read_csv.__doc__)
```
{numref}`help_read_csv` shows the documentation that will pop up,
including a high-level description of the function, its arguments, 
a description of each, and more. Note that you may find some of the
text in the documentation a bit too technical right now 
Fear not: as you work through this book, many of these terms will be introduced
to you, and slowly but surely you will become more adept at understanding and navigating 
documentation like that shown in {numref}`help_read_csv`. But do keep in mind that the documentation
is not written to *teach* you about a function; it is just there as a reference to *remind*
you about the different arguments and usage of functions that you have already learned about elsewhere.



+++

```{figure} img/help_read_csv.png
---
height: 700px
name: help_read_csv
---
The documentation for the read_csv function including a high-level description, a list of arguments and their meanings, and more.
```

## Exercises

Practice exercises for the material covered in this chapter 
can be found in the accompanying 
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the "R and the tidyverse" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in Chapter \@ref(move-to-your-own-machine). This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.
