---
jupytext:
  formats: py:percent,md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(classification)=
# Classification I: training & predicting

```{code-cell} ipython3
:tags: [remove-cell]

import random

import altair as alt
from altair_saver import save
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, plot
from IPython.display import HTML
from myst_nb import glue

alt.data_transformers.disable_max_rows()

# alt.renderers.enable('altair_saver', fmts=['vega-lite', 'png'])

# # Handle large data sets by not embedding them in the notebook
# alt.data_transformers.enable('data_server')

# # Save a PNG blob as a backup for when the Altair plots do not render
# alt.renderers.enable('mimetype')
```

## Overview 
In previous chapters, we focused solely on descriptive and exploratory
data analysis questions. 
This chapter and the next together serve as our first
foray into answering *predictive* questions about data. In particular, we will
focus on *classification*, i.e., using one or more 
variables to predict the value of a categorical variable of interest. This chapter
will cover the basics of classification, how to preprocess data to make it
suitable for use in a classifier, and how to use our observed data to make
predictions. The next chapter will focus on how to evaluate how accurate the
predictions from our classifier are, as well as how to improve our classifier
(where possible) to maximize its accuracy.

## Chapter learning objectives 

By the end of the chapter, readers will be able to do the following:

- Recognize situations where a classifier would be appropriate for making predictions.
- Describe what a training data set is and how it is used in classification.
- Interpret the output of a classifier.
- Compute, by hand, the straight-line (Euclidean) distance between points on a graph when there are two predictor variables.
- Explain the $K$-nearest neighbor classification algorithm.
- Perform $K$-nearest neighbor classification in Python using `scikit-learn`.
- Use `StandardScaler` to preprocess data to be centered, scaled, and balanced.
- Combine preprocessing and model training using `make_pipeline`.

+++

## The classification problem

In many situations, we want to make `predictions`\\index{predictive question} based on the current situation
as well as past experiences. For instance, a doctor may want to diagnose a
patient as either diseased or healthy based on their symptoms and the doctor's
past experience with patients; an email provider might want to tag a given
email as "spam" or "not spam" based on the email's text and past email text data; 
or a credit card company may want to predict whether a purchase is fraudulent based
on the current purchase item, amount, and location as well as past purchases.
These tasks are all examples of \index{classification} **classification**, i.e., predicting a
categorical class (sometimes called a *label*) \index{class}\index{categorical variable} for an observation given its
other variables (sometimes called *features*). \index{feature|see{predictor}}

Generally, a classifier assigns an observation without a known class (e.g., a new patient) 
to a class (e.g., diseased or healthy) on the basis of how similar it is to other observations
for which we do know the class (e.g., previous patients with known diseases and
symptoms). These observations with known classes that we use as a basis for
prediction are called a **training set**; \index{training set} this name comes from the fact that
we use these data to train, or teach, our classifier. Once taught, we can use
the classifier to make predictions on new data for which we do not know the class.

There are many possible methods that we could use to predict
a categorical class/label for an observation. In this book, we will
focus on the widely used **$K$-nearest neighbors** \index{K-nearest neighbors} algorithm {cite:p}`knnfix,knncover`.
In your future studies, you might encounter decision trees, support vector machines (SVMs),
logistic regression, neural networks, and more; see the additional resources
section at the end of the next chapter for where to begin learning more about
these other methods. It is also worth mentioning that there are many
variations on the basic classification problem. For example, 
we focus on the setting of **binary classification** \index{classification!binary} where only two
classes are involved (e.g., a diagnosis of either healthy or diseased), but you may 
also run into multiclass classification problems with more than two
categories (e.g., a diagnosis of healthy, bronchitis, pneumonia, or a common cold).

## Exploring a data set

In this chapter and the next, we will study a data set of 
[digitized breast cancer image features](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29),
created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian {cite:p}`streetbreastcancer`. \index{breast cancer} 
Each row in the data set represents an
image of a tumor sample, including the diagnosis (benign or malignant) and
several other measurements (nucleus texture, perimeter, area, and more).
Diagnosis for each image was conducted by physicians. 

As with all data analyses, we first need to formulate a precise question that
we want to answer. Here, the question is *predictive*: \index{question!classification} can 
we use the tumor
image measurements available to us to predict whether a future tumor image
(with unknown diagnosis) shows a benign or malignant tumor? Answering this
question is important because traditional, non-data-driven methods for tumor
diagnosis are quite subjective and dependent upon how skilled and experienced
the diagnosing physician is. Furthermore, benign tumors are not normally
dangerous; the cells stay in the same place, and the tumor stops growing before
it gets very large. By contrast, in malignant tumors, the cells invade the
surrounding tissue and spread into nearby organs, where they can cause serious
damage {cite:p}`streetbreastcancer`.
Thus, it is important to quickly and accurately diagnose the tumor type to
guide patient treatment.

+++

### Loading the cancer data

Our first step is to load, wrangle, and explore the data using visualizations
in order to better understand the data we are working with. We start by
loading the `pandas` package needed for our analysis.

```{code-cell} ipython3
import pandas as pd
```

In this case, the file containing the breast cancer data set is a `.csv`
file with headers. We'll use the `read_csv` function with no additional
arguments, and then inspect its contents:

\index{read function!read\_csv}

```{code-cell} ipython3
cancer = pd.read_csv("data/wdbc.csv")
cancer
```

### Describing the variables in the cancer data set

Breast tumors can be diagnosed by performing a *biopsy*, a process where
tissue is removed from the body and examined for the presence of disease.
Traditionally these procedures were quite invasive; modern methods such as fine
needle aspiration, used to collect the present data set, extract only a small
amount of tissue and are less invasive. Based on a digital image of each breast
tissue sample collected for this data set, ten different variables were measured
for each cell nucleus in the image (items 3&ndash;12 of the list of variables below), and then the mean 
 for each variable across the nuclei was recorded. As part of the
data preparation, these values have been *standardized (centered and scaled)*; we will discuss what this
means and why we do it later in this chapter. Each image additionally was given
a unique ID and a diagnosis by a physician.  Therefore, the
total set of variables per image in this data set is:

1. ID: identification number 
2. Class: the diagnosis (M = malignant or B = benign)
3. Radius: the mean of distances from center to points on the perimeter
4. Texture: the standard deviation of gray-scale values
5. Perimeter: the length of the surrounding contour 
6. Area: the area inside the contour
7. Smoothness: the local variation in radius lengths
8. Compactness: the ratio of squared perimeter and area
9. Concavity: severity of concave portions of the contour 
10. Concave Points: the number of concave portions of the contour
11. Symmetry: how similar the nucleus is when mirrored 
12. Fractal Dimension: a measurement of how "rough" the perimeter is

+++

Below we use `.info()` \index{glimpse} to preview the data frame. This method can 
make it easier to inspect the data when we have a lot of columns, 
as it prints the data such that the columns go down 
the page (instead of across).

```{code-cell} ipython3
cancer.info()
```

From the summary of the data above, we can see that `Class` is of type object.

+++

Given that we only have two different values in our `Class` column (B for benign and M 
for malignant), we only expect to get two names back.

```{code-cell} ipython3
cancer['Class'].unique()
```

```{code-cell} ipython3
:tags: [remove-cell]

## The above was based on the following text and code in R textbook. ##
#######################################################################
# Below we use `glimpse` \index{glimpse} to preview the data frame. This function can
# make it easier to inspect the data when we have a lot of columns,
# as it prints the data such that the columns go down
# the page (instead of across).

# ```{r 05-glimpse}
# glimpse(cancer)
# ```

# From the summary of the data above, we can see that `Class` is of type character
# (denoted by `<chr>`). Since we will be working with `Class` as a
# categorical statistical variable, we will convert it to a factor using the
# function `as_factor`. \index{factor!as\_factor}

# ```{r 05-class}
# cancer <- cancer |>
#   mutate(Class = as_factor(Class))
# glimpse(cancer)
# ```

# Recall that factors have what are called "levels", which you can think of as categories. We
# can verify the levels of the `Class` column by using the `levels` \index{levels}\index{factor!levels} function.
# This function should return the name of each category in that column. Given
# that we only have two different values in our `Class` column (B for benign and M
# for malignant), we only expect to get two names back.  Note that the `levels` function requires a *vector* argument;
# so we use the `pull` function to extract a single column (`Class`) and
# pass that into the `levels` function to see the categories
# in the `Class` column.

# ```{r 05-levels}
# cancer |>
#   pull(Class) |>
#   levels()
# ```
```

### Exploring the cancer data

Before we start doing any modeling, let's explore our data set. Below we use
the `.groupby()`, `.count()` \index{group\_by}\index{summarize} methods to find the number and percentage 
of benign and malignant tumor observations in our data set. When paired with `.groupby()`, `.count()` counts the number of observations in each `Class` group. 
Then we calculate the percentage in each group by dividing by the total number of observations. We have 357 (63\%) benign and 212 (37\%) malignant tumor observations.

```{code-cell} ipython3
num_obs = len(cancer)
explore_cancer = pd.DataFrame()
explore_cancer['count'] = cancer.groupby('Class')['ID'].count()
explore_cancer['percentage'] = explore_cancer['count'] / num_obs * 100
explore_cancer
```

Next, let's draw a scatter plot \index{visualization!scatter} to visualize the relationship between the
perimeter and concavity variables. Rather than use `altair's` default palette,
we select our own colorblind-friendly colors&mdash;`"#efb13f"` 
for light orange and `"#86bfef"` for light blue&mdash;and
 pass them as the `scale` argument in the `color` argument. 
We also make the category labels ("B" and "M") more readable by 
changing them to "Benign" and "Malignant" using `.apply()` method on the dataframe.

```{code-cell} ipython3
:tags: []

colors = ["#86bfef", "#efb13f"]
cancer["Class"] = cancer["Class"].apply(
    lambda x: "Malignant" if (x == "M") else "Benign"
)
perim_concav = (
    alt.Chart(cancer)
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)
perim_concav
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-scatter
:figclass: caption-hack

Scatter plot of concavity versus perimeter colored by diagnosis label.
```

+++

In {numref}`fig:05-scatter`, we can see that malignant observations typically fall in
the upper right-hand corner of the plot area. By contrast, benign
observations typically fall in the lower left-hand corner of the plot. In other words,
benign observations tend to have lower concavity and perimeter values, and malignant
ones tend to have larger values. Suppose we
obtain a new observation not in the current data set that has all the variables
measured *except* the label (i.e., an image without the physician's diagnosis
for the tumor class). We could compute the standardized perimeter and concavity values,
resulting in values of, say, 1 and 1. Could we use this information to classify
that observation as benign or malignant? Based on the scatter plot, how might 
you classify that new observation? If the standardized concavity and perimeter
values are 1 and 1 respectively, the point would lie in the middle of the
orange cloud of malignant points and thus we could probably classify it as
malignant. Based on our visualization, it seems like 
the *prediction of an unobserved label* might be possible.

+++

## Classification with $K$-nearest neighbors

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [2, 4]
attrs = ["Perimeter", "Concavity"]
points_df = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df = pd.concat((cancer, points_df), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal dataset
my_distances = euclidean_distances(perim_concav_with_new_point_df.loc[:, attrs])[
    len(cancer)
][:-1]
glue("1-new_point_0", new_point[0])
glue("1-new_point_1", new_point[1])
```

In order to actually make predictions for new observations in practice, we
will need a classification algorithm. 
In this book, we will use the $K$-nearest neighbors \index{K-nearest neighbors!classification} classification algorithm.
To predict the label of a new observation (here, classify it as either benign
or malignant), the $K$-nearest neighbors classifier generally finds the $K$
"nearest" or "most similar" observations in our training set, and then uses
their diagnoses to make a prediction for the new observation's diagnosis. $K$ 
is a number that we must choose in advance; for now, we will assume that someone has chosen
$K$ for us. We will cover how to choose $K$ ourselves in the next chapter. 

To illustrate the concept of $K$-nearest neighbors classification, we 
will walk through an example.  Suppose we have a
new observation, with standardized perimeter of {glue:}`1-new_point_0` and standardized concavity of {glue:}`1-new_point_1`, whose 
diagnosis "Class" is unknown. This new observation is depicted by the red, diamond point in {numref}`fig:05-knn-2`.

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point = (
    alt.Chart(
        perim_concav_with_new_point_df,
        # title="Scatter plot of concavity versus perimeter with new observation represented as a red diamond.",
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)
glue('fig:05-knn-2', perim_concav_with_new_point, display=True)
```

:::{glue:figure} fig:05-knn-2
:name: fig:05-knn-2

Scatter plot of concavity versus perimeter with new observation represented as a red diamond.
:::

```{code-cell} ipython3
:tags: [remove-cell]

near_neighbor_df = pd.concat(
    (
        cancer.loc[np.argmin(my_distances), attrs],
        perim_concav_with_new_point_df.loc[len(cancer), attrs],
    ),
    axis=1,
).T
glue("1-neighbor_per", round(near_neighbor_df.iloc[0, :]['Perimeter'], 1))
glue("1-neighbor_con", round(near_neighbor_df.iloc[0, :]['Concavity'], 1))
```

{numref}`fig:05-knn-3` shows that the nearest point to this new observation is **malignant** and
located at the coordinates ({glue:}`1-neighbor_per`, {glue:}`1-neighbor_con`). The idea here is that if a point is close to another in the scatter plot,
then the perimeter and concavity values are similar, and so we may expect that
they would have the same diagnosis.

```{code-cell} ipython3
:tags: [remove-cell]

line = (
    alt.Chart(near_neighbor_df)
    .mark_line()
    .encode(x="Perimeter", y="Concavity", color=alt.value("black"))
)

glue('fig:05-knn-3', (perim_concav_with_new_point + line), display=True)
```

:::{glue:figure} fig:05-knn-3
:name: fig:05-knn-3

Scatter plot of concavity versus perimeter. The new observation is represented as a red diamond with a line to the one nearest neighbor, which has a malignant label.
:::

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0.2, 3.3]
attrs = ["Perimeter", "Concavity"]
points_df2 = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df2 = pd.concat((cancer, points_df2), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal dataset
my_distances2 = euclidean_distances(perim_concav_with_new_point_df2.loc[:, attrs])[
    len(cancer)
][:-1]
glue("2-new_point_0", new_point[0])
glue("2-new_point_1", new_point[1])
```

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point2 = (
    alt.Chart(
        perim_concav_with_new_point_df2,
        # title="Scatter plot of concavity versus perimeter with new observation represented as a red diamond.",
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)

near_neighbor_df2 = pd.concat(
    (
        cancer.loc[np.argmin(my_distances2), attrs],
        perim_concav_with_new_point_df2.loc[len(cancer), attrs],
    ),
    axis=1,
).T
line2 = alt.Chart(near_neighbor_df2).mark_line().encode(
    x='Perimeter',
    y='Concavity',
    color=alt.value('black')
)

glue("2-neighbor_per", round(near_neighbor_df2.iloc[0, :]['Perimeter'], 1))
glue("2-neighbor_con", round(near_neighbor_df2.iloc[0, :]['Concavity'], 1))
glue('fig:05-knn-4', (perim_concav_with_new_point2 + line2), display=True)
```

Suppose we have another new observation with standardized perimeter {glue:}`2-new_point_0` and
concavity of {glue:}`2-new_point_1`. Looking at the scatter plot in {numref}`fig:05-knn-4`, how would you
classify this red, diamond observation? The nearest neighbor to this new point is a
**benign** observation at ({glue:}`2-neighbor_per`, {glue:}`2-neighbor_con`).
Does this seem like the right prediction to make for this observation? Probably 
not, if you consider the other nearby points.

+++

:::{glue:figure} fig:05-knn-4
:name: fig:05-knn-4

Scatter plot of concavity versus perimeter. The new observation is represented as a red diamond with a line to the one nearest neighbor, which has a benign label.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# The index of 3 rows that has smallest distance to the new point
min_3_idx = np.argpartition(my_distances2, 3)[:3]
near_neighbor_df3 = pd.concat(
    (
        cancer.loc[min_3_idx[1], attrs],
        perim_concav_with_new_point_df2.loc[len(cancer), attrs],
    ),
    axis=1,
).T
near_neighbor_df4 = pd.concat(
    (
        cancer.loc[min_3_idx[2], attrs],
        perim_concav_with_new_point_df2.loc[len(cancer), attrs],
    ),
    axis=1,
).T
```

```{code-cell} ipython3
:tags: [remove-cell]

line3 = alt.Chart(near_neighbor_df3).mark_line().encode(
    x='Perimeter',
    y='Concavity',
    color=alt.value('black')
)
line4 = alt.Chart(near_neighbor_df4).mark_line().encode(
    x='Perimeter',
    y='Concavity',
    color=alt.value('black')
)
glue('fig:05-knn-5', (perim_concav_with_new_point2 + line2 + line3 + line4), display=True)
```

To improve the prediction we can consider several
neighboring points, say $K = 3$, that are closest to the new observation
to predict its diagnosis class. Among those 3 closest points, we use the
*majority class* as our prediction for the new observation. As shown in {numref}`fig:05-knn-5`, we
see that the diagnoses of 2 of the 3 nearest neighbors to our new observation
are malignant. Therefore we take majority vote and classify our new red, diamond
observation as malignant.

+++

:::{glue:figure} fig:05-knn-5
:name: fig:05-knn-5

Scatter plot of concavity versus perimeter with three nearest neighbors.
:::

+++

Here we chose the $K=3$ nearest observations, but there is nothing special
about $K=3$. We could have used $K=4, 5$ or more (though we may want to choose
an odd number to avoid ties). We will discuss more about choosing $K$ in the
next chapter.

+++

### Distance between points

We decide which points are the $K$ "nearest" to our new observation
using the *straight-line distance* (we will often just refer to this as *distance*). \index{distance!K-nearest neighbors}\index{straight line!distance}
Suppose we have two observations $a$ and $b$, each having two predictor variables, $x$ and $y$.
Denote $a_x$ and $a_y$ to be the values of variables $x$ and $y$ for observation $a$;
$b_x$ and $b_y$ have similar definitions for observation $b$.
Then the straight-line distance between observation $a$ and $b$ on the x-y plane can 
be computed using the following formula: 

$$\mathrm{Distance} = \sqrt{(a_x -b_x)^2 + (a_y - b_y)^2}$$

+++

To find the $K$ nearest neighbors to our new observation, we compute the distance
from that new observation to each observation in our training data, and select the $K$ observations corresponding to the
$K$ *smallest* distance values. For example, suppose we want to use $K=5$ neighbors to classify a new 
observation with perimeter of {glue:}`3-new_point_0` and 
concavity of {glue:}`3-new_point_1`, shown as a red diamond in {numref}`fig:05-multiknn-1`. Let's calculate the distances
between our new point and each of the observations in the training set to find
the $K=5$ neighbors that are nearest to our new point. 
You will see in the code below \index{mutate}, we compute the straight-line
distance using the formula above: we square the differences between the two observations' perimeter 
and concavity coordinates, add the squared differences, and then take the square root.

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0, 3.5]
attrs = ["Perimeter", "Concavity"]
points_df3 = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df3 = pd.concat((cancer, points_df3), ignore_index=True)
perim_concav_with_new_point3 = (
    alt.Chart(
        perim_concav_with_new_point_df3,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)

glue("3-new_point_0", new_point[0])
glue("3-new_point_1", new_point[1])
glue("fig:05-multiknn-1", perim_concav_with_new_point3)
```

:::{glue:figure} fig:05-multiknn-1
:name: fig:05-multiknn-1

Scatter plot of concavity versus perimeter with new observation represented as a red diamond.
:::

```{code-cell} ipython3
new_obs_Perimeter = 0
new_obs_Concavity = 3.5
cancer_dist = cancer.loc[:, ["ID", "Perimeter", "Concavity", "Class"]]
cancer_dist["dist_from_new"] = np.sqrt(
    (cancer_dist["Perimeter"] - new_obs_Perimeter) ** 2
    + (cancer_dist["Concavity"] - new_obs_Concavity) ** 2
)
# sort the rows in ascending order and take the first 5 rows
cancer_dist = cancer_dist.sort_values(by="dist_from_new").head(5)
cancer_dist
```

In {numref}`tab:05-multiknn-mathtable` we show in mathematical detail how
we computed the `dist_from_new` variable (the
distance to the new observation) for each of the 5 nearest neighbors in the
training data.

```{code-cell} ipython3
:tags: [remove-cell]

## Couldn't find ways to have nice Latex equations in pandas dataframe

# cancer_dist_eq = cancer_dist.copy()
# cancer_dist_eq['Perimeter'] = round(cancer_dist_eq['Perimeter'], 2)
# cancer_dist_eq['Concavity'] = round(cancer_dist_eq['Concavity'], 2)
# for i in list(cancer_dist_eq.index):
#     cancer_dist_eq.loc[
#         i, "Distance"
#     ] = f"[({new_obs_Perimeter} - {round(cancer_dist_eq.loc[i, 'Perimeter'], 2)})² + ({new_obs_Concavity} - {round(cancer_dist_eq.loc[i, 'Concavity'], 2)})²]¹/² = {round(cancer_dist_eq.loc[i, 'dist_from_new'], 2)}"
# cancer_dist_eq[["Perimeter", "Concavity", "Distance", "Class"]]
```

```{table} Evaluating the distances from the new observation to each of its 5 nearest neighbors
:name: tab:05-multiknn-mathtable
| Perimeter | Concavity | Distance            | Class |
|-----------|-----------|----------------------------------------|-------|
| 0.24      | 2.65      | $\sqrt{(0-0.24)^2+(3.5-2.65)^2}=0.88$| B     |
| 0.75      | 2.87      | $\sqrt{(0-0.75)^2+(3.5-2.87)^2}=0.98$| M     |
| 0.62      | 2.54      | $\sqrt{(0-0.62)^2+(3.5-2.54)^2}=1.14$| M     |
| 0.42      | 2.31      | $\sqrt{(0-0.42)^2+(3.5-2.31)^2}=1.26$| M     |
| -1.16     | 4.04      | $\sqrt{(0-(-1.16))^2+(3.5-4.04)^2}=1.28$| B     |
```

+++

The result of this computation shows that 3 of the 5 nearest neighbors to our new observation are
malignant (`M`); since this is the majority, we classify our new observation as malignant. 
These 5 neighbors are circled in {numref}`fig:05-multiknn-3`.

```{code-cell} ipython3
:tags: [remove-cell]

circle_path_df = pd.DataFrame(
    {
        "Perimeter": new_point[0] + 1.4 * np.cos(np.linspace(0, 2 * np.pi, 100)),
        "Concavity": new_point[1] + 1.4 * np.sin(np.linspace(0, 2 * np.pi, 100)),
    }
)
circle = alt.Chart(circle_path_df.reset_index()).mark_line(color='black').encode(
    x='Perimeter',
    y='Concavity',
    order='index'
)

glue("fig:05-multiknn-3", (perim_concav_with_new_point3 + circle))
```

:::{glue:figure} fig:05-multiknn-3
:name: fig:05-multiknn-3

Scatter plot of concavity versus perimeter with 5 nearest neighbors circled.
:::

+++

### More than two explanatory variables 

Although the above description is directed toward two predictor variables, 
exactly the same $K$-nearest neighbors algorithm applies when you
have a higher number of predictor variables.  Each predictor variable may give us new
information to help create our classifier.  The only difference is the formula
for the distance between points. Suppose we have $m$ predictor
variables for two observations $a$ and $b$, i.e., 
$a = (a_{1}, a_{2}, \dots, a_{m})$ and
$b = (b_{1}, b_{2}, \dots, b_{m})$.

The distance formula becomes \index{distance!more than two variables}

$$\mathrm{Distance} = \sqrt{(a_{1} -b_{1})^2 + (a_{2} - b_{2})^2 + \dots + (a_{m} - b_{m})^2}.$$

This formula still corresponds to a straight-line distance, just in a space
with more dimensions. Suppose we want to calculate the distance between a new
observation with a perimeter of 0, concavity of 3.5, and symmetry of 1, and
another observation with a perimeter, concavity, and symmetry of 0.417, 2.31, and
0.837 respectively. We have two observations with three predictor variables:
perimeter, concavity, and symmetry. Previously, when we had two variables, we
added up the squared difference between each of our (two) variables, and then
took the square root. Now we will do the same, except for our three variables.
We calculate the distance as follows

$$\mathrm{Distance} =\sqrt{(0 - 0.417)^2 + (3.5 - 2.31)^2 + (1 - 0.837)^2} = 1.27.$$

Let's calculate the distances between our new observation and each of the
observations in the training set to find the $K=5$ neighbors when we have these
three predictors.

```{code-cell} ipython3
new_obs_Perimeter = 0
new_obs_Concavity = 3.5
new_obs_Symmetry = 1
cancer_dist2 = cancer.loc[:, ["ID", "Perimeter", "Concavity", "Symmetry", "Class"]]
cancer_dist2["dist_from_new"] = np.sqrt(
    (cancer_dist2["Perimeter"] - new_obs_Perimeter) ** 2
    + (cancer_dist2["Concavity"] - new_obs_Concavity) ** 2
    + (cancer_dist2["Symmetry"] - new_obs_Symmetry) ** 2
)
# sort the rows in ascending order and take the first 5 rows
cancer_dist2 = cancer_dist2.sort_values(by="dist_from_new").head(5)
cancer_dist2
```

Based on $K=5$ nearest neighbors with these three predictors we would classify the new observation as malignant since 4 out of 5 of the nearest neighbors are malignant class. 
{numref}`fig:05-more` shows what the data look like when we visualize them 
as a 3-dimensional scatter with lines from the new observation to its five nearest neighbors.

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0, 3.5, 1]
attrs = ["Perimeter", "Concavity", "Symmetry"]
points_df4 = pd.DataFrame(
    {
        "Perimeter": new_point[0],
        "Concavity": new_point[1],
        "Symmetry": new_point[2],
        "Class": ["Unknown"],
    }
)
perim_concav_with_new_point_df4 = pd.concat((cancer, points_df4), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal dataset
my_distances4 = euclidean_distances(perim_concav_with_new_point_df4.loc[:, attrs])[
    len(cancer)
][:-1]
```

```{code-cell} ipython3
:tags: [remove-cell]

# The index of 5 rows that has smallest distance to the new point
min_5_idx = np.argpartition(my_distances4, 5)[:5]

neighbor_df_list = []
for idx in min_5_idx:
    neighbor_df = pd.concat(
        (
            cancer.loc[idx, attrs + ['Class']],
            perim_concav_with_new_point_df4.loc[len(cancer), attrs + ['Class']],
        ),
        axis=1,
    ).T
    neighbor_df_list.append(neighbor_df)
```

```{code-cell} ipython3
:tags: [remove-cell]

fig = px.scatter_3d(
    perim_concav_with_new_point_df4,
    x="Perimeter",
    y="Concavity",
    z="Symmetry",
    color="Class",
    symbol="Class",
    opacity=0.5,
)
# specify trace names and symbols in a dict
symbols = {"Malignant": "circle", "Benign": "circle", "Unknown": "diamond"}

# set all symbols in fig
for i, d in enumerate(fig.data):
    fig.data[i].marker.symbol = symbols[fig.data[i].name]

# specify trace names and colors in a dict
colors = {"Malignant": "#86bfef", "Benign": "#efb13f", "Unknown": "red"}

# set all colors in fig
for i, d in enumerate(fig.data):
    fig.data[i].marker.color = colors[fig.data[i].name]

# set a fixed custom marker size
fig.update_traces(marker={"size": 5})

# add lines
for neighbor_df in neighbor_df_list:
    fig.add_trace(
        go.Scatter3d(
            x=neighbor_df["Perimeter"],
            y=neighbor_df["Concavity"],
            z=neighbor_df["Symmetry"],
            line_color=colors[neighbor_df.iloc[0]["Class"]],
            name=neighbor_df.iloc[0]["Class"],
            mode="lines",
            line=dict(width=2),
            showlegend=False,
        )
    )


# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=1), template="plotly_white")

plot(fig, filename="img/classification1/fig05-more.html", auto_open=False)
# display(HTML("img/classification1/fig05-more.html"))
```

```{code-cell} ipython3
:tags: [remove-input]

display(HTML("img/classification1/fig05-more.html"))
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-more
:figclass: caption-hack

3D scatter plot of the standardized symmetry, concavity, and perimeter variables. Note that in general we recommend against using 3D visualizations; here we show the data in 3D only to illustrate what higher dimensions and nearest neighbors look like, for learning purposes.
```

+++

### Summary of $K$-nearest neighbors algorithm

In order to classify a new observation using a $K$-nearest neighbor classifier, we have to do the following:

1. Compute the distance between the new observation and each observation in the training set.
2. Sort the data table in ascending order according to the distances.
3. Choose the top $K$ rows of the sorted table.
4. Classify the new observation based on a majority vote of the neighbor classes.

+++

## $K$-nearest neighbors with `scikit-learn`

Coding the $K$-nearest neighbors algorithm in Python ourselves can get complicated,
especially if we want to handle multiple classes, more than two variables,
or predict the class for multiple new observations. Thankfully, in Python,
the $K$-nearest neighbors algorithm is 
implemented in [the `scikit-learn` package](https://scikit-learn.org/stable/index.html) along with 
many [other models](https://scikit-learn.org/stable/user_guide.html) that you will encounter in this and future chapters of the book. Using the functions 
in the `scikit-learn` package will help keep our code simple, readable and accurate; the 
less we have to code ourselves, the fewer mistakes we will likely make. We 
start by importing `KNeighborsClassifier` from the `sklearn.neighbors` module.

```{code-cell} ipython3
:tags: [remove-cell]

## The above was based on:

# Coding the $K$-nearest neighbors algorithm in R ourselves can get complicated,
# especially if we want to handle multiple classes, more than two variables,
# or predict the class for multiple new observations. Thankfully, in R,
# the $K$-nearest neighbors algorithm is
# implemented in [the `parsnip` R package](https://parsnip.tidymodels.org/) [@parsnip]
# included in `tidymodels`, along with
# many [other models](https://www.tidymodels.org/find/parsnip/) \index{tidymodels}\index{parsnip}
#  that you will encounter in this and future chapters of the book. The `tidymodels` collection
# provides tools to help make and use models, such as classifiers.  Using the packages
# in this collection will help keep our code simple, readable and accurate; the
# less we have to code ourselves, the fewer mistakes we will likely make. We
# start by loading `tidymodels`.
```

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier
```

Let's walk through how to use `KNeighborsClassifier` to perform $K$-nearest neighbors classification. 
We will use the `cancer` data set from above, with
perimeter and concavity as predictors and $K = 5$ neighbors to build our classifier. Then
we will use the classifier to predict the diagnosis label for a new observation with
perimeter 0, concavity 3.5, and an unknown diagnosis label. Let's pick out our two desired
predictor variables and class label and store them as a new data set named `cancer_train`:

```{code-cell} ipython3
cancer_train = cancer.loc[:, ['Class', 'Perimeter', 'Concavity']]
cancer_train
```

Next, we create a *model specification* for $K$-nearest neighbors classification
by creating a `KNeighborsClassifier` object, specifying that we want to use $K = 5$ neighbors
(we will discuss how to choose $K$ in the next chapter) and the straight-line 
distance (`weights="uniform"`). The `weights` argument controls
how neighbors vote when classifying a new observation; by setting it to `"uniform"`,
each of the $K$ nearest neighbors gets exactly 1 vote as described above. Other choices, 
which weigh each neighbor's vote differently, can be found on 
[the `scikit-learn` website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier).

```{code-cell} ipython3
:tags: [remove-cell]

## The above was based on:

# Next, we create a *model specification* for \index{tidymodels!model specification} $K$-nearest neighbors classification
# by calling the `nearest_neighbor` function, specifying that we want to use $K = 5$ neighbors
# (we will discuss how to choose $K$ in the next chapter) and the straight-line 
# distance (`weight_func = "rectangular"`). The `weight_func` argument controls
# how neighbors vote when classifying a new observation; by setting it to `"rectangular"`,
# each of the $K$ nearest neighbors gets exactly 1 vote as described above. Other choices, 
# which weigh each neighbor's vote differently, can be found on 
# [the `parsnip` website](https://parsnip.tidymodels.org/reference/nearest_neighbor.html).
# In the `set_engine` \index{tidymodels!engine} argument, we specify which package or system will be used for training
# the model. Here `kknn` is the R package we will use for performing $K$-nearest neighbors classification.
# Finally, we specify that this is a classification problem with the `set_mode` function.
```

```{code-cell} ipython3
knn_spec = KNeighborsClassifier(n_neighbors=5)
knn_spec
```

In order to fit the model on the breast cancer data, we need to call `fit` on the classifier object and pass the data in the argument. We also need to specify what variables to use as predictors and what variable to use as the target. Below, the `X=cancer_train[["Perimeter", "Concavity"]]` and the `y=cancer_train['Class']` argument specifies 
that `Class` is the target variable (the one we want to predict),
and both `Perimeter` and `Concavity` are to be used as the predictors.

```{code-cell} ipython3
:tags: [remove-cell]

## The above was based on:

# In order to fit the model on the breast cancer data, we need to pass the model specification
# and the data set to the `fit` function. We also need to specify what variables to use as predictors
# and what variable to use as the target. Below, the `Class ~ Perimeter + Concavity` argument specifies 
# that `Class` is the target variable (the one we want to predict),
# and both `Perimeter` and `Concavity` are to be used as the predictors.


# We can also use a convenient shorthand syntax using a period, `Class ~ .`, to indicate
# that we want to use every variable *except* `Class` \index{tidymodels!model formula} as a predictor in the model.
# In this particular setup, since `Concavity` and `Perimeter` are the only two predictors in the `cancer_train`
# data frame, `Class ~ Perimeter + Concavity` and `Class ~ .` are equivalent.
# In general, you can choose individual predictors using the `+` symbol, or you can specify to
# use *all* predictors using the `.` symbol.
```

```{code-cell} ipython3
knn_spec.fit(X=cancer_train[["Perimeter", "Concavity"]], y=cancer_train["Class"]);
```

```{code-cell} ipython3
:tags: [remove-cell]

# Here you can see the final trained model summary. It confirms that the computational engine used
# to train the model  was `kknn::train.kknn`. It also shows the fraction of errors made by
# the nearest neighbor model, but we will ignore this for now and discuss it in more detail
# in the next chapter.
# Finally, it shows (somewhat confusingly) that the "best" weight function 
# was "rectangular" and "best" setting of $K$ was 5; but since we specified these earlier,
# R is just repeating those settings to us here. In the next chapter, we will actually
# let R find the value of $K$ for us. 
```

Finally, we make the prediction on the new observation by calling `predict` \index{tidymodels!predict} on the classifier object,
passing the new observation itself. As above, 
when we ran the $K$-nearest neighbors
classification algorithm manually, the `knn_fit` object classifies the new observation as "Malignant". Note that the `predict` function outputs a `numpy` array with the model's prediction.

```{code-cell} ipython3
new_obs = pd.DataFrame({'Perimeter': [0], 'Concavity': [3.5]})
knn_spec.predict(new_obs)
```

Is this predicted malignant label the true class for this observation? 
Well, we don't know because we do not have this
observation's diagnosis&mdash; that is what we were trying to predict! The 
classifier's prediction is not necessarily correct, but in the next chapter, we will 
learn ways to quantify how accurate we think our predictions are.

+++

## Data preprocessing with `scikit-learn`

### Centering and scaling

When using $K$-nearest neighbor classification, the *scale* \index{scaling} of each variable
(i.e., its size and range of values) matters. Since the classifier predicts
classes by identifying observations nearest to it, any variables with 
a large scale will have a much larger effect than variables with a small
scale. But just because a variable has a large scale *doesn't mean* that it is
more important for making accurate predictions. For example, suppose you have a
data set with two features, salary (in dollars) and years of education, and
you want to predict the corresponding type of job. When we compute the
neighbor distances, a difference of \$1000 is huge compared to a difference of
10 years of education. But for our conceptual understanding and answering of
the problem, it's the opposite; 10 years of education is huge compared to a
difference of \$1000 in yearly salary!

+++

In many other predictive models, the *center* of each variable (e.g., its mean)
matters as well. For example, if we had a data set with a temperature variable
measured in degrees Kelvin, and the same data set with temperature measured in
degrees Celsius, the two variables would differ by a constant shift of 273
(even though they contain exactly the same information). Likewise, in our
hypothetical job classification example, we would likely see that the center of
the salary variable is in the tens of thousands, while the center of the years
of education variable is in the single digits. Although this doesn't affect the
$K$-nearest neighbor classification algorithm, this large shift can change the
outcome of using many other predictive models.  \index{centering}

To scale and center our data, we need to find
our variables' *mean* (the average, which quantifies the "central" value of a 
set of numbers) and *standard deviation* (a number quantifying how spread out values are). 
For each observed value of the variable, we subtract the mean (i.e., center the variable) 
and divide by the standard deviation (i.e., scale the variable). When we do this, the data 
is said to be *standardized*, \index{standardization!K-nearest neighbors} and all variables in a data set will have a mean of 0 
and a standard deviation of 1. To illustrate the effect that standardization can have on the $K$-nearest
neighbor algorithm, we will read in the original, unstandardized Wisconsin breast
cancer data set; we have been using a standardized version of the data set up
until now. To keep things simple, we will just use the `Area`, `Smoothness`, and `Class`
variables:

```{code-cell} ipython3
unscaled_cancer = pd.read_csv("data/unscaled_wdbc.csv")
unscaled_cancer = unscaled_cancer[['Class', 'Area', 'Smoothness']]
unscaled_cancer
```

Looking at the unscaled and uncentered data above, you can see that the differences
between the values for area measurements are much larger than those for
smoothness. Will this affect
predictions? In order to find out, we will create a scatter plot of these two
predictors (colored by diagnosis) for both the unstandardized data we just
loaded, and the standardized version of that same data. But first, we need to
standardize the `unscaled_cancer` data set with `scikit-learn`.

In the `scikit-learn` framework, all data preprocessing and modeling can be built using a [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?highlight=pipeline#sklearn.pipeline.Pipeline), or a more convenient function [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) for simple pipeline construction.
Here we will initialize a preprocessor using `make_column_transformer` for 
the `unscaled_cancer` data above, specifying
that we want to standardize the predictors `Area` and `Smoothness`:

```{code-cell} ipython3
:tags: [remove-cell]

## The above was based on:

# In the `tidymodels` framework, all data preprocessing happens 
# using a `recipe` from [the `recipes` R package](https://recipes.tidymodels.org/) [@recipes]
# Here we will initialize a recipe \index{recipe} \index{tidymodels!recipe|see{recipe}} for 
# the `unscaled_cancer` data above, specifying
# that the `Class` variable is the target, and all other variables are predictors:
```

```{code-cell} ipython3
preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
preprocessor
```

So far, we have built a preprocessor so that each of the predictors have a mean of 0 and standard deviation of 1.

You can now see that the recipe includes a scaling and centering step for all predictor variables.
Note that when you add a step to a `ColumnTransformer`, you must specify what columns to apply the step to.
Here we specified that `StandardScaler` should be applied to 
all predictor variables.

At this point, the data are not yet scaled and centered. To actually scale and center 
the data, we need to call `fit` and `transform` on the unscaled data ( can be combined into `fit_transform`).

```{code-cell} ipython3
:tags: [remove-cell]

# So far, there is not much in the recipe; just a statement about the number of targets
# and predictors. Let's add 
# scaling (`step_scale`) \index{recipe!step\_scale} and 
# centering (`step_center`) \index{recipe!step\_center} steps for 
# all of the predictors so that they each have a mean of 0 and standard deviation of 1.
# Note that `tidyverse` actually provides `step_normalize`, which does both centering and scaling in
# a single recipe step; in this book we will keep `step_scale` and `step_center` separate
# to emphasize conceptually that there are two steps happening. 
# The `prep` function finalizes the recipe by using the data (here, `unscaled_cancer`)  \index{tidymodels!prep}\index{prep|see{tidymodels}}
# to compute anything necessary to run the recipe (in this case, the column means and standard
# deviations):
```

```{code-cell} ipython3
:tags: [remove-cell]

# You can now see that the recipe includes a scaling and centering step for all predictor variables.
# Note that when you add a step to a recipe, you must specify what columns to apply the step to.
# Here we used the `all_predictors()` \index{recipe!all\_predictors} function to specify that each step should be applied to 
# all predictor variables. However, there are a number of different arguments one could use here,
# as well as naming particular columns with the same syntax as the `select` function. 
# For example:

# - `all_nominal()` and `all_numeric()`: specify all categorical or all numeric variables
# - `all_predictors()` and `all_outcomes()`: specify all predictor or all target variables
# - `Area, Smoothness`: specify both the `Area` and `Smoothness` variable
# - `-Class`: specify everything except the `Class` variable

# You can find a full set of all the steps and variable selection functions
# on the [`recipes` reference page](https://recipes.tidymodels.org/reference/index.html).

# At this point, we have calculated the required statistics based on the data input into the 
# recipe, but the data are not yet scaled and centered. To actually scale and center 
# the data, we need to apply the `bake` \index{tidymodels!bake} \index{bake|see{tidymodels}} function to the unscaled data.
```

```{code-cell} ipython3
preprocessor.fit(unscaled_cancer)
scaled_cancer = preprocessor.transform(unscaled_cancer)
# scaled_cancer = preprocessor.fit_transform(unscaled_cancer)
scaled_cancer = pd.DataFrame(scaled_cancer, columns=['Area', 'Smoothness'])
scaled_cancer['Class'] = unscaled_cancer['Class']
scaled_cancer
```

It may seem redundant that we had to both `fit` *and* `transform` to scale and center the data.
 However, we do this in two steps so we can specify a different data set in the `transform` step if we want. 
 For example, we may want to specify new data that were not part of the training set. 

You may wonder why we are doing so much work just to center and
scale our variables. Can't we just manually scale and center the `Area` and
`Smoothness` variables ourselves before building our $K$-nearest neighbor model? Well,
technically *yes*; but doing so is error-prone.  In particular, we might
accidentally forget to apply the same centering / scaling when making
predictions, or accidentally apply a *different* centering / scaling than what
we used while training. Proper use of a `ColumnTransformer` helps keep our code simple,
readable, and error-free. Furthermore, note that using `fit` and `transform` on the preprocessor is required only when you want to inspect the result of the preprocessing steps
yourself. You will see further on in
Section {ref}`08:puttingittogetherworkflow` that `scikit-learn` provides tools to
automatically streamline the preprocesser and the model so that you can call`fit` 
and `transform` on the `Pipeline` as necessary without additional coding effort.

```{code-cell} ipython3
:tags: [remove-cell]

# It may seem redundant that we had to both `bake` *and* `prep` to scale and center the data.
#  However, we do this in two steps so we can specify a different data set in the `bake` step if we want. 
#  For example, we may want to specify new data that were not part of the training set. 

# You may wonder why we are doing so much work just to center and
# scale our variables. Can't we just manually scale and center the `Area` and
# `Smoothness` variables ourselves before building our $K$-nearest neighbor model? Well,
# technically *yes*; but doing so is error-prone.  In particular, we might
# accidentally forget to apply the same centering / scaling when making
# predictions, or accidentally apply a *different* centering / scaling than what
# we used while training. Proper use of a `recipe` helps keep our code simple,
# readable, and error-free. Furthermore, note that using `prep` and `bake` is
# required only when you want to inspect the result of the preprocessing steps
# yourself. You will see further on in Section
# \@ref(puttingittogetherworkflow) that `tidymodels` provides tools to
# automatically apply `prep` and `bake` as necessary without additional coding effort.
```

{numref}`fig:05-scaling-plt` shows the two scatter plots side-by-side&mdash;one for `unscaled_cancer` and one for
`scaled_cancer`. Each has the same new observation annotated with its $K=3$ nearest neighbors.
In the original unstandardized data plot, you can see some odd choices
for the three nearest neighbors. In particular, the "neighbors" are visually
well within the cloud of benign observations, and the neighbors are all nearly
vertically aligned with the new observation (which is why it looks like there
is only one black line on this plot). {numref}`fig:05-scaling-plt-zoomed`
shows a close-up of that region on the unstandardized plot. Here the computation of nearest
neighbors is dominated by the much larger-scale area variable. The plot for standardized data 
on the right in {numref}`fig:05-scaling-plt` shows a much more intuitively reasonable
selection of nearest neighbors. Thus, standardizing the data can change things
in an important way when we are using predictive algorithms. 
Standardizing your data should be a part of the preprocessing you do
before predictive modeling and you should always think carefully about your problem domain and
whether you need to standardize your data.

```{code-cell} ipython3
:tags: [remove-cell]

def class_dscp(x):
    if x == "M":
        return "Malignant"
    elif x == "B":
        return "Benign"
    else:
        return x


attrs = ["Area", "Smoothness"]
new_obs = pd.DataFrame({"Class": ["Unknwon"], "Area": 400, "Smoothness": 0.135})
unscaled_cancer["Class"] = unscaled_cancer["Class"].apply(class_dscp)
area_smoothness_new_df = pd.concat((unscaled_cancer, new_obs), ignore_index=True)
my_distances = euclidean_distances(area_smoothness_new_df.loc[:, attrs])[
    len(unscaled_cancer)
][:-1]
area_smoothness_new_point = (
    alt.Chart(
        area_smoothness_new_df,
        title=alt.TitleParams(text="Unstandardized data", anchor="start"),
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area"),
        y=alt.Y("Smoothness"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)

# The index of 3 rows that has smallest distance to the new point
min_3_idx = np.argpartition(my_distances, 3)[:3]
neighbor1 = pd.concat(
    (
        unscaled_cancer.loc[min_3_idx[0], attrs],
        new_obs[attrs].T,
    ),
    axis=1,
).T
neighbor2 = pd.concat(
    (
        unscaled_cancer.loc[min_3_idx[1], attrs],
        new_obs[attrs].T,
    ),
    axis=1,
).T
neighbor3 = pd.concat(
    (
        unscaled_cancer.loc[min_3_idx[2], attrs],
        new_obs[attrs].T,
    ),
    axis=1,
).T

line1 = (
    alt.Chart(neighbor1)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line2 = (
    alt.Chart(neighbor2)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line3 = (
    alt.Chart(neighbor3)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)

area_smoothness_new_point = area_smoothness_new_point + line1 + line2 + line3
```

```{code-cell} ipython3
:tags: [remove-cell]

attrs = ["Area", "Smoothness"]
new_obs_scaled = pd.DataFrame({"Class": ["Unknwon"], "Area": -0.72, "Smoothness": 2.8})
scaled_cancer["Class"] = scaled_cancer["Class"].apply(class_dscp)
area_smoothness_new_df_scaled = pd.concat(
    (scaled_cancer, new_obs_scaled), ignore_index=True
)
my_distances_scaled = euclidean_distances(area_smoothness_new_df_scaled.loc[:, attrs])[
    len(scaled_cancer)
][:-1]
area_smoothness_new_point_scaled = (
    alt.Chart(
        area_smoothness_new_df_scaled,
        title=alt.TitleParams(text="Standardized data", anchor="start"),
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area", title="Area (standardized)"),
        y=alt.Y("Smoothness", title="Smoothness (standardized)"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)
min_3_idx_scaled = np.argpartition(my_distances_scaled, 3)[:3]
neighbor1_scaled = pd.concat(
    (
        scaled_cancer.loc[min_3_idx_scaled[0], attrs],
        new_obs_scaled[attrs].T,
    ),
    axis=1,
).T
neighbor2_scaled = pd.concat(
    (
        scaled_cancer.loc[min_3_idx_scaled[1], attrs],
        new_obs_scaled[attrs].T,
    ),
    axis=1,
).T
neighbor3_scaled = pd.concat(
    (
        scaled_cancer.loc[min_3_idx_scaled[2], attrs],
        new_obs_scaled[attrs].T,
    ),
    axis=1,
).T

line1_scaled = (
    alt.Chart(neighbor1_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line2_scaled = (
    alt.Chart(neighbor2_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line3_scaled = (
    alt.Chart(neighbor3_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)

area_smoothness_new_point_scaled = (
    area_smoothness_new_point_scaled + line1_scaled + line2_scaled + line3_scaled
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue(
    "fig:05-scaling-plt",
    (area_smoothness_new_point | area_smoothness_new_point_scaled).configure_legend(
        orient="bottom", titleAnchor="middle"
    ),
)
```

:::{glue:figure} fig:05-scaling-plt
:name: fig:05-scaling-plt

Comparison of K = 3 nearest neighbors with standardized and unstandardized data.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# Could not find something mimicing `facet_zoom` in R, here are 2 plots trying to
# illustrate similar points
# 1. interactive plot which allows zooming in/out
glue('fig:05-scaling-plt-interactive', area_smoothness_new_point.interactive())
```

+++ {"tags": ["remove-cell"]}

:::{glue:figure} fig:05-scaling-plt-interactive
:name: fig:05-scaling-plt-interactive

Close-up of three nearest neighbors for unstandardized data.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# 2. Static plot, Zoom-in
zoom_area_smoothness_new_point = (
    alt.Chart(
        area_smoothness_new_df,
        title=alt.TitleParams(text="Unstandardized data", anchor="start"),
    )
    .mark_point(clip=True, opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area", scale=alt.Scale(domain=(380, 420))),
        y=alt.Y("Smoothness", scale=alt.Scale(domain=(0.08, 0.14))),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)
zoom_area_smoothness_new_point + line1 + line2 + line3
glue("fig:05-scaling-plt-zoomed", (zoom_area_smoothness_new_point + line1 + line2 + line3))
```

:::{glue:figure} fig:05-scaling-plt-zoomed
:name: fig:05-scaling-plt-zoomed

Close-up of three nearest neighbors for unstandardized data.
:::

+++

### Balancing

Another potential issue in a data set for a classifier is *class imbalance*, \index{balance}\index{imbalance}
i.e., when one label is much more common than another. Since classifiers like
the $K$-nearest neighbor algorithm use the labels of nearby points to predict
the label of a new point, if there are many more data points with one label
overall, the algorithm is more likely to pick that label in general (even if
the "pattern" of data suggests otherwise). Class imbalance is actually quite a
common and important problem: from rare disease diagnosis to malicious email
detection, there are many cases in which the "important" class to identify
(presence of disease, malicious email) is much rarer than the "unimportant"
class (no disease, normal email).

To better illustrate the problem, let's revisit the scaled breast cancer data, 
`cancer`; except now we will remove many of the observations of malignant tumors, simulating
what the data would look like if the cancer was rare. We will do this by
picking only 3 observations from the malignant group, and keeping all
of the benign observations. We choose these 3 observations using the `.head()`
method, which takes the number of rows to select from the top (`n`).
The new imbalanced data is shown in {numref}`fig:05-unbalanced`.

```{code-cell} ipython3
:tags: [remove-cell]

# To better illustrate the problem, let's revisit the scaled breast cancer data, 
# `cancer`; except now we will remove many of the observations of malignant tumors, simulating
# what the data would look like if the cancer was rare. We will do this by
# picking only 3 observations from the malignant group, and keeping all
# of the benign observations. We choose these 3 observations using the `slice_head`
# function, which takes two arguments: a data frame-like object,
# and the number of rows to select from the top (`n`).
# The new imbalanced data is shown in Figure \@ref(fig:05-unbalanced).
```

```{code-cell} ipython3
cancer = pd.read_csv("data/wdbc.csv")
rare_cancer = pd.concat(
    (cancer.query("Class == 'B'"), cancer.query("Class == 'M'").head(3))
)
colors = ["#86bfef", "#efb13f"]
rare_cancer["Class"] = rare_cancer["Class"].apply(
    lambda x: "Malignant" if (x == "M") else "Benign"
)
rare_plot = (
    alt.Chart(
        rare_cancer
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)
rare_plot
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-unbalanced
:figclass: caption-hack

Imbalanced data.
```

+++

Suppose we now decided to use $K = 7$ in $K$-nearest neighbor classification.
With only 3 observations of malignant tumors, the classifier 
will *always predict that the tumor is benign, no matter what its concavity and perimeter
are!* This is because in a majority vote of 7 observations, at most 3 will be
malignant (we only have 3 total malignant observations), so at least 4 must be
benign, and the benign vote will always win. For example, {numref}`fig:05-upsample`
shows what happens for a new tumor observation that is quite close to three observations
in the training data that were tagged as malignant.

```{code-cell} ipython3
:tags: [remove-cell]

attrs = ["Perimeter", "Concavity"]
new_point = [2, 2]
new_point_df = pd.DataFrame(
    {"Class": ["Unknwon"], "Perimeter": new_point[0], "Concavity": new_point[1]}
)
rare_cancer["Class"] = rare_cancer["Class"].apply(class_dscp)
rare_cancer_with_new_df = pd.concat((rare_cancer, new_point_df), ignore_index=True)
my_distances = euclidean_distances(rare_cancer_with_new_df.loc[:, attrs])[
    len(rare_cancer)
][:-1]

# First layer: scatter plot, with unknwon point labeled as red "unknown" diamond
rare_plot = (
    alt.Chart(
        rare_cancer_with_new_df
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            scale=alt.Scale(range=["#86bfef", "#efb13f", "red"]),
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30))
    )
)

# Find the 7 NNs
min_7_idx = np.argpartition(my_distances, 7)[:7]

# For loop: each iteration adds a line segment of corresponding color
for i in range(7):
    clr = "#86bfef"
    if rare_cancer.iloc[min_7_idx[i], :]["Class"] == "Malignant":
        clr = "#efb13f"
    neighbor = pd.concat(
        (
            rare_cancer.iloc[min_7_idx[i], :][attrs],
            new_point_df[attrs].T,
        ),
        axis=1,
    ).T
    rare_plot = rare_plot + (
        alt.Chart(neighbor)
        .mark_line(opacity=0.3)
        .encode(x="Perimeter", y="Concavity", color=alt.value(clr))
    )

glue('fig:05-upsample', rare_plot)
```

:::{glue:figure} fig:05-upsample
:name: fig:05-upsample

Imbalanced data with 7 nearest neighbors to a new observation highlighted.
:::

+++

{numref}`fig:05-upsample-2` shows what happens if we set the background color of 
each area of the plot to the predictions the $K$-nearest neighbor 
classifier would make. We can see that the decision is 
always "benign," corresponding to the blue color.

```{code-cell} ipython3
:tags: [remove-cell]

knn_spec = KNeighborsClassifier(n_neighbors=7)
knn_spec.fit(X=rare_cancer.loc[:, ["Perimeter", "Concavity"]], y=rare_cancer["Class"])

# create a prediction pt grid
per_grid = np.linspace(
    rare_cancer["Perimeter"].min(), rare_cancer["Perimeter"].max(), 100
)
con_grid = np.linspace(
    rare_cancer["Concavity"].min(), rare_cancer["Concavity"].max(), 100
)
pcgrid = np.array(np.meshgrid(per_grid, con_grid)).reshape(2, -1).T
pcgrid = pd.DataFrame(pcgrid, columns=["Perimeter", "Concavity"])
knnPredGrid = knn_spec.predict(pcgrid)
prediction_table = pcgrid.copy()
prediction_table["Class"] = knnPredGrid

# create the scatter plot
rare_plot = (
    alt.Chart(
        rare_cancer,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
prediction_plot = (
    alt.Chart(
        prediction_table,
        title="Imbalanced data",
    )
    .mark_point(opacity=0.02, filled=True, size=200)
    .encode(
        x=alt.X(
            "Perimeter",
            title="Perimeter (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Perimeter"].min(), rare_cancer["Perimeter"].max())
            ),
        ),
        y=alt.Y(
            "Concavity",
            title="Concavity (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Concavity"].min(), rare_cancer["Concavity"].max())
            ),
        ),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)
rare_plot + prediction_plot
glue("fig:05-upsample-2", (rare_plot + prediction_plot))
```

:::{glue:figure} fig:05-upsample-2
:name: fig:05-upsample-2

Imbalanced data with background color indicating the decision of the classifier and the points represent the labeled data.
:::

+++

Despite the simplicity of the problem, solving it in a statistically sound manner is actually
fairly nuanced, and a careful treatment would require a lot more detail and mathematics than we will cover in this textbook.
For the present purposes, it will suffice to rebalance the data by *oversampling* the rare class. \index{oversampling}
In other words, we will replicate rare observations multiple times in our data set to give them more
voting power in the $K$-nearest neighbor algorithm. In order to do this, we will need an oversampling
step with the `resample` function from the `sklearn` Python package. \index{recipe!step\_upsample}
We show below how to do this, and also
use the `.groupby()` and `.count()` methods to see that our classes are now balanced:

```{code-cell} ipython3
:tags: [remove-cell]

# Despite the simplicity of the problem, solving it in a statistically sound manner is actually
# fairly nuanced, and a careful treatment would require a lot more detail and mathematics than we will cover in this textbook.
# For the present purposes, it will suffice to rebalance the data by *oversampling* the rare class. \index{oversampling}
# In other words, we will replicate rare observations multiple times in our data set to give them more
# voting power in the $K$-nearest neighbor algorithm. In order to do this, we will add an oversampling
# step to the earlier `uc_recipe` recipe with the `step_upsample` function from the `themis` R package. \index{recipe!step\_upsample}
# We show below how to do this, and also
# use the `group_by` and `summarize` functions to see that our classes are now balanced:
```

```{code-cell} ipython3
rare_cancer['Class'].value_counts()
```

```{code-cell} ipython3
from sklearn.utils import resample

malignant_cancer = rare_cancer[rare_cancer["Class"] == "Malignant"]
benign_cancer = rare_cancer[rare_cancer["Class"] == "Benign"]
malignant_cancer_upsample = resample(
    malignant_cancer, n_samples=len(benign_cancer), random_state=100
)
upsampled_cancer = pd.concat((malignant_cancer_upsample, benign_cancer))
upsampled_cancer.groupby(by='Class')['Class'].count()
```

Now suppose we train our $K$-nearest neighbor classifier with $K=7$ on this *balanced* data. 
{numref}`fig:05-upsample-plot` shows what happens now when we set the background color 
of each area of our scatter plot to the decision the $K$-nearest neighbor 
classifier would make. We can see that the decision is more reasonable; when the points are close
to those labeled malignant, the classifier predicts a malignant tumor, and vice versa when they are 
closer to the benign tumor observations.

```{code-cell} ipython3
:tags: [remove-cell]

knn_spec = KNeighborsClassifier(n_neighbors=7)
knn_spec.fit(
    X=upsampled_cancer.loc[:, ["Perimeter", "Concavity"]], y=upsampled_cancer["Class"]
)

# create a prediction pt grid
knnPredGrid = knn_spec.predict(pcgrid)
prediction_table = pcgrid
prediction_table["Class"] = knnPredGrid

# create the scatter plot
rare_plot = (
    alt.Chart(rare_cancer)
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X(
            "Perimeter",
            title="Perimeter (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Perimeter"].min(), rare_cancer["Perimeter"].max())
            ),
        ),
        y=alt.Y(
            "Concavity",
            title="Concavity (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Concavity"].min(), rare_cancer["Concavity"].max())
            ),
        ),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
upsampled_plot = (
    alt.Chart(prediction_table)
    .mark_point(opacity=0.02, filled=True, size=200)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)
rare_plot + upsampled_plot
glue("fig:05-upsample-plot", (rare_plot + upsampled_plot))
```

:::{glue:figure} fig:05-upsample-plot
:name: fig:05-upsample-plot

Upsampled data with background color indicating the decision of the classifier.
:::

+++

(08:puttingittogetherworkflow)=
## Putting it together in a `pipeline`

The `scikit-learn` package collection also provides the `pipeline`, a way to chain together multiple data analysis steps without a lot of otherwise necessary code for intermediate steps.
To illustrate the whole pipeline, let's start from scratch with the `unscaled_wdbc.csv` data.
First we will load the data, create a model, and specify a preprocessor for how the data should be preprocessed:

```{code-cell} ipython3
:tags: [remove-cell]

# The `tidymodels` package collection also provides the `workflow`, 
# a way to chain\index{tidymodels!workflow}\index{workflow|see{tidymodels}} 
# together multiple data analysis steps without a lot of otherwise necessary code 
# for intermediate steps.
# To illustrate the whole pipeline, let's start from scratch with the `unscaled_wdbc.csv` data.
# First we will load the data, create a model, 
# and specify a recipe for how the data should be preprocessed:
```

```{code-cell} ipython3
# load the unscaled cancer data
unscaled_cancer = pd.read_csv("data/unscaled_wdbc.csv")

# create the KNN model
knn_spec = KNeighborsClassifier(n_neighbors=7)

# create the centering / scaling preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
```

You will also notice that we did not call `.fit()` on the preprocessor; this is unnecessary when it is
placed in a `Pipeline`.

We will now place these steps in a `Pipeline` using the `make_pipeline` function, 
and finally we will call `.fit()` to run the whole `Pipeline` on the `unscaled_cancer` data.

```{code-cell} ipython3
:tags: [remove-cell]

# Note that each of these steps is exactly the same as earlier, except for one major difference:
# we did not use the `select` function to extract the relevant variables from the data frame,
# and instead simply specified the relevant variables to use via the 
# formula `Class ~ Area + Smoothness` (instead of `Class ~ .`) in the recipe.
# You will also notice that we did not call `prep()` on the recipe; this is unnecessary when it is
# placed in a workflow.

# We will now place these steps in a `workflow` using the `add_recipe` and `add_model` functions, \index{tidymodels!add\_recipe}\index{tidymodels!add\_model}
# and finally we will use the `fit` function to run the whole workflow on the `unscaled_cancer` data.
# Note another difference from earlier here: we do not include a formula in the `fit` function. This \index{tidymodels!fit}
# is again because we included the formula in the recipe, so there is no need to respecify it:
```

```{code-cell} ipython3
knn_fit = make_pipeline(preprocessor, knn_spec).fit(
    X=unscaled_cancer.loc[:, ["Area", "Smoothness"]], y=unscaled_cancer["Class"]
)

knn_fit
```

As before, the fit object lists the function that trains the model. But now the fit object also includes information about
the overall workflow, including the standardizing preprocessing step.
In other words, when we use the `predict` function with the `knn_fit` object to make a prediction for a new
observation, it will first apply the same preprocessing steps to the new observation. 
As an example, we will predict the class label of two new observations:
one with `Area = 500` and `Smoothness = 0.075`, and one with `Area = 1500` and `Smoothness = 0.1`.

```{code-cell} ipython3
:tags: [remove-cell]

# As before, the fit object lists the function that trains the model as well as the "best" settings
# for the number of neighbors and weight function (for now, these are just the values we chose
#  manually when we created `knn_spec` above). But now the fit object also includes information about
# the overall workflow, including the centering and scaling preprocessing steps.
# In other words, when we use the `predict` function with the `knn_fit` object to make a prediction for a new
# observation, it will first apply the same recipe steps to the new observation. 
# As an example, we will predict the class label of two new observations:
# one with `Area = 500` and `Smoothness = 0.075`, and one with `Area = 1500` and `Smoothness = 0.1`.
```

```{code-cell} ipython3
new_observation = pd.DataFrame({"Area": [500, 1500], "Smoothness": [0.075, 0.1]})
prediction = knn_fit.predict(new_observation)
prediction
```

The classifier predicts that the first observation is benign ("B"), while the second is
malignant ("M"). {numref}`fig:05-workflow-plot-show` visualizes the predictions that this 
trained $K$-nearest neighbor model will make on a large range of new observations.
Although you have seen colored prediction map visualizations like this a few times now,
we have not included the code to generate them, as it is a little bit complicated.
For the interested reader who wants a learning challenge, we now include it below. 
The basic idea is to create a grid of synthetic new observations using the `numpy.meshgrid` function, 
predict the label of each, and visualize the predictions with a colored scatter having a very high transparency 
(low `opacity` value) and large point radius. See if you can figure out what each line is doing!

> **Note:** Understanding this code is not required for the remainder of the
> textbook. It is included for those readers who would like to use similar
> visualizations in their own data analyses.

```{code-cell} ipython3
# create the grid of area/smoothness vals, and arrange in a data frame
are_grid = np.linspace(
    unscaled_cancer["Area"].min(), unscaled_cancer["Area"].max(), 100
)
smo_grid = np.linspace(
    unscaled_cancer["Smoothness"].min(), unscaled_cancer["Smoothness"].max(), 100
)
asgrid = np.array(np.meshgrid(are_grid, smo_grid)).reshape(2, -1).T
asgrid = pd.DataFrame(asgrid, columns=["Area", "Smoothness"])

# use the fit workflow to make predictions at the grid points
knnPredGrid = knn_fit.predict(asgrid)

# bind the predictions as a new column with the grid points
prediction_table = asgrid.copy()
prediction_table["Class"] = knnPredGrid

# plot:
# 1. the colored scatter of the original data
unscaled_plot = (
    alt.Chart(
        unscaled_cancer,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X(
            "Area",
            title="Area (standardized)",
            scale=alt.Scale(
                domain=(unscaled_cancer["Area"].min(), unscaled_cancer["Area"].max())
            ),
        ),
        y=alt.Y(
            "Smoothness",
            title="Smoothness (standardized)",
            scale=alt.Scale(
                domain=(
                    unscaled_cancer["Smoothness"].min(),
                    unscaled_cancer["Smoothness"].max(),
                )
            ),
        ),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)

# 2. the faded colored scatter for the grid points
prediction_plot = (
    alt.Chart(prediction_table)
    .mark_point(opacity=0.02, filled=True, size=200)
    .encode(
        x=alt.X("Area"),
        y=alt.Y("Smoothness"),
        color=alt.Color("Class", scale=alt.Scale(range=colors), title="Diagnosis"),
    )
)

unscaled_plot + prediction_plot
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-workflow-plot-show
:figclass: caption-hack

Scatter plot of smoothness versus area where background color indicates the decision of the classifier.
```

+++

## Exercises

Practice exercises for the material covered in this chapter 
can be found in the accompanying 
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the "Classification I: training and predicting" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in Chapter \@ref(move-to-your-own-machine). This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

```{bibliography}
```
