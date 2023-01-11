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

```{code-cell} ipython3
:tags: [remove-cell]
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from myst_nb import glue
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import HTML
```

(classification)=
# Classification I: training & predicting

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
- Use `StandardScaler` and `make_column_transformer` to preprocess data to be centered and scaled.
- Use `resample` to preprocess data to be balanced.
- Combine preprocessing and model training using `make_pipeline`.

+++

## The classification problem

```{index} predictive question, classification, class, categorical variable
```

```{index} see: feature ; predictor
```

In many situations, we want to make predictions based on the current situation
as well as past experiences. For instance, a doctor may want to diagnose a
patient as either diseased or healthy based on their symptoms and the doctor's
past experience with patients; an email provider might want to tag a given
email as "spam" or "not spam" based on the email's text and past email text data; 
or a credit card company may want to predict whether a purchase is fraudulent based
on the current purchase item, amount, and location as well as past purchases.
These tasks are all examples of **classification**, i.e., predicting a
categorical class (sometimes called a *label*) for an observation given its
other variables (sometimes called *features*).

```{index} training set
```

Generally, a classifier assigns an observation without a known class (e.g., a new patient) 
to a class (e.g., diseased or healthy) on the basis of how similar it is to other observations
for which we do know the class (e.g., previous patients with known diseases and
symptoms). These observations with known classes that we use as a basis for
prediction are called a **training set**; this name comes from the fact that
we use these data to train, or teach, our classifier. Once taught, we can use
the classifier to make predictions on new data for which we do not know the class.

```{index} K-nearest neighbors, classification; binary
```

There are many possible methods that we could use to predict
a categorical class/label for an observation. In this book, we will
focus on the widely used **$K$-nearest neighbors** algorithm {cite:p}`knnfix,knncover`.
In your future studies, you might encounter decision trees, support vector machines (SVMs),
logistic regression, neural networks, and more; see the additional resources
section at the end of the next chapter for where to begin learning more about
these other methods. It is also worth mentioning that there are many
variations on the basic classification problem. For example, 
we focus on the setting of **binary classification** where only two
classes are involved (e.g., a diagnosis of either healthy or diseased), but you may 
also run into multiclass classification problems with more than two
categories (e.g., a diagnosis of healthy, bronchitis, pneumonia, or a common cold).

## Exploring a data set

```{index} breast cancer, question; classification
```

In this chapter and the next, we will study a data set of 
[digitized breast cancer image features](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29),
created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian {cite:p}`streetbreastcancer`.
Each row in the data set represents an
image of a tumor sample, including the diagnosis (benign or malignant) and
several other measurements (nucleus texture, perimeter, area, and more).
Diagnosis for each image was conducted by physicians. 

As with all data analyses, we first need to formulate a precise question that
we want to answer. Here, the question is *predictive*: can 
we use the tumor
image measurements available to us to predict whether a future tumor image
(with unknown diagnosis) shows a benign or malignant tumor? Answering this
question is important because traditional, non-data-driven methods for tumor
diagnosis are quite subjective and dependent upon how skilled and experienced
the diagnosing physician is. Furthermore, benign tumors are not normally
dangerous; the cells stay in the same place, and the tumor stops growing before
it gets very large. By contrast, in malignant tumors, the cells invade the
surrounding tissue and spread into nearby organs, where they can cause serious
damage {cite:p}`stanfordhealthcare`.
Thus, it is important to quickly and accurately diagnose the tumor type to
guide patient treatment.

+++

### Loading the cancer data

Our first step is to load, wrangle, and explore the data using visualizations
in order to better understand the data we are working with. We start by
loading the `pandas` and `altair` packages needed for our analysis.

```{code-cell} ipython3
import pandas as pd
import altair as alt
```

In this case, the file containing the breast cancer data set is a `.csv`
file with headers. We'll use the `read_csv` function with no additional
arguments, and then inspect its contents:

```{index} read function; read\_csv
```

```{code-cell} ipython3
:tags: ["output_scroll"]
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

```{index} info
```

Below we use the `info` method to preview the data frame. This method can 
make it easier to inspect the data when we have a lot of columns:
it prints only the column names down the page (instead of across),
as well as their data types and the number of non-missing entries.

```{code-cell} ipython3
cancer.info()
```

```{index} unique
```

From the summary of the data above, we can see that `Class` is of type `object`.
We can use the `unique` method on the `Class` column to see all unique values
present in that column. We see that there are two diagnoses:
benign, represented by `'B'`, and malignant, represented by `'M'`.

```{code-cell} ipython3
cancer['Class'].unique()
```

We will also improve the readability of our analysis
by renaming `'M'` to `'Malignant'` and `'B'` to `'Benign'` using the `replace`
method. The `replace` method takes one argument: a dictionary that maps
previous values to desired new values. 
Furthermore, since we will be working with `Class` as a categorical statistical variable,
it is a good idea to convert it to the `category` type using the `astype` method 
on the `cancer` data frame. We will verify the result using the `info` 
and `unique` methods again.

```{index} replace
```

```{code-cell} ipython3
cancer['Class'] = cancer['Class'].replace({
    'M' : 'Malignant',
    'B' : 'Benign'
}).astype('category')
cancer.info()
```

```{code-cell} ipython3
cancer['Class'].unique()
```

### Exploring the cancer data

```{index} groupby, count
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("benign_count", cancer['Class'].value_counts()['Benign'])
glue("benign_pct", int(np.round(100*cancer['Class'].value_counts(normalize=True)['Benign'])))
glue("malignant_count", cancer['Class'].value_counts()['Malignant'])
glue("malignant_pct", int(np.round(100*cancer['Class'].value_counts(normalize=True)['Malignant'])))
```

Before we start doing any modeling, let's explore our data set. Below we use
the `groupby` and `count` methods to find the number and percentage 
of benign and malignant tumor observations in our data set. When paired with
`groupby`, `count` counts the number of observations for each value of the `Class`
variable. Then we calculate the percentage in each group by dividing by the total
number of observations and multiplying by 100. We have 
{glue:}`benign_count` ({glue:}`benign_pct`\%) benign and
{glue:}`malignant_count` ({glue:}`malignant_pct`\%) malignant 
tumor observations.

```{code-cell} ipython3
explore_cancer = pd.DataFrame()
explore_cancer['count'] = cancer.groupby('Class')['ID'].count()
explore_cancer['percentage'] = 100 * explore_cancer['count']/len(cancer)
explore_cancer
```

```{index} value_counts
```

The `pandas` package also has a more convenient specialized `value_counts` method for 
counting the number of occurrences of each value in a column. If we pass no arguments
to the method, it outputs a series containing the number of occurences
of each value. If we instead pass the argument `normalize=True`, it instead prints the fraction
of occurrences of each value.

```{code-cell} ipython3
cancer['Class'].value_counts()
```

```{code-cell} ipython3
cancer['Class'].value_counts(normalize=True)
```

```{index} visualization; scatter
```

Next, let's draw a colored scatter plot to visualize the relationship between the
perimeter and concavity variables. Recall that `altair's` default palette
is colorblind-friendly, so we can stick with that here.

```{code-cell} ipython3
perim_concav = (
    alt.Chart(cancer)
    .mark_circle()
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
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
it may be possible to make accurate predictions of the `Class` variable (i.e., a diagnosis) for
tumor images with unknown diagnoses.

+++

## Classification with $K$-nearest neighbors

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [2, 4]
glue("new_point_1_0", new_point[0])
glue("new_point_1_1", new_point[1])
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
```

```{index} K-nearest neighbors; classification
```

In order to actually make predictions for new observations in practice, we
will need a classification algorithm. 
In this book, we will use the $K$-nearest neighbors classification algorithm.
To predict the label of a new observation (here, classify it as either benign
or malignant), the $K$-nearest neighbors classifier generally finds the $K$
"nearest" or "most similar" observations in our training set, and then uses
their diagnoses to make a prediction for the new observation's diagnosis. $K$ 
is a number that we must choose in advance; for now, we will assume that someone has chosen
$K$ for us. We will cover how to choose $K$ ourselves in the next chapter. 

To illustrate the concept of $K$-nearest neighbors classification, we 
will walk through an example.  Suppose we have a
new observation, with standardized perimeter 
of {glue:}`new_point_1_0` and standardized concavity 
of {glue:}`new_point_1_1`, whose 
diagnosis "Class" is unknown. This new observation is 
depicted by the red, diamond point in {numref}`fig:05-knn-2`.

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point = (
    alt.Chart(
        perim_concav_with_new_point_df,
    )
    .mark_circle()
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(100), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value('black'), alt.value(None)),
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

{numref}`fig:05-knn-3` shows that the nearest point to this new observation is
**malignant** and located at the coordinates ({glue:}`1-neighbor_per`,
{glue:}`1-neighbor_con`). The idea here is that if a point is close to another
in the scatter plot, then the perimeter and concavity values are similar, 
and so we may expect that they would have the same diagnosis.

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

Scatter plot of concavity versus perimeter. The new observation is represented
as a red diamond with a line to the one nearest neighbor, which has a malignant
label.
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
glue("new_point_2_0", new_point[0])
glue("new_point_2_1", new_point[1])
```

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point2 = (
    alt.Chart(
        perim_concav_with_new_point_df2,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
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

Suppose we have another new observation with standardized perimeter
{glue:}`new_point_2_0` and concavity of {glue:}`new_point_2_1`. Looking at the
scatter plot in {numref}`fig:05-knn-4`, how would you classify this red,
diamond observation? The nearest neighbor to this new point is a
**benign** observation at ({glue:}`2-neighbor_per`, {glue:}`2-neighbor_con`).
Does this seem like the right prediction to make for this observation? Probably 
not, if you consider the other nearby points.

+++

:::{glue:figure} fig:05-knn-4
:name: fig:05-knn-4

Scatter plot of concavity versus perimeter. The new observation is represented
as a red diamond with a line to the one nearest neighbor, which has a benign
label.
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

```{index} distance; K-nearest neighbors, straight line; distance
```

We decide which points are the $K$ "nearest" to our new observation using the
*straight-line distance* (we will often just refer to this as *distance*).
Suppose we have two observations $a$ and $b$, each having two predictor
variables, $x$ and $y$.  Denote $a_x$ and $a_y$ to be the values of variables
$x$ and $y$ for observation $a$; $b_x$ and $b_y$ have similar definitions for
observation $b$.  Then the straight-line distance between observation $a$ and
$b$ on the x-y plane can be computed using the following formula: 

$$\mathrm{Distance} = \sqrt{(a_x -b_x)^2 + (a_y - b_y)^2}$$

+++

To find the $K$ nearest neighbors to our new observation, we compute the distance
from that new observation to each observation in our training data, and select the $K$ observations corresponding to the
$K$ *smallest* distance values. For example, suppose we want to use $K=5$ neighbors to classify a new 
observation with perimeter of {glue:}`3-new_point_0` and 
concavity of {glue:}`3-new_point_1`, shown as a red diamond in {numref}`fig:05-multiknn-1`. Let's calculate the distances
between our new point and each of the observations in the training set to find
the $K=5$ neighbors that are nearest to our new point. 
You will see in the code below, we compute the straight-line
distance using the formula above: we square the differences between the two observations' perimeter 
and concavity coordinates, add the squared differences, and then take the square root.
In order to find the $K=5$ nearest neighbors, we will use the `nsmallest` function from `pandas`.

> **Note:** Recall that in the {ref}`intro` chapter, we used `sort_values` followed by `head` to obtain
> the ten rows with the *largest* values of a variable. We could have instead used the `nlargest` function
> from `pandas` for this purpose. The `nsmallest` and `nlargest` functions achieve the same goal 
> as `sort_values` followed by `head`, but are slightly more efficient because they are specialized for this purpose.
> In general, it is good to use more specialized functions when they are available!

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

```{index} pandas.DataFrame; assign
```

```{code-cell} ipython3
new_obs_Perimeter = 0
new_obs_Concavity = 3.5
(
    cancer
   [["Perimeter", "Concavity", "Class"]]
   .assign(dist_from_new = (
       (cancer["Perimeter"] - new_obs_Perimeter) ** 2
     + (cancer["Concavity"] - new_obs_Concavity) ** 2
   )**(1/2))
   .nsmallest(5, "dist_from_new")
)
```

In {numref}`tab:05-multiknn-mathtable` we show in mathematical detail how
we computed the `dist_from_new` variable (the
distance to the new observation) for each of the 5 nearest neighbors in the
training data.

```{table} Evaluating the distances from the new observation to each of its 5 nearest neighbors
:name: tab:05-multiknn-mathtable
| Perimeter | Concavity | Distance            | Class |
|-----------|-----------|----------------------------------------|-------|
| 0.24      | 2.65      | $\sqrt{(0-0.24)^2+(3.5-2.65)^2}=0.88$| Benign     |
| 0.75      | 2.87      | $\sqrt{(0-0.75)^2+(3.5-2.87)^2}=0.98$| Malignant     |
| 0.62      | 2.54      | $\sqrt{(0-0.62)^2+(3.5-2.54)^2}=1.14$| Malignant     |
| 0.42      | 2.31      | $\sqrt{(0-0.42)^2+(3.5-2.31)^2}=1.26$| Malignant     |
| -1.16     | 4.04      | $\sqrt{(0-(-1.16))^2+(3.5-4.04)^2}=1.28$| Benign     |
```

+++

The result of this computation shows that 3 of the 5 nearest neighbors to our new observation are
malignant; since this is the majority, we classify our new observation as malignant. 
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

```{index} distance; more than two variables
```

The distance formula becomes 

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
(
    cancer
    [["Perimeter", "Concavity", "Symmetry", "Class"]]
    .assign(dist_from_new = (
        (cancer["Perimeter"] - new_obs_Perimeter) ** 2
        + (cancer["Concavity"] - new_obs_Concavity) ** 2
        + (cancer["Symmetry"] - new_obs_Symmetry) ** 2
    )**(1/2))
    .nsmallest(5, "dist_from_new")
)
```

Based on $K=5$ nearest neighbors with these three predictors we would classify 
the new observation as malignant since 4 out of 5 of the nearest neighbors are malignant class. 
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
colors = {"Malignant": "#ff7f0e", "Benign": "#1f77b4", "Unknown": "red"}

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
```

```{code-cell} ipython3
:tags: [remove-input]

display(HTML("img/classification1/fig05-more.html"))
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-more
:figclass: caption-hack

3D scatter plot of the standardized symmetry, concavity, and perimeter
variables. Note that in general we recommend against using 3D visualizations;
here we show the data in 3D only to illustrate what higher dimensions and
nearest neighbors look like, for learning purposes.
```

+++

### Summary of $K$-nearest neighbors algorithm

In order to classify a new observation using a $K$-nearest neighbor classifier, we have to do the following:

1. Compute the distance between the new observation and each observation in the training set.
2. Find the $K$ rows corresponding to the $K$ smallest distances.
3. Classify the new observation based on a majority vote of the neighbor classes.

+++

## $K$-nearest neighbors with `scikit-learn`

```{index} scikit-learn
```

Coding the $K$-nearest neighbors algorithm in Python ourselves can get complicated,
especially if we want to handle multiple classes, more than two variables,
or predict the class for multiple new observations. Thankfully, in Python,
the $K$-nearest neighbors algorithm is 
implemented in [the `scikit-learn` Python package](https://scikit-learn.org/stable/index.html) {cite:p}`sklearn_api` along with 
many [other models](https://scikit-learn.org/stable/user_guide.html) that you will encounter in this and future chapters of the book. Using the functions 
in the `scikit-learn` package (named `sklearn` in Python) will help keep our code simple, readable and accurate; the 
less we have to code ourselves, the fewer mistakes we will likely make. 
Before getting started with $K$-nearest neighbors, we need to tell the `sklearn` package 
that we prefer using `pandas` data frames over regular arrays via the `set_config` function. 
```{code-cell} ipython3
from sklearn import set_config
set_config(transform_output="pandas")
```

We can now get started with $K$-nearest neighbors. The first step is to
 import the `KNeighborsClassifier` from the `sklearn.neighbors` module.

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier
```

Let's walk through how to use `KNeighborsClassifier` to perform $K$-nearest neighbors classification. 
We will use the `cancer` data set from above, with
perimeter and concavity as predictors and $K = 5$ neighbors to build our classifier. Then
we will use the classifier to predict the diagnosis label for a new observation with
perimeter 0, concavity 3.5, and an unknown diagnosis label. Let's pick out our two desired
predictor variables and class label and store them with the name `cancer_train`:

```{code-cell} ipython3
cancer_train = cancer[['Class', 'Perimeter', 'Concavity']]
cancer_train
```

```{index} scikit-learn; model object, scikit-learn; KNeighborsClassifier
```

Next, we create a *model object* for $K$-nearest neighbors classification
by creating a `KNeighborsClassifier` instance, specifying that we want to use $K = 5$ neighbors;
we will discuss how to choose $K$ in the next chapter.

> **Note:** You can specify the `weights` argument in order to control
> how neighbors vote when classifying a new observation. The default is `"uniform"`, where
> each of the $K$ nearest neighbors gets exactly 1 vote as described above. Other choices, 
> which weigh each neighbor's vote differently, can be found on 
> [the `scikit-learn` website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier).


```{code-cell} ipython3
knn = KNeighborsClassifier(n_neighbors=5)
knn
```

```{index} scikit-learn; X & y
```

In order to fit the model on the breast cancer data, we need to call `fit` on
the model object. The `X` argument is used to specify the data for the predictor
variables, while the `y` argument is used to specify the data for the response variable.
So below, we set `X=cancer_train[["Perimeter", "Concavity"]]` and
`y=cancer_train['Class']` to specify that `Class` is the target
variable (the one we want to predict), and both `Perimeter` and `Concavity` are
to be used as the predictors. Note that the `fit` function might look like it does not
do much from the outside, but it is actually doing all the heavy lifting to train
the K-nearest neighbors model, and modifies the `knn` model object.

```{code-cell} ipython3
knn.fit(X=cancer_train[["Perimeter", "Concavity"]], y=cancer_train["Class"]);
```

```{index} scikit-learn; predict
```

After using the `fit` function, we can make a prediction on a new observation
by calling `predict` on the classifier object, passing the new observation
itself. As above, when we ran the $K$-nearest neighbors classification
algorithm manually, the `knn` model object classifies the new observation as
"Malignant". Note that the `predict` function outputs an `array` with the
model's prediction; you can actually make multiple predictions at the same
time using the `predict` function, which is why the output is stored as an `array`.

```{code-cell} ipython3
new_obs = pd.DataFrame({'Perimeter': [0], 'Concavity': [3.5]})
knn.predict(new_obs)
```

Is this predicted malignant label the true class for this observation? 
Well, we don't know because we do not have this
observation's diagnosis&mdash; that is what we were trying to predict! The 
classifier's prediction is not necessarily correct, but in the next chapter, we will 
learn ways to quantify how accurate we think our predictions are.

+++

## Data preprocessing with `scikit-learn`

### Centering and scaling

```{index} scaling
```

When using $K$-nearest neighbor classification, the *scale* of each variable
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

```{index} centering
```

In many other predictive models, the *center* of each variable (e.g., its mean)
matters as well. For example, if we had a data set with a temperature variable
measured in degrees Kelvin, and the same data set with temperature measured in
degrees Celsius, the two variables would differ by a constant shift of 273
(even though they contain exactly the same information). Likewise, in our
hypothetical job classification example, we would likely see that the center of
the salary variable is in the tens of thousands, while the center of the years
of education variable is in the single digits. Although this doesn't affect the
$K$-nearest neighbor classification algorithm, this large shift can change the
outcome of using many other predictive models.

```{index} standardization; K-nearest neighbors
```

To scale and center our data, we need to find
our variables' *mean* (the average, which quantifies the "central" value of a 
set of numbers) and *standard deviation* (a number quantifying how spread out values are). 
For each observed value of the variable, we subtract the mean (i.e., center the variable) 
and divide by the standard deviation (i.e., scale the variable). When we do this, the data 
is said to be *standardized*, and all variables in a data set will have a mean of 0 
and a standard deviation of 1. To illustrate the effect that standardization can have on the $K$-nearest
neighbor algorithm, we will read in the original, unstandardized Wisconsin breast
cancer data set; we have been using a standardized version of the data set up
until now. We will apply the same initial wrangling steps as we did earlier,
and to keep things simple we will just use the `Area`, `Smoothness`, and `Class`
variables:

```{code-cell} ipython3
unscaled_cancer = pd.read_csv("data/unscaled_wdbc.csv")[['Class', 'Area', 'Smoothness']]
unscaled_cancer['Class'] = unscaled_cancer['Class'].replace({
   'M' : 'Malignant',
   'B' : 'Benign'
}).astype('category')
unscaled_cancer
unscaled_cancer
```

Looking at the unscaled and uncentered data above, you can see that the differences
between the values for area measurements are much larger than those for
smoothness. Will this affect predictions? In order to find out, we will create a scatter plot of these two
predictors (colored by diagnosis) for both the unstandardized data we just
loaded, and the standardized version of that same data. But first, we need to
standardize the `unscaled_cancer` data set with `scikit-learn`.

```{index} pipeline, scikit-learn; make_column_transformer
```

```{index} double: scikit-learn; pipeline
```

The `scikit-learn` framework provides a collection of *preprocessors* used to manipulate
data in the [`preprocessing` module](https://scikit-learn.org/stable/modules/preprocessing.html).
Here we will use the `StandardScaler` transformer to standardize the predictor variables in
the `unscaled_cancer` data. In order to tell the `StandardScaler` which variables to standardize,
we wrap it in a 
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) object
using the [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer) function. 
`ColumnTransformer` objects also enable the use of multiple preprocessors at
once, which is especially handy when you want to apply different preprocessing to each of the predictor variables. 
The primary argument of the `make_column_transformer` function is a sequence of
pairs of (1) a preprocessor, and (2) the columns to which you want to apply that preprocessor.
In the present case, we just have the one `StandardScaler` preprocessor to apply to the `Area` and `Smoothness` columns.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
preprocessor
```

```{index} scikit-learn; ColumnTransformer, scikit-learn; StandardScaler, scikit-learn; fit_transform
```

```{index} ColumnTransformer; StandardScaler
```

```{index} scikit-learn; fit, scikit-learn; transform
```

You can see that the preprocessor includes a single standardization step
that is applied to the `Area` and `Smoothness` columns. 
Note that here we specified which columns to apply the preprocessing step to 
by individual names; this approach can become quite difficult, e.g., when we have many
predictor variables. Rather than writing out the column names individually,
we can instead use the 
[`make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector) function. For
example, if we wanted to standardize all *numerical* predictors,
we would use `make_column_selector` and specify the `dtype_include` argument to be `'number'`. 
This creates a preprocessor equivalent to the one we created previously.

```{code-cell} ipython3
from sklearn.compose import make_column_selector

preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include='number')),
)
preprocessor
```

```{index} see: fit, transform, fit_transform; scikit-learn
```

We are now ready to standardize the numerical predictor columns in the `unscaled_cancer` data frame.
This happens in two steps. We first use the `fit` function to compute the values necessary to apply
the standardization (the mean and standard deviation of each variable), passing the `unscaled_cancer` data as an argument.
Then we use the `transform` function to actually apply the standardization.  
It may seem a bit unnecessary to use two steps---`fit` *and* `transform`---to standardize the data.
However, we do this in two steps so that we can specify a different data set in the `transform` step if we want. 
This enables us to compute the quantities needed to standardize using one data set, and then 
apply that standardization to another data set.

```{code-cell} ipython3
preprocessor.fit(unscaled_cancer)
scaled_cancer = preprocessor.transform(unscaled_cancer)
scaled_cancer
```
```{code-cell} ipython3
:tags: [remove-cell]
glue('scaled-cancer-column-0', scaled_cancer.columns[0])
glue('scaled-cancer-column-1', scaled_cancer.columns[1])
```
It looks like our `Smoothness` and `Area` variables have been standardized. Woohoo!
But there are two important things to notice about the new `scaled_cancer` data frame. First, it only keeps
the columns from the input to `transform` (here, `unscaled_cancer`) that had a preprocessing step applied
to them. The default behavior of the `ColumnTransformer` that we build using `make_column_transformer` 
is to *drop* the remaining columns. This default behavior works well with the rest of `sklearn` (as we will see below
in the {ref}`08:puttingittogetherworkflow` section), but for visualizing the result of preprocessing it can be useful to keep the other columns
in our original data frame, such as the `Class` variable here.
To keep other columns, we need to set the `remainder` argument to `'passthrough'` in the `make_column_transformer` function.
 Furthermore, you can see that the new column names---{glue:}`scaled-cancer-column-0`
and {glue:}`scaled-cancer-column-1`---include the name
of the preprocessing step separated by underscores. This default behavior is useful in `sklearn` because we sometimes want to apply
multiple different preprocessing steps to the same columns; but again, for visualization it can be useful to preserve
the original column names. To keep original column names, we need to set the `verbose_feature_names_out` argument to `False`.

> **Note:** Only specify the `remainder` and `verbose_feature_names_out` arguments when you want to examine the result
> of your preprocessing step. In most cases, you should leave these arguments at their default values.

```{code-cell} ipython3
preprocessor_keep_all = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include='number')),
    remainder='passthrough',
    verbose_feature_names_out=False
)
preprocessor_keep_all.fit(unscaled_cancer)
scaled_cancer_all = preprocessor_keep_all.transform(unscaled_cancer)
scaled_cancer_all
```

You may wonder why we are doing so much work just to center and
scale our variables. Can't we just manually scale and center the `Area` and
`Smoothness` variables ourselves before building our $K$-nearest neighbor model? Well,
technically *yes*; but doing so is error-prone.  In particular, we might
accidentally forget to apply the same centering / scaling when making
predictions, or accidentally apply a *different* centering / scaling than what
we used while training. Proper use of a `ColumnTransformer` helps keep our code simple,
readable, and error-free. Furthermore, note that using `fit` and `transform` on
the preprocessor is required only when you want to inspect the result of the
preprocessing steps
yourself. You will see further on in the
{ref}`08:puttingittogetherworkflow` section that `scikit-learn` provides tools to
automatically streamline the preprocesser and the model so that you can call `fit` 
and `transform` on the `Pipeline` as necessary without additional coding effort.

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
new_obs = pd.DataFrame({"Class": ["Unknown"], "Area": 400, "Smoothness": 0.135})
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
new_obs_scaled = pd.DataFrame({"Class": ["Unknown"], "Area": -0.72, "Smoothness": 2.8})
scaled_cancer_all["Class"] = scaled_cancer_all["Class"].apply(class_dscp)
area_smoothness_new_df_scaled = pd.concat(
    (scaled_cancer_all, new_obs_scaled), ignore_index=True
)
my_distances_scaled = euclidean_distances(area_smoothness_new_df_scaled.loc[:, attrs])[
    len(scaled_cancer_all)
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
        scaled_cancer_all.loc[min_3_idx_scaled[0], attrs],
        new_obs_scaled[attrs].T,
    ),
    axis=1,
).T
neighbor2_scaled = pd.concat(
    (
        scaled_cancer_all.loc[min_3_idx_scaled[1], attrs],
        new_obs_scaled[attrs].T,
    ),
    axis=1,
).T
neighbor3_scaled = pd.concat(
    (
        scaled_cancer_all.loc[min_3_idx_scaled[2], attrs],
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

```{index} balance, imbalance
```

Another potential issue in a data set for a classifier is *class imbalance*, 
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
We use the [`concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) 
function from `pandas` to glue the two resulting filtered
data frames back together by passing them together in a sequence.
The new imbalanced data is shown in {numref}`fig:05-unbalanced`, 
and we print the counts of the classes using the `value_counts` function.

```{code-cell} ipython3
rare_cancer = pd.concat((
    cancer[cancer["Class"] == 'Benign'],
    cancer[cancer["Class"] == 'Malignant'].head(3)
))

rare_plot = (
    alt.Chart(rare_cancer)
    .mark_circle()
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)
rare_plot
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-unbalanced
:figclass: caption-hack

Imbalanced data.
```

```{code-cell} ipython3
rare_cancer['Class'].value_counts()
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
    {"Class": ["Unknown"], "Perimeter": new_point[0], "Concavity": new_point[1]}
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
    clr = "#1f77b4"
    if rare_cancer.iloc[min_7_idx[i], :]["Class"] == "Malignant":
        clr = "#ff7f0e"
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

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X=rare_cancer.loc[:, ["Perimeter", "Concavity"]], y=rare_cancer["Class"])

# create a prediction pt grid
per_grid = np.linspace(
    rare_cancer["Perimeter"].min(), rare_cancer["Perimeter"].max(), 50
)
con_grid = np.linspace(
    rare_cancer["Concavity"].min(), rare_cancer["Concavity"].max(), 50
)
pcgrid = np.array(np.meshgrid(per_grid, con_grid)).reshape(2, -1).T
pcgrid = pd.DataFrame(pcgrid, columns=["Perimeter", "Concavity"])
pcgrid

knnPredGrid = knn.predict(pcgrid)
prediction_table = pcgrid.copy()
prediction_table["Class"] = knnPredGrid
prediction_table

# create the scatter plot
rare_plot = (
    alt.Chart(
        rare_cancer,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
prediction_plot = (
    alt.Chart(
        prediction_table,
        title="Imbalanced data",
    )
    .mark_point(opacity=0.05, filled=True, size=300)
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
        color=alt.Color("Class", title="Diagnosis"),
    )
)
#rare_plot + prediction_plot
glue("fig:05-upsample-2", (rare_plot + prediction_plot))
```

:::{glue:figure} fig:05-upsample-2
:name: fig:05-upsample-2

Imbalanced data with background color indicating the decision of the classifier and the points represent the labeled data.
:::

+++

```{index} oversampling, scikit-learn; resample
```

Despite the simplicity of the problem, solving it in a statistically sound manner is actually
fairly nuanced, and a careful treatment would require a lot more detail and mathematics than we will cover in this textbook.
For the present purposes, it will suffice to rebalance the data by *oversampling* the rare class. 
In other words, we will replicate rare observations multiple times in our data set to give them more
voting power in the $K$-nearest neighbor algorithm. In order to do this, we will 
first separate the classes out into their own data frames by filtering.
Then, we will
use the [`resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) function 
from the `sklearn` package to increase the number of `Malignant` observations to be the same as the number 
of `Benign` observations. We set the `n_samples` argument to be the number of `Malignant` observations we want. 
Finally, we use the `value_counts` method to see that our classes are now balanced.
Note that `resample` picks which data to replicate *randomly*; we will learn more about properly handling randomness
in data analysis in the {ref}`classification2` chapter.

```{code-cell} ipython3
:tags: [remove-cell]
# hidden seed call to make the below resample reproducible
# we haven't taught students about seeds / prngs yet, so 
# for now just hide this.
np.random.seed(1)
```

```{code-cell} ipython3
from sklearn.utils import resample

malignant_cancer = rare_cancer[rare_cancer["Class"] == "Malignant"]
benign_cancer = rare_cancer[rare_cancer["Class"] == "Benign"]
malignant_cancer_upsample = resample(
    malignant_cancer, n_samples=len(benign_cancer)
)
upsampled_cancer = pd.concat((malignant_cancer_upsample, benign_cancer))
upsampled_cancer['Class'].value_counts()
```

Now suppose we train our $K$-nearest neighbor classifier with $K=7$ on this *balanced* data. 
{numref}`fig:05-upsample-plot` shows what happens now when we set the background color 
of each area of our scatter plot to the decision the $K$-nearest neighbor 
classifier would make. We can see that the decision is more reasonable; when the points are close
to those labeled malignant, the classifier predicts a malignant tumor, and vice versa when they are 
closer to the benign tumor observations.

```{code-cell} ipython3
:tags: [remove-cell]

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(
    X=upsampled_cancer.loc[:, ["Perimeter", "Concavity"]], y=upsampled_cancer["Class"]
)

# create a prediction pt grid
knnPredGrid = knn.predict(pcgrid)
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
        color=alt.Color("Class", title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
upsampled_plot = (
    alt.Chart(prediction_table)
    .mark_point(opacity=0.05, filled=True, size=300)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)
#rare_plot + upsampled_plot
glue("fig:05-upsample-plot", (rare_plot + upsampled_plot))
```

:::{glue:figure} fig:05-upsample-plot
:name: fig:05-upsample-plot

Upsampled data with background color indicating the decision of the classifier.
:::

+++

(08:puttingittogetherworkflow)=
## Putting it together in a `Pipeline`

```{index} scikit-learn; pipeline
```

The `scikit-learn` package collection also provides the [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?highlight=pipeline#sklearn.pipeline.Pipeline), 
a  way to chain together multiple data analysis steps without a lot of otherwise necessary code for intermediate steps.
To illustrate the whole workflow, let's start from scratch with the `unscaled_wdbc.csv` data.
First we will load the data, create a model, and specify a preprocessor for the data.

```{code-cell} ipython3
# load the unscaled cancer data, make Class readable
unscaled_cancer = pd.read_csv("data/unscaled_wdbc.csv")
unscaled_cancer['Class'] = unscaled_cancer['Class'].replace({
   'M' : 'Malignant',
   'B' : 'Benign'
}).astype('category')
unscaled_cancer


# create the KNN model
knn = KNeighborsClassifier(n_neighbors=7)

# create the centering / scaling preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
```

```{index} scikit-learn; make_pipeline, scikit-learn; fit
```

Next we place these steps in a `Pipeline` using 
the [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) function.
The `make_pipeline` function takes a list of steps to apply in your data analysis; in this
case, we just have the `preprocessor` and `knn` steps.
Finally, we call `fit` on the pipeline.
Notice that we do not need to separately call `fit` and `transform` on the `preprocessor`; the
pipeline handles doing this properly for us.
Also notice that when we call `fit` on the pipeline, we can pass
the whole `unscaled_cancer` data frame to the `X` argument, since the preprocessing
step drops all the variables except the two we listed: `Area` and `Smoothness`.
For the `y` response variable argument, we pass the `unscaled_cancer["Class"]` series as before.

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline

knn_fit = make_pipeline(preprocessor, knn).fit(
    X=unscaled_cancer, 
    y=unscaled_cancer["Class"]
)

knn_fit
```

As before, the fit object lists the function that trains the model. But now the fit object also includes information about
the overall workflow, including the standardization preprocessing step.
In other words, when we use the `predict` function with the `knn_fit` object to make a prediction for a new
observation, it will first apply the same preprocessing steps to the new observation. 
As an example, we will predict the class label of two new observations:
one with `Area = 500` and `Smoothness = 0.075`, and one with `Area = 1500` and `Smoothness = 0.1`.

```{code-cell} ipython3
new_observation = pd.DataFrame({"Area": [500, 1500], "Smoothness": [0.075, 0.1]})
prediction = knn_fit.predict(new_observation)
prediction
```

The classifier predicts that the first observation is benign, while the second is
malignant. {numref}`fig:05-workflow-plot` visualizes the predictions that this 
trained $K$-nearest neighbor model will make on a large range of new observations.
Although you have seen colored prediction map visualizations like this a few times now,
we have not included the code to generate them, as it is a little bit complicated.
For the interested reader who wants a learning challenge, we now include it below. 
The basic idea is to create a grid of synthetic new observations using the `meshgrid` function from `numpy`, 
predict the label of each, and visualize the predictions with a colored scatter having a very high transparency 
(low `opacity` value) and large point radius. See if you can figure out what each line is doing!

> **Note:** Understanding this code is not required for the remainder of the
> textbook. It is included for those readers who would like to use similar
> visualizations in their own data analyses.

```{code-cell} ipython3
:tags: [remove-output]
import numpy as np

# create the grid of area/smoothness vals, and arrange in a data frame
are_grid = np.linspace(
    unscaled_cancer["Area"].min(), unscaled_cancer["Area"].max(), 50
)
smo_grid = np.linspace(
    unscaled_cancer["Smoothness"].min(), unscaled_cancer["Smoothness"].max(), 50
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
        color=alt.Color("Class", title="Diagnosis"),
    )
)

# 2. the faded colored scatter for the grid points
prediction_plot = (
    alt.Chart(prediction_table)
    .mark_point(opacity=0.05, filled=True, size=300)
    .encode(
        x=alt.X("Area"),
        y=alt.Y("Smoothness"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)
unscaled_plot + prediction_plot
```

```{code-cell} ipython3
:tags: [remove-input]
glue("fig:05-workflow-plot", (unscaled_plot + prediction_plot))
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-workflow-plot
:figclass: caption-hack

Scatter plot of smoothness versus area where background color indicates the decision of the classifier.
```

+++

## Exercises

Practice exercises for the material covered in this chapter 
can be found in the accompanying 
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-python-worksheets#readme)
in the "Classification I: training and predicting" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in the {ref}`move-to-your-own-machine` chapter. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++
## References

```{bibliography}
:filter: docname in docnames
```
