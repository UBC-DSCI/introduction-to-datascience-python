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

(clustering)=
# Clustering

```{code-cell} ipython3
:tags: [remove-cell]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import altair as alt
import os


# Use PNG images in the PDF version of the books to make sure that they render
if os.environ['BOOK_BUILD_TYPE'] == 'PDF':
    alt.renderers.enable('png', scale_factor=0.7, ppi=300)
else:
    # Reduce chart sizes and allow to plot up to 100k graphical objects (not the same as rows in the data frame)
    alt.data_transformers.enable('vegafusion')
```

## Overview

As part of exploratory data analysis, it is often helpful to see if there are
meaningful subgroups (or *clusters*) in the data.
This grouping can be used for many purposes,
such as generating new questions or improving predictive analyses.
This chapter provides an introduction to clustering
using the K-means algorithm,
including techniques to choose the number of clusters.

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

* Describe a case where clustering is appropriate,
and what insight it might extract from the data.
* Explain the K-means clustering algorithm.
* Interpret the output of a K-means analysis.
* Differentiate between clustering and classification.
* Identify when it is necessary to scale variables before clustering and do this using Python
* Perform k-means clustering in Python using `scikit-learn`
* Use the elbow method to choose the number of clusters for K-means.
* Visualize the output of k-means clustering in Python using a coloured scatter plot
* Describe advantages, limitations and assumptions of the kmeans clustering algorithm.

## Clustering

```{index} clustering
```

Clustering is a data analysis task
involving separating a data set into subgroups of related data.
For example, we might use clustering to separate a
data set of documents into groups that correspond to topics, a data set of
human genetic information into groups that correspond to ancestral
subpopulations, or a data set of online customers into groups that correspond
to purchasing behaviors.  Once the data are separated, we can, for example,
use the subgroups to generate new questions about the data and follow up with a
predictive modeling exercise. In this course, clustering will be used only for
exploratory analysis, i.e., uncovering patterns in the data.

```{index} classification, regression, supervised, unsupervised
```

Note that clustering is a fundamentally different kind of task than
classification or regression.  In particular, both classification and
regression are *supervised tasks* where there is a *response variable* (a
category label or value), and we have examples of past data with labels/values
that help us predict those of future data.  By contrast, clustering is an
*unsupervised task*, as we are trying to understand and examine the structure
of data without any response variable labels or values to help us.  This
approach has both advantages and disadvantages.  Clustering requires no
additional annotation or input on the data.  For example, while it would be
nearly impossible to annotate all the articles on Wikipedia with human-made
topic labels, we can cluster the articles without this information to find
groupings corresponding to topics automatically.  However, given that there is
no response variable, it is not as easy to evaluate the "quality" of a
clustering.  With classification, we can use a test data set to assess
prediction performance. In clustering, there is not a single good choice for
evaluation. In this book, we will use visualization to ascertain the quality of
a clustering, and leave rigorous evaluation for more advanced courses.

Given that there is no response variable, it is not as easy to evaluate
the "quality" of a clustering.  With classification, we can use a test data set
to assess prediction performance. In clustering, there is not a single good
choice for evaluation. In this book, we will use visualization to ascertain the
quality of a clustering, and leave rigorous evaluation for more advanced
courses.

```{index} K-means
```

As in the case of classification,
there are many possible methods that we could use to cluster our observations
to look for subgroups.
In this book, we will focus on the widely used K-means algorithm {cite:p}`kmeans`.
In your future studies, you might encounter hierarchical clustering,
principal component analysis, multidimensional scaling, and more;
see the additional resources section at the end of this chapter
for where to begin learning more about these other methods.

```{index} semisupervised
```

```{note}
There are also so-called *semisupervised* tasks,
where only some of the data come with response variable labels/values,
but the vast majority don't.
The goal is to try to uncover underlying structure in the data
that allows one to guess the missing labels.
This sort of task is beneficial, for example,
when one has an unlabeled data set that is too large to manually label,
but one is willing to provide a few informative example labels as a "seed"
to guess the labels for all the data.
```

**An illustrative example**

```{index} Palmer penguins
```

Here we will present an illustrative example using a data set from
[the `palmerpenguins` R package](https://allisonhorst.github.io/palmerpenguins/) {cite:p}`palmerpenguins`. This
data set was collected by Dr. Kristen Gorman and
the Palmer Station, Antarctica Long Term Ecological Research Site, and includes
measurements for adult penguins found near there {cite:p}`penguinpaper`. We have
modified the data set for use in this chapter. Here we will focus on using two
variables&mdash;penguin bill and flipper length, both in millimeters&mdash;to determine whether
there are distinct types of penguins in our data.
Understanding this might help us with species discovery and classification in a data-driven
way.

```{figure} img/gentoo.jpg
---
height: 400px
name: 09-penguins
---
Gentoo penguin.
```

To learn about K-means clustering
we will work with `penguin_data` in this chapter.
`penguin_data` is a subset of 18 observations of the original data,
which has already been standardized
(remember from Chapter {ref}`classification1`
that scaling is part of the standardization process).
We will discuss scaling for K-means in more detail later in this chapter.

Before we get started, we will set a random seed.
This will ensure that our analysis will be reproducible.
As we will learn in more detail later in the chapter,
setting the seed here is important
because the K-means clustering algorithm uses randomness
when choosing a starting position for each cluster.

```{index} seed; numpy.random.seed
```

```{code-cell} ipython3
import numpy as np

np.random.seed(149)
```

```{index} read function; read_csv
```

Now we can load and preview the data.

```{code-cell} ipython3
:tags: [remove-cell]

from myst_nb import glue
import pandas as pd

data = pd.read_csv(
    'data/penguins_toy.csv'
).replace(
    [2, 3],
    [0, 2]
)
```

```{code-cell} ipython3
import pandas as pd

penguin_data = pd.read_csv("data/penguins_standardized.csv")
penguin_data
```

Next, we can create a scatter plot using this data set
to see if we can detect subtypes or groups in our data set.

```{code-cell} ipython3
import altair as alt

scatter_plot = alt.Chart(penguin_data).mark_circle().encode(
    x=alt.X("flipper_length_standardized").title("Flipper Length (standardized)"),
    y=alt.Y("bill_length_standardized").title("Bill Length (standardized)")
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue('scatter_plot', scatter_plot, display=True)
```

:::{glue:figure} scatter_plot
:figwidth: 700px
:name: scatter_plot

Scatter plot of standardized bill length versus standardized flipper length.
:::

```{index} altair, altair; mark_circle
```

Based on the visualization
in {numref}`scatter_plot`,
we might suspect there are a few subtypes of penguins within our data set.
We can see roughly 3 groups of observations in {numref}`scatter_plot`,
including:

1. a small flipper and bill length group,
2. a small flipper length, but large bill length group, and
3. a large  flipper and bill length group.

```{index} K-means, elbow method
```

Data visualization is a great tool to give us a rough sense of such patterns
when we have a small number of variables.
But if we are to group data&mdash;and select the number of groups&mdash;as part of
a reproducible analysis, we need something a bit more automated.
Additionally, finding groups via visualization becomes more difficult
as we increase the number of variables we consider when clustering.
The way to rigorously separate the data into groups
is to use a clustering algorithm.
In this chapter, we will focus on the *K-means* algorithm,
a widely used and often very effective clustering method,
combined with the *elbow method*
for selecting the number of clusters.
This procedure will separate the data into groups;
{numref}`colored_scatter_plot` shows these groups
denoted by colored scatter points.

```{code-cell} ipython3
:tags: [remove-cell]

colored_scatter_plot = alt.Chart(data).mark_circle().encode(
    x=alt.X("flipper_length_standardized", title="Flipper Length (standardized)"),
    y=alt.Y("bill_length_standardized", title="Bill Length (standardized)"),
    color=alt.Color('cluster:N')
)

glue('colored_scatter_plot', colored_scatter_plot, display=True)
```

:::{glue:figure} colored_scatter_plot
:figwidth: 700px
:name: colored_scatter_plot

Scatter plot of standardized bill length versus standardized flipper length with colored groups.
:::


What are the labels for these groups? Unfortunately, we don't have any. K-means,
like almost all clustering algorithms, just outputs meaningless "cluster labels"
that are typically whole numbers: 0, 1, 2, 3, etc. But in a simple case like this,
where we can easily visualize the clusters on a scatter plot, we can give
human-made labels to the groups using their positions on
the plot:

- small flipper length and small bill length (<font color="#f59518">orange cluster</font>),
- small flipper length and large bill length (<font color="#4c78a8">blue cluster</font>).
- and large flipper length and large bill  length (<font color="#e45756">red cluster</font>).

Once we have made these determinations, we can use them to inform our species
classifications or ask further questions about our data. For example, we might
be interested in understanding the relationship between flipper length and bill
length, and that relationship may differ depending on the type of penguin we
have.

## K-means

### Measuring cluster quality

```{code-cell} ipython3
:tags: [remove-cell]

clus = data[data["cluster"] == 2][["bill_length_standardized", "flipper_length_standardized"]]
```

```{index} see: within-cluster sum-of-squared-distances; WSSD
```

```{index} WSSD
```

The K-means algorithm is a procedure that groups data into K clusters.
It starts with an initial clustering of the data, and then iteratively
improves it by making adjustments to the assignment of data
to clusters until it cannot improve any further. But how do we measure
the "quality" of a clustering, and what does it mean to improve it?
In K-means clustering, we measure the quality of a cluster by its
*within-cluster sum-of-squared-distances* (WSSD), also called *intertia*. Computing this involves two steps.
First, we find the cluster centers by computing the mean of each variable
over data points in the cluster. For example, suppose we have a
cluster containing four observations, and we are using two variables, $x$ and $y$, to cluster the data.
Then we would compute the coordinates, $\mu_x$ and $\mu_y$, of the cluster center via



$\mu_x = \frac{1}{4}(x_1+x_2+x_3+x_4) \quad \mu_y = \frac{1}{4}(y_1+y_2+y_3+y_4)$

```{code-cell} ipython3
:tags: [remove-cell]

clus_rows = clus.shape[0]

mean_flipper_len_std = round(np.mean(clus['flipper_length_standardized']),2)
mean_bill_len_std = round(np.mean(clus['bill_length_standardized']),2)

glue("clus_rows_glue", clus_rows)
glue("mean_flipper_len_std_glue",mean_flipper_len_std)
glue("mean_bill_len_std_glue", mean_bill_len_std)
```

In the first cluster from the example, there are {glue:}`clus_rows_glue` data points. These are shown with their cluster center
(flipper_length_standardized = {glue:}`mean_flipper_len_std_glue` and bill_length_standardized = {glue:}`mean_bill_len_std_glue`) highlighted
in {numref}`toy-example-clus1-center-1`


```{figure} img/toy-example-clus1-center-1.png
---
height: 400px
name: toy-example-clus1-center-1
---
Cluster 1 from the penguin_data data set example. Observations are in blue, with the cluster center highlighted in red.
```

```{index} distance; K-means
```

The second step in computing the WSSD is to add up the squared distance
between each point in the cluster
and the cluster center.
We use the straight-line / Euclidean distance formula
that we learned about in Chapter {ref}`classification1`.
In the {glue:}`clus_rows_glue`-observation cluster example above,
we would compute the WSSD $S^2$ via


$S^2 = \left((x_1 - \mu_x)^2 + (y_1 - \mu_y)^2\right) + \left((x_2 - \mu_x)^2 + (y_2 - \mu_y)^2\right) + \left((x_3 - \mu_x)^2 + (y_3 - \mu_y)^2\right)  +  \left((x_4 - \mu_x)^2 + (y_4 - \mu_y)^2\right)$

These distances are denoted by lines in {numref}`toy-example-clus1-dists-1` for the first cluster of the penguin data example.

```{figure} img/toy-example-clus1-dists-1.png
---
height: 400px
name: toy-example-clus1-dists-1
---
Cluster 1 from the penguin_data data set example. Observations are in blue, with the cluster center highlighted in red. The distances from the observations to the cluster center are represented as black lines.
```

The larger the value of $S^2$, the more spread out the cluster is, since large $S^2$ means that points are far from the cluster center.
Note, however, that "large" is relative to *both* the scale of the variables for clustering *and* the number of points in the cluster. A cluster where points are very close to the center might still have a large $S^2$ if there are many data points in the cluster.

After we have calculated the WSSD for all the clusters,
we sum them together to get the *total WSSD*.
For our example,
this means adding up all the squared distances for the 18 observations.
These distances are denoted by black lines in
{numref}`toy-example-all-clus-dists-1`

```{figure} img/toy-example-all-clus-dists-1.png
---
height: 400px
name: toy-example-all-clus-dists-1
---
All clusters from the penguin_data data set example. Observations are in orange, blue, and yellow with the cluster center highlighted in red. The distances from the observations to each of the respective cluster centers are represented as black lines.


```

### The clustering algorithm

```{index} K-means; algorithm
```

We begin the K-means algorithm by picking K,
and randomly assigning a roughly equal number of observations
to each of the K clusters.
An example random initialization is shown in {numref}`toy-kmeans-init-1`



```{figure} img/toy-kmeans-init-1.png
---
height: 400px
name: toy-kmeans-init-1
---
Random initialization of labels.

```

```{index} WSSD; total
```

Then K-means consists of two major steps that attempt to minimize the
sum of WSSDs over all the clusters, i.e., the *total WSSD*:

1. **Center update:** Compute the center of each cluster.
2. **Label update:** Reassign each data point to the cluster with the nearest center.

These two steps are repeated until the cluster assignments no longer change.
We show what the first four iterations of K-means would look like in
{numref}`toy-kmeans-iter-1`
There each row corresponds to an iteration,
where the left column depicts the center update,
and the right column depicts the reassignment of data to clusters.

```{figure} img/toy-kmeans-iter-1.png
---
height: 400px
name: toy-kmeans-iter-1
---
First four iterations of K-means clustering on the penguin_data example data set. Each pair of plots corresponds to an iteration. Within the pair, the first plot depicts the center update, and the second plot depicts the reassignment of data to clusters. Cluster centers are indicated by larger points that are outlined in black.

```

Note that at this point, we can terminate the algorithm since none of the assignments changed
in the fourth iteration; both the centers and labels will remain the same from this point onward.

```{index} K-means; termination
```

```{note}
Is K-means *guaranteed* to stop at some point, or could it iterate forever? As it turns out,
thankfully, the answer is that K-means is guaranteed to stop after *some* number of iterations. For the interested reader, the
logic for this has three steps: (1) both the label update and the center update decrease total WSSD in each iteration,
(2) the total WSSD is always greater than or equal to 0, and (3) there are only a finite number of possible
ways to assign the data to clusters. So at some point, the total WSSD must stop decreasing, which means none of the assignments
are changing, and the algorithm terminates.
```

What kind of data is suitable for K-means clustering?
In the simplest version of K-means clustering that we have presented here,
the straight-line distance is used to measure the
distance between observations and cluster centers.
This means that only quantitative data should be used with this algorithm.
There are variants on the K-means algorithm,
as well as other clustering algorithms entirely,
that use other distance metrics
to allow for non-quantitative data to be clustered.
These, however, are beyond the scope of this book.

### Random restarts

```{index} K-means; init argument
```

Unlike the classification and regression models we studied in previous chapters, K-means can get "stuck" in a bad solution.
For example, {numref}`toy-kmeans-bad-init-1` illustrates an unlucky random initialization by K-means.

```{figure} img/toy-kmeans-bad-init-1.png
---
height: 400px
name: toy-kmeans-bad-init-1
---
Random initialization of labels.
```

{numref}`toy-kmeans-bad-iter-1` shows what the iterations of K-means would look like with the unlucky random initialization shown in {numref}`toy-kmeans-bad-init-1`



```{figure} img/toy-kmeans-bad-iter-1.png
---
height: 700px
name: toy-kmeans-bad-iter-1
---
First five iterations of K-means clustering on the penguin_data example data set with a poor random initialization. Each pair of plots corresponds to an iteration. Within the pair, the first plot depicts the center update, and the second plot depicts the reassignment of data to clusters. Cluster centers are indicated by larger points that are outlined in black.
```



This looks like a relatively bad clustering of the data, but K-means cannot improve it.
To solve this problem when clustering data using K-means, we should randomly re-initialize the labels a few times, run K-means for each initialization,
and pick the clustering that has the lowest final total WSSD.

### Choosing K

In order to cluster data using K-means,
we also have to pick the number of clusters, K.
But unlike in classification, we have no response variable
and cannot perform cross-validation with some measure of model prediction error.
Further, if K is chosen too small, then multiple clusters get grouped together;
if K is too large, then clusters get subdivided.
In both cases, we will potentially miss interesting structure in the data.
{numref}`toy-kmeans-vary-k-1` illustrates the impact of K
on K-means clustering of our penguin flipper and bill length data
by showing the different clusterings for K's ranging from 1 to 9.

```{figure} img/toy-kmeans-vary-k-1.png
---
height: 700px
name: toy-kmeans-vary-k-1
---
Clustering of the penguin data for K clusters ranging from 1 to 9. Cluster centers are indicated by larger points that are outlined in black.
```

```{index} elbow method
```

If we set K less than 3, then the clustering merges separate groups of data; this causes a large
total WSSD, since the cluster center (denoted by an "x") is not close to any of the data in the cluster. On
the other hand, if we set K greater than 3, the clustering subdivides subgroups of data; this does indeed still
decrease the total WSSD, but by only a *diminishing amount*. If we plot the total WSSD versus the number of
clusters, we see that the decrease in total WSSD levels off (or forms an "elbow shape") when we reach roughly
the right number of clusters ({numref}`toy-kmeans-elbow-1`)).

```{figure} img/toy-kmeans-elbow-1.png
---
height: 400px
name: toy-kmeans-elbow-1
---
Total WSSD for K clusters ranging from 1 to 9.
```

## K-means in Python

```{index} K-means; kmeans function, scikit-learn; KMeans
```

We can perform K-means in Python using a workflow similar to those
in the earlier classification and regression chapters. We will begin
by reading the original (i.e., unstandardized) subset of 18 observations
from the penguins dataset.

```{code-cell} ipython3
:tags: [remove-cell]

unstandardized_data = pd.read_csv("data/penguins_toy.csv", usecols=["bill_length_mm", "flipper_length_mm"])
unstandardized_data.to_csv("data/penguins.csv", index=False)
```

```{code-cell} ipython3
penguins = pd.read_csv("data/penguins.csv")
penguins
```


Recall that K-means clustering uses straight-line distance to decide which points are similar to
each other. Therefore, the *scale* of each of the variables in the data
will influence which cluster data points end up being assigned.
Variables with a large scale will have a much larger
effect on deciding cluster assignment than variables with a small scale.
To address this problem, we typically standardize our data before clustering,
which ensures that each variable has a mean of 0 and standard deviation of 1.
The `StandardScaler` function in `scikit-learn` can be used to do this.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn import set_config

# Output dataframes instead of arrays
set_config(transform_output="pandas")

preprocessor = make_column_transformer(
    (StandardScaler(), ['bill_length_mm', 'flipper_length_mm']),
    verbose_feature_names_out=False,
)
preprocessor
```

To indicate that we are performing K-means clustering, we will create a `KMeans`
model object. It takes at
least one argument: the number of clusters `n_clusters`, which we set to 3.

```{code-cell} ipython3
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans
```

To actually run the K-means clustering, we combine the preprocessor and model object
in a `Pipeline`, and use the `fit` function. Note that the K-means
algorithm uses a random initialization of assignments, but since we set 
the random seed in the beginning of this chapter, the clustering will be reproducible.

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline

penguin_clust = make_pipeline(preprocessor, kmeans).fit(penguins)
penguin_clust
```

```{index} K-means; inertia_, K-means; cluster_centers_, K-means; labels_, K-means; predict
```

The fit `KMeans` object&mdash;which is the second item in the 
pipeline, and can be accessed as `penguin_clust[1]`&mdash;has a lot of information
that can be used to visualize the clusters, pick K, and evaluate the total WSSD.
Let's start by visualizing the clusters as a colored scatter plot! In 
order to do that, we first need to augment our 
original `penguins` data frame with the cluster assignments. 
We can access these using the `labels_` attribute of the clustering object 
("labels" is a common alternative term to "assignments" in clustering), and 
add them to the data frame using `assign`.

```{code-cell} ipython3
clustered_data = penguins.assign(cluster = penguin_clust[1].labels_)
clustered_data
```

Let's start by visualizing the clustering
as a colored scatter plot. To do that,
we will add a new column and store assign the above predictions to that. The final
data frame will contain the data and the cluster assignments for
each point:

Now that we have the cluster assignments included in the `clustered_data` data frame, we can 
visualize them as shown in {numref}`cluster_plot`.
Note that we are plotting the *un-standardized* data here; if we for some reason wanted to 
visualize the *standardized* data, we would need to use the `fit` and `transform` functions
on the `StandardScaler` preprocessor directly to obtain that first.

```{code-cell} ipython3
cluster_plot=alt.Chart(clustered_data).mark_circle().encode(
    x=alt.X("flipper_length_mm").title("Flipper Length").scale(zero=False),
    y=alt.Y("bill_length_mm").title("Bill Length").scale(zero=False),
    color=alt.Color("cluster:N").title("Cluster"),
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue('cluster_plot', cluster_plot, display=True)
```

:::{glue:figure} cluster_plot
:figwidth: 700px
:name: cluster_plot

The data colored by the cluster assignments returned by K-means.
:::

```{index} WSSD; total, K-means; inertia_
```

```{index} see: WSSD; K-means inertia
```

As mentioned above,
we also need to select K
by finding where the "elbow" occurs in the plot of total WSSD versus the number of clusters.
The total WSSD is stored in the `.inertia_` attribute
of the clustering object ("inertia" is the term `scikit-learn` uses to denote WSSD).

```{code-cell} ipython3
penguin_clust[1].inertia_
```

To calculate the total WSSD for a variety of Ks, we will
create a data frame that contains different values of `k`
and the WSSD of running KMeans with each values of k.
To create this dataframe,
we will use what is called a "list comprehension" in Python,
where we repeat an operation multiple times
and return a list with the result.
Here is an examples of a list comprehension that stores the numbers 0-2 in a list:

```{code-cell} ipython3
[n for n in range(3)]
```

We can change the variable `n` to be called whatever we prefer
and we can also perform any operation we want as part of the list comprehension.
For example,
we could square all the numbers from 1-4 and store them in a list:

```{code-cell} ipython3
[number ** 2 for number in range(1, 5)]
```

Next, we will use this approach to compute the WSSD for the K-values 1 through 9.
For each value of K,
we create a new KMeans model
and wrap it in a `scikit-learn` pipeline
with the preprocessor we created earlier.
We store the WSSD values in a list that we will use to create a dataframe 
of both the K-values and their corresponding WSSDs.

```{note}
We are creating the variable `ks` to store the range of possible k-values,
so that we only need to change this range in one place
if we decide to change which values of k we want to explore.
Otherwise it would be easy to forget to update it
in either the list comprehension or in the data frame assignment.
If you are using a value multiple times,
it is always the safest to assign it to a variable name for reuse.
```

```{code-cell} ipython3
ks = range(1, 10)
wssds = [
    make_pipeline(
    	preprocessor, 
    	KMeans(n_clusters=k)  # Create a new KMeans model with `k` clusters
    ).fit(penguins)[1].inertia_
    for k in ks
]

penguin_clust_ks = pd.DataFrame({
    'k': ks,
    'wssd': wssds,
})

penguin_clust_ks
```

Now that we have `wssd` and `k` as columns in a data frame, we can make a line plot
({numref}`elbow_plot`) and search for the "elbow" to find which value of K to use.

```{code-cell} ipython3
elbow_plot = alt.Chart(penguin_clust_ks).mark_line(point=True).encode(
    x=alt.X("k").title("Number of clusters"),
    y=alt.Y("wssd").title("Total within-cluster sum of squares"),
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue('elbow_plot', elbow_plot, display=True)
```

:::{glue:figure} elbow_plot
:figwidth: 700px
:name: elbow_plot

A plot showing the total WSSD versus the number of clusters.
:::

```{index} K-means; init argument
```

It looks like three clusters is the right choice for this data,
since that is where the "elbow" of the line is the most distinct.
In the plot,
you can also see that the WSSD is always decreasing,
as we would expect when we add more clusters.
However,
it is possible to have an elbow plot
where the WSSD increases at one of the steps,
causing a small bump in the line.
This is because K-means can get "stuck" in a bad solution
due to an unlucky initialization of the initial centroid positions
as we mentioned earlier in the chapter.

```{note}
It is rare that the KMeans function from `scikit-learn`
gets stuck in a bad solution,
because the selection of the centroid starting points
is optimized to prevent this from happening.
If you still find yourself in a situation where you have a bump in the elbow plot,
you can increase the `n_init` parameter
to try more different starting points for the centroids.
The larger the value the better from an analysis perspective,
but there is a trade-off that doing many clusterings could take a long time.
```

## Exercises

Practice exercises for the material covered in this chapter
can be found in the accompanying
[worksheets repository](https://worksheets.python.datasciencebook.ca)
in the "Clustering" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in Chapter {ref}`move-to-your-own-machine`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

## Additional resources

- Chapter 10 of *An Introduction to Statistical
  Learning* {cite:p}`james2013introduction` provides a
  great next stop in the process of learning about clustering and unsupervised
  learning in general. In the realm of clustering specifically, it provides a
  great companion introduction to K-means, but also covers *hierarchical*
  clustering for when you expect there to be subgroups, and then subgroups within
  subgroups, etc., in your data. In the realm of more general unsupervised
  learning, it covers *principal components analysis (PCA)*, which is a very
  popular technique for reducing the number of predictors in a dataset.

## References

+++

```{bibliography}
:filter: docname in docnames
```
