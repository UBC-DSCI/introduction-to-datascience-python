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

# Clustering {#clustering}




```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

import altair as alt

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
Clustering \index{clustering} is a data analysis task 
involving separating a data set into subgroups of related data. 
For example, we might use clustering to separate a
data set of documents into groups that correspond to topics, a data set of
human genetic information into groups that correspond to ancestral
subpopulations, or a data set of online customers into groups that correspond
to purchasing behaviors.  Once the data are separated, we can, for example,
use the subgroups to generate new questions about the data and follow up with a
predictive modeling exercise. In this course, clustering will be used only for
exploratory analysis, i.e., uncovering patterns in the data.  

Note that clustering is a fundamentally different kind of task 
than classification or regression. 
In particular, both classification and regression are *supervised tasks* 
\index{classification}\index{regression}\index{supervised} 
where there is a *response variable* (a category label or value), 
and we have examples of past data with labels/values 
that help us predict those of future data. 
By contrast, clustering is an *unsupervised task*, 
\index{unsupervised} as we are trying to understand 
and examine the structure of data without any response variable labels 
or values to help us. 
This approach has both advantages and disadvantages. 
Clustering requires no additional annotation or input on the data. 
For example, it would be nearly impossible to annotate 
all the articles on Wikipedia with human-made topic labels. 
However, we can still cluster the articles without this information 
to find groupings corresponding to topics automatically. 

Given that there is no response variable, it is not as easy to evaluate
the "quality" of a clustering.  With classification, we can use a test data set
to assess prediction performance. In clustering, there is not a single good
choice for evaluation. In this book, we will use visualization to ascertain the
quality of a clustering, and leave rigorous evaluation for more advanced
courses.  

As in the case of classification, 
there are many possible methods that we could use to cluster our observations 
to look for subgroups. 
In this book, we will focus on the widely used K-means \index{K-means} algorithm [@kmeans]. 
In your future studies, you might encounter hierarchical clustering,
principal component analysis, multidimensional scaling, and more; 
see the additional resources section at the end of this chapter 
for where to begin learning more about these other methods.

\newpage

> **Note:** There are also so-called *semisupervised* tasks, \index{semisupervised} 
> where only some of the data come with response variable labels/values, 
> but the vast majority don't. 
> The goal is to try to uncover underlying structure in the data 
> that allows one to guess the missing labels. 
> This sort of task is beneficial, for example, 
> when one has an unlabeled data set that is too large to manually label, 
> but one is willing to provide a few informative example labels as a "seed" 
> to guess the labels for all the data.

**An illustrative example** 

Here we will present an illustrative example using a data set \index{Palmer penguins} from
[the `palmerpenguins` R package](https://allisonhorst.github.io/palmerpenguins/) [@palmerpenguins]. This 
data set was collected by Dr. Kristen Gorman and
the Palmer Station, Antarctica Long Term Ecological Research Site, and includes
measurements for adult penguins found near there [@penguinpaper]. We have
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
(remember from Chapter \@ref(classification) 
that scaling is part of the standardization process). 
We will discuss scaling for K-means in more detail later in this chapter. 
\index{mutate}\index{read function!read\_csv} 

Before we get started, we will set a random seed.
This will ensure that our analysis will be reproducible.
As we will learn in more detail later in the chapter, 
setting the seed here is important 
because the K-means clustering algorithm uses random numbers.



\index{seed!set.seed}



```{code-cell} ipython3

np.random.seed(1)
```


Now we can load and preview the data.

```{code-cell} ipython3
:tags: ["remove-cell"]
data = pd.read_csv('data/toy_penguins.csv')
data
```

```{code-cell} ipython3
penguin_data = pd.read_csv("data/penguins_standardized.csv")
penguin_data

```

Next, we can create a scatter plot using this data set 
to see if we can detect subtypes or groups in our data set.

\newpage



```{code-cell} ipython3
scatter_plot = alt.Chart(penguin_data, title="Scatter plot of standardized bill length versus standardized flipper length.").mark_circle(color='black').encode(
    x = alt.X("flipper_length_standardized", title="Flipper Length (standardized)"),
    y = alt.Y("bill_length_standardized", title="Bill Length (standardized)")).configure_axis(
    labelFontSize=12,
    titleFontSize=12
).configure_title(fontSize=12)
 
```
```{code-cell} ipython3
:tags: ["remove-cell"]
glue('scatter_plot', scatter_plot, display=True)
```

:::{glue:figure} scatter_plot 
:figwidth: 700px 
:name: scatter_plot

Scatter plot of standardized bill length versus standardized flipper length.
:::




Based \index{ggplot}\index{ggplot!geom\_point} on the visualization 
in Figure \@ref(fig:10-toy-example-plot), 
we might suspect there are a few subtypes of penguins within our data set.
We can see roughly 3 groups of observations in Figure \@ref(fig:10-toy-example-plot),
including:

1. a small flipper and bill length group,
2. a small flipper length, but large bill length group, and
3. a large  flipper and bill length group.

Data visualization is a great tool to give us a rough sense of such patterns
when we have a small number of variables. 
But if we are to group data&mdash;and select the number of groups&mdash;as part of 
a reproducible analysis, we need something a bit more automated.
Additionally, finding groups via visualization becomes more difficult 
as we increase the number of variables we consider when clustering.
The way to rigorously separate the data into groups 
is to use a clustering algorithm.
In this chapter, we will focus on the *K-means* algorithm, 
\index{K-means} a widely used and often very effective clustering method, 
combined with the *elbow method* \index{elbow method} 
for selecting the number of clusters. 
This procedure will separate the data into groups;
Figure \@ref(fig:10-toy-example-clustering) shows these groups
denoted by colored scatter points.




```{code-cell} ipython3
colors = ["orange", "blue", "brown"]

colored_scatter_plot = alt.Chart(data, title="Scatter plot of standardized bill length versus standardized flipper length with colored groups.").mark_circle().encode(
    x = alt.X("flipper_length_standardized", title="Flipper Length (standardized)"),
    y = alt.Y("bill_length_standardized", title="Bill Length (standardized)"),
    color = alt.Color('cluster:N', scale=alt.Scale(range=colors))).configure_axis(
    labelFontSize=12,
    titleFontSize=12
).configure_title(fontSize=12)
 
 
```


```{code-cell} ipython3
:tags: ["remove-cell"]
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

- small flipper length and small bill length (<font color="#D55E00">orange cluster</font>), 
- small flipper length and large bill length (<font color="#0072B2">blue cluster</font>).
- and large flipper length and large bill  length (<font color="#F0E442">yellow cluster</font>).

Once we have made these determinations, we can use them to inform our species
classifications or ask further questions about our data. For example, we might
be interested in understanding the relationship between flipper length and bill
length, and that relationship may differ depending on the type of penguin we
have. 

## K-means 

### Measuring cluster quality



```{code-cell} ipython3
clus = data[data["cluster"] == 2].loc[:,["bill_length_standardized", "flipper_length_standardized"]]
```

The K-means algorithm is a procedure that groups data into K clusters.
It starts with an initial clustering of the data, and then iteratively
improves it by making adjustments to the assignment of data
to clusters until it cannot improve any further. But how do we measure
the "quality" of a clustering, and what does it mean to improve it? 
In K-means clustering, we measure the quality of a cluster by its
\index{within-cluster sum-of-squared-distances|see{WSSD}}\index{WSSD}
*within-cluster sum-of-squared-distances* (WSSD). Computing this involves two steps.
First, we find the cluster centers by computing the mean of each variable 
over data points in the cluster. For example, suppose we have a 
cluster containing four observations, and we are using two variables, $x$ and $y$, to cluster the data.
Then we would compute the coordinates, $\mu_x$ and $\mu_y$, of the cluster center via



$\mu_x = \frac{1}{4}(x_1+x_2+x_3+x_4) \quad \mu_y = \frac{1}{4}(y_1+y_2+y_3+y_4)$

```{code-cell} ipython3
:tags: ["remove-cell"]
from myst_nb import glue
clus_rows = clus.shape[0]

mean_flipper_len_std = round(np.mean(clus['flipper_length_standardized']),2)
mean_bill_len_std = round(np.mean(clus['bill_length_standardized']),2)

glue("clus_rows_glue", clus_rows)
glue("mean_flipper_len_std_glue",mean_flipper_len_std)
glue("mean_bill_len_std_glue", mean_bill_len_std)
```


In the first cluster from the example, there are {glue:}`clus_rows_glue` data points. These are shown with their cluster center 
(flipper_length_standardized = {glue:}`mean_flipper_len_std_glue` and bill_length_standardized = {glue:}`mean_bill_len_std_glue`) highlighted 
in Figure \@ref(fig:10-toy-example-clus1-center).

(ref:10-toy-example-clus1-center) Cluster 1 from the `penguin_data` data set example. Observations are in blue, with the cluster center highlighted in red.

```{figure} img/toy-example-clus1-center-1.png
---
height: 400px
name: toy-example-clus1-center-1
---
Cluster 1 from the penguin_data data set example. Observations are in blue, with the cluster center highlighted in red.
```


The second step in computing the WSSD is to add up the squared distance 
\index{distance!K-means} between each point in the cluster 
and the cluster center.
We use the straight-line / Euclidean distance formula 
that we learned about in Chapter \@ref(classification).
In the {glue:}`clus_rows_glue`-observation cluster example above, 
we would compute the WSSD $S^2$ via


$S^2 = \left((x_1 - \mu_x)^2 + (y_1 - \mu_y)^2\right) + \left((x_2 - \mu_x)^2 + (y_2 - \mu_y)^2\right) + \left((x_3 - \mu_x)^2 + (y_3 - \mu_y)^2\right)  +  \left((x_4 - \mu_x)^2 + (y_4 - \mu_y)^2\right)$

These distances are denoted by lines in Figure \@ref(fig:10-toy-example-clus1-dists) for the first cluster of the penguin data example. 

(ref:10-toy-example-clus1-dists) Cluster 1 from the `penguin_data` data set example. Observations are in blue, with the cluster center highlighted in red. The distances from the observations to the cluster center are represented as black lines.

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
Figure \@ref(fig:10-toy-example-all-clus-dists).

(ref:10-toy-example-all-clus-dists) All clusters from the `penguin_data` data set example. Observations are in orange, blue, and yellow with the cluster center highlighted in red. The distances from the observations to each of the respective cluster centers are represented as black lines.


```{figure} img/toy-example-all-clus-dists-1.png
---
height: 400px
name: toy-example-all-clus-dists-1
---
All clusters from the penguin_data data set example. Observations are in orange, blue, and yellow with the cluster center highlighted in red. The distances from the observations to each of the respective cluster centers are represented as black lines.


```


\newpage

### The clustering algorithm

We begin the K-means \index{K-means!algorithm} algorithm by picking K, 
and randomly assigning a roughly equal number of observations 
to each of the K clusters.
An example random initialization is shown in Figure \@ref(fig:10-toy-kmeans-init).



```{figure} img/toy-kmeans-init-1.png
---
height: 400px
name: toy-kmeans-init-1
---
Random initialization of labels.

```


Then K-means consists of two major steps that attempt to minimize the
sum of WSSDs over all the clusters, i.e., the \index{WSSD!total} *total WSSD*:

1. **Center update:** Compute the center of each cluster.
2. **Label update:** Reassign each data point to the cluster with the nearest center.

These two steps are repeated until the cluster assignments no longer change.
We show what the first four iterations of K-means would look like in  
Figure \@ref(fig:10-toy-kmeans-iter). 
There each row corresponds to an iteration,
where the left column depicts the center update, 
and the right column depicts the reassignment of data to clusters.

(ref:10-toy-kmeans-iter) First four iterations of K-means clustering on the `penguin_data` example data set. Each pair of plots corresponds to an iteration. Within the pair, the first plot depicts the center update, and the second plot depicts the reassignment of data to clusters. Cluster centers are indicated by larger points that are outlined in black.


```{figure} img/toy-kmeans-iter-1.png
---
height: 400px
name: toy-kmeans-iter-1
---
First four iterations of K-means clustering on the penguin_data example data set. Each pair of plots corresponds to an iteration. Within the pair, the first plot depicts the center update, and the second plot depicts the reassignment of data to clusters. Cluster centers are indicated by larger points that are outlined in black.

```




Note that at this point, we can terminate the algorithm since none of the assignments changed
in the fourth iteration; both the centers and labels will remain the same from this point onward.

> **Note:** Is K-means *guaranteed* to stop at some point, or could it iterate forever? As it turns out,
> thankfully, the answer is that K-means \index{K-means!termination} is guaranteed to stop after *some* number of iterations. For the interested reader, the
> logic for this has three steps: (1) both the label update and the center update decrease total WSSD in each iteration,
> (2) the total WSSD is always greater than or equal to 0, and (3) there are only a finite number of possible
> ways to assign the data to clusters. So at some point, the total WSSD must stop decreasing, which means none of the assignments
> are changing, and the algorithm terminates.

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

Unlike the classification and regression models we studied in previous chapters, K-means \index{K-means!restart, nstart} can get "stuck" in a bad solution.
For example, Figure \@ref(fig:10-toy-kmeans-bad-init) illustrates an unlucky random initialization by K-means.



```{figure} img/toy-kmeans-bad-init-1.png
---
height: 400px
name: toy-kmeans-bad-init-1
---
Random initialization of labels.
```





Figure \@ref(fig:10-toy-kmeans-bad-iter) shows what the iterations of K-means would look like with the unlucky random initialization shown in Figure \@ref(fig:10-toy-kmeans-bad-init).

(ref:10-toy-kmeans-bad-iter) First five iterations of K-means clustering on the `penguin_data` example data set with a poor random initialization. Each pair of plots corresponds to an iteration. Within the pair, the first plot depicts the center update, and the second plot depicts the reassignment of data to clusters. Cluster centers are indicated by larger points that are outlined in black.

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
Figure \@ref(fig:10-toy-kmeans-vary-k) illustrates the impact of K 
on K-means clustering of our penguin flipper and bill length data 
by showing the different clusterings for K's ranging from 1 to 9.

```{figure} img/toy-kmeans-vary-k-1.png
---
height: 700px
name: toy-kmeans-vary-k-1
---
Clustering of the penguin data for K clusters ranging from 1 to 9. Cluster centers are indicated by larger points that are outlined in black.
```




If we set K less than 3, then the clustering merges separate groups of data; this causes a large 
total WSSD, since the cluster center (denoted by an "x") is not close to any of the data in the cluster. On 
the other hand, if we set K greater than 3, the clustering subdivides subgroups of data; this does indeed still 
decrease the total WSSD, but by only a *diminishing amount*. If we plot the total WSSD versus the number of 
clusters, we see that the decrease in total WSSD levels off (or forms an "elbow shape") \index{elbow method} when we reach roughly 
the right number of clusters (Figure \@ref(fig:10-toy-kmeans-elbow)).


```{figure} img/toy-kmeans-elbow-1.png
---
height: 400px
name: toy-kmeans-elbow-1
---
Total WSSD for K clusters ranging from 1 to 9.
```



## Data pre-processing for K-means

Similar to K-nearest neighbors classification and regression, K-means 
clustering uses straight-line distance to decide which points are similar to 
each other. Therefore, the *scale* of each of the variables in the data
will influence which cluster data points end up being assigned.
Variables with a large scale will have a much larger 
effect on deciding cluster assignment than variables with a small scale. 
To address this problem, we typically standardize \index{standardization!K-means}\index{K-means!standardization} our data before clustering,
which ensures that each variable has a mean of 0 and standard deviation of 1.
The `StandardScaler()` function in Python can be used to do this. 
We show an example of how to use this function 
below using an unscaled and unstandardized version of the data set in this chapter.



```{code-cell} ipython3
:tags: ["remove-cell"]
unstandardized_data = pd.read_csv("data/toy_penguins.csv", usecols=["bill_length_mm", "flipper_length_mm"]) 
unstandardized_data.to_csv("data/penguins_not_standardized.csv", index=False)
unstandardized_data
```

First, here is what the raw (i.e., not standardized) data looks like:



```{code-cell} ipython3
not_standardized_data = pd.read_csv("data/penguins_not_standardized.csv")
not_standardized_data
```

And then we apply the `StandardScaler()` function to both the columns in the data frame 
using `fit_transform()`



```{code-cell} ipython3
scaler = StandardScaler()
standardized_data = pd.DataFrame(
    scaler.fit_transform(not_standardized_data), columns = ['bill_length_mm', 'flipper_length_mm'])
    
standardized_data
```



## K-means in Python

To perform K-means clustering in Python, we use the `KMeans` function. \index{K-means!kmeans function} It takes at
least two arguments: the data frame containing the data you wish to cluster,
and K, the number of clusters (here we choose K = 3). Note that since the K-means
algorithm uses a random initialization of assignments, but since we set the random seed
earlier, the clustering will be reproducible.



```{code-cell} ipython3
np.random.seed(1234)
```


```{code-cell} ipython3
from sklearn.cluster import KMeans
penguin_clust = KMeans(n_clusters=3, random_state=1234).fit(standardized_data)
penguin_clust

```

As you can see above, the clustering object returned by `KMeans` has a lot of information
that can be used to visualize the clusters, pick K, and evaluate the total WSSD.
To obtain this information in a tidy format, we will call in help 
from the `broom` package. \index{broom} Let's start by visualizing the clustering
as a colored scatter plot. To do that,
we use the `augment` function, \index{K-means!augment} \index{augment} which takes in the model and the original data
frame, and returns a data frame with the data and the cluster assignments for
each point:



```{code-cell} ipython3
predictions = penguin_clust.predict(standardized_data)
predictions

```

```{code-cell} ipython3
clustered_data = standardized_data
clustered_data = clustered_data.assign(clusters = predictions)
clustered_data
```

Now that we have this information in a tidy data frame, we can make a visualization
of the cluster assignments for each point, as shown in Figure \@ref(fig:10-plot-clusters-2).


```{code-cell} ipython3
cluster_plot = (
    alt.Chart(clustered_data)
    .mark_circle()
    .encode(
         x = alt.X("flipper_length_mm", title="Flipper Length (standardized)"),
        y = alt.Y("bill_length_mm", title="Bill Length (standardized)"),
    color=alt.Color("clusters:O", title="Cluster", scale=alt.Scale(scheme="dark2")),
    ).properties(width=400, height=400)
    .configure_axis(labelFontSize=20, titleFontSize=20)
)


cluster_plot


```

```{code-cell} ipython3
:tags: ["remove-cell"]
glue('cluster_plot', cluster_plot, display=True)
```

:::{glue:figure} cluster_plot 
:figwidth: 700px 
:name: cluster_plot

The data colored by the cluster assignments returned by K-means.
:::



As mentioned above, we also need to select K by finding
where the "elbow" occurs in the plot of total WSSD versus the number of clusters. 
We can obtain the total WSSD (`tot.withinss`) \index{WSSD!total} from our
clustering using `broom`'s `glance` function. For example:


```{code-cell} ipython3
penguin_clust.inertia_
```

To calculate the total WSSD for a variety of Ks, we will
create a data frame with a column named `k` with rows containing
each value of K we want to run K-means with (here, 1 to 9). 



```{code-cell} ipython3
import numpy as np
penguin_clust_ks = pd.DataFrame({"k": np.array(range(1, 10)).transpose()})
```

Then we use `rowwise` \index{rowwise} + `mutate` to apply the `kmeans` function 
within each row to each K. 
However, given that the `kmeans` function 
returns a model object to us (not a vector),
we will need to store the results as a list column.
This works because both vectors and lists are legitimate 
data structures for data frame columns. 
To make this work, 
we have to put each model object in a list using the `list` function.
We demonstrate how to do this below:



```{code-cell} ipython3
penguin_clust_ks = penguin_clust_ks.assign(
    penguin_clusts=penguin_clust_ks['k'].apply(
        lambda x: KMeans(n_clusters=x, n_init=3, random_state=2020).fit(standardized_data)
    )
)
penguin_clust_ks
```
```{code-cell} ipython3
penguin_clust_ks = penguin_clust_ks.assign(
    inertia=penguin_clust_ks["penguin_clusts"].apply(lambda x: x.inertia_)
).drop(columns=['penguin_clusts'])
penguin_clust_ks

```

If we take a look at our data frame `penguin_clust_ks` now, 
we see that it has two columns: one with the value for K, 
and the other holding the clustering model object in a list column.


If we wanted to get one of the clusterings out 
of the list column in the data frame,
we could use a familiar friend: `pull`.
`pull` will return to us a data frame column as a simpler data structure,
here that would be a list.
And then to extract the first item of the list, 
we can use the `pluck` function. We pass  
it the index for the element we would like to extract 
(here, `1`).

```{r}
penguin_clust_ks |>
  pull(penguin_clusts) |>
  pluck(1)
```

Next, we use `mutate` again to apply `glance` \index{glance} 
to each of the K-means clustering objects to get the clustering statistics 
(including WSSD). 
The output of `glance` is a data frame, 
and so we need to create another list column (using `list`) for this to work. 
This results in a complex data frame with 3 columns, one for K, one for the 
K-means clustering objects, and one for the clustering statistics:

```{r 10-choose-k-part3}
penguin_clust_ks <- tibble(k = 1:9) |>
  rowwise() |>
  mutate(penguin_clusts = list(kmeans(standardized_data, k)),
         glanced = list(glance(penguin_clusts)))

penguin_clust_ks
```

Finally we extract the total WSSD from the column named `glanced`. 
Given that each item in this list column is a data frame, 
we will need to use the `unnest` function 
to unpack the data frames into simpler column data types. 

```{r 10-get-total-within-sumsqs}
clustering_statistics <- penguin_clust_ks |>
  unnest(glanced)

clustering_statistics
```

Now that we have `tot.withinss` and `k` as columns in a data frame, we can make a line plot 
(Figure \@ref(fig:10-plot-choose-k)) and search for the "elbow" to find which value of K to use. 

```{r 10-plot-choose-k, fig.height = 3.25, fig.width = 4.25, fig.align = "center", fig.pos = "H", out.extra="", fig.cap = "A plot showing the total WSSD versus the number of clusters."}
elbow_plot <- ggplot(clustering_statistics, aes(x = k, y = tot.withinss)) +
  geom_point() +
  geom_line() +
  xlab("K") +
  ylab("Total within-cluster sum of squares") +
  scale_x_continuous(breaks = 1:9) + 
  theme(text = element_text(size = 12))

elbow_plot
```

```{code-cell} ipython3
elbow_plot=(
    alt.Chart(penguin_clust_ks)
    .mark_line(point=True, color='black')
    .encode(
        x=alt.X("k", title="K"),
        y=alt.Y("inertia", title="Total within-cluster sum of squares"),
    )
    .properties(width=400, height=400)
    .configure_axis(labelFontSize=15, titleFontSize=20)
)
### END SOLUTION 
elbow_plot
```
```{code-cell} ipython3
:tags: ["remove-cell"]
glue('elbow_plot', elbow_plot, display=True)
```

:::{glue:figure} elbow_plot 
:figwidth: 700px 
:name: elbow_plot

A plot showing the total WSSD versus the number of clusters.
:::




It looks like 3 clusters is the right choice for this data.
But why is there a "bump" in the total WSSD plot here? 
Shouldn't total WSSD always decrease as we add more clusters? 
Technically yes, but remember:  K-means can get "stuck" in a bad solution. 
Unfortunately, for K = 8 we had an unlucky initialization
and found a bad clustering! \index{K-means!restart, nstart} 
We can help prevent finding a bad clustering 
by trying a few different random initializations 
via the `nstart` argument (Figure \@ref(fig:10-choose-k-nstart) 
shows a setup where we use 10 restarts). 
When we do this, K-means clustering will be performed 
the number of times specified by the `nstart` argument,
and R will return to us the best clustering from this.
The more times we perform K-means clustering,
the more likely we are to find a good clustering (if one exists).
What value should you choose for `nstart`? The answer is that it depends
on many factors: the size and characteristics of your data set,
as well as the speed and size of your computer.
The larger the `nstart` value the better from an analysis perspective, 
but there is a trade-off that doing many clusterings 
could take a long time.
So this is something that needs to be balanced.

```{r 10-choose-k-nstart, fig.height = 3.25, fig.width = 4.25, fig.pos = "H", out.extra="", message= FALSE, warning = FALSE, fig.align = "center", fig.cap = "A plot showing the total WSSD versus the number of clusters when K-means is run with 10 restarts."}
penguin_clust_ks <- tibble(k = 1:9) |>
  rowwise() |>
  mutate(penguin_clusts = list(kmeans(standardized_data, nstart = 10, k)),
         glanced = list(glance(penguin_clusts)))

clustering_statistics <- penguin_clust_ks |>
  unnest(glanced)

elbow_plot <- ggplot(clustering_statistics, aes(x = k, y = tot.withinss)) +
  geom_point() +
  geom_line() +
  xlab("K") +
  ylab("Total within-cluster sum of squares") +
  scale_x_continuous(breaks = 1:9) + 
  theme(text = element_text(size = 12))

elbow_plot
```

## Exercises

Practice exercises for the material covered in this chapter 
can be found in the accompanying 
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the "Clustering" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in Chapter \@ref(move-to-your-own-machine). This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

## Additional resources
- Chapter 10 of *An Introduction to Statistical
  Learning* [@james2013introduction] provides a
  great next stop in the process of learning about clustering and unsupervised
  learning in general. In the realm of clustering specifically, it provides a
  great companion introduction to K-means, but also covers *hierarchical*
  clustering for when you expect there to be subgroups, and then subgroups within
  subgroups, etc., in your data. In the realm of more general unsupervised
  learning, it covers *principal components analysis (PCA)*, which is a very
  popular technique for reducing the number of predictors in a dataset.
