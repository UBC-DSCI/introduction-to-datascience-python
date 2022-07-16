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

(regression1)=
# Regression I: K-nearest neighbors

```{code-cell} ipython3
:tags: [remove-cell]

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
from IPython.display import HTML

from myst_nb import glue
```

## Overview 

This chapter continues our foray into answering predictive questions.
Here we will focus on predicting *numerical* variables 
and will use *regression* to perform this task.
This is unlike the past two chapters, which focused on predicting categorical
variables via classification. However, regression does have many similarities
to classification: for example, just as in the case of classification,
we will split our data into training, validation, and test sets, we will 
use `scikit-learn` workflows, we will use a K-nearest neighbors (KNN) 
approach to make predictions, and we will use cross-validation to choose K.
Because of how similar these procedures are, make sure to read Chapters
{ref}`classification` and {ref}`classification2` before reading 
this one&mdash;we will move a little bit faster here with the
concepts that have already been covered.
This chapter will primarily focus on the case where there is a single predictor, 
but the end of the chapter shows how to perform
regression with more than one predictor variable, i.e., *multivariable regression*.
It is important to note that regression 
can also be used to answer inferential and causal questions, 
however that is beyond the scope of this book.

+++

## Chapter learning objectives 
By the end of the chapter, readers will be able to do the following:

* Recognize situations where a simple regression analysis would be appropriate for making predictions.
* Explain the K-nearest neighbor (KNN) regression algorithm and describe how it differs from KNN classification.
* Interpret the output of a KNN regression.
* In a dataset with two or more variables, perform K-nearest neighbor regression in Python using a `scikit-learn` workflow.
* Execute cross-validation in Python to choose the number of neighbors.
* Evaluate KNN regression prediction accuracy in Python using a test data set and the root mean squared prediction error (RMSPE).
* In the context of KNN regression, compare and contrast goodness of fit and prediction properties (namely RMSE vs RMSPE).
* Describe the advantages and disadvantages of K-nearest neighbors regression.

+++

## The regression problem

Regression, like classification, is a predictive \index{predictive question} problem setting where we want
to use past information to predict future observations. But in the case of
regression, the goal is to predict *numerical* values instead of *categorical* values. 
The variable that you want to predict is often called the *response variable*. \index{response variable}
For example, we could try to use the number of hours a person spends on
exercise each week to predict their race time in the annual Boston marathon. As 
another example, we could try to use the size of a house to
predict its sale price. Both of these response variables&mdash;race time and sale price&mdash;are 
numerical, and so predicting them given past data is considered a regression problem.

Just like in the \index{classification!comparison to regression} 
classification setting, there are many possible methods that we can use 
to predict numerical response variables. In this chapter we will
focus on the **K-nearest neighbors** algorithm {cite:p}`knnfix,knncover`, and in the next chapter
we will study **linear regression**.
In your future studies, you might encounter regression trees, splines,
and general local regression methods; see the additional resources
section at the end of the next chapter for where to begin learning more about
these other methods. 

Many of the concepts from classification map over to the setting of regression. For example, 
a regression model predicts a new observation's response variable based on the response variables
for similar observations in the data set of past observations. When building a regression model,
we first split the data into training and test sets, in order to ensure that we assess the performance
of our method on observations not seen during training. And finally, we can use cross-validation to evaluate different
choices of model parameters (e.g., K in a K-nearest neighbors model). The major difference
is that we are now predicting numerical variables instead of categorical variables.


> **Note:** You can usually tell whether a\index{categorical variable}\index{numerical variable} variable is numerical or
> categorical&mdash;and therefore whether you need to perform regression or
> classification&mdash;by taking two response variables X and Y from your data,
> and asking the question, "is response variable X *more* than response
> variable Y?" If the variable is categorical, the question will make no sense.
> (Is blue more than red?  Is benign more than malignant?) If the variable is
> numerical, it will make sense. (Is 1.5 hours more than 2.25 hours? Is
> \$500,000 more than \$400,000?) Be careful when applying this heuristic,
> though: sometimes categorical variables will be encoded as numbers in your
> data (e.g., "1" represents "benign", and "0" represents "malignant"). In
> these cases you have to ask the question about the *meaning* of the labels
> ("benign" and "malignant"), not their values ("1" and "0").

+++

## Exploring a data set

In this chapter and the next, we will study 
a data set \index{Sacramento real estate} of 
[932 real estate transactions in Sacramento, California](https://support.spatialkey.com/spatialkey-sample-csv-data/) 
originally reported in the *Sacramento Bee* newspaper.
We first need to formulate a precise question that
we want to answer. In this example, our question is again predictive:
\index{question!regression} Can we use the size of a house in the Sacramento, CA area to predict
its sale price? A rigorous, quantitative answer to this question might help
a realtor advise a client as to whether the price of a particular listing 
is fair, or perhaps how to set the price of a new listing.
We begin the analysis by loading and examining the data.

```{code-cell} ipython3
:tags: [remove-cell]

# In this chapter and the next, we will study 
# a data set \index{Sacramento real estate} of 
# [932 real estate transactions in Sacramento, California](https://support.spatialkey.com/spatialkey-sample-csv-data/) 
# originally reported in the *Sacramento Bee* newspaper.
# We first need to formulate a precise question that
# we want to answer. In this example, our question is again predictive:
# \index{question!regression} Can we use the size of a house in the Sacramento, CA area to predict
# its sale price? A rigorous, quantitative answer to this question might help
# a realtor advise a client as to whether the price of a particular listing 
# is fair, or perhaps how to set the price of a new listing.
# We begin the analysis by loading and examining the data, and setting the seed value.

# \index{seed!set.seed}
```

```{code-cell} ipython3
import pandas as pd
import altair as alt

sacramento = pd.read_csv("data/sacramento.csv")
sacramento
```

The scientific question guides our initial exploration: the columns in the
data that we are interested in are `sqft` (house size, in livable square feet)
and `price` (house sale price, in US dollars (USD)).  The first step is to visualize
the data as a scatter plot where we place the predictor variable
(house size) on the x-axis, and we place the target/response variable that we
want to predict (sale price) on the y-axis.
\index{ggplot!geom\_point}
\index{visualization!scatter}

> **Note:** Given that the y-axis unit is dollars in {numref}`fig:07-edaRegr`, 
> we format the axis labels to put dollar signs in front of the house prices, 
> as well as commas to increase the readability of the larger numbers.
> We can do this in `altair` by passing the `axis=alt.Axis(format='$,.0f')` argument 
> to the `y` encoding channel in an `altair` specification.

```{code-cell} ipython3
:tags: [remove-output]

eda = (
    alt.Chart(sacramento)
    .mark_circle(opacity=0.4, color='black')
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y("price", title="Price (USD)", axis=alt.Axis(format='$,.0f')),
    )
)

eda
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-edaRegr", eda)
```

:::{glue:figure} fig:07-edaRegr
:name: fig:07-edaRegr

Scatter plot of price (USD) versus house size (square feet).
:::

+++

The plot is shown in {numref}`fig:07-edaRegr`.
We can see that in Sacramento, CA, as the
size of a house increases, so does its sale price. Thus, we can reason that we
may be able to use the size of a not-yet-sold house (for which we don't know
the sale price) to predict its final sale price. Note that we do not suggest here
that a larger house size *causes* a higher sale price; just that house price
tends to increase with house size, and that we may be able to use the latter to 
predict the former.

+++

## K-nearest neighbors regression

Much like in the case of classification, 
we can use a K-nearest neighbors-based \index{K-nearest neighbors!regression} 
approach in regression to make predictions. 
Let's take a small sample of the data in {numref}`fig:07-edaRegr` 
and walk through how K-nearest neighbors (KNN) works
in a regression context before we dive in to creating our model and assessing
how well it predicts house sale price. This subsample is taken to allow us to
illustrate the mechanics of KNN regression with a few data points; later in
this chapter we will use all the data.

To take a small random sample of size 30, we'll use the 
`sample` method of a `pandas.DataFrame` object, and input the number of rows
to randomly select (`n`) and the random seed (`random_state`). \index{slice\_sample}

```{code-cell} ipython3
small_sacramento = sacramento.sample(n=30, random_state=10)
```

Next let's say we come across a  2,000 square-foot house in Sacramento we are
interested in purchasing, with an advertised list price of \$350,000. Should we
offer to pay the asking price for this house, or is it overpriced and we should
offer less? Absent any other information, we can get a sense for a good answer
to this question by using the data we have to predict the sale price given the
sale prices we have already observed. But in {numref}`fig:07-small-eda-regr`,
you can see that we have no
observations of a house of size *exactly* 2,000 square feet. How can we predict
the sale price?

```{code-cell} ipython3
:tags: [remove-output]

small_plot = (
    alt.Chart(small_sacramento)
    .mark_circle(color='black')
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y("price", title="Price (USD)", axis=alt.Axis(format='$,.0f')),
    )
)

# add an overlay to the base plot
line_df = pd.DataFrame({"x": [2000]})
rule = alt.Chart(line_df).mark_rule(strokeDash=[2, 4]).encode(x="x")

small_plot + rule
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-small-eda-regr", (small_plot + rule))
```

:::{glue:figure} fig:07-small-eda-regr
:name: fig:07-small-eda-regr

Scatter plot of price (USD) versus house size (square feet) with vertical line indicating 2,000 square feet on x-axis.
:::

+++

We will employ the same intuition from the classification chapter, and use the
neighboring points to the new point of interest to suggest/predict what its
sale price might be. 
For the example shown in {numref}`fig:07-small-eda-regr`, 
we find and label the 5 nearest neighbors to our observation 
of a house that is 2,000 square feet.
\index{mutate}\index{slice}\index{arrange}\index{abs}

```{code-cell} ipython3
nearest_neighbors = (
    small_sacramento.assign(diff=abs(2000 - small_sacramento["sqft"]))
    .sort_values("diff")
    .iloc[:5]
)

nearest_neighbors
```

```{code-cell} ipython3
:tags: [remove-cell]

nn_plot = small_plot + rule

# plot horizontal lines which is perpendicular to x=2000
for i in range(5):
    h_line_df = pd.concat(
        (
            pd.DataFrame(nearest_neighbors.iloc[i, [4, 6]]).T,
            pd.DataFrame({"sqft": [2000], "price": [nearest_neighbors.iloc[i, 6]]}),
        ),
        ignore_index=True,
    )
    h_line = alt.Chart(h_line_df).mark_line(color="orange").encode(x="sqft", y="price")
    nn_plot += h_line

nn_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-knn5-example", nn_plot)
```

:::{glue:figure} fig:07-knn5-example
:name: fig:07-knn5-example

Scatter plot of price (USD) versus house size (square feet) with lines to 5 nearest neighbors.
:::

+++

{numref}`fig:07-knn5-example` illustrates the difference between the house sizes
of the 5 nearest neighbors (in terms of house size) to our new
2,000 square-foot house of interest. Now that we have obtained these nearest neighbors, 
we can use their values to predict the
sale price for the new home.  Specifically, we can take the mean (or
average) of these 5 values as our predicted value, as illustrated by
the red point in {numref}`fig:07-predictedViz-knn`.

```{code-cell} ipython3
prediction = nearest_neighbors["price"].mean()
prediction
```

```{code-cell} ipython3
:tags: [remove-cell]

nn_plot_pred = nn_plot + alt.Chart(
    pd.DataFrame({"sqft": [2000], "price": [prediction]})
).mark_circle(size=40).encode(x="sqft", y="price", color=alt.value("red"))
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("knn-5-pred", "{0:,.0f}".format(prediction))
glue("fig:07-predictedViz-knn", nn_plot_pred)
```

:::{glue:figure} fig:07-predictedViz-knn
:name: fig:07-predictedViz-knn

Scatter plot of price (USD) versus house size (square feet) with predicted price for a 2,000 square-foot house based on 5 nearest neighbors represented as a red dot.
:::

+++

Our predicted price is \${glue:text}`knn-5-pred`
(shown as a red point in {numref}`fig:07-predictedViz-knn`), which is much less than \$350,000; perhaps we
might want to offer less than the list price at which the house is advertised.
But this is only the very beginning of the story. We still have all the same
unanswered questions here with KNN regression that we had with KNN
classification: which $K$ do we choose, and is our model any good at making
predictions? In the next few sections, we will address these questions in the
context of KNN regression.

One strength of the KNN regression algorithm 
that we would like to draw attention to at this point
is its ability to work well with non-linear relationships
(i.e., if the relationship is not a straight line).
This stems from the use of nearest neighbors to predict values.
The algorithm really has very few assumptions 
about what the data must look like for it to work.

+++ {"tags": []}

## Training, evaluating, and tuning the model

As usual, 
we must start by putting some test data away in a lock box 
that we will come back to only after we choose our final model. 
Let's take care of that now. 
Note that for the remainder of the chapter 
we'll be working with the entire Sacramento data set, 
as opposed to the smaller sample of 30 points 
that we used earlier in the chapter ({numref}`fig:07-small-eda-regr`).
\index{training data}
\index{test data}

+++

> Note that we are not specifying the `stratify` argument here as we did in Chapter {ref}`classification2`,
> since `sklearn.model_selection.train_test_split` cannot stratify based on a continuous variable. However, some functions inside other packages are able to do that.

```{code-cell} ipython3
sacramento_train, sacramento_test = train_test_split(
    sacramento, train_size=0.75, random_state=5
)
```

Next, we'll use cross-validation \index{cross-validation} to choose $K$. In KNN classification, we used
accuracy to see how well our predictions matched the true labels. We cannot use
the same metric in the regression setting, since our predictions will almost never
*exactly* match the true response variable values. Therefore in the
context of KNN regression we will use root mean square prediction error \index{root mean square prediction error|see{RMSPE}}\index{RMSPE}
(RMSPE) instead. The mathematical formula for calculating RMSPE is: 

$$\text{RMSPE} = \sqrt{\frac{1}{n}\sum\limits_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

where:

- $n$ is the number of observations,
- $y_i$ is the observed value for the $i^\text{th}$ observation, and
- $\hat{y}_i$ is the forecasted/predicted value for the $i^\text{th}$ observation.

In other words, we compute the *squared* difference between the predicted and true response 
value for each observation in our test (or validation) set, compute the average, and then finally
take the square root. The reason we use the *squared* difference (and not just the difference)
is that the differences can be positive or negative, i.e., we can overshoot or undershoot the true
response value. {numref}`fig:07-verticalerrors` illustrates both positive and negative differences
between predicted and true response values.
So if we want to measure error&mdash;a notion of distance between our predicted and true response values&mdash;we 
want to make sure that we are only adding up positive values, with larger positive values representing larger
mistakes.
If the predictions are very close to the true values, then
RMSPE will be small. If, on the other-hand, the predictions are very
different from the true values, then RMSPE will be quite large. When we
use cross-validation, we will choose the $K$ that gives
us the smallest RMSPE.

```{code-cell} ipython3
:tags: [remove-cell]

# (synthetic) new prediction points
pts = pd.DataFrame({"sqft": [1250, 1850, 2250], "price": [250000, 200000, 500000]})
finegrid = pd.DataFrame({"sqft": np.arange(900, 3901, 10)})

# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor(n_neighbors=4))

# fit the model
X = small_sacramento[["sqft"]]
y = small_sacramento[["price"]]
sacr_pipeline.fit(X, y)

# predict on the full grid and new data pts
sacr_full_preds_hid = pd.concat(
    (finegrid, pd.DataFrame(sacr_pipeline.predict(finegrid), columns=["predicted"])),
    axis=1,
)

sacr_new_preds_hid = pd.concat(
    (pts, pd.DataFrame(sacr_pipeline.predict(pts), columns=["predicted"])),
    axis=1,
)

# to make altair mark_line works, need to create separate dataframes for each vertical error line
sacr_new_preds_melted_df = sacr_new_preds_hid.melt(id_vars=["sqft"])
errors_plot = (
    small_plot
    + alt.Chart(sacr_full_preds_hid).mark_line().encode(x="sqft", y="predicted")
    + alt.Chart(sacr_new_preds_hid)
    .mark_circle(color="black")
    .encode(x="sqft", y="price")
)
for i in pts["sqft"]:
    line_df = sacr_new_preds_melted_df.query("sqft == @i")
    errors_plot += alt.Chart(line_df).mark_line(color="red").encode(x="sqft", y="value")

errors_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-verticalerrors", errors_plot, display=False)
```

:::{glue:figure} fig:07-verticalerrors
:name: fig:07-verticalerrors

Scatter plot of price (USD) versus house size (square feet) with example predictions (blue line) and the error in those predictions compared with true response values for three selected observations (vertical red lines).
:::

+++

> **Note:** When using many code packages, the evaluation output 
> we will get to assess the prediction quality of
> our KNN regression models is labeled "RMSE", or "root mean squared
> error". Why is this so, and why not RMSPE? \index{RMSPE!comparison with RMSE}
> In statistics, we try to be very precise with our
> language to indicate whether we are calculating the prediction error on the
> training data (*in-sample* prediction) versus on the testing data 
> (*out-of-sample* prediction). When predicting and evaluating prediction quality on the training data, we 
>  say RMSE. By contrast, when predicting and evaluating prediction quality
> on the testing or validation data, we say RMSPE. 
> The equation for calculating RMSE and RMSPE is exactly the same; all that changes is whether the $y$s are
> training or testing data. But many people just use RMSE for both, 
> and rely on context to denote which data the root mean squared error is being calculated on.

Now that we know how we can assess how well our model predicts a numerical
value, let's use Python to perform cross-validation and to choose the optimal $K$.
First, we will create a recipe for preprocessing our data.
Note that we include standardization
in our preprocessing to build good habits, but since we only have one 
predictor, it is technically not necessary; there is no risk of comparing two predictors
of different scales.
Next we create a model pipeline for K-nearest neighbors regression. Note 
that we use `KNeighborsRegressor`
now in the model specification to denote a regression problem, as opposed to the classification
problems from the previous chapters. 
The use of `KNeighborsRegressor` essentially
 tells `scikit-learn` that we need to use different metrics (instead of accuracy)
for tuning and evaluation. 
Next we specify a dictionary of parameter grid containing the numbers of neighbors ranging from 1 to 200. 
Then we create a 5-fold `GridSearchCV` object, and pass in the pipeline and parameter grid. We also
need to specify that `scoring="neg_root_mean_squared_error"` to get *negative* RMSPE from `scikit-learn`.
The reason that `scikit-learn` negates the regular RMSPE is that the function always tries to maximize 
the scores, while RMSPE should be minimized. Hence, in order to see the actual RMSPE, we need to negate 
back the `mean_test_score`.

+++

In the output of the `sacr_results`
results data frame, we see that the `param_kneighborsregressor__n_neighbors` variable contains the values of $K$,
the `RMSPE` variable contains the value of the RMSPE estimated via cross-validation, 
which was obtained through negating the `mean_test_score` variable, 
and the standard error (`std_test_score`) contains a value corresponding to a measure of how uncertain we are in the mean value. A detailed treatment of this
is beyond the scope of this chapter; but roughly, if your estimated mean is 100,000 and standard
error is 1,000, you can expect the *true* RMSPE to be somewhere roughly between 99,000 and 101,000 (although it may
fall outside this range). You may ignore the other columns in the metrics data frame,
as they do not provide any additional insight.

```{code-cell} ipython3
:tags: [remove-cell]

# Now that we know how we can assess how well our model predicts a numerical
# value, let's use R to perform cross-validation and to choose the optimal $K$.
# First, we will create a recipe for preprocessing our data.
# Note that we include standardization
# in our preprocessing to build good habits, but since we only have one 
# predictor, it is technically not necessary; there is no risk of comparing two predictors
# of different scales.
# Next we create a model specification for K-nearest neighbors regression. Note 
# that we use `set_mode("regression")`
# now in the model specification to denote a regression problem, as opposed to the classification
# problems from the previous chapters. 
# The use of `set_mode("regression")` essentially
#  tells `tidymodels` that we need to use different metrics (RMSPE, not accuracy)
# for tuning and evaluation.
# Then we create a 5-fold cross-validation object, and put the recipe and model specification together
# in a workflow.
# \index{tidymodels}\index{recipe}\index{workflow}
```

```{code-cell} ipython3
# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor())

# create the predictor and target
X = sacramento_train[["sqft"]]
y = sacramento_train[["price"]]

# create the 5-fold GridSearchCV object
param_grid = {
    "kneighborsregressor__n_neighbors": range(1, 201, 3),
}
sacr_gridsearch = GridSearchCV(
    estimator=sacr_pipeline,
    param_grid=param_grid,
    n_jobs=-1,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

# fit the GridSearchCV object
sacr_gridsearch.fit(X, y)

# retrieve the CV scores
sacr_results = pd.DataFrame(sacr_gridsearch.cv_results_)
# negate mean_test_score (negative RMSPE) to get the actual RMSPE
sacr_results["RMSPE"] = -sacr_results["mean_test_score"]

sacr_results[
    [
        "param_kneighborsregressor__n_neighbors",
        "RMSPE",
        "mean_test_score",
        "std_test_score",
    ]
]
```

```{code-cell} ipython3
:tags: [remove-cell]

# Next we run cross-validation for a grid of numbers of neighbors ranging from 1 to 200. 
# The following code tunes
# the model and returns the RMSPE for each number of neighbors. In the output of the `sacr_results`
# results data frame, we see that the `neighbors` variable contains the value of $K$,
# the mean (`mean`) contains the value of the RMSPE estimated via cross-validation,
# and the standard error (`std_err`) contains a value corresponding to a measure of how uncertain we are in the mean value. A detailed treatment of this
# is beyond the scope of this chapter; but roughly, if your estimated mean is 100,000 and standard
# error is 1,000, you can expect the *true* RMSPE to be somewhere roughly between 99,000 and 101,000 (although it may
# fall outside this range). You may ignore the other columns in the metrics data frame,
# as they do not provide any additional insight.
# \index{cross-validation!collect\_metrics}
```

```{code-cell} ipython3
:tags: [remove-cell]

sacr_tunek_plot = alt.Chart(sacr_results).mark_line(point=True).encode(
    x=alt.X("param_kneighborsregressor__n_neighbors", title="Neighbors"),
    y=alt.Y("RMSPE", scale=alt.Scale(zero=False))
)

sacr_tunek_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-choose-k-knn-plot", sacr_tunek_plot, display=False)
```

:::{glue:figure} fig:07-choose-k-knn-plot
:name: fig:07-choose-k-knn-plot

Effect of the number of neighbors on the RMSPE.
:::

+++

{numref}`fig:07-choose-k-knn-plot` visualizes how the RMSPE varies with the number of neighbors $K$.
We take the *minimum* RMSPE to find the best setting for the number of neighbors:

```{code-cell} ipython3
# show only the row of minimum RMSPE
sacr_min = sacr_results[sacr_results["RMSPE"] == min(sacr_results["RMSPE"])]
sacr_min[
    [
        "param_kneighborsregressor__n_neighbors",
        "RMSPE",
        "mean_test_score",
        "std_test_score",
    ]
]
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("kmin", int(sacr_min['param_kneighborsregressor__n_neighbors']))
```

The smallest RMSPE occurs when $K =$ {glue:}`kmin`.

+++

## Underfitting and overfitting
Similar to the setting of classification, by setting the number of neighbors
to be too small or too large, we cause the RMSPE to increase, as shown in
{numref}`fig:07-choose-k-knn-plot`. What is happening here? 

{numref}`fig:07-howK` visualizes the effect of different settings of $K$ on the
regression model. Each plot shows the predicted values for house sale price from
our KNN regression model for 6 different values for $K$: 1, 3, {glue:}`kmin`, 41, 250, and 699 (i.e., the training data).
For each model, we predict prices for the range of possible home sizes we
observed in the data set (here 500 to 5,000 square feet) and we plot the
predicted prices as a blue line.

```{code-cell} ipython3
:tags: [remove-cell]

# Figure \@ref(fig:07-howK) visualizes the effect of different settings of $K$ on the
# regression model. Each plot shows the predicted values for house sale price from
# our KNN regression model for 6 different values for $K$: 1, 3, `r kmin`, 41, 250, and 932 (i.e., the entire dataset).
# For each model, we predict prices for the range of possible home sizes we
# observed in the data set (here 500 to 5,000 square feet) and we plot the
# predicted prices as a blue line.
```

```{code-cell} ipython3
:tags: [remove-cell]

gridvals = [
    1,
    3,
    int(sacr_min["param_kneighborsregressor__n_neighbors"]),
    41,
    250,
    len(sacramento_train),
]

plots = list()

sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
X = sacramento_train[["sqft"]]
y = sacramento_train[["price"]]

base_plot = (
    alt.Chart(sacramento_train)
    .mark_circle(opacity=0.4, color="black")
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y("price", title="Price (USD)", axis=alt.Axis(format="$,.0f")),
    )
)
for i in range(len(gridvals)):
    # make the pipeline based on n_neighbors
    sacr_pipeline = make_pipeline(
        sacr_preprocessor, KNeighborsRegressor(n_neighbors=gridvals[i])
    )
    sacr_pipeline.fit(X, y)
    # predictions
    sacr_preds = sacramento_train
    sacr_preds = sacr_preds.assign(predicted=sacr_pipeline.predict(sacramento_train))
    # overlay the plots
    plots.append(
        base_plot
        + alt.Chart(sacr_preds, title=f"K = {gridvals[i]}")
        .mark_line(color="blue")
        .encode(x="sqft", y="predicted")
    )
```

```{code-cell} ipython3
:tags: [remove-cell]

glue(
    "fig:07-howK", (plots[0] | plots[1]) & (plots[2] | plots[3]) & (plots[4] | plots[5])
)
```

:::{glue:figure} fig:07-howK
:name: fig:07-howK

Predicted values for house price (represented as a blue line) from KNN regression models for six different values for $K$.
:::

+++

{numref}`fig:07-howK` shows that when $K$ = 1, the blue line runs perfectly
through (almost) all of our training observations. 
This happens because our
predicted values for a given region (typically) depend on just a single observation. 
In general, when $K$ is too small, the line follows the training data quite
closely, even if it does not match it perfectly.
If we used a different training data set of house prices and sizes
from the Sacramento real estate market, we would end up with completely different
predictions. In other words, the model is *influenced too much* by the data.
Because the model follows the training data so closely, it will not make accurate
predictions on new observations which, generally, will not have the same fluctuations
as the original training data.
Recall from the classification
chapters that this behavior&mdash;where the model is influenced too much
by the noisy data&mdash;is called *overfitting*; we use this same term 
in the context of regression. \index{overfitting!regression}

What about the plots in {numref}`fig:07-howK` where $K$ is quite large, 
say, $K$ = 250 or 699? 
In this case the blue line becomes extremely smooth, and actually becomes flat
once $K$ is equal to the number of datapoints in the entire data set. 
This happens because our predicted values for a given x value (here, home
size), depend on many neighboring observations; in the case where $K$ is equal 
to the size of the dataset, the prediction is just the mean of the house prices
in the dataset (completely ignoring the house size). 
In contrast to the $K=1$ example, 
the smooth, inflexible blue line does not follow the training observations very closely.
In other words, the model is *not influenced enough* by the training data.
Recall from the classification
chapters that this behavior is called *underfitting*; we again use this same
term in the context of regression.  \index{underfitting!regression}

Ideally, what we want is neither of the two situations discussed above. Instead,
we would like a model that (1) follows the overall "trend" in the training data, so the model
actually uses the training data to learn something useful, and (2) does not follow
the noisy fluctuations, so that we can be confident that our model will transfer/generalize
well to other new data. If we explore 
the other values for $K$, in particular $K$ = {glue:}`kmin` (as suggested by cross-validation),
we can see it achieves this goal: it follows the increasing trend of house price
versus house size, but is not influenced too much by the idiosyncratic variations
in price. All of this is similar to how
the choice of $K$ affects K-nearest neighbors classification, as discussed in the previous
chapter.

```{code-cell} ipython3
:tags: [remove-cell]

# Changed from ...

# What about the plots in Figure \@ref(fig:07-howK) where $K$ is quite large, 
# say, $K$ = 250 or 932? 
```

## Evaluating on the test set

To assess how well our model might do at predicting on unseen data, we will
assess its RMSPE on the test data. To do this, we will first
re-train our KNN regression model on the entire training data set, 
using $K =$ {glue:}`kmin` neighbors. Then we will
use `predict` to make predictions on the test data, and use the `mean_squared_error`
function to compute the mean squared prediction error (MSPE). Finally take the 
square root of MSPE to get RMSPE. The reason that we do not use `score` method (as we
did in Classification chapter) is that the scoring metric `score` uses for `KNeighborsRegressor` 
is $R^2$ instead of RMSPE.

```{code-cell} ipython3
:tags: [remove-cell]

# To assess how well our model might do at predicting on unseen data, we will
# assess its RMSPE on the test data. To do this, we will first
# re-train our KNN regression model on the entire training data set, 
# using $K =$ `r sacr_min |> pull(neighbors)` neighbors. Then we will
# use `predict` to make predictions on the test data, and use the `metrics`
# function again to compute the summary of regression quality. Because
# we specify that we are performing regression in `set_mode`, the `metrics`
# function knows to output a quality summary related to regression, and not, say, classification.
```

```{code-cell} ipython3
kmin = int(sacr_min["param_kneighborsregressor__n_neighbors"])

# re-make the pipeline
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor(n_neighbors=kmin))

# create the predictor and target
X = sacramento_train[["sqft"]]
y = sacramento_train[["price"]]

# fit the pipeline object
sacr_pipeline.fit(X, y)

# predict on test data
sacr_preds = sacramento_test
sacr_preds = sacr_preds.assign(predicted=sacr_pipeline.predict(sacramento_test))

# calculate RMSPE
from sklearn.metrics import mean_squared_error

RMSPE = np.sqrt(
    mean_squared_error(y_true=sacr_preds["price"], y_pred=sacr_preds["predicted"])
)

RMSPE
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("test_RMSPE", "{0:,.0f}".format(int(RMSPE)))
glue("cv_RMSPE", "{0:,.0f}".format(int(sacr_min['RMSPE'])))
```

Our final model's test error as assessed by RMSPE 
is \$ {glue:text}`test_RMSPE`. 
Note that RMSPE is measured in the same units as the response variable.
In other words, on new observations, we expect the error in our prediction to be 
*roughly* \$ {glue:text}`test_RMSPE`. 
From one perspective, this is good news: this is about the same as the cross-validation
RMSPE estimate of our tuned model 
(which was \$ {glue:text}`cv_RMSPE`, 
so we can say that the model appears to generalize well
to new data that it has never seen before.
However, much like in the case of KNN classification, whether this value for RMSPE is *good*&mdash;i.e.,
whether an error of around \$ {glue:text}`test_RMSPE`
is acceptable&mdash;depends entirely on the application. 
In this application, this error
is not prohibitively large, but it is not negligible either; 
\$ {glue:text}`test_RMSPE`
might represent a substantial fraction of a home buyer's budget, and
could make or break whether or not they could afford put an offer on a house. 

Finally, {numref}`fig:07-predict-all` shows the predictions that our final model makes across
the range of house sizes we might encounter in the Sacramento area&mdash;from 500 to 5000 square feet. 
You have already seen a few plots like this in this chapter, but here we also provide the code that generated it
as a learning challenge.

```{code-cell} ipython3
:tags: [remove-output]

sacr_preds = pd.DataFrame({"sqft": np.arange(500, 5001, 10)})
sacr_preds = pd.concat(
    (sacr_preds, pd.DataFrame(sacr_pipeline.predict(sacr_preds), columns=["predicted"])),
    axis=1,
)

# the base plot: the training data scatter plot
base_plot = (
    alt.Chart(sacramento_train)
    .mark_circle(opacity=0.4, color="black")
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y("price", title="Price (USD)", axis=alt.Axis(format="$,.0f")),
    )
)

# add the prediction layer
plot_final = base_plot + alt.Chart(sacr_preds, title=f"K = {kmin}").mark_line(
    color="blue"
).encode(x="sqft", y="predicted")

plot_final
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-predict-all", plot_final)
```

:::{glue:figure} fig:07-predict-all
:name: fig:07-predict-all

Predicted values of house price (blue line) for the final KNN regression model.
:::

+++

## Multivariable KNN regression

As in KNN classification, we can use multiple predictors in KNN regression.
In this setting, we have the same concerns regarding the scale of the predictors. Once again, 
 predictions are made by identifying the $K$
observations that are nearest to the new point we want to predict; any
variables that are on a large scale will have a much larger effect than
variables on a small scale. Hence, we should re-define the preprocessor in the 
pipeline to incorporate all predictor variables.

```{code-cell} ipython3
:tags: [remove-cell]

# As in KNN classification, we can use multiple predictors in KNN regression.
# In this setting, we have the same concerns regarding the scale of the predictors. Once again, 
#  predictions are made by identifying the $K$
# observations that are nearest to the new point we want to predict; any
# variables that are on a large scale will have a much larger effect than
# variables on a small scale. But since the `recipe` we built above scales and centers
# all predictor variables, this is handled for us.
```

Note that we also have the same concern regarding the selection of predictors
in KNN regression as in KNN classification: having more predictors is **not** always
better, and the choice of which predictors to use has a potentially large influence
on the quality of predictions. Fortunately, we can use the predictor selection 
algorithm from the classification chapter in KNN regression as well.
As the algorithm is the same, we will not cover it again in this chapter.

We will now demonstrate a multivariable KNN regression \index{K-nearest neighbors!multivariable regression}  analysis of the 
Sacramento real estate \index{Sacramento real estate} data using `scikit-learn`. This time we will use
house size (measured in square feet) as well as number of bedrooms as our
predictors, and continue to use house sale price as our outcome/target variable
that we are trying to predict.
It is always a good practice to do exploratory data analysis, such as
visualizing the data, before we start modeling the data. {numref}`fig:07-bedscatter`
shows that the number of bedrooms might provide useful information
to help predict the sale price of a house.

```{code-cell} ipython3
:tags: [remove-output]

plot_beds = (
    alt.Chart(sacramento)
    .mark_circle(opacity=0.4, color="black")
    .encode(
        x=alt.X("beds", title="Number of Bedrooms"),
        y=alt.Y("price", title="Price (USD)", axis=alt.Axis(format="$,.0f")),
    )
)

plot_beds
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-bedscatter", plot_beds)
```

:::{glue:figure} fig:07-bedscatter
:name: fig:07-bedscatter

Scatter plot of the sale price of houses versus the number of bedrooms.
:::

+++

{numref}`fig:07-bedscatter` shows that as the number of bedrooms increases,
the house sale price tends to increase as well, but that the relationship
is quite weak. Does adding the number of bedrooms
to our model improve our ability to predict price? To answer that
question, we will have to create a new KNN regression
model using house size and number of bedrooms, and then we can compare it to
the model we previously came up with that only used house
size. Let's do that now!

First we'll build a new model specification and preprocessor for the analysis. Note that
we pass in `["sqft", "beds"]` to denote that we have two predictors in `make_column_transformer`.
Moreover, we do not specify `n_neighbors` in `KNeighborsRegressor` inside the pipeline, and pass this pipeline to the `GridSearchCV` to tell `scikit-learn` to tune the number of neighbors for us.

```{code-cell} ipython3
:tags: [remove-cell]

# First we'll build a new model specification and recipe for the analysis. Note that
# we use the formula `price ~ sqft + beds` to denote that we have two predictors,
# and set `neighbors = tune()` to tell `tidymodels` to tune the number of neighbors for us.
```

```{code-cell} ipython3
# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft", "beds"]))
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor())

# create the predictor and target
X = sacramento_train[["sqft", "beds"]]
y = sacramento_train[["price"]]
```

Next, we'll use 5-fold cross-validation through `GridSearchCV` to choose the number of neighbors via the minimum RMSPE:

```{code-cell} ipython3
# create the 5-fold GridSearchCV object
param_grid = {
    "kneighborsregressor__n_neighbors": range(1, 201),
}
sacr_gridsearch = GridSearchCV(
    estimator=sacr_pipeline,
    param_grid=param_grid,
    n_jobs=-1,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

# fit the GridSearchCV object
sacr_gridsearch.fit(X, y)

# retrieve the CV scores
sacr_results = pd.DataFrame(sacr_gridsearch.cv_results_)
# negate mean_test_score (negative RMSPE) to get the actual RMSPE
sacr_results["RMSPE"] = -sacr_results["mean_test_score"]

# show only the row of minimum RMSPE
sacr_multi = sacr_results[sacr_results["RMSPE"] == min(sacr_results["RMSPE"])]
sacr_k = int(sacr_multi["param_kneighborsregressor__n_neighbors"])

sacr_multi
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sacr_k", sacr_k)
glue("cv_RMSPE_2pred", "{0:,.0f}".format(int(sacr_multi["RMSPE"])))
```

Here we see that the smallest estimated RMSPE from cross-validation occurs when $K =$ {glue:}`sacr_k`.
If we want to compare this multivariable KNN regression model to the model with only a single
predictor *as part of the model tuning process* (e.g., if we are running forward selection as described
in the chapter on evaluating and tuning classification models),
then we must compare the accuracy estimated using only the training data via cross-validation.
Looking back, the estimated cross-validation accuracy for the single-predictor 
model was {glue:}`cv_RMSPE`.
The estimated cross-validation accuracy for the multivariable model is
{glue:text}`cv_RMSPE_2pred`.
Thus in this case, we did not improve the model 
by a large amount by adding this additional predictor.

Regardless, let's continue the analysis to see how we can make predictions with a multivariable KNN regression model
and evaluate its performance on test data. We first need to re-train the model on the entire
training data set with $K =$ {glue:}`sacr_k`, and then use that model to make predictions
on the test data.

```{code-cell} ipython3
# re-make the pipeline
sacr_pipeline_mult = make_pipeline(
    sacr_preprocessor, KNeighborsRegressor(n_neighbors=sacr_k)
)

# fit the pipeline object
sacr_pipeline_mult.fit(X, y)

# predict on test data
sacr_preds = sacramento_test
sacr_preds = sacr_preds.assign(predicted=sacr_pipeline_mult.predict(sacramento_test))

# calculate RMSPE
from sklearn.metrics import mean_squared_error

RMSPE_mult = np.sqrt(
    mean_squared_error(y_true=sacr_preds["price"], y_pred=sacr_preds["predicted"])
)

RMSPE_mult
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("RMSPE_mult", "{0:,.0f}".format(RMSPE_mult))
```

This time, when we performed KNN regression on the same data set, but also
included number of bedrooms as a predictor, we obtained a RMSPE test error 
of {glue:text}`RMSPE_mult`.
{numref}`fig:07-knn-mult-viz` visualizes the model's predictions overlaid on top of the data. This 
time the predictions are a surface in 3D space, instead of a line in 2D space, as we have 2
predictors instead of 1.

```{code-cell} ipython3
:tags: [remove-cell]

# create a prediction pt grid
xvals = np.linspace(
    sacramento_train["sqft"].min(), sacramento_train["sqft"].max(), 50
)
yvals = np.linspace(
    sacramento_train["beds"].min(), sacramento_train["beds"].max(), 50
)
xygrid = np.array(np.meshgrid(xvals, yvals)).reshape(2, -1).T
xygrid = pd.DataFrame(xygrid, columns=["sqft", "beds"])

# add prediction
knnPredGrid = sacr_pipeline_mult.predict(xygrid)

fig = px.scatter_3d(
    sacramento_train,
    x="sqft",
    y="beds",
    z="price",
    opacity=0.4,
    labels={"sqft": "Size (sq ft)", "beds": "Bedrooms", "price": "Price (USD)"},
)

fig.update_traces(marker={"size": 2, "color": "red"})

fig.add_trace(
    go.Surface(
        x=xvals,
        y=yvals,
        z=knnPredGrid.reshape(50, -1),
        name="Predictions",
        colorscale="viridis",
        colorbar={"title": "Price (USD)"}
    )
)

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=1),
    template="plotly_white",
)

plot(fig, filename="img/regression1/fig07-knn-mult-viz.html", auto_open=False)
```

```{code-cell} ipython3
:tags: [remove-input]

display(HTML("img/regression1/fig07-knn-mult-viz.html"))
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:07-knn-mult-viz
:figclass: caption-hack

KNN regression model’s predictions represented as a surface in 3D space overlaid on top of the data using three predictors (price, house size, and the number of bedrooms). Note that in general we recommend against using 3D visualizations; here we use a 3D visualization only to illustrate what the surface of predictions looks like for learning purposes.
```

+++

We can see that the predictions in this case, where we have 2 predictors, form
a surface instead of a line. Because the newly added predictor (number of bedrooms) is 
related to price (as price changes, so does number of bedrooms)
and is not totally determined by house size (our other predictor),
we get additional and useful information for making our
predictions. For example, in this model we would predict that the cost of a
house with a size of 2,500 square feet generally increases slightly as the number
of bedrooms increases. Without having the additional predictor of number of
bedrooms, we would predict the same price for these two houses.

+++

## Strengths and limitations of KNN regression

As with KNN classification (or any prediction algorithm for that matter), KNN 
regression has both strengths and weaknesses. Some are listed here:

**Strengths:** K-nearest neighbors regression

1. is a simple, intuitive algorithm,
2. requires few assumptions about what the data must look like, and 
3. works well with non-linear relationships (i.e., if the relationship is not a straight line).

**Weaknesses:** K-nearest neighbors regression

1. becomes very slow as the training data gets larger,
2. may not perform well with a large number of predictors, and
3. may not predict well beyond the range of values input in your training data.

+++

## Exercises

Practice exercises for the material covered in this chapter 
can be found in the accompanying 
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the "Regression I: K-nearest neighbors" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in Chapter {ref}`move-to-your-own-machine`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

## References

+++

```{bibliography}
```
