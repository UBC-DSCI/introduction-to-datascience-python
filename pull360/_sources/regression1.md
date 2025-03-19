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

from chapter_preamble import *
from IPython.display import HTML
from IPython.display import Image
import plotly.express as px
import plotly.graph_objects as go
```

## Overview

This chapter continues our foray into answering predictive questions.
Here we will focus on predicting *numerical* variables
and will use *regression* to perform this task.
This is unlike the past two chapters, which focused on predicting categorical
variables via classification. However, regression does have many similarities
to classification: for example, just as in the case of classification,
we will split our data into training, validation, and test sets, we will
use `scikit-learn` workflows, we will use a K-nearest neighbors (K-NN)
approach to make predictions, and we will use cross-validation to choose K.
Because of how similar these procedures are, make sure to read
{numref}`Chapters %s <classification1>` and {numref}`%s <classification2>` before reading
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

- Recognize situations where a regression analysis would be appropriate for making predictions.
- Explain the K-nearest neighbors (K-NN) regression algorithm and describe how it differs from K-NN classification.
- Interpret the output of a K-NN regression.
- In a data set with two or more variables, perform K-nearest neighbors regression in Python.
- Evaluate K-NN regression prediction quality in Python using the root mean squared prediction error (RMSPE).
- Estimate the RMSPE in Python using cross-validation or a test set.
- Choose the number of neighbors in K-nearest neighbors regression by minimizing estimated cross-validation RMSPE.
- Describe underfitting and overfitting, and relate it to the number of neighbors in K-nearest neighbors regression.
- Describe the advantages and disadvantages of K-nearest neighbors regression.

+++

## The regression problem

```{index} predictive question, response variable
```

Regression, like classification, is a predictive problem setting where we want
to use past information to predict future observations. But in the case of
regression, the goal is to predict *numerical* values instead of *categorical* values.
The variable that you want to predict is often called the *response variable*.
For example, we could try to use the number of hours a person spends on
exercise each week to predict their race time in the annual Boston marathon. As
another example, we could try to use the size of a house to
predict its sale price. Both of these response variables&mdash;race time and sale price&mdash;are
numerical, and so predicting them given past data is considered a regression problem.

```{index} classification; comparison to regression
```

```{index} regression; comparison to classification
```

Just like in the classification setting, there are many possible methods that we can use
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

```{index} categorical variable, numerical variable
```

```{note}
You can usually tell whether a variable is numerical or
categorical&mdash;and therefore whether you need to perform regression or
classification&mdash;by taking the response variable for two observations X and Y from your data,
and asking the question, "is response variable X *more* than response
variable Y?" If the variable is categorical, the question will make no sense.
(Is blue more than red?  Is benign more than malignant?) If the variable is
numerical, it will make sense. (Is 1.5 hours more than 2.25 hours? Is
\$500,000 more than \$400,000?) Be careful when applying this heuristic,
though: sometimes categorical variables will be encoded as numbers in your
data (e.g., "1" represents "benign", and "0" represents "malignant"). In
these cases you have to ask the question about the *meaning* of the labels
("benign" and "malignant"), not their values ("1" and "0").
```

+++

## Exploring a data set

```{index} Sacramento real estate, question; regression
```

In this chapter and the next, we will study
a data set of
[932 real estate transactions in Sacramento, California](https://support.spatialkey.com/spatialkey-sample-csv-data/)
originally reported in the *Sacramento Bee* newspaper.
We first need to formulate a precise question that
we want to answer. In this example, our question is again predictive:
Can we use the size of a house in the Sacramento, CA area to predict
its sale price? A rigorous, quantitative answer to this question might help
a realtor advise a client as to whether the price of a particular listing
is fair, or perhaps how to set the price of a new listing.
We begin the analysis by loading and examining the data,
as well as setting the seed value.

```{index} seed;numpy.random.seed
```

```{code-cell} ipython3
import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

# Output dataframes instead of arrays
set_config(transform_output="pandas")

np.random.seed(10)

sacramento = pd.read_csv("data/sacramento.csv")
sacramento
```

```{index} altair; mark_circle, visualization; scatter
```

The scientific question guides our initial exploration: the columns in the
data that we are interested in are `sqft` (house size, in livable square feet)
and `price` (house sale price, in US dollars (USD)).  The first step is to visualize
the data as a scatter plot where we place the predictor variable
(house size) on the x-axis, and we place the response variable that we
want to predict (sale price) on the y-axis.

```{note}
Given that the y-axis unit is dollars in {numref}`fig:07-edaRegr`,
we format the axis labels to put dollar signs in front of the house prices,
as well as commas to increase the readability of the larger numbers.
We can do this in `altair` by using `.axis(format="$,.0f")` on
the `y` encoding channel.
```

```{code-cell} ipython3
:tags: [remove-output]

scatter = alt.Chart(sacramento).mark_circle().encode(
    x=alt.X("sqft")
        .scale(zero=False)
        .title("House size (square feet)"),
    y=alt.Y("price")
        .axis(format="$,.0f")
        .title("Price (USD)")
)

scatter
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("fig:07-edaRegr", scatter)
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

```{index} K-nearest neighbors, K-nearest neighbors; regression
```

Much like in the case of classification,
we can use a K-nearest neighbors-based
approach in regression to make predictions.
Let's take a small sample of the data in {numref}`fig:07-edaRegr`
and walk through how K-nearest neighbors (K-NN) works
in a regression context before we dive in to creating our model and assessing
how well it predicts house sale price. This subsample is taken to allow us to
illustrate the mechanics of K-NN regression with a few data points; later in
this chapter we will use all the data.

```{index} DataFrame; sample
```

To take a small random sample of size 30, we'll use the
`sample` method on the `sacramento` data frame, specifying
that we want to select `n=30` rows.

```{code-cell} ipython3
small_sacramento = sacramento.sample(n=30)
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

small_plot = alt.Chart(small_sacramento).mark_circle(opacity=1).encode(
    x=alt.X("sqft")
        .scale(zero=False)
        .title("House size (square feet)"),
    y=alt.Y("price")
        .axis(format="$,.0f")
        .title("Price (USD)")
)

# add an overlay to the base plot
line_df = pd.DataFrame({"x": [2000]})
rule = alt.Chart(line_df).mark_rule(strokeDash=[6], size=1.5, color="black").encode(x="x")

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

```{index} DataFrame; abs, DataFrame; nsmallest
```

We will employ the same intuition from {numref}`Chapters %s <classification1>` and {numref}`%s <classification2>`, and use the
neighboring points to the new point of interest to suggest/predict what its
sale price might be.
For the example shown in {numref}`fig:07-small-eda-regr`,
we find and label the 5 nearest neighbors to our observation
of a house that is 2,000 square feet.

```{code-cell} ipython3
small_sacramento["dist"] = (2000 - small_sacramento["sqft"]).abs()
nearest_neighbors = small_sacramento.nsmallest(5, "dist")
nearest_neighbors
```

```{code-cell} ipython3
:tags: [remove-cell]

nn_plot = small_plot + rule

# plot horizontal lines which is perpendicular to x=2000
h_lines = []
for i in range(5):
    h_line_df = pd.DataFrame({
        "sqft": [nearest_neighbors.iloc[i, 4], 2000],
        "price": [nearest_neighbors.iloc[i, 6]] * 2
    })
    h_lines.append(alt.Chart(h_line_df).mark_line(color="black").encode(x="sqft", y="price"))

# highlight the nearest neighbors in orange
orange_neighbrs = alt.Chart(nearest_neighbors).mark_circle(opacity=1, color="#ff7f0e").encode(
    x=alt.X("sqft")
        .scale(zero=False)
        .title("House size (square feet)"),
    y=alt.Y("price")
        .axis(format="$,.0f")
        .title("Price (USD)")
)

nn_plot = alt.layer(*h_lines, small_plot, orange_neighbrs, rule)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-knn5-example", nn_plot)
```

:::{glue:figure} fig:07-knn5-example
:name: fig:07-knn5-example

Scatter plot of price (USD) versus house size (square feet) with lines to 5 nearest neighbors (highlighted in orange).
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
).mark_circle(size=80, opacity=1, color="#d62728").encode(x="sqft", y="price")
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
unanswered questions here with K-NN regression that we had with K-NN
classification: which $K$ do we choose, and is our model any good at making
predictions? In the next few sections, we will address these questions in the
context of K-NN regression.

One strength of the K-NN regression algorithm
that we would like to draw attention to at this point
is its ability to work well with non-linear relationships
(i.e., if the relationship is not a straight line).
This stems from the use of nearest neighbors to predict values.
The algorithm really has very few assumptions
about what the data must look like for it to work.

+++

## Training, evaluating, and tuning the model

```{index} training set, test set
```

As usual, we must start by putting some test data away in a lock box
that we will come back to only after we choose our final model.
Let's take care of that now.
Note that for the remainder of the chapter
we'll be working with the entire Sacramento data set,
as opposed to the smaller sample of 30 points
that we used earlier in the chapter ({numref}`fig:07-small-eda-regr`).

+++

```{note}
We are not specifying the `stratify` argument here like we did in
{numref}`Chapter %s <classification2>`, since
the `train_test_split` function cannot stratify based on a
quantitative variable.
```

```{code-cell} ipython3
:tags: [remove-cell]
# fix seed right before train/test split for reproducibility with next chapter
# make sure this seed is always the same as the one used before the split in Regression 2
np.random.seed(1)
```

```{code-cell} ipython3
sacramento_train, sacramento_test = train_test_split(
    sacramento, train_size=0.75
)
```

```{index} cross-validation, RMSPE
```

```{index} see: root mean square prediction error; RMSPE
```

Next, we'll use cross-validation to choose $K$. In K-NN classification, we used
accuracy to see how well our predictions matched the true labels. We cannot use
the same metric in the regression setting, since our predictions will almost never
*exactly* match the true response variable values. Therefore in the
context of K-NN regression we will use root mean square prediction error (RMSPE) instead.
The mathematical formula for calculating RMSPE is:

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

from sklearn.neighbors import KNeighborsRegressor

# (synthetic) new prediction points
pts = pd.DataFrame({"sqft": [1200, 1850, 2250], "price": [300000, 200000, 500000]})
finegrid = pd.DataFrame({"sqft": np.arange(600, 3901, 10)})

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
    (small_sacramento[["sqft", "price"]].reset_index(), pd.DataFrame(sacr_pipeline.predict(small_sacramento[["sqft", "price"]]), columns=["predicted"])),
    axis=1,
).drop(columns=["index"])

# to make altair mark_line works, need to create separate dataframes for each vertical error line
errors_plot = (
    small_plot
    + alt.Chart(sacr_full_preds_hid).mark_line(color="#ff7f0e").encode(x="sqft", y="predicted")
    + alt.Chart(sacr_new_preds_hid)
    .mark_circle(opacity=1)
    .encode(x="sqft", y="price")
)
sacr_new_preds_melted_df = sacr_new_preds_hid.melt(id_vars=["sqft"])
v_lines = []
for i in sacr_new_preds_hid["sqft"]:
    line_df = sacr_new_preds_melted_df.query(f"sqft == {i}")
    v_lines.append(alt.Chart(line_df).mark_line(color="black").encode(x="sqft", y="value"))

errors_plot = alt.layer(*v_lines, errors_plot)
errors_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-verticalerrors", errors_plot, display=False)
```

:::{glue:figure} fig:07-verticalerrors
:name: fig:07-verticalerrors

Scatter plot of price (USD) versus house size (square feet) with example predictions (orange line) and the error in those predictions compared with true response values (vertical lines).
:::

+++

```{index} RMSPE; comparison with RMSE
```

```{note}
When using many code packages, the evaluation output
we will get to assess the prediction quality of
our K-NN regression models is labeled "RMSE", or "root mean squared
error". Why is this so, and why not RMSPE?
In statistics, we try to be very precise with our
language to indicate whether we are calculating the prediction error on the
training data (*in-sample* prediction) versus on the testing data
(*out-of-sample* prediction). When predicting and evaluating prediction quality on the training data, we
say RMSE. By contrast, when predicting and evaluating prediction quality
on the testing or validation data, we say RMSPE.
The equation for calculating RMSE and RMSPE is exactly the same; all that changes is whether the $y$s are
training or testing data. But many people just use RMSE for both,
and rely on context to denote which data the root mean squared error is being calculated on.
```

```{index} scikit-learn, scikit-learn; Pipeline, scikit-learn; make_pipeline, scikit-learn; make_column_transformer
```

Now that we know how we can assess how well our model predicts a numerical
value, let's use Python to perform cross-validation and to choose the optimal
$K$.  First, we will create a column transformer for preprocessing our data.  Note
that we include standardization in our preprocessing to build good habits, but
since we only have one predictor, it is technically not necessary; there is no
risk of comparing two predictors of different scales.  Next we create a model
pipeline for K-nearest neighbors regression. Note that we use the
`KNeighborsRegressor` model object now to denote a regression problem, as
opposed to the classification problems from the previous chapters.  The use of
`KNeighborsRegressor` essentially tells `scikit-learn` that we need to use
different metrics (instead of accuracy) for tuning and evaluation.  Next we
specify a parameter grid containing numbers of neighbors
ranging from 1 to 200.  Then we create a 5-fold `GridSearchCV` object, and
pass in the pipeline and parameter grid.
There is one additional slight complication: unlike classification models in `scikit-learn`---which
by default use accuracy for tuning, as desired---regression models in `scikit-learn`
do not use the RMSPE for tuning by default.
So we need to specify that we want to use the RMSPE for tuning by setting the
`scoring` argument to `"neg_root_mean_squared_error"`.

```{note}
We obtained the identifier of the parameter representing the number
of neighbours, `"kneighborsregressor__n_neighbors"` by examining the output
of `sacr_pipeline.get_params()`, as we did in {numref}`Chapter %s <classification1>`.
```

```{index} scikit-learn; GridSearchCV
```

```{code-cell} ipython3
# import the K-NN regression model
from sklearn.neighbors import KNeighborsRegressor

# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor())

# create the 5-fold GridSearchCV object
param_grid = {
    "kneighborsregressor__n_neighbors": range(1, 201, 3),
}
sacr_gridsearch = GridSearchCV(
    estimator=sacr_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
)
```

Next, we use the run cross validation by calling the `fit` method
on `sacr_gridsearch`. Note the use of two brackets for the input features
(`sacramento_train[["sqft"]]`), which creates a data frame with a single column.
As we learned in {numref}`Chapter %s <wrangling>`, we can obtain a data frame with a
subset of columns by passing a list of column names; `["sqft"]` is a list with one
item, so we obtain a data frame with one column. If instead we used
just one bracket (`sacramento_train["sqft"]`), we would obtain a series.
In `scikit-learn`, it is easier to work with the input features as a data frame
rather than a series, so we opt for two brackets here. On the other hand, the response variable
can be a series, so we use just one bracket there (`sacramento_train["price"]`).

As in {numref}`Chapter %s <classification2>`, once the model has been fit
we will wrap the `cv_results_` output in a data frame, extract
only the relevant columns, compute the standard error based on 5 folds,
and rename the parameter column to be more readable.


```{code-cell} ipython3
# fit the GridSearchCV object
sacr_gridsearch.fit(
    sacramento_train[["sqft"]],  # A single-column data frame
    sacramento_train["price"]  # A series
)

# Retrieve the CV scores
sacr_results = pd.DataFrame(sacr_gridsearch.cv_results_)
sacr_results["sem_test_score"] = sacr_results["std_test_score"] / 5**(1/2)
sacr_results = (
    sacr_results[[
        "param_kneighborsregressor__n_neighbors",
        "mean_test_score",
        "sem_test_score"
    ]]
    .rename(columns={"param_kneighborsregressor__n_neighbors": "n_neighbors"})
)
sacr_results
```

In the `sacr_results` results data frame, we see that the
`n_neighbors` variable contains the values of $K$,
and `mean_test_score` variable contains the value of the RMSPE estimated via
cross-validation...Wait a moment! Isn't the RMSPE supposed to be nonnegative?
Recall that when we specified the `scoring` argument in the `GridSearchCV` object,
we used the value `"neg_root_mean_squared_error"`. See the `neg_` at the start?
That stands for *negative*! As it turns out, `scikit-learn` always tries to *maximize* a score
when it tunes a model. But we want to *minimize* the RMSPE when we tune a regression
model. So `scikit-learn` gets around this by working with the *negative* RMSPE instead.
It is a little convoluted, but we need to add one more step to convert the negative
RMSPE back to the regular RMSPE.

```{code-cell} ipython3
sacr_results["mean_test_score"] = -sacr_results["mean_test_score"]
sacr_results
```

Alright, now the `mean_test_score` variable actually has values of the RMSPE
for different numbers of neighbors. Finally, the `sem_test_score` variable
contains the standard error of our cross-validation RMSPE estimate, which
is a measure of how uncertain we are in the mean value. Roughly, if
your estimated mean RMSPE is \$100,000 and standard error is \$1,000, you can expect the
*true* RMSPE to be somewhere roughly between \$99,000 and \$101,000 (although it
may fall outside this range).

{numref}`fig:07-choose-k-knn-plot` visualizes how the RMSPE varies with the number of neighbors $K$.
We take the *minimum* RMSPE to find the best setting for the number of neighbors.
The smallest RMSPE occurs when $K$ is {glue:text}`best_k_sacr`.

```{code-cell} ipython3
:tags: [remove-cell]
best_k_sacr = sacr_results["n_neighbors"][sacr_results["mean_test_score"].idxmin()]
best_cv_RMSPE = min(sacr_results["mean_test_score"])
glue("best_k_sacr", "{:d}".format(best_k_sacr))
glue("cv_RMSPE", "{0:,.0f}".format(best_cv_RMSPE))
```

```{code-cell} ipython3
:tags: [remove-cell]

sacr_tunek_plot = alt.Chart(sacr_results).mark_line(point=True).encode(
    x=alt.X("n_neighbors:Q", title="Neighbors"),
    y=alt.Y("mean_test_score", scale=alt.Scale(zero=False), title="Cross-Validation RMSPE Estimate")
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

To see which parameter value corresponds to the minimum RMSPE,
we can also access the `best_params_` attribute of the original fit `GridSearchCV` object.
Note that it is still useful to visualize the results as we did above
since this provides additional information on how the model performance varies.

```{code-cell} ipython3
sacr_gridsearch.best_params_
```

+++

## Underfitting and overfitting
Similar to the setting of classification, by setting the number of neighbors
to be too small or too large, we cause the RMSPE to increase, as shown in
{numref}`fig:07-choose-k-knn-plot`. What is happening here?

{numref}`fig:07-howK` visualizes the effect of different settings of $K$ on the
regression model. Each plot shows the predicted values for house sale price from
our K-NN regression model for 6 different values for $K$: 1, 3, 25, {glue:text}`best_k_sacr`, 250, and 699 (i.e., all of the training data).
For each model, we predict prices for the range of possible home sizes we
observed in the data set (here 500 to 5,000 square feet) and we plot the
predicted prices as a orange line.

```{code-cell} ipython3
:tags: [remove-cell]

gridvals = [
    1,
    3,
    25,
    best_k_sacr,
    250,
    len(sacramento_train),
]

plots = list()

sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
X = sacramento_train[["sqft"]]
y = sacramento_train[["price"]]

base_plot = (
    alt.Chart(sacramento_train)
    .mark_circle()
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
        .mark_line(color="#ff7f0e")
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

Predicted values for house price (represented as a orange line) from K-NN regression models for six different values for $K$.
:::

+++

```{index} overfitting; regression
```

{numref}`fig:07-howK` shows that when $K$ = 1, the orange line runs perfectly
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
in the context of regression.

```{index} underfitting; regression
```

What about the plots in {numref}`fig:07-howK` where $K$ is quite large,
say, $K$ = 250 or 699?
In this case the orange line becomes extremely smooth, and actually becomes flat
once $K$ is equal to the number of datapoints in the entire data set.
This happens because our predicted values for a given x value (here, home
size), depend on many neighboring observations; in the case where $K$ is equal
to the size of the data set, the prediction is just the mean of the house prices
in the data set (completely ignoring the house size).
In contrast to the $K=1$ example,
the smooth, inflexible orange line does not follow the training observations very closely.
In other words, the model is *not influenced enough* by the training data.
Recall from the classification
chapters that this behavior is called *underfitting*; we again use this same
term in the context of regression.

Ideally, what we want is neither of the two situations discussed above. Instead,
we would like a model that (1) follows the overall "trend" in the training data, so the model
actually uses the training data to learn something useful, and (2) does not follow
the noisy fluctuations, so that we can be confident that our model will transfer/generalize
well to other new data. If we explore
the other values for $K$, in particular $K$ = {glue:text}`best_k_sacr` (as suggested by cross-validation),
we can see it achieves this goal: it follows the increasing trend of house price
versus house size, but is not influenced too much by the idiosyncratic variations
in price. All of this is similar to how
the choice of $K$ affects K-nearest neighbors classification, as discussed in the previous
chapter.

## Evaluating on the test set

To assess how well our model might do at predicting on unseen data, we will
assess its RMSPE on the test data. To do this, we first need to retrain the
K-NN regression model on the entire training data set using $K =$ {glue:text}`best_k_sacr`
neighbors. As we saw in {numref}`Chapter %s <classification2>` we do not have to do this ourselves manually; `scikit-learn`
does it for us automatically. To make predictions with the best model on the test data,
we can use the `predict` method of the fit `GridSearchCV` object.
We then use the `mean_squared_error` function (with the `y_true` and `y_pred` arguments)
to compute the mean squared prediction error, and finally take the
square root to get the RMSPE. The reason that we do not just use the `score`
method---as in {numref}`Chapter %s <classification2>`---is that the `KNeighborsRegressor`
model uses a different default scoring metric than the RMSPE.

```{code-cell} ipython3
from sklearn.metrics import mean_squared_error

sacramento_test["predicted"] = sacr_gridsearch.predict(sacramento_test)
RMSPE = mean_squared_error(
    y_true=sacramento_test["price"],
    y_pred=sacramento_test["predicted"]
)**(1/2)
RMSPE
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("test_RMSPE", "{0:,.0f}".format(RMSPE))
```

Our final model's test error as assessed by RMSPE
is \${glue:text}`test_RMSPE`.
Note that RMSPE is measured in the same units as the response variable.
In other words, on new observations, we expect the error in our prediction to be
*roughly* \${glue:text}`test_RMSPE`.
From one perspective, this is good news: this is about the same as the cross-validation
RMSPE estimate of our tuned model
(which was \${glue:text}`cv_RMSPE`,
so we can say that the model appears to generalize well
to new data that it has never seen before.
However, much like in the case of K-NN classification, whether this value for RMSPE is *good*&mdash;i.e.,
whether an error of around \${glue:text}`test_RMSPE`
is acceptable&mdash;depends entirely on the application.
In this application, this error
is not prohibitively large, but it is not negligible either;
\${glue:text}`test_RMSPE`
might represent a substantial fraction of a home buyer's budget, and
could make or break whether or not they could afford put an offer on a house.

Finally, {numref}`fig:07-predict-all` shows the predictions that our final
model makes across the range of house sizes we might encounter in the
Sacramento area.
Note that instead of predicting the house price only for those house sizes that happen to appear in our data,
we predict it for evenly spaced values between the minimum and maximum in the data set
(roughly 500 to 5000 square feet).
We superimpose this prediction line on a scatter
plot of the original housing price data,
so that we can qualitatively assess if the model seems to fit the data well.
You have already seen a
few plots like this in this chapter, but here we also provide the code that
generated it as a learning opportunity.

```{code-cell} ipython3
:tags: [remove-output]

# Create a grid of evenly spaced values along the range of the sqft data
sqft_prediction_grid = pd.DataFrame({
    "sqft": np.arange(sacramento["sqft"].min(), sacramento["sqft"].max(), 10)
})
# Predict the price for each of the sqft values in the grid
sqft_prediction_grid["predicted"] = sacr_gridsearch.predict(sqft_prediction_grid)

# Plot all the houses
base_plot = alt.Chart(sacramento).mark_circle(opacity=0.4).encode(
    x=alt.X("sqft")
        .scale(zero=False)
        .title("House size (square feet)"),
    y=alt.Y("price")
        .axis(format="$,.0f")
        .title("Price (USD)")
)

# Add the predictions as a line
sacr_preds_plot = base_plot + alt.Chart(
    sqft_prediction_grid,
    title=f"K = {best_k_sacr}"
).mark_line(
    color="#ff7f0e"
).encode(
    x="sqft",
    y="predicted"
)

sacr_preds_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:07-predict-all", sacr_preds_plot)
```

:::{glue:figure} fig:07-predict-all
:name: fig:07-predict-all

Predicted values of house price (orange line) for the final K-NN regression model.
:::

+++

## Multivariable K-NN regression

As in K-NN classification, we can use multiple predictors in K-NN regression.
In this setting, we have the same concerns regarding the scale of the predictors. Once again,
 predictions are made by identifying the $K$
observations that are nearest to the new point we want to predict; any
variables that are on a large scale will have a much larger effect than
variables on a small scale. Hence, we should re-define the preprocessor in the
pipeline to incorporate all predictor variables.

Note that we also have the same concern regarding the selection of predictors
in K-NN regression as in K-NN classification: having more predictors is **not** always
better, and the choice of which predictors to use has a potentially large influence
on the quality of predictions. Fortunately, we can use the predictor selection
algorithm from {numref}`Chapter %s <classification2>` in K-NN regression as well.
As the algorithm is the same, we will not cover it again in this chapter.

```{index} K-nearest neighbors; multivariable regression, Sacramento real estate
```

We will now demonstrate a multivariable K-NN regression analysis of the
Sacramento real estate data using `scikit-learn`. This time we will use
house size (measured in square feet) as well as number of bedrooms as our
predictors, and continue to use house sale price as our response variable
that we are trying to predict.
It is always a good practice to do exploratory data analysis, such as
visualizing the data, before we start modeling the data. {numref}`fig:07-bedscatter`
shows that the number of bedrooms might provide useful information
to help predict the sale price of a house.

```{code-cell} ipython3
:tags: [remove-output]

plot_beds = alt.Chart(sacramento).mark_circle().encode(
    x=alt.X("beds").title("Number of Bedrooms"),
    y=alt.Y("price").title("Price (USD)").axis(format="$,.0f"),
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
question, we will have to create a new K-NN regression
model using house size and number of bedrooms, and then we can compare it to
the model we previously came up with that only used house
size. Let's do that now!

First we'll build a new model object and preprocessor for the analysis.
Note that we pass the list `["sqft", "beds"]` into the `make_column_transformer`
function to denote that we have two predictors.  Moreover, we do not specify `n_neighbors` in
`KNeighborsRegressor`, indicating that we want this parameter to be tuned by `GridSearchCV`.

```{code-cell} ipython3
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft", "beds"]))
sacr_pipeline = make_pipeline(sacr_preprocessor, KNeighborsRegressor())
```

Next, we'll use 5-fold cross-validation with a `GridSearchCV` object
to choose the number of neighbors via the minimum RMSPE:

```{code-cell} ipython3
# create the 5-fold GridSearchCV object
param_grid = {
    "kneighborsregressor__n_neighbors": range(1, 50),
}

sacr_gridsearch = GridSearchCV(
    estimator=sacr_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

sacr_gridsearch.fit(
  sacramento_train[["sqft", "beds"]],
  sacramento_train["price"]
)

# retrieve the CV scores
sacr_results = pd.DataFrame(sacr_gridsearch.cv_results_)
sacr_results["sem_test_score"] = sacr_results["std_test_score"] / 5**(1/2)
sacr_results["mean_test_score"] = -sacr_results["mean_test_score"]
sacr_results = (
    sacr_results[[
        "param_kneighborsregressor__n_neighbors",
        "mean_test_score",
        "sem_test_score"
    ]]
    .rename(columns={"param_kneighborsregressor__n_neighbors" : "n_neighbors"})
)

# show only the row of minimum RMSPE
sacr_results.nsmallest(1, "mean_test_score")
```

```{code-cell} ipython3
:tags: [remove-cell]

best_k_sacr_multi = sacr_results["n_neighbors"][sacr_results["mean_test_score"].idxmin()]
min_rmspe_sacr_multi = min(sacr_results["mean_test_score"])
glue("best_k_sacr_multi", "{:d}".format(best_k_sacr_multi))
glue("cv_RMSPE_2pred", "{0:,.0f}".format(min_rmspe_sacr_multi))
```

Here we see that the smallest estimated RMSPE from cross-validation occurs when $K =$ {glue:text}`best_k_sacr_multi`.
If we want to compare this multivariable K-NN regression model to the model with only a single
predictor *as part of the model tuning process* (e.g., if we are running forward selection as described
in the chapter on evaluating and tuning classification models),
then we must compare the RMSPE estimated using only the training data via cross-validation.
Looking back, the estimated cross-validation RMSPE for the single-predictor
model was \${glue:text}`cv_RMSPE`.
The estimated cross-validation RMSPE for the multivariable model is
\${glue:text}`cv_RMSPE_2pred`.
Thus in this case, we did not improve the model
by a large amount by adding this additional predictor.

Regardless, let's continue the analysis to see how we can make predictions with a multivariable K-NN regression model
and evaluate its performance on test data. As previously, we will use the best model to make predictions on the test data
via the `predict` method of the fit `GridSearchCV` object. Finally, we will use the `mean_squared_error` function
to compute the RMSPE.

```{code-cell} ipython3
sacramento_test["predicted"] = sacr_gridsearch.predict(sacramento_test)
RMSPE_mult = mean_squared_error(
    y_true=sacramento_test["price"],
    y_pred=sacramento_test["predicted"]
)**(1/2)
RMSPE_mult

```

```{code-cell} ipython3
:tags: [remove-cell]

glue("RMSPE_mult", "{0:,.0f}".format(RMSPE_mult))
```

This time, when we performed K-NN regression on the same data set, but also
included number of bedrooms as a predictor, we obtained a RMSPE test error
of \${glue:text}`RMSPE_mult`.
{numref}`fig:07-knn-mult-viz` visualizes the model's predictions overlaid on top of the data. This
time the predictions are a surface in 3D space, instead of a line in 2D space, as we have 2
predictors instead of 1.

```{code-cell} ipython3
:tags: [remove-input]

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
knnPredGrid = sacr_gridsearch.predict(xygrid)

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

# if HTML, use the plotly 3d image; if PDF, use static image
if "BOOK_BUILD_TYPE" in os.environ and os.environ["BOOK_BUILD_TYPE"] == "PDF":
    glue("fig:07-knn-mult-viz", Image("img/regression1/plot3d_knn_regression.png"))
else:
    glue("fig:07-knn-mult-viz", fig)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:07-knn-mult-viz
:figclass: caption-hack

K-NN regression model's predictions represented as a surface in 3D space overlaid on top of the data using three predictors (price, house size, and the number of bedrooms). Note that in general we recommend against using 3D visualizations; here we use a 3D visualization only to illustrate what the surface of predictions looks like for learning purposes.
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

## Strengths and limitations of K-NN regression

As with K-NN classification (or any prediction algorithm for that matter), K-NN
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
[worksheets repository](https://worksheets.python.datasciencebook.ca)
in the "Regression I: K-nearest neighbors" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in {numref}`Chapter %s <move-to-your-own-machine>`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

## References

```{bibliography}
:filter: docname in docnames
```
