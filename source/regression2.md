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

(regression2)=
# Regression II: linear regression

```{code-cell} ipython3
:tags: [remove-cell]

from chapter_preamble import *
from IPython.display import HTML
from IPython.display import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```

## Overview
Up to this point, we have solved all of our predictive problems&mdash;both classification
and regression&mdash;using K-nearest neighbors (K-NN)-based approaches. In the context of regression,
there is another commonly used method known as *linear regression*. This chapter provides an introduction
to the basic concept of linear regression, shows how to use `scikit-learn` to perform linear regression in Python,
and characterizes its strengths and weaknesses compared to K-NN regression. The focus is, as usual,
on the case where there is a single predictor and single response variable of interest; but the chapter
concludes with an example using *multivariable linear regression* when there is more than one
predictor.

## Chapter learning objectives
By the end of the chapter, readers will be able to do the following:

- Use Python to fit simple and multivariable linear regression models on training data.
- Evaluate the linear regression model on test data.
- Compare and contrast predictions obtained from K-nearest neighbors regression to those obtained using linear regression from the same data set.
- Describe how linear regression is affected by outliers and multicollinearity.

+++

## Simple linear regression

```{index} regression; linear
```

At the end of the previous chapter, we noted some limitations of K-NN regression.
While the method is simple and easy to understand, K-NN regression does not
predict well beyond the range of the predictors in the training data, and
the method gets significantly slower as the training data set grows.
Fortunately, there is an alternative to K-NN regression&mdash;*linear regression*&mdash;that addresses
both of these limitations. Linear regression is also very commonly
used in practice because it provides an interpretable mathematical equation that describes
the relationship between the predictor and response variables. In this first part of the chapter, we will focus on *simple* linear regression,
which involves only one predictor variable and one response variable; later on, we will consider
 *multivariable* linear regression, which involves multiple predictor variables.
 Like K-NN regression, simple linear regression involves
predicting a numerical response variable (like race time, house price, or height);
but *how* it makes those predictions for a new observation is quite different from K-NN regression.
 Instead of looking at the K nearest neighbors and averaging
over their values for a prediction, in simple linear regression, we create a
straight line of best fit through the training data and then
"look up" the prediction using the line.

+++

```{index} regression; logistic
```

```{note}
Although we did not cover it in earlier chapters, there
is another popular method for classification called *logistic
regression* (it is used for classification even though the name, somewhat confusingly,
has the word "regression" in it). In logistic regression&mdash;similar to linear regression&mdash;you
"fit" the model to the training data and then "look up" the prediction for each new observation.
Logistic regression and K-NN classification have an advantage/disadvantage comparison
similar to that of linear regression and K-NN
regression. It is useful to have a good understanding of linear regression before learning about
logistic regression. After reading this chapter, see the "Additional Resources" section at the end of the
classification chapters to learn more about logistic regression.
```

+++

```{index} Sacramento real estate, question; regression
```

Let's return to the Sacramento housing data from {numref}`Chapter %s <regression1>` to learn
how to apply linear regression and compare it to K-NN regression. For now, we
will consider
a smaller version of the housing data to help make our visualizations clear.
Recall our predictive question: can we use the size of a house in the Sacramento, CA area to predict
its sale price? In particular, recall that we have come across a new 2,000 square-foot house we are interested
in purchasing with an advertised list price of
\$350,000. Should we offer the list price, or is that over/undervalued?
To answer this question using simple linear regression, we use the data we have
to draw the straight line of best fit through our existing data points.
The small subset of data as well as the line of best fit are shown
in {numref}`fig:08-lin-reg1`.

```{code-cell} ipython3
:tags: [remove-cell]

import pandas as pd

np.random.seed(2)

sacramento = pd.read_csv("data/sacramento.csv")

small_sacramento = sacramento.sample(n=30)

small_plot = (
    alt.Chart(small_sacramento)
    .mark_circle(opacity=1)
    .encode(
        x=alt.X("sqft")
            .scale(zero=False)
            .title("House size (square feet)"),
        y=alt.Y("price")
            .axis(format="$,.0f")
            .scale(zero=False)
            .title("Price (USD)"),
    )
)


# create df_lines with one fake/empty line (for starting at 2nd color later)
df_lines = {"x": [500, 500], "y": [100000, 100000], "number": ["-1", "-1"]}

# set the domains (range of x values) of lines
min_x = small_sacramento["sqft"].min()
max_x = small_sacramento["sqft"].max()

# add the line of best fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(small_sacramento[["sqft"]], small_sacramento[["price"]])
pred_min = float(lm.predict(pd.DataFrame({"sqft": [min_x]})))
pred_max = float(lm.predict(pd.DataFrame({"sqft": [max_x]})))

df_lines["x"].extend([min_x, max_x])
df_lines["y"].extend([pred_min, pred_max])
df_lines["number"].extend(["0", "0"])

# add other similar looking lines
intercept_l = [-64542.23, -6900, -64542.23]
slope_l = [190, 175, 160]
for i in range(len(slope_l)):
    df_lines["x"].extend([min_x, max_x])
    df_lines["y"].extend([
                        intercept_l[i] + slope_l[i] * min_x,
                        intercept_l[i] + slope_l[i] * max_x,
                    ])
    df_lines["number"].extend([f"{i+1}", f"{i+1}"])

df_lines = pd.DataFrame(df_lines)

# plot the bogus line to skip the same color as the scatter
small_plot += alt.Chart(
    df_lines[df_lines["number"] == "-1"]
).mark_line().encode(
    x="x", y="y", color=alt.Color("number", legend=None)
)
# plot the real line with 2nd color
small_plot += alt.Chart(
    df_lines[df_lines["number"] == "0"]
).mark_line().encode(
    x="x", y="y", color=alt.Color("number", legend=None)
)

small_plot
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("fig:08-lin-reg1", small_plot)
```

:::{glue:figure} fig:08-lin-reg1
:name: fig:08-lin-reg1

Scatter plot of sale price versus size with line of best fit for subset of the Sacramento housing data.
:::

+++

```{index} straight line; equation
```

```{index} see: line; straight line
```

The equation for the straight line is:

$$\text{house sale price} = \beta_0 + \beta_1 \cdot (\text{house size}),$$
where

- $\beta_0$ is the *vertical intercept* of the line (the price when house size is 0)
- $\beta_1$ is the *slope* of the line (how quickly the price increases as you increase house size)

Therefore using the data to find the line of best fit is equivalent to finding coefficients
$\beta_0$ and $\beta_1$ that *parametrize* (correspond to) the line of best fit.
Now of course, in this particular problem, the idea of a 0 square-foot house is a bit silly;
but you can think of $\beta_0$ here as the "base price," and
$\beta_1$ as the increase in price for each square foot of space.
Let's push this thought even further: what would happen in the equation for the line if you
tried to evaluate the price of a house with size 6 *million* square feet?
Or what about *negative* 2,000 square feet? As it turns out, nothing in the formula breaks; linear
regression will happily make predictions for crazy predictor values if you ask it to. But even though
you *can* make these wild predictions, you shouldn't. You should only make predictions roughly within
the range of your original data, and perhaps a bit beyond it only if it makes sense. For example,
the data in {numref}`fig:08-lin-reg1` only reaches around 600 square feet on the low end, but
it would probably be reasonable to use the linear regression model to make a prediction at 500 square feet, say.

Back to the example! Once we have the coefficients $\beta_0$ and $\beta_1$, we can use the equation
above to evaluate the predicted sale price given the value we have for the
predictor variable&mdash;here 2,000 square feet. {numref}`fig:08-lin-reg2` demonstrates this process.

```{code-cell} ipython3
:tags: [remove-cell]
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(small_sacramento[["sqft"]], small_sacramento[["price"]])
prediction = float(lm.predict(pd.DataFrame({"sqft": [2000]})))

# the vertical dotted line
line_df = pd.DataFrame({"x": [2000]})
rule = alt.Chart(line_df).mark_rule(strokeDash=[6], size=1.5).encode(x="x")

# the red point
point_df = pd.DataFrame({"x": [2000], "y": [prediction]})
point = alt.Chart(point_df).mark_circle(color="red", size=80, opacity=1).encode(x="x", y="y")

# overlay all plots
small_plot_2000_pred = (
    small_plot
    + rule
    + point
    # add the text
    + alt.Chart(
        pd.DataFrame(
            {
                "x": [2450],
                "y": [prediction - 41000],
                "prediction": ["$" + "{0:,.0f}".format(prediction)],
            }
        )
    )
    .mark_text(dy=-5, size=15)
    .encode(x="x", y="y", text="prediction")
)

small_plot_2000_pred
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:08-lin-reg2", small_plot_2000_pred)
glue("pred_2000", "{0:,.0f}".format(prediction))
```

:::{glue:figure} fig:08-lin-reg2
:name: fig:08-lin-reg2

Scatter plot of sale price versus size with line of best fit and a red dot at the predicted sale price for a 2,000 square-foot home.
:::

+++

By using simple linear regression on this small data set to predict the sale price
for a 2,000 square-foot house, we get a predicted value of
\${glue:text}`pred_2000`. But wait a minute...how
exactly does simple linear regression choose the line of best fit? Many
different lines could be drawn through the data points.
Some plausible examples are shown in {numref}`fig:08-several-lines`.

```{code-cell} ipython3
:tags: [remove-cell]

several_lines_plot = small_plot.copy()

several_lines_plot += alt.Chart(
    df_lines[df_lines["number"] != "0"]
).mark_line().encode(x="x", y="y", color=alt.Color("number",legend=None))

several_lines_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:08-several-lines", several_lines_plot)
```

:::{glue:figure} fig:08-several-lines
:name: fig:08-several-lines

Scatter plot of sale price versus size with many possible lines that could be drawn through the data points.
:::

+++

```{index} RMSPE
```

Simple linear regression chooses the straight line of best fit by choosing
the line that minimizes the **average squared vertical distance** between itself and
each of the observed data points in the training data (equivalent to minimizing the RMSE). {numref}`fig:08-verticalDistToMin` illustrates
these vertical distances as lines. Finally, to assess the predictive
accuracy of a simple linear regression model,
we use RMSPE&mdash;the same measure of predictive performance we used with K-NN regression.

```{code-cell} ipython3
:tags: [remove-cell]

small_sacramento_pred = small_sacramento
# get prediction
small_sacramento_pred = small_sacramento_pred.assign(
    predicted=lm.predict(small_sacramento[["sqft"]])
)
# melt the dataframe to create separate df to create lines
small_sacramento_pred = small_sacramento_pred[["sqft", "price", "predicted"]].melt(
    id_vars=["sqft"]
)

v_lines = []
for i in range(len(small_sacramento)):
    sqft_val = small_sacramento.iloc[i]["sqft"]
    line_df = small_sacramento_pred.query("sqft == @sqft_val")
    v_lines.append(alt.Chart(line_df).mark_line(color="black").encode(x="sqft", y="value"))

error_plot = alt.layer(*v_lines, small_plot).configure_circle(opacity=1)
error_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:08-verticalDistToMin", error_plot)
```

:::{glue:figure} fig:08-verticalDistToMin
:name: fig:08-verticalDistToMin

Scatter plot of sale price versus size with lines denoting the vertical distances between the predicted values and the observed data points.
:::

+++

## Linear regression in Python

+++

```{index} scikit-learn
```

We can perform simple linear regression in Python using `scikit-learn` in a
very similar manner to how we performed K-NN regression.
To do this, instead of creating a `KNeighborsRegressor` model object,
we use a `LinearRegression` model object;
and as usual, we first have to import it from `sklearn`.
Another difference is that we do not need to choose $K$ in the
context of linear regression, and so we do not need to perform cross-validation.
Below we illustrate how we can use the usual `scikit-learn` workflow to predict house sale
price given house size. We use a simple linear regression approach on the full
Sacramento real estate data set.

```{index} seed; numpy.random.seed
```

As usual, we start by loading packages, setting the seed, loading data, and
putting some test data away in a lock box that we
can come back to after we choose our final model. Let's take care of that now.

```{code-cell} ipython3
import numpy as np
import altair as alt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import set_config

# Output dataframes instead of arrays
set_config(transform_output="pandas")

np.random.seed(1)

sacramento = pd.read_csv("data/sacramento.csv")

sacramento_train, sacramento_test = train_test_split(
    sacramento, train_size=0.75
)
```

Now that we have our training data, we will create
and fit the linear regression model object.
We will also extract the slope of the line
via the `coef_[0]` property, as well as the
intercept of the line via the `intercept_` property.

```{index} scikit-learn; fit
```

```{code-cell} ipython3
# fit the linear regression model
lm = LinearRegression()
lm.fit(
   sacramento_train[["sqft"]],  # A single-column data frame
   sacramento_train["price"]  # A series
)

# make a dataframe containing slope and intercept coefficients
pd.DataFrame({"slope": [lm.coef_[0]], "intercept": [lm.intercept_]})
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("train_lm_slope", "{:0.0f}".format(lm.coef_[0]))
glue("train_lm_intercept", "{:0.0f}".format(lm.intercept_))
glue("train_lm_slope_f", "{0:,.0f}".format(lm.coef_[0]))
glue("train_lm_intercept_f", "{0:,.0f}".format(lm.intercept_))
```

```{index} standardization
```

```{note}
An additional difference that you will notice here is that we do
not standardize (i.e., scale and center) our
predictors. In K-nearest neighbors models, recall that the model fit changes
depending on whether we standardize first or not. In linear regression,
standardization does not affect the fit (it *does* affect the coefficients in
the equation, though!).  So you can standardize if you want&mdash;it won't
hurt anything&mdash;but if you leave the predictors in their original form,
the best fit coefficients are usually easier to interpret afterward.
```

+++

Our coefficients are
(intercept) $\beta_0=$ {glue:text}`train_lm_intercept`
and (slope) $\beta_1=$ {glue:text}`train_lm_slope`.
This means that the equation of the line of best fit is

$\text{house sale price} =$ {glue:text}`train_lm_intercept` $+$ {glue:text}`train_lm_slope` $\cdot (\text{house size}).$

In other words, the model predicts that houses
start at \${glue:text}`train_lm_intercept_f` for 0 square feet, and that
every extra square foot increases the cost of
the house by \${glue:text}`train_lm_slope_f`. Finally,
we predict on the test data set to assess how well our model does.

```{code-cell} ipython3
# make predictions
sacramento_test["predicted"] = lm.predict(sacramento_test[["sqft"]])

# calculate RMSPE
RMSPE = mean_squared_error(
    y_true=sacramento_test["price"],
    y_pred=sacramento_test["predicted"]
)**(1/2)

RMSPE
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sacr_RMSPE", "{0:,.0f}".format(RMSPE))
```

```{index} RMSPE
```

Our final model's test error as assessed by RMSPE
is \${glue:text}`sacr_RMSPE`.
Remember that this is in units of the response variable, and here that
is US Dollars (USD). Does this mean our model is "good" at predicting house
sale price based off of the predictor of home size? Again, answering this is
tricky and requires knowledge of how you intend to use the prediction.

To visualize the simple linear regression model, we can plot the predicted house
sale price across all possible house sizes we might encounter.
Since our model is linear,
we only need to compute the predicted price of the minimum and maximum house size,
and then connect them with a straight line.
We superimpose this prediction line on a scatter
plot of the original housing price data,
so that we can qualitatively assess if the model seems to fit the data well.
{numref}`fig:08-lm-predict-all` displays the result.

```{code-cell} ipython3
:tags: [remove-output]
sqft_prediction_grid = sacramento[["sqft"]].agg(["min", "max"])
sqft_prediction_grid["predicted"] = lm.predict(sqft_prediction_grid)

all_points = alt.Chart(sacramento).mark_circle().encode(
    x=alt.X("sqft")
        .scale(zero=False)
        .title("House size (square feet)"),
    y=alt.Y("price")
        .axis(format="$,.0f")
        .scale(zero=False)
        .title("Price (USD)")
)

sacr_preds_plot = all_points + alt.Chart(sqft_prediction_grid).mark_line(
    color="#ff7f0e"
).encode(
    x="sqft",
    y="predicted"
)

sacr_preds_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:08-lm-predict-all", sacr_preds_plot)
```

:::{glue:figure} fig:08-lm-predict-all
:name: fig:08-lm-predict-all

Scatter plot of sale price versus size with line of best fit for the full Sacramento housing data.
:::

## Comparing simple linear and K-NN regression

```{index} regression; comparison of methods
```

Now that we have a general understanding of both simple linear and K-NN
regression, we can start to compare and contrast these methods as well as the
predictions made by them. To start, let's look at the visualization of the
simple linear regression model predictions for the Sacramento real estate data
(predicting price from house size) and the "best" K-NN regression model
obtained from the same problem, shown in {numref}`fig:08-compareRegression`.

```{code-cell} ipython3
:tags: [remove-cell]
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
sacr_pipeline_knn = make_pipeline(
    sacr_preprocessor, KNeighborsRegressor(n_neighbors=55)
)  # 55 is the best parameter obtained through cross validation in regression1 chapter

sacr_pipeline_knn.fit(sacramento_train[["sqft"]], sacramento_train[["price"]])

# knn in-sample predictions (on training split)
sacr_preds_knn = sacramento_train
sacr_preds_knn = sacr_preds_knn.assign(
    knn_predicted=sacr_pipeline_knn.predict(sacramento_train)
)

# knn out-of-sample predictions (on test split)
sacr_preds_knn_test = sacramento_test
sacr_preds_knn_test = sacr_preds_knn_test.assign(
    knn_predicted=sacr_pipeline_knn.predict(sacramento_test)
)

sacr_rmspe_knn = np.sqrt(
    mean_squared_error(
        y_true=sacr_preds_knn_test["price"], y_pred=sacr_preds_knn_test["knn_predicted"]
    )
)

# plot knn in-sample predictions overlaid on scatter plot
knn_plot_final = (
    alt.Chart(sacr_preds_knn, title="K-NN regression")
    .mark_circle()
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "price",
            title="Price (USD)",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        ),
    )
)

knn_plot_final = (
    knn_plot_final
    + knn_plot_final.mark_line(color="#ff7f0e").encode(x="sqft", y="knn_predicted")
    + alt.Chart(  # add the text
        pd.DataFrame(
            {
                "x": [3500],
                "y": [100000],
                "rmspe": [f"RMSPE = {round(sacr_rmspe_knn)}"],
            }
        )
    )
    .mark_text(dy=-5, size=15)
    .encode(x="x", y="y", text="rmspe")
)


# add more components to lm_plot_final
lm_plot_final = (
    alt.Chart(sacramento_train, title="linear regression")
    .mark_circle()
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "price",
            title="Price (USD)",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        ),
    )
)

lm_plot_final = (
    lm_plot_final
    + lm_plot_final.transform_regression("sqft", "price").mark_line(color="#ff7f0e")
    + alt.Chart(  # add the text
        pd.DataFrame(
            {
                "x": [3500],
                "y": [100000],
                "rmspe": [f"RMSPE = {round(RMSPE)}"],
            }
        )
    )
    .mark_text(dy=-5, size=15)
    .encode(x="x", y="y", text="rmspe")
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:08-compareRegression", (lm_plot_final | knn_plot_final))
```

:::{glue:figure} fig:08-compareRegression
:name: fig:08-compareRegression

Comparison of simple linear regression and K-NN regression.
:::

+++

What differences do we observe in {numref}`fig:08-compareRegression`? One obvious
difference is the shape of the orange lines. In simple linear regression we are
restricted to a straight line, whereas in K-NN regression our line is much more
flexible and can be quite wiggly. But there is a major interpretability advantage in limiting the
model to a straight line. A
straight line can be defined by two numbers, the
vertical intercept and the slope. The intercept tells us what the prediction is when
all of the predictors are equal to 0; and the slope tells us what unit increase in the response
variable we predict given a unit increase in the predictor
variable. K-NN regression, as simple as it is to implement and understand, has no such
interpretability from its wiggly line.

```{index} underfitting; regression
```

There can, however, also be a disadvantage to using a simple linear regression
model in some cases, particularly when the relationship between the response variable and
the predictor is not linear, but instead some other shape (e.g., curved or oscillating). In
these cases the prediction model from a simple linear regression
will underfit, meaning that model/predicted values do not
match the actual observed values very well. Such a model would probably have a
quite high RMSE when assessing model goodness of fit on the training data and
a quite high RMSPE when assessing model prediction quality on a test data
set. On such a data set, K-NN regression may fare better. Additionally, there
are other types of regression you can learn about in future books that may do
even better at predicting with such data.

How do these two models compare on the Sacramento house prices data set? In
{numref}`fig:08-compareRegression`, we also printed the RMSPE as calculated from
predicting on the test data set that was not used to train/fit the models. The RMSPE for the simple linear
regression model is slightly lower than the RMSPE for the K-NN regression model.
Considering that the simple linear regression model is also more interpretable,
if we were comparing these in practice we would likely choose to use the simple
linear regression model.

```{index} extrapolation
```

Finally, note that the K-NN regression model becomes "flat"
at the left and right boundaries of the data, while the linear model
predicts a constant slope. Predicting outside the range of the observed
data is known as *extrapolation*; K-NN and linear models behave quite differently
when extrapolating. Depending on the application, the flat
or constant slope trend may make more sense. For example, if our housing
data were slightly different, the linear model may have actually predicted
a *negative* price for a small house (if the intercept $\beta_0$ was negative),
which obviously does not match reality. On the other hand, the trend of increasing
house size corresponding to increasing house price probably continues for large houses,
so the "flat" extrapolation of K-NN likely does not match reality.

+++

## Multivariable linear regression

+++

```{index} regression; multivariable linear, regression; multivariable linear equation
```

```{index} see: multivariable linear equation; plane equation
```

As in K-NN classification and K-NN regression, we can move beyond the simple
case of only one predictor to the case with multiple predictors,
known as *multivariable linear regression*.
To do this, we follow a very similar approach to what we did for
K-NN regression: we just specify the training data by adding more predictors.
But recall that we do not need to use cross-validation to choose any parameters,
nor do we need to standardize (i.e., center and scale) the data for linear regression.
Note once again that we have the same concerns regarding multiple predictors
 as in the settings of multivariable K-NN regression and classification: having more predictors is **not** always
better. But because the same predictor selection
algorithm from {numref}`Chapter %s <classification2>` extends to the setting of linear regression,
it will not be covered again in this chapter.

```{index} Sacramento real estate
```

We will demonstrate multivariable linear regression using the Sacramento real estate
data with both house size
(measured in square feet) as well as number of bedrooms as our predictors, and
continue to use house sale price as our response variable.
The `scikit-learn` framework makes this easy to do: we just need to set
both the `sqft` and `beds` variables as predictors, and then use the `fit`
method as usual.

```{code-cell} ipython3

mlm = LinearRegression()
mlm.fit(
    sacramento_train[["sqft", "beds"]],
    sacramento_train["price"]
)
```
Finally, we make predictions on the test data set to assess the quality of our model.

```{index} scikit-learn;predict, scikit-learn;mean_squared_error
```
```{index} see: mean_squared_error;scikit-learn
```
```{index} see: predict;scikit-learn
```

```{code-cell} ipython3
sacramento_test["predicted"] = mlm.predict(sacramento_test[["sqft","beds"]])

lm_mult_test_RMSPE = mean_squared_error(
    y_true=sacramento_test["price"],
    y_pred=sacramento_test["predicted"]
)**(1/2)
lm_mult_test_RMSPE
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sacr_mult_RMSPE", "{0:,.0f}".format(lm_mult_test_RMSPE))
```

Our model's test error as assessed by RMSPE
is \${glue:text}`sacr_mult_RMSPE`.
In the case of two predictors, we can plot the predictions made by our linear regression creates a *plane* of best fit, as
shown in {numref}`fig:08-3DlinReg`.

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
mlmPredGrid = mlm.predict(xygrid)

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
        z=mlmPredGrid.reshape(50, -1),
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
    glue("fig:08-3DlinReg", Image("img/regression2/plot3d_linear_regression.png"))
else:
    glue("fig:08-3DlinReg", fig)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:08-3DlinReg
:figclass: caption-hack

Linear regression plane of best fit overlaid on top of the data (using price,
house size, and number of bedrooms as predictors). Note that in general we
recommend against using 3D visualizations; here we use a 3D visualization only
to illustrate what the regression plane looks like for learning purposes.
```

+++

We see that the predictions from linear regression with two predictors form a
flat plane. This is the hallmark of linear regression, and differs from the
wiggly, flexible surface we get from other methods such as K-NN regression.
 As discussed, this can be advantageous in one aspect, which is that for each
predictor, we can get slopes/intercept from linear regression, and thus describe the
plane mathematically. We can extract those slope values from the `coef_` property
of our model object, and the intercept from the `intercept_` property,
as shown below.

```{code-cell} ipython3
mlm.coef_
```

```{code-cell} ipython3
mlm.intercept_
```

When we have multiple predictor variables, it is not easy to
know which variable goes with which coefficient in `mlm.coef_`. In particular,
you will see that `mlm.coef_` above is just an array of values without any variable names.
Unfortunately you have to do this mapping yourself: the coefficients in `mlm.coef_` appear
in the *same order* as the columns of the predictor data frame you used when training.
So since we used `sacramento_train[["sqft", "beds"]]` when training,
we have that `mlm.coef_[0]` corresponds to `sqft`, and `mlm.coef_[1]` corresponds to `beds`.
Once you sort out the correspondence, you can then use those slopes to write a mathematical equation to describe the prediction plane:

```{index} plane equation
```



$$\text{house sale price} = \beta_0 + \beta_1\cdot(\text{house size}) + \beta_2\cdot(\text{number of bedrooms}),$$
where:

- $\beta_0$ is the *vertical intercept* of the hyperplane (the price when both house size and number of bedrooms are 0)
- $\beta_1$ is the *slope* for the first predictor (how quickly the price increases as you increase house size)
- $\beta_2$ is the *slope* for the second predictor (how quickly the price increases as you increase the number of bedrooms)

Finally, we can fill in the values for $\beta_0$, $\beta_1$ and $\beta_2$ from the model output above
to create the equation of the plane of best fit to the data:

```{code-cell} ipython3
:tags: [remove-cell]

icept = "{0:,.0f}".format(mlm.intercept_)
sqftc = "{0:,.0f}".format(mlm.coef_[0])
bedsc = "{0:,.0f}".format(mlm.coef_[1])
glue("icept", icept)
glue("sqftc", sqftc)
glue("bedsc", bedsc)
```

$\text{house sale price} =$ {glue:text}`icept` $+$ {glue:text}`sqftc` $\cdot (\text{house size})$ {glue:text}`bedsc` $\cdot (\text{number of bedrooms})$

This model is more interpretable than the multivariable K-NN
regression model; we can write a mathematical equation that explains how
each predictor is affecting the predictions. But as always, we should
question how well multivariable linear regression is doing compared to
the other tools we have, such as simple linear regression
and multivariable K-NN regression. If this comparison is part of
the model tuning process&mdash;for example, if we are trying
 out many different sets of predictors for multivariable linear
and K-NN regression&mdash;we must perform this comparison using
cross-validation on only our training data. But if we have already
decided on a small number (e.g., 2 or 3) of tuned candidate models and
we want to make a final comparison, we can do so by comparing the prediction
error of the methods on the test data.

```{code-cell} ipython3
lm_mult_test_RMSPE
```

```{index} RMSPE
```

We obtain an RMSPE for the multivariable linear regression model
of \${glue:text}`sacr_mult_RMSPE`. This prediction error
 is less than the prediction error for the multivariable K-NN regression model,
indicating that we should likely choose linear regression for predictions of
house sale price on this data set. Revisiting the simple linear regression model
with only a single predictor from earlier in this chapter, we see that the RMSPE for that model was
\${glue:text}`sacr_RMSPE`,
which is slightly higher than that of our more complex model. Our model with two predictors
provided a slightly better fit on test data than our model with just one.
As mentioned earlier, this is not always the case: sometimes including more
predictors can negatively impact the prediction performance on unseen
test data.

+++

## Multicollinearity and outliers

What can go wrong when performing (possibly multivariable) linear regression?
This section will introduce two common issues&mdash;*outliers* and *collinear predictors*&mdash;and
illustrate their impact on predictions.

+++

### Outliers

```{index} outliers
```

Outliers are data points that do not follow the usual pattern of the rest of the data.
In the setting of linear regression, these are points that
 have a vertical distance to the line of best fit that is either much higher or much lower
than you might expect based on the rest of the data. The problem with outliers is that
they can have *too much influence* on the line of best fit. In general, it is very difficult
to judge accurately which data are outliers without advanced techniques that are beyond
the scope of this book.

But to illustrate what can happen when you have outliers, {numref}`fig:08-lm-outlier`
shows a small subset of the Sacramento housing data again, except we have added a *single* data point (highlighted
in red). This house is 5,000 square feet in size, and sold for only \$50,000. Unbeknownst to the
data analyst, this house was sold by a parent to their child for an absurdly low price. Of course,
this is not representative of the real housing market values that the other data points follow;
the data point is an *outlier*. In orange we plot the original line of best fit, and in red
we plot the new line of best fit including the outlier. You can see how different the red line
is from the orange line, which is entirely caused by that one extra outlier data point.

```{code-cell} ipython3
:tags: [remove-cell]

sacramento_train_small = sacramento_train.sample(100, random_state=2)
sacramento_outlier = pd.DataFrame({"sqft": [5000], "price": [50000]})
sacramento_concat_df = pd.concat((sacramento_train_small, sacramento_outlier))

lm_plot_outlier = (
    alt.Chart(sacramento_train_small)
    .mark_circle()
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "price",
            title="Price (USD)",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        ),
    )
)
lm_plot_outlier += lm_plot_outlier.transform_regression("sqft", "price").mark_line(
    color="#ff7f0e"
)

outlier_pt = (
    alt.Chart(sacramento_outlier)
    .mark_circle(color="#d62728", size=100)
    .encode(x="sqft", y="price")
)

outlier_line = (
    (
        alt.Chart(sacramento_concat_df)
        .mark_circle()
        .encode(
            x=alt.X(
                "sqft", title="House size (square feet)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "price",
                title="Price (USD)",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False),
            ),
        )
    )
    .transform_regression("sqft", "price")
    .mark_line(color="#d62728")
)

lm_plot_outlier += outlier_pt + outlier_line

glue("fig:08-lm-outlier", lm_plot_outlier)
```

:::{glue:figure} fig:08-lm-outlier
:name: fig:08-lm-outlier

Scatter plot of a subset of the data, with outlier highlighted in red.
:::

+++

Fortunately, if you have enough data, the inclusion of one or two
outliers&mdash;as long as their values are not *too* wild&mdash;will
typically not have a large effect on the line of best fit. {numref}`fig:08-lm-outlier-2` shows how that same outlier data point from earlier
influences the line of best fit when we are working with the entire original
Sacramento training data. You can see that with this larger data set, the line
changes much less when adding the outlier.
Nevertheless, it is still important when working with linear regression to critically
think about how much any individual data point is influencing the model.

```{code-cell} ipython3
:tags: [remove-cell]

sacramento_concat_df = pd.concat((sacramento_train, sacramento_outlier))

lm_plot_outlier_large = (
    alt.Chart(sacramento_train)
    .mark_circle()
    .encode(
        x=alt.X("sqft", title="House size (square feet)", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "price",
            title="Price (USD)",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        ),
    )
)
lm_plot_outlier_large += lm_plot_outlier_large.transform_regression(
    "sqft", "price"
).mark_line(color="#ff7f0e")

outlier_line = (
    (
        alt.Chart(sacramento_concat_df)
        .mark_circle()
        .encode(
            x=alt.X(
                "sqft", title="House size (square feet)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "price",
                title="Price (USD)",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False),
            ),
        )
    )
    .transform_regression("sqft", "price")
    .mark_line(color="#d62728")
)

lm_plot_outlier_large += outlier_pt + outlier_line

glue("fig:08-lm-outlier-2", lm_plot_outlier_large)
```

:::{glue:figure} fig:08-lm-outlier-2
:name: fig:08-lm-outlier-2

Scatter plot of the full data, with outlier highlighted in red.
:::

+++

### Multicollinearity

```{index} multicollinearity
```

The second, and much more subtle, issue can occur when performing multivariable
linear regression.  In particular, if you include multiple predictors that are
strongly linearly related to one another, the coefficients that describe the
plane of best fit can be very unreliable&mdash;small changes to the data can
result in large changes in the coefficients. Consider an extreme example using
the Sacramento housing data where the house was measured twice by two people.
Since the two people are each slightly inaccurate, the two measurements might
not agree exactly, but they are very strongly linearly related to each other,
as shown in {numref}`fig:08-lm-multicol`.

```{code-cell} ipython3
:tags: [remove-cell]

np.random.seed(1)
sacramento_train = sacramento_train.assign(
    sqft1=sacramento_train["sqft"]
    + 100
    * np.random.choice(range(1000000), size=len(sacramento_train), replace=True)
    / 1000000
)
sacramento_train = sacramento_train.assign(
    sqft2=sacramento_train["sqft"]
    + 100
    * np.random.choice(range(1000000), size=len(sacramento_train), replace=True)
    / 1000000
)
sacramento_train = sacramento_train.assign(
    sqft3=sacramento_train["sqft"]
    + 100
    * np.random.choice(range(1000000), size=len(sacramento_train), replace=True)
    / 1000000
)
sacramento_train

lm_plot_multicol_1 = (
    alt.Chart(sacramento_train)
    .mark_circle()
    .encode(
        x=alt.X("sqft", title="House size measurement 1 (square feet)"),
        y=alt.Y("sqft1", title="House size measurement 2 (square feet)"),
    )
)

glue("fig:08-lm-multicol", lm_plot_multicol_1)
```

:::{glue:figure} fig:08-lm-multicol
:name: fig:08-lm-multicol

Scatter plot of house size (in square feet) measured by person 1 versus house size (in square feet) measured by person 2.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# first LM
lm_fit1 = LinearRegression()
X_train = sacramento_train[["sqft", "sqft1"]]
y_train = sacramento_train[["price"]]

lm_fit1.fit(X_train, y_train)

icept1 = "{0:,.0f}".format(lm_fit1.intercept_[0])
sqft1 = "{0:,.0f}".format(lm_fit1.coef_[0][0])
sqft11 = "{0:,.0f}".format(lm_fit1.coef_[0][1])
glue("icept1", icept1)
glue("sqft1", sqft1)
glue("sqft11", sqft11)

# second LM
lm_fit2 = LinearRegression()
X_train = sacramento_train[["sqft", "sqft2"]]
y_train = sacramento_train[["price"]]

lm_fit2.fit(X_train, y_train)

icept2 = "{0:,.0f}".format(lm_fit2.intercept_[0])
sqft2 = "{0:,.0f}".format(lm_fit2.coef_[0][0])
sqft22 = "{0:,.0f}".format(lm_fit2.coef_[0][1])
glue("icept2", icept2)
glue("sqft2", sqft2)
glue("sqft22", sqft22)

# third LM
lm_fit3 = LinearRegression()
X_train = sacramento_train[["sqft", "sqft3"]]
y_train = sacramento_train[["price"]]

lm_fit3.fit(X_train, y_train)

icept3 = "{0:,.0f}".format(lm_fit3.intercept_[0])
sqft3 = "{0:,.0f}".format(lm_fit3.coef_[0][0])
sqft33 = "{0:,.0f}".format(lm_fit3.coef_[0][1])
glue("icept3", icept3)
glue("sqft3", sqft3)
glue("sqft33", sqft33)
```

 If we again fit the multivariable linear regression model on this data, then the plane of best fit
has regression coefficients that are very sensitive to the exact values in the data. For example,
if we change the data ever so slightly&mdash;e.g., by running cross-validation, which splits
up the data randomly into different chunks&mdash;the coefficients vary by large amounts:

Best Fit 1: $\text{house sale price} =$ {glue:text}`icept1` $+$ {glue:text}`sqft1` $\cdot (\text{house size 1}$ $(\text{ft}^2)) +$ {glue:text}`sqft11` $\cdot (\text{house size 2}$ $(\text{ft}^2)).$

Best Fit 2: $\text{house sale price} =$ {glue:text}`icept2` $+$ {glue:text}`sqft2` $\cdot (\text{house size 1}$ $(\text{ft}^2)) +$ {glue:text}`sqft22` $\cdot (\text{house size 2}$ $(\text{ft}^2)).$

Best Fit 3: $\text{house sale price} =$ {glue:text}`icept3` $+$ {glue:text}`sqft3` $\cdot (\text{house size 1}$ $(\text{ft}^2)) +$ {glue:text}`sqft33` $\cdot (\text{house size 2}$ $(\text{ft}^2)).$

 Therefore, when performing multivariable linear regression, it is important to avoid including very
linearly related predictors. However, techniques for doing so are beyond the scope of this
book; see the list of additional resources at the end of this chapter to find out where you can learn more.

+++

## Designing new predictors

We were quite fortunate in our initial exploration to find a predictor variable (house size)
that seems to have a meaningful and nearly linear relationship with our response variable (sale price).
But what should we do if we cannot immediately find such a nice variable?
Well, sometimes it is just a fact that the variables in the data do not have enough of
a relationship with the response variable to provide useful predictions. For example,
if the only available predictor was "the current house owner's favorite ice cream flavor",
we likely would have little hope of using that variable to predict the house's sale price
(barring any future remarkable scientific discoveries about the relationship between
the housing market and homeowner ice cream preferences). In cases like these,
the only option is to obtain measurements of more useful variables.

There are, however, a wide variety of cases where the predictor variables do have a
meaningful relationship with the response variable, but that relationship does not fit
the assumptions of the regression method you have chosen. For example, a data frame `df`
with two variables&mdash;`x` and `y`&mdash;with a nonlinear relationship between the two variables
will not be fully captured by simple linear regression, as shown in {numref}`fig:08-predictor-design`.

```{code-cell} ipython3
:tags: [remove-cell]

np.random.seed(3)
df = pd.DataFrame({"x": np.random.choice(range(10000), size=100, replace=True) / 10000})
df = df.assign(
    y=df["x"] ** 3
    + 0.2 * np.random.choice(range(10000), size=100, replace=True) / 10000
    - 0.1
)
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
:tags: [remove-cell]

curve_plt = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x=alt.X("x", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "y",
            scale=alt.Scale(zero=False),
        ),
    )
)


curve_plt += curve_plt.transform_regression("x", "y").mark_line(color="#ff7f0e")

glue("fig:08-predictor-design", curve_plt)
```

:::{glue:figure} fig:08-predictor-design
:name: fig:08-predictor-design

Example of a data set with a nonlinear relationship between the predictor and the response.
:::

+++

```{index} predictor design
```

Instead of trying to predict the response `y` using a linear regression on `x`,
we might have some scientific background about our problem to suggest that `y`
should be a cubic function of `x`. So before performing regression,
we might *create a new predictor variable* `z`:

```{code-cell} ipython3
df["z"] = df["x"] ** 3
```

Then we can perform linear regression for `y` using the predictor variable `z`,
as shown in {numref}`fig:08-predictor-design-2`.
Here you can see that the transformed predictor `z` helps the
linear regression model make more accurate predictions.
Note that none of the `y` response values have changed between {numref}`fig:08-predictor-design`
and {numref}`fig:08-predictor-design-2`; the only change is that the `x` values
have been replaced by `z` values.

```{code-cell} ipython3
:tags: [remove-cell]

curve_plt2 = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x=alt.X("z", title="z = x³" ,scale=alt.Scale(zero=False)),
        y=alt.Y(
            "y",
            scale=alt.Scale(zero=False),
        ),
    )
)


curve_plt2 += curve_plt2.transform_regression("z", "y").mark_line(color="#ff7f0e")

glue("fig:08-predictor-design-2", curve_plt2)
```

:::{glue:figure} fig:08-predictor-design-2
:name: fig:08-predictor-design-2

Relationship between the transformed predictor and the response.
:::

+++

```{index} see: feature engineering; predictor design
```

The process of
transforming predictors (and potentially combining multiple predictors in the process)
is known as *feature engineering*. In real data analysis
problems, you will need to rely on
a deep understanding of the problem&mdash;as well as the wrangling tools
from previous chapters&mdash;to engineer useful new features that improve
predictive performance.

```{note}
Feature engineering
is *part of tuning your model*, and as such you must not use your test data
to evaluate the quality of the features you produce. You are free to use
cross-validation, though!
```

+++

## The other sides of regression

So far in this textbook we have used regression only in the context of
prediction. However, regression can also be seen as a method to understand and
quantify the effects of individual variables on a response variable of interest.
In the housing example from this chapter, beyond just using past data
to predict future sale prices,
we might also be interested in describing the
individual relationships of house size and the number of bedrooms with house price,
quantifying how strong each of these relationships are, and assessing how accurately we
can estimate their magnitudes. And even beyond that, we may be interested in
understanding whether the predictors *cause* changes in the price.
These sides of regression are well beyond the scope of this book; but
the material you have learned here should give you a foundation of knowledge
that will serve you well when moving to more advanced books on the topic.

+++

## Exercises

Practice exercises for the material covered in this chapter
can be found in the accompanying
[worksheets repository](https://worksheets.python.datasciencebook.ca)
in the "Regression II: linear regression" row.
You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
If you instead decide to download the worksheet and run it on your own machine,
make sure to follow the instructions for computer setup
found in {numref}`Chapter %s <move-to-your-own-machine>`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

## Additional resources

- The [`scikit-learn` website](https://scikit-learn.org/stable/) is an excellent
  reference for more details on, and advanced usage of, the functions and
  packages in the past two chapters. Aside from that, it also offers many
  useful [tutorials](https://scikit-learn.org/stable/tutorial/index.html) and [an extensive list
  of more advanced examples](https://scikit-learn.org/stable/auto_examples/index.html#general-examples)
  that you can use to continue learning beyond the scope of this book.
- *An Introduction to Statistical Learning* {cite:p}`james2013introduction` provides
  a great next stop in the process of
  learning about regression. Chapter 3 covers linear regression at a slightly
  more mathematical level than we do here, but it is not too large a leap and so
  should provide a good stepping stone. Chapter 6 discusses how to pick a subset
  of "informative" predictors when you have a data set with many predictors, and
  you expect only a few of them to be relevant. Chapter 7 covers regression
  models that are more flexible than linear regression models but still enjoy the
  computational efficiency of linear regression. In contrast, the K-NN methods we
  covered earlier are indeed more flexible but become very slow when given lots
  of data.

+++

## References

```{bibliography}
:filter: docname in docnames
```
