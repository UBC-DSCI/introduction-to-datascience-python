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
# (regression2)=
# # Regression II: linear regression

# %% tags=["remove-cell"]
import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

# %% [markdown]
# ## Overview 
# Up to this point, we have solved all of our predictive problems&mdash;both classification
# and regression&mdash;using K-nearest neighbors (KNN)-based approaches. In the context of regression, 
# there is another commonly used method known as *linear regression*. This chapter provides an introduction
# to the basic concept of linear regression, shows how to use `scikit-learn` to perform linear regression in Python,
# and characterizes its strengths and weaknesses compared to KNN regression. The focus is, as usual,
# on the case where there is a single predictor and single response variable of interest; but the chapter
# concludes with an example using *multivariable linear regression* when there is more than one
# predictor.
#
# ## Chapter learning objectives 
# By the end of the chapter, readers will be able to do the following:
#
# * Use Python and `scikit-learn` to fit a linear regression model on training data.
# * Evaluate the linear regression model on test data.
# * Compare and contrast predictions obtained from K-nearest neighbor regression to those obtained using linear regression from the same data set.
# * In `altair`, overlay predictions from linear regression on a scatter plot of data using `transform_regression`.

# %% [markdown]
# ## Simple linear regression
#
# At the end of the previous chapter, we noted some limitations of KNN regression.
# While the method is simple and easy to understand, KNN regression does not
# predict well beyond the range of the predictors in the training data, and
# the method gets significantly slower as the training data set grows. \index{regression!linear}
# Fortunately, there is an alternative to KNN regression&mdash;*linear regression*&mdash;that addresses
# both of these limitations. Linear regression is also very commonly 
# used in practice because it provides an interpretable mathematical equation that describes
# the relationship between the predictor and response variables. In this first part of the chapter, we will focus on *simple* linear regression,
# which involves only one predictor variable and one response variable; later on, we will consider
#  *multivariable* linear regression, which involves multiple predictor variables.
#  Like KNN regression, simple linear regression involves
# predicting a numerical response variable (like race time, house price, or height);
# but *how* it makes those predictions for a new observation is quite different from KNN regression.
#  Instead of looking at the K nearest neighbors and averaging
# over their values for a prediction, in simple linear regression, we create a 
# straight line of best fit through the training data and then
# "look up" the prediction using the line.

# %% [markdown]
# > **Note:** Although we did not cover it in earlier chapters, there 
# > is another popular method for classification called *logistic
# > regression* (it is used for classification even though the name, somewhat confusingly,
# > has the word "regression" in it). In logistic regression&mdash;similar to linear regression&mdash;you
# > "fit" the model to the training data and then "look up" the prediction for each new observation.
# > Logistic regression and KNN classification have an advantage/disadvantage comparison 
# > similar to that of linear regression and KNN
# > regression. It is useful to have a good understanding of linear regression before learning about
# > logistic regression. After reading this chapter, see the "Additional Resources" section at the end of the
# > classification chapters to learn more about logistic regression. \index{regression!logistic}

# %% [markdown]
# Let's return to the Sacramento housing data \index{Sacramento real estate} from Chapter {ref}`regression1` to learn
# how to apply linear regression and compare it to KNN regression. For now, we
# will consider 
# a smaller version of the housing data to help make our visualizations clear.
# Recall our predictive question: can we use the size of a house in the Sacramento, CA area to predict
# its sale price? \index{question!regression} In particular, recall that we have come across a new 2,000 square-foot house we are interested
# in purchasing with an advertised list price of
# \$350,000. Should we offer the list price, or is that over/undervalued?
# To answer this question using simple linear regression, we use the data we have
# to draw the straight line of best fit through our existing data points.
# The small subset of data as well as the line of best fit are shown
# in {numref}`fig:08-lin-reg1`.

# %% tags=["remove-cell"]
sacramento = pd.read_csv("data/sacramento.csv")

small_sacramento = sacramento.sample(n=30, random_state=2)

small_plot = (
    alt.Chart(small_sacramento)
    .mark_circle(color="black")
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

small_plot += small_plot.transform_regression("sqft", "price").mark_line()

small_plot

# %% tags=["remove-cell"]
glue("fig:08-lin-reg1", small_plot)

# %% [markdown]
# :::{glue:figure} fig:08-lin-reg1
# :name: fig:08-lin-reg1
#
# Scatter plot of sale price versus size with line of best fit for subset of the Sacramento housing data.
# :::

# %% [markdown]
# The equation for the straight line is: \index{straight line!equation}
#
# $$\text{house sale price} = \beta_0 + \beta_1 \cdot (\text{house size}),$$
# where
#
# - $\beta_0$ is the *vertical intercept* of the line (the price when house size is 0)
# - $\beta_1$ is the *slope* of the line (how quickly the price increases as you increase house size)
#
# Therefore using the data to find the line of best fit is equivalent to finding coefficients 
# $\beta_0$ and $\beta_1$ that *parametrize* (correspond to) the line of best fit.
# Now of course, in this particular problem, the idea of a 0 square-foot house is a bit silly;
# but you can think of $\beta_0$ here as the "base price," and 
# $\beta_1$ as the increase in price for each square foot of space.
# Let's push this thought even further: what would happen in the equation for the line if you 
# tried to evaluate the price of a house with size 6 *million* square feet?
# Or what about *negative* 2,000 square feet? As it turns out, nothing in the formula breaks; linear
# regression will happily make predictions for crazy predictor values if you ask it to. But even though
# you *can* make these wild predictions, you shouldn't. You should only make predictions roughly within
# the range of your original data, and perhaps a bit beyond it only if it makes sense. For example,
# the data in {numref}`fig:08-lin-reg1` only reaches around 600 square feet on the low end, but 
# it would probably be reasonable to use the linear regression model to make a prediction at 500 square feet, say.
#
# Back to the example! Once we have the coefficients $\beta_0$ and $\beta_1$, we can use the equation
# above to evaluate the predicted sale price given the value we have for the
# predictor variable&mdash;here 2,000 square feet. {numref}`fig:08-lin-reg2` demonstrates this process.

# %% tags=["remove-cell"]
lm = LinearRegression()
lm.fit(small_sacramento[["sqft"]], small_sacramento[["price"]])
prediction = float(lm.predict(pd.DataFrame({"sqft": [2000]})))

# the vertical dotted line
line_df = pd.DataFrame({"x": [2000]})
rule = alt.Chart(line_df).mark_rule(strokeDash=[2, 4]).encode(x="x")

# the red point
point_df = pd.DataFrame({"x": [2000], "y": [prediction]})
point = alt.Chart(point_df).mark_circle(color="red", size=100).encode(x="x", y="y")

# overlay all plots
small_plot_2000_pred = (
    small_plot
    + rule
    + point
    # add the text
    + alt.Chart(
        pd.DataFrame(
            {
                "x": [2350],
                "y": [prediction - 41000],
                "prediction": ["$" + "{0:,.0f}".format(prediction)],
            }
        )
    )
    .mark_text(dy=-5, size=15)
    .encode(x="x", y="y", text="prediction")
)

small_plot_2000_pred

# %% tags=["remove-cell"]
glue("fig:08-lin-reg2", small_plot_2000_pred)
glue("pred_2000", int(prediction))

# %% [markdown]
# :::{glue:figure} fig:08-lin-reg2
# :name: fig:08-lin-reg2
#
# Scatter plot of sale price versus size with line of best fit and a red dot at the predicted sale price for a 2,000 square-foot home.
# :::

# %% [markdown]
# By using simple linear regression on this small data set to predict the sale price
# for a 2,000 square-foot house, we get a predicted value of 
# \${glue:}`pred_2000`. But wait a minute...how
# exactly does simple linear regression choose the line of best fit? Many
# different lines could be drawn through the data points. 
# Some plausible examples are shown in {numref}`fig:08-several-lines`.

# %% tags=["remove-cell"]
intercept_l = [-64542.23, -6900, -64542.23]
slope_l = [190, 175, 160]
line_color_l = ["green", "purple", "red"]

# set the domains (range of x values) of lines
min_x = small_sacramento["sqft"].min()
max_x = small_sacramento["sqft"].max()

several_lines_plot = small_plot.copy()

for i in range(len(slope_l)):
    several_lines_plot += (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [min_x, max_x],
                    "y": [
                        intercept_l[i] + slope_l[i] * min_x,
                        intercept_l[i] + slope_l[i] * max_x,
                    ],
                }
            )
        )
        .mark_line(color=line_color_l[i])
        .encode(x=alt.X("x"), y=alt.Y("y"))
    )

several_lines_plot

# %% tags=["remove-cell"]
glue("fig:08-several-lines", several_lines_plot)

# %% [markdown]
# :::{glue:figure} fig:08-several-lines
# :name: fig:08-several-lines
#
# Scatter plot of sale price versus size with many possible lines that could be drawn through the data points.
# :::

# %% [markdown]
# Simple linear regression chooses the straight line of best fit by choosing
# the line that minimizes the **average squared vertical distance** between itself and
# each of the observed data points in the training data. {numref}`fig:08-verticalDistToMin` illustrates 
# these vertical distances as red lines. Finally, to assess the predictive 
# accuracy of a simple linear regression model,
# we use RMSPE&mdash;the same measure of predictive performance we used with KNN regression.
# \index{RMSPE}

# %% tags=["remove-cell"]
small_sacramento_pred = small_sacramento
# get prediction
small_sacramento_pred = small_sacramento_pred.assign(
    predicted=lm.predict(small_sacramento[["sqft"]])
)
# melt the dataframe to create separate df to create lines
small_sacramento_pred = small_sacramento_pred[["sqft", "price", "predicted"]].melt(
    id_vars=["sqft"]
)

error_plot = small_plot.copy()

for i in range(len(small_sacramento)):
    sqft_val = small_sacramento.iloc[i]["sqft"]
    line_df = small_sacramento_pred.query("sqft == @sqft_val")
    error_plot += alt.Chart(line_df).mark_line(color="red").encode(x="sqft", y="value")

error_plot

# %% tags=["remove-cell"]
glue("fig:08-verticalDistToMin", error_plot)

# %% [markdown]
# :::{glue:figure} fig:08-verticalDistToMin
# :name: fig:08-verticalDistToMin
#
# Scatter plot of sale price versus size with red lines denoting the vertical distances between the predicted values and the observed data points.
# :::

# %% [markdown]
# ## Linear regression in R

# %% [markdown]
# We can perform simple linear regression in Python using `scikit-learn` in a
# very similar manner to how we performed KNN regression. 
# To do this, instead of creating a `KNeighborsRegressor` model specification, 
# we use a `LinearRegression` model specification.
# Another difference is that we do not need to choose $K$ in the
# context of linear regression, and so we do not need to perform cross-validation.
# Below we illustrate how we can use the usual `scikit-learn` workflow to predict house sale
# price given house size using a simple linear regression approach using the full
# Sacramento real estate data set.
#
# As usual, we start by loading packages, setting the seed, loading data, and putting some test data away in a lock box (setting the `random_state`) that we
# can come back to after we choose our final model. Let's take care of that now.

# %% tags=["remove-cell"]
# We can perform simple linear regression in R using `tidymodels` \index{tidymodels} in a
# very similar manner to how we performed KNN regression. 
# To do this, instead of creating a `nearest_neighbor` model specification with
# the `kknn` engine, we use a `linear_reg` model specification
# with the `lm` engine. Another difference is that we do not need to choose $K$ in the
# context of linear regression, and so we do not need to perform cross-validation.
# Below we illustrate how we can use the usual `tidymodels` workflow to predict house sale
# price given house size using a simple linear regression approach using the full
# Sacramento real estate data set.

# %% tags=["remove-cell"]
# As usual, we start by loading packages, setting the seed, loading data, and putting some test data away in a lock box that we
# can come back to after we choose our final model. Let's take care of that now.
# \index{seed!set.seed}

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

sacramento = pd.read_csv("data/sacramento.csv")

sacramento_train, sacramento_test = train_test_split(
    sacramento, train_size=0.6, random_state=1234
)

# %% [markdown]
# Now that we have our training data, we will create the model specification
# and recipe, and fit our simple linear regression model:

# %%
lm = LinearRegression()
X_train = sacramento_train[["sqft"]]
y_train = sacramento_train[["price"]]

# fit the linear regression model
lm.fit(X_train, y_train)

# make a dataframe containing slope and intercept coefficients
pd.DataFrame({"slope": lm.coef_[0], "intercept": lm.intercept_})

# %% tags=["remove-cell"]
glue("train_lm_slope", round(lm.coef_[0][0]))
glue("train_lm_intercept", round(lm.intercept_[0]))

# %% [markdown]
# > **Note:** An additional difference that you will notice here is that we do
# > not standardize \index{standardization} (i.e., scale and center) our
# > predictors. In K-nearest neighbors models, recall that the model fit changes
# > depending on whether we standardize first or not. In linear regression,
# > standardization does not affect the fit (it *does* affect the coefficients in
# > the equation, though!).  So you can standardize if you want&mdash;it won't
# > hurt anything&mdash;but if you leave the predictors in their original form,
# > the best fit coefficients are usually easier to interpret afterward.

# %% [markdown]
# Our coefficients are 
# (intercept) $\beta_0=$ {glue:}`train_lm_intercept`
# and (slope) $\beta_1=$ {glue:}`train_lm_slope`.
# This means that the equation of the line of best fit is
#
# $\text{house sale price} =$ {glue:}`train_lm_intercept` $+$ {glue:}`train_lm_slope` $\cdot (\text{house size}).$
#
# In other words, the model predicts that houses 
# start at \${glue:}`train_lm_intercept` for 0 square feet, and that
# every extra square foot increases the cost of 
# the house by \${glue:}`train_lm_slope`. Finally, 
# we predict on the test data set to assess how well our model does:

# %%
X_test = sacramento_test[["sqft"]]
y_test = sacramento_test[["price"]]

# predict on test data
sacr_preds = sacramento_test
sacr_preds = sacr_preds.assign(predicted=lm.predict(X_test))

# calculate RMSPE
from sklearn.metrics import mean_squared_error

RMSPE = np.sqrt(
    mean_squared_error(y_true=sacr_preds["price"], y_pred=sacr_preds["predicted"])
)

RMSPE

# %% tags=["remove-cell"]
glue("sacr_RMSPE", round(RMSPE))

# %% [markdown]
# Our final model's test error as assessed by RMSPE \index{RMSPE}
# is {glue:}`sacr_RMSPE`. 
# Remember that this is in units of the target/response variable, and here that
# is US Dollars (USD). Does this mean our model is "good" at predicting house
# sale price based off of the predictor of home size? Again, answering this is
# tricky and requires knowledge of how you intend to use the prediction.
#
# To visualize the simple linear regression model, we can plot the predicted house
# sale price across all possible house sizes we might encounter superimposed on a scatter
# plot of the original housing price data. There is a plotting function in 
# the `altair`, `transform_regression`, that
# allows us to add a layer on our plot with the simple
# linear regression predicted line of best fit. 
# {numref}`fig:08-lm-predict-all` displays the result.

# %% tags=["remove-cell"]
# To visualize the simple linear regression model, we can plot the predicted house
# sale price across all possible house sizes we might encounter superimposed on a scatter
# plot of the original housing price data. There is a plotting function in 
# the `tidyverse`, `geom_smooth`, that
# allows us to add a layer on our plot with the simple
# linear regression predicted line of best fit. By default `geom_smooth` adds some other information
# to the plot that we are not interested in at this point; we provide the argument `se = FALSE` to
# tell `geom_smooth` not to show that information. Figure \@ref(fig:08-lm-predict-all) displays the result.

# %% tags=["remove-output"]
lm_plot_final = (
    alt.Chart(sacramento_train)
    .mark_circle(color="black", opacity=0.3)
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

lm_plot_final += lm_plot_final.transform_regression("sqft", "price").mark_line(color="blue")

lm_plot_final

# %% tags=["remove-cell"]
glue("fig:08-lm-predict-all", lm_plot_final)

# %% [markdown]
# :::{glue:figure} fig:08-lm-predict-all
# :name: fig:08-lm-predict-all
#
# Scatter plot of sale price versus size with line of best fit for the full Sacramento housing data.
# :::

# %% tags=["remove-cell"]
# We can extract the coefficients from our model by accessing the
# fit object that is output by the `fit` \index{tidymodels!fit} function; we first have to extract
# it from the workflow using the `pull_workflow_fit` function, and then apply
# the `tidy` function to convert the result into a data frame:

# ```{r 08-lm-get-coeffs}
# coeffs <- lm_fit |>
#              pull_workflow_fit() |>
#              tidy()
# coeffs
# ```

# %% [markdown]
# ## Comparing simple linear and KNN regression
#
# Now that we have a general understanding of both simple linear and KNN
# regression, we \index{regression!comparison of methods} can start to compare and contrast these methods as well as the
# predictions made by them. To start, let's look at the visualization of the
# simple linear regression model predictions for the Sacramento real estate data
# (predicting price from house size) and the "best" KNN regression model
# obtained from the same problem, shown in {numref}`fig:08-compareRegression`.

# %% tags=["remove-cell"]
# preprocess the data, make the pipeline
sacr_preprocessor = make_column_transformer((StandardScaler(), ["sqft"]))
sacr_pipeline_knn = make_pipeline(
    sacr_preprocessor, KNeighborsRegressor(n_neighbors=31)
)  # 31 is the best parameter obtained in CV in regression1 chapter

sacr_pipeline_knn.fit(X_train, y_train)

# in-sample predictions (on training split)
sacr_preds_knn = sacramento_train
sacr_preds_knn = sacr_preds_knn.assign(
    knn_predicted=sacr_pipeline_knn.predict(sacramento_train)
)

sacr_rmse_knn = np.sqrt(
    mean_squared_error(
        y_true=sacr_preds_knn["price"], y_pred=sacr_preds_knn["knn_predicted"]
    )
)

# out-of-sample predictions (on test split)
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
    alt.Chart(sacr_preds_knn, title="KNN regression")
    .mark_circle(color="black", opacity=0.3)
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
    + knn_plot_final.mark_line(color="blue").encode(x="sqft", y="knn_predicted")
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
    .mark_circle(color="black", opacity=0.3)
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
    + lm_plot_final.transform_regression("sqft", "price").mark_line(color="blue")
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

# %% tags=["remove-cell"]
glue("fig:08-compareRegression", (lm_plot_final | knn_plot_final))

# %% [markdown]
# :::{glue:figure} fig:08-compareRegression
# :name: fig:08-compareRegression
#
# Comparison of simple linear regression and KNN regression.
# :::

# %% [markdown]
# What differences do we observe in {numref}`fig:08-compareRegression`? One obvious
# difference is the shape of the blue lines. In simple linear regression we are
# restricted to a straight line, whereas in KNN regression our line is much more
# flexible and can be quite wiggly. But there is a major interpretability advantage in limiting the
# model to a straight line. A 
# straight line can be defined by two numbers, the
# vertical intercept and the slope. The intercept tells us what the prediction is when
# all of the predictors are equal to 0; and the slope tells us what unit increase in the target/response
# variable we predict given a unit increase in the predictor
# variable. KNN regression, as simple as it is to implement and understand, has no such
# interpretability from its wiggly line. 
#
# There can, however, also be a disadvantage to using a simple linear regression
# model in some cases, particularly when the relationship between the target and
# the predictor is not linear, but instead some other shape (e.g., curved or oscillating). In 
# these cases the prediction model from a simple linear regression
# will underfit \index{underfitting!regression} (have high bias), meaning that model/predicted values do not
# match the actual observed values very well. Such a model would probably have a
# quite high RMSE when assessing model goodness of fit on the training data and
# a quite high RMSPE when assessing model prediction quality on a test data
# set. On such a data set, KNN regression may fare better. Additionally, there
# are other types of regression you can learn about in future books that may do
# even better at predicting with such data.
#
# How do these two models compare on the Sacramento house prices data set? In
# {numref}`fig:08-compareRegression`, we also printed the RMSPE as calculated from 
# predicting on the test data set that was not used to train/fit the models. The RMSPE for the simple linear
# regression model is slightly lower than the RMSPE for the KNN regression model.
# Considering that the simple linear regression model is also more interpretable,
# if we were comparing these in practice we would likely choose to use the simple
# linear regression model.
#
# Finally, note that the KNN regression model becomes "flat"
# at the left and right boundaries of the data, while the linear model
# predicts a constant slope. Predicting outside the range of the observed
# data is known as *extrapolation*; \index{extrapolation} KNN and linear models behave quite differently
# when extrapolating. Depending on the application, the flat
# or constant slope trend may make more sense. For example, if our housing
# data were slightly different, the linear model may have actually predicted 
# a *negative* price for a small house (if the intercept $\beta_0$ was negative),
# which obviously does not match reality. On the other hand, the trend of increasing
# house size corresponding to increasing house price probably continues for large houses, 
# so the "flat" extrapolation of KNN likely does not match reality.

# %% [markdown]
# ## Multivariable linear regression
#
# As in KNN classification and KNN regression, we can move beyond the simple
# case of only one predictor to the case with multiple predictors, 
# known as *multivariable linear regression*. \index{regression!multivariable linear}\index{regression!multivariable linear equation|see{plane equation}}
# To do this, we follow a very similar approach to what we did for
# KNN regression: we just add more predictors to the model formula in the 
# recipe. But recall that we do not need to use cross-validation to choose any parameters,
# nor do we need to standardize (i.e., center and scale) the data for linear regression. 
# Note once again that we have the same concerns regarding multiple predictors
#  as in the settings of multivariable KNN regression and classification: having more predictors is **not** always
# better. But because the same predictor selection 
# algorithm from the classification chapter extends to the setting of linear regression,
# it will not be covered again in this chapter.
#
# We will demonstrate multivariable linear regression using the Sacramento real estate  \index{Sacramento real estate}
# data with both house size
# (measured in square feet) as well as number of bedrooms as our predictors, and
# continue to use house sale price as our response variable. We will start by 
# changing the formula in the recipe to 
# include both the `sqft` and `beds` variables as predictors:

# %% [markdown]
# ```{r 08-lm-mult-test-train-split}
# mlm_recipe <- recipe(price ~ sqft + beds, data = sacramento_train)
# ```

# %% [markdown]
# Now we can build our workflow and fit the model:

# %% [markdown]
# ```{r 08-fitlm, results = 'hide', echo = TRUE}
# mlm_fit <- workflow() |>
#   add_recipe(mlm_recipe) |>
#   add_model(lm_spec) |>
#   fit(data = sacramento_train)
#
# mlm_fit
# ```
#
# ```{r echo = FALSE}
# print_tidymodels(mlm_fit)
# ```

# %% [markdown]
# And finally, we make predictions on the test data set to assess the quality of our model:

# %% [markdown]
# ```{r 08-assessFinal-multi}
# lm_mult_test_results <- mlm_fit |>
#   predict(sacramento_test) |>
#   bind_cols(sacramento_test) |>
#   metrics(truth = price, estimate = .pred)
#
# lm_mult_test_results
# ```

# %% [markdown]
# Our model's test error as assessed by RMSPE
# is `r format(round(lm_mult_test_results |> filter(.metric == 'rmse') |> pull(.estimate)), big.mark=",", nsmall=0, scientific=FALSE)`.
# In the case of two predictors, we can plot the predictions made by our linear regression creates a *plane* of best fit, as
# shown in Figure \@ref(fig:08-3DlinReg).

# %% [markdown]
# ```{r 08-3DlinReg, echo = FALSE, message = FALSE, warning = FALSE, fig.cap = "Linear regression plane of best fit overlaid on top of the data (using price, house size, and number of bedrooms as predictors). Note that in general we recommend against using 3D visualizations; here we use a 3D visualization only to illustrate what the regression plane looks like for learning purposes.", out.width="100%"}
# xvals <- seq(from = min(sacramento_train$sqft), 
#              to = max(sacramento_train$sqft), 
#              length = 50)
# yvals <- seq(from = min(sacramento_train$beds), 
#              to = max(sacramento_train$beds), 
#              length = 50)
#
# zvals <- mlm_fit |>
#   predict(crossing(xvals, yvals) |> 
#             mutate(sqft = xvals, beds = yvals)) |>
#   pull(.pred)
#
# zvalsm <- matrix(zvals, nrow = length(xvals))
#
# plot_3d <- plot_ly() |>
#   add_markers(
#     data = sacramento_train,
#     x = ~sqft,
#     y = ~beds,
#     z = ~price,
#     marker = list(size = 2, opacity = 0.4, color = "red")
#   ) |>
#   layout(scene = list(
#     xaxis = list(title = "Size (sq ft)"),
#     zaxis = list(title = "Price (USD)"),
#     yaxis = list(title = "Bedrooms")
#   )) |>
#   add_surface(
#     x = ~xvals,
#     y = ~yvals,
#     z = ~zvalsm,
#     colorbar = list(title = "Price (USD)")
#   )
#
# if(!is_latex_output()){  
#   plot_3d
# } else {
#   scene = list(camera = list(eye = list(x = -2.1, y = -2.2, z = 0.75)))
#   plot_3d <- plot_3d  |> layout(scene = scene)
#   save_image(plot_3d, "img/plot3d_linear_regression.png", scale = 10)
#   knitr::include_graphics("img/plot3d_linear_regression.png")
# }
# ```

# %% [markdown]
# We see that the predictions from linear regression with two predictors form a
# flat plane. This is the hallmark of linear regression, and differs from the 
# wiggly, flexible surface we get from other methods such as KNN regression. 
#  As discussed, this can be advantageous in one aspect, which is that for each
# predictor, we can get slopes/intercept from linear regression, and thus describe the
# plane mathematically. We can extract those slope values from our model object
# as shown below:

# %% [markdown]
# ```{r 08-lm-multi-get-coeffs}
# mcoeffs <- mlm_fit |>
#              pull_workflow_fit() |>
#              tidy()
#
# mcoeffs
# ```

# %% [markdown]
# And then use those slopes to write a mathematical equation to describe the prediction plane: \index{plane equation}
#
# $$\text{house sale price} = \beta_0 + \beta_1\cdot(\text{house size}) + \beta_2\cdot(\text{number of bedrooms}),$$
# where:
#
# - $\beta_0$ is the *vertical intercept* of the hyperplane (the price when both house size and number of bedrooms are 0)
# - $\beta_1$ is the *slope* for the first predictor (how quickly the price increases as you increase house size)
# - $\beta_2$ is the *slope* for the second predictor (how quickly the price increases as you increase the number of bedrooms)
#
# Finally, we can fill in the values for $\beta_0$, $\beta_1$ and $\beta_2$ from the model output above
# to create the equation of the plane of best fit to the data:

# %% [markdown]
# ```{r 08-lm-multi-get-coeffs-hidden, echo = FALSE}
# icept <- format(round(mcoeffs |> 
#                         filter(term == "(Intercept)") |> 
#                         pull(estimate)), scientific = FALSE)
# sqftc <- format(round(mcoeffs |> 
#                         filter(term == "sqft") |> 
#                         pull(estimate)), scientific = FALSE)
# bedsc <- format(round(mcoeffs |> 
#                         filter(term == "beds") |> 
#                         pull(estimate)), scientific = FALSE)
# ```

# %% [markdown]
# $$\text{house sale price} = `r icept` + `r sqftc`\cdot (\text{house size})  `r bedsc` \cdot (\text{number of bedrooms})$$
#
# This model is more interpretable than the multivariable KNN
# regression model; we can write a mathematical equation that explains how
# each predictor is affecting the predictions. But as always, we should 
# question how well multivariable linear regression is doing compared to
# the other tools we have, such as simple linear regression 
# and multivariable KNN regression. If this comparison is part of
# the model tuning process&mdash;for example, if we are trying
#  out many different sets of predictors for multivariable linear
# and KNN regression&mdash;we must perform this comparison using
# cross-validation on only our training data. But if we have already
# decided on a small number (e.g., 2 or 3) of tuned candidate models and
# we want to make a final comparison, we can do so by comparing the prediction
# error of the methods on the test data.

# %% [markdown]
# ```{r 08-get-RMSPE}
# lm_mult_test_results
# ```

# %% [markdown]
# We obtain an RMSPE \index{RMSPE} for the multivariable linear regression model 
# of `r format(lm_mult_test_results |> filter(.metric == 'rmse') |> pull(.estimate), big.mark=",", nsmall=0, scientific = FALSE)`. This prediction error
#  is less than the prediction error for the multivariable KNN regression model,
# indicating that we should likely choose linear regression for predictions of
# house sale price on this data set. Revisiting the simple linear regression model
# with only a single predictor from earlier in this chapter, we see that the RMSPE for that model was 
# `r format(lm_test_results |> filter(.metric == 'rmse') |> pull(.estimate), big.mark=",", nsmall=0, scientific = FALSE)`, 
# which is slightly higher than that of our more complex model. Our model with two predictors
# provided a slightly better fit on test data than our model with just one. 
# As mentioned earlier, this is not always the case: sometimes including more
# predictors can negatively impact the prediction performance on unseen 
# test data.

# %% [markdown]
# ## Multicollinearity and outliers
#
# What can go wrong when performing (possibly multivariable) linear regression? 
# This section will introduce two common issues&mdash;*outliers* and *collinear predictors*&mdash;and 
# illustrate their impact on predictions.

# %% [markdown]
# ### Outliers
#
# Outliers \index{outliers} are data points that do not follow the usual pattern of the rest of the data.
# In the setting of linear regression, these are points that
#  have a vertical distance to the line of best fit that is either much higher or much lower 
# than you might expect based on the rest of the data. The problem with outliers is that
# they can have *too much influence* on the line of best fit. In general, it is very difficult
# to judge accurately which data are outliers without advanced techniques that are beyond
# the scope of this book.
#
# But to illustrate what can happen when you have outliers, Figure \@ref(fig:08-lm-outlier)
# shows a small subset of the Sacramento housing data again, except we have added a *single* data point (highlighted
# in red). This house is 5,000 square feet in size, and sold for only \$50,000. Unbeknownst to the
# data analyst, this house was sold by a parent to their child for an absurdly low price. Of course,
# this is not representative of the real housing market values that the other data points follow; 
# the data point is an *outlier*. In blue we plot the original line of best fit, and in red
# we plot the new line of best fit including the outlier. You can see how different the red line
# is from the blue line, which is entirely caused by that one extra outlier data point.

# %% [markdown]
# ```{r 08-lm-outlier, fig.height = 3.5, fig.width = 4.5, message = FALSE, warning = FALSE, echo = FALSE, fig.cap = "Scatter plot of a subset of the data, with outlier highlighted in red."}
# sacramento_train_small <- sacramento_train |> sample_n(100)
# sacramento_outlier <- tibble(sqft = 5000, price = 50000)
#
# lm_plot_outlier <- ggplot(sacramento_train_small, aes(x = sqft, y = price)) +
#   geom_point(alpha = 0.4) +
#   geom_point(data = sacramento_outlier, 
#              mapping = aes(x = sqft, y = price), color="red", size = 2.5) +
#   xlab("House size (square feet)") +
#   ylab("Price (USD)") +
#   scale_y_continuous(labels = dollar_format()) +
#   geom_smooth(method = "lm", se = FALSE) + 
#   geom_smooth(data = sacramento_train_small |> 
#                 add_row(sqft = 5000, price = 50000), 
#               method = "lm", se = FALSE, color = "red")
#
# lm_plot_outlier
# ```

# %% [markdown]
# Fortunately, if you have enough data, the inclusion of one or two
# outliers&mdash;as long as their values are not *too* wild&mdash;will
# typically not have a large effect on the line of best fit. Figure
# \@ref(fig:08-lm-outlier-2) shows how that same outlier data point from earlier
# influences the line of best fit when we are working with the entire original
# Sacramento training data. You can see that with this larger data set, the line
# changes much less when adding the outlier.
# Nevertheless, it is still important when working with linear regression to critically
# think about how much any individual data point is influencing the model.

# %% [markdown]
# ```{r 08-lm-outlier-2, fig.height = 3.5, fig.width = 4.5, warning = FALSE, message = FALSE, echo = FALSE, fig.cap = "Scatter plot of the full data, with outlier highlighted in red."}
# sacramento_outlier <- tibble(sqft = 5000, price = 50000)
#
# lm_plot_outlier_large <- ggplot(sacramento_train, aes(x = sqft, y = price)) +
#   geom_point(alpha = 0.4) +
#   geom_point(data = sacramento_outlier, 
#              mapping = aes(x = sqft, y = price), 
#              color="red", 
#              size = 2.5) +
#   xlab("House size (square feet)") +
#   ylab("Price (USD)") +
#   scale_y_continuous(labels = dollar_format()) +
#   geom_smooth(method = "lm", se = FALSE) + 
#   geom_smooth(data = sacramento_train |> 
#                 add_row(sqft = 5000, price = 50000), 
#               method = "lm", 
#               se = FALSE, 
#               color = "red")
#
# lm_plot_outlier_large
# ```

# %% [markdown]
# ### Multicollinearity
#
# The second, and much more subtle, issue can occur when performing multivariable
# linear regression.  In particular, if you include multiple predictors that are \index{colinear}\index{multicolinear|see{colinear}}
# strongly linearly related to one another, the coefficients that describe the
# plane of best fit can be very unreliable&mdash;small changes to the data can
# result in large changes in the coefficients. Consider an extreme example using
# the Sacramento housing data where the house was measured twice by two people.
# Since the two people are each slightly inaccurate, the two measurements might
# not agree exactly, but they are very strongly linearly related to each other,
# as shown in Figure \@ref(fig:08-lm-multicol).

# %% [markdown]
# ```{r 08-lm-multicol, fig.height = 3.5, fig.width = 4.5, warning = FALSE, echo = FALSE, fig.cap = "Scatter plot of house size (in square inches) versus house size (in square feet)."}
# sacramento_train <- sacramento_train |>
#                           mutate(sqft1 = sqft + 100 * sample(1000000,
#                                                              size=nrow(sacramento_train),
#                                                              replace=TRUE)/1000000) |>
#                           mutate(sqft2 = sqft + 100 * sample(1000000, 
#                                                              size=nrow(sacramento_train),
#                                                              replace=TRUE)/1000000) |>
#                           mutate(sqft3 = sqft + 100 * sample(1000000, 
#                                                              size=nrow(sacramento_train),
#                                                              replace=TRUE)/1000000) 
#
# lm_plot_multicol_1 <- ggplot(sacramento_train) +
#   geom_point(aes(x = sqft, y = sqft1), alpha=0.4)+
#   xlab("House size (square feet)") +
#   ylab("House size (square inches)") 
# lm_plot_multicol_1
# ```

# %% [markdown]
# ```{r 08-lm-multicol-2, echo = FALSE, warning = FALSE}
# lm_recipe1 <- recipe(price ~ sqft + sqft1, data = sacramento_train)
#
# lm_fit1 <- workflow() |>
#   add_recipe(lm_recipe1) |>
#   add_model(lm_spec) |>
#   fit(data = sacramento_train)
#
# coeffs <- tidy(pull_workflow_fit(lm_fit1))
#
# icept1 <- format(round(coeffs |> 
#                          filter(term == "(Intercept)") |>  
#                          pull(estimate)), 
#                  scientific = FALSE)
# sqft1 <- format(round(coeffs |> 
#                         filter(term == "sqft") |>  
#                         pull(estimate)), 
#                 scientific = FALSE)
# sqft11 <- format(round(coeffs |> 
#                          filter(term == "sqft1") |>  
#                          pull(estimate)), 
#                  scientific = FALSE)
#
# lm_recipe2 <- recipe(price ~ sqft + sqft2, data = sacramento_train)
#
# lm_fit2 <- workflow() |>
#   add_recipe(lm_recipe2) |>
#   add_model(lm_spec) |>
#   fit(data = sacramento_train)
#
# coeffs <- tidy(pull_workflow_fit(lm_fit2))
# icept2 <- format(round(coeffs |> 
#                          filter(term == "(Intercept)") |> 
#                          pull(estimate)), 
#                  scientific = FALSE)
# sqft2 <- format(round(coeffs |> 
#                         filter(term == "sqft") |> 
#                         pull(estimate)), 
#                 scientific = FALSE)
# sqft22 <- format(round(coeffs |> 
#                          filter(term == "sqft2") |> 
#                          pull(estimate)), 
#                  scientific = FALSE)
#
# lm_recipe3 <- recipe(price ~ sqft + sqft3, data = sacramento_train)
#
# lm_fit3 <- workflow() |>
#   add_recipe(lm_recipe3) |>
#   add_model(lm_spec) |>
#   fit(data = sacramento_train)
#
# coeffs <- tidy(pull_workflow_fit(lm_fit3))
# icept3 <- format(round(coeffs |> 
#                          filter(term == "(Intercept)") |> 
#                          pull(estimate)), 
#                  scientific = FALSE)
# sqft3 <- format(round(coeffs |> 
#                         filter(term == "sqft") |> 
#                         pull(estimate)), 
#                 scientific = FALSE)
# sqft33 <- format(round(coeffs |> 
#                          filter(term == "sqft3") |> 
#                          pull(estimate)), 
#                  scientific = FALSE)
#
# ```

# %% [markdown]
#  If we again fit the multivariable linear regression model on this data, then the plane of best fit
# has regression coefficients that are very sensitive to the exact values in the data. For example,
# if we change the data ever so slightly&mdash;e.g., by running cross-validation, which splits
# up the data randomly into different chunks&mdash;the coefficients vary by large amounts:
#
# Best Fit 1: $\text{house sale price} = `r icept1` + `r sqft1`\cdot (\text{house size 1 (ft$^2$)}) + `r sqft11` \cdot (\text{house size 2 (ft$^2$)}).$
#
# Best Fit 2: $\text{house sale price} = `r icept2` + `r sqft2`\cdot (\text{house size 1 (ft$^2$)}) + `r sqft22` \cdot (\text{house size 2 (ft$^2$)}).$
#
# Best Fit 3: $\text{house sale price} = `r icept3` + `r sqft3`\cdot (\text{house size 1 (ft$^2$)}) + `r sqft33` \cdot (\text{house size 2 (ft$^2$)}).$
#
#  Therefore, when performing multivariable linear regression, it is important to avoid including very
# linearly related predictors. However, techniques for doing so are beyond the scope of this
# book; see the list of additional resources at the end of this chapter to find out where you can learn more.

# %% [markdown]
# ## Designing new predictors
#
# We were quite fortunate in our initial exploration to find a predictor variable (house size)
# that seems to have a meaningful and nearly linear relationship with our response variable (sale price).
# But what should we do if we cannot immediately find such a nice variable?
# Well, sometimes it is just a fact that the variables in the data do not have enough of 
# a relationship with the response variable to provide useful predictions. For example,
# if the only available predictor was "the current house owner's favorite ice cream flavor",
# we likely would have little hope of using that variable to predict the house's sale price
# (barring any future remarkable scientific discoveries about the relationship between 
# the housing market and homeowner ice cream preferences). In cases like these, 
# the only option is to obtain measurements of more useful variables.
#
# There are, however, a wide variety of cases where the predictor variables do have a 
# meaningful relationship with the response variable, but that relationship does not fit
# the assumptions of the regression method you have chosen. For example, a data frame `df`
# with two variables&mdash;`x` and `y`&mdash;with a nonlinear relationship between the two variables
# will not be fully captured by simple linear regression, as shown in Figure \@ref(fig:08-predictor-design).

# %% [markdown]
# ```{r 08-predictor-design-df, echo = FALSE, message = FALSE, warning = FALSE}
# set.seed(1)
#
# df = tibble(x = sample(10000, 100, replace=TRUE) / 10000)
#
# df <- df |> 
#   mutate(y = x^3 + 0.2 * sample(10000, 100, replace=TRUE)/10000 - 0.1)
# ```

# %% [markdown]
# ```{r 08-predictor-design-df-2, message = FALSE, warning = FALSE}
# df
# ```

# %% [markdown]
# ```{r 08-predictor-design, message = FALSE, warning = FALSE, echo = FALSE, fig.height = 3.5, fig.width = 4.5, fig.cap = "Example of a data set with a nonlinear relationship between the predictor and the response."}
# curve_plt <- ggplot(df, aes(x = x, y = y)) +
#   geom_point() +
#   xlab("x") +
#   ylab("y") +
#   geom_smooth(method = "lm", se = FALSE)
# curve_plt
# ```

# %% [markdown]
# Instead of trying to predict the response `y` using a linear regression on `x`,
# we might have some scientific background about our problem to suggest that `y`
# should be a cubic function of `x`. So before performing regression,
# we might *create a new predictor variable* `z` using the `mutate` function: \index{predictor design}

# %% [markdown]
# ```{r 08-predictor-design-3, message = FALSE, warning = FALSE}
# df <- df |>
#         mutate(z = x^3)
# ```

# %% [markdown]
# Then we can perform linear regression for `y` using the predictor variable `z`,
# as shown in Figure \@ref(fig:08-predictor-design-2).
# Here you can see that the transformed predictor `z` helps the 
# linear regression model make more accurate predictions. 
# Note that none of the `y` response values have changed between Figures \@ref(fig:08-predictor-design)
# and \@ref(fig:08-predictor-design-2); the only change is that the `x` values
# have been replaced by `z` values.

# %% [markdown]
# ```{r 08-predictor-design-2, message = FALSE, warning = FALSE, echo = FALSE, fig.height = 3.5, fig.width = 4.5, fig.cap = "Relationship between the transformed predictor and the response."}
# curve_plt2 <- ggplot(df, aes(x = z, y = y)) +
#   geom_point() +
#   xlab(expression(paste("z = x"^"3"))) +
#   ylab("y") +
#   geom_smooth(method = "lm", se = FALSE)
#
# curve_plt2
# ```

# %% [markdown]
# The process of
# transforming predictors (and potentially combining multiple predictors in the process)
# is known as *feature engineering*. \index{feature engineering|see{predictor design}} In real data analysis
# problems, you will need to rely on
# a deep understanding of the problem&mdash;as well as the wrangling tools 
# from previous chapters&mdash;to engineer useful new features that improve
# predictive performance.
#
# > **Note:** Feature engineering
# > is *part of tuning your model*, and as such you must not use your test data
# > to evaluate the quality of the features you produce. You are free to use
# > cross-validation, though!

# %% [markdown]
# ## The other sides of regression
#
# So far in this textbook we have used regression only in the context of
# prediction. However, regression can also be seen as a method to understand and
# quantify the effects of individual variables on a response / outcome of interest.
# In the housing example from this chapter, beyond just using past data
# to predict future sale prices, 
# we might also be interested in describing the
# individual relationships of house size and the number of bedrooms with house price,
# quantifying how strong each of these relationships are, and assessing how accurately we
# can estimate their magnitudes. And even beyond that, we may be interested in 
# understanding whether the predictors *cause* changes in the price.
# These sides of regression are well beyond the scope of this book; but
# the material you have learned here should give you a foundation of knowledge
# that will serve you well when moving to more advanced books on the topic.

# %% [markdown]
# ## Exercises
#
# Practice exercises for the material covered in this chapter 
# can be found in the accompanying 
# [worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
# in the "Regression II: linear regression" row.
# You can launch an interactive version of the worksheet in your browser by clicking the "launch binder" button.
# You can also preview a non-interactive version of the worksheet by clicking "view worksheet."
# If you instead decide to download the worksheet and run it on your own machine,
# make sure to follow the instructions for computer setup
# found in Chapter \@ref(move-to-your-own-machine). This will ensure that the automated feedback
# and guidance that the worksheets provide will function as intended.

# %% [markdown]
# ## Additional resources
# - The [`tidymodels` website](https://tidymodels.org/packages) is an excellent
#   reference for more details on, and advanced usage of, the functions and
#   packages in the past two chapters. Aside from that, it also has a [nice
#   beginner's tutorial](https://www.tidymodels.org/start/) and [an extensive list
#   of more advanced examples](https://www.tidymodels.org/learn/) that you can use
#   to continue learning beyond the scope of this book. 
# - *Modern Dive* [@moderndive] is another textbook that uses the
#   `tidyverse` / `tidymodels` framework. Chapter 6 complements the material in
#   the current chapter well; it covers some slightly more advanced concepts than
#   we do without getting mathematical. Give this chapter a read before moving on
#   to the next reference. It is also worth noting that this book takes a more
#   "explanatory" / "inferential" approach to regression in general (in Chapters 5,
#   6, and 10), which provides a nice complement to the predictive tack we take in
#   the present book.
# - *An Introduction to Statistical Learning* [@james2013introduction] provides 
#   a great next stop in the process of
#   learning about regression. Chapter 3 covers linear regression at a slightly
#   more mathematical level than we do here, but it is not too large a leap and so
#   should provide a good stepping stone. Chapter 6 discusses how to pick a subset
#   of "informative" predictors when you have a data set with many predictors, and
#   you expect only a few of them to be relevant. Chapter 7 covers regression
#   models that are more flexible than linear regression models but still enjoy the
#   computational efficiency of linear regression. In contrast, the KNN methods we
#   covered earlier are indeed more flexible but become very slow when given lots
#   of data.

# %% [markdown]
# ## References

# %% [markdown]
# ```{bibliography}
# ```
