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

(inference)=
# Statistical inference

```{code-cell} ipython3
:tags: [remove-cell]

from chapter_preamble import *
```

## Overview

A typical data analysis task in practice is to draw conclusions about some
unknown aspect of a population of interest based on observed data sampled from
that population; we typically do not get data on the *entire* population.  Data
analysis questions regarding how summaries, patterns, trends, or relationships
in a data set extend to the wider population are called *inferential
questions*. This chapter will start with the fundamental ideas of sampling from
populations and then introduce two common techniques in statistical inference:
*point estimation* and *interval estimation*.

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

- Describe real-world examples of questions that can be answered with statistical inference.
- Define common population parameters (e.g., mean, proportion, standard deviation) that are often estimated using sampled data, and estimate these from a sample.
- Define the following statistical sampling terms: population, sample, population parameter, point estimate, and sampling distribution.
- Explain the difference between a population parameter and a sample point estimate.
- Use Python to draw random samples from a finite population.
- Use Python to create a sampling distribution from a finite population.
- Describe how sample size influences the sampling distribution.
- Define bootstrapping.
- Use Python to create a bootstrap distribution to approximate a sampling distribution.
- Contrast the bootstrap and sampling distributions.

+++

## Why do we need sampling?

We often need to understand how quantities we observe in a subset
of data relate to the same quantities in the broader population. For example, suppose a
retailer is considering selling iPhone accessories, and they want to estimate
how big the market might be. Additionally, they want to strategize how they can
market their products on North American college and university campuses. This
retailer might formulate the following question:

*What proportion of all undergraduate students in North America own an iPhone?*

```{index} population, population; parameter
```

In the above question, we are interested in making a conclusion about *all*
undergraduate students in North America; this is referred to as the **population**. In
general, the population is the complete collection of individuals or cases we
are interested in studying.  Further, in the above question, we are interested
in computing a quantity&mdash;the proportion of iPhone owners&mdash;based on
the entire population. This proportion is referred to as a **population parameter**. In
general, a population parameter is a numerical characteristic of the entire
population. To compute this number in the example above, we would need to ask
every single undergraduate in North America whether they own an iPhone. In
practice, directly computing population parameters is often time-consuming and
costly, and sometimes impossible.

```{index} sample, sample; estimate, inference
```

```{index} see: statistical inference; inference
```

A more practical approach would be to make measurements for a **sample**, i.e., a
subset of individuals collected from the population. We can then compute a
**sample estimate**&mdash;a numerical characteristic of the sample&mdash;that
estimates the population parameter. For example, suppose we randomly selected
ten undergraduate students across North America (the sample) and computed the
proportion of those students who own an iPhone (the sample estimate). In that
case, we might suspect that proportion is a reasonable estimate of the
proportion of students who own an iPhone in the entire population.
{numref}`fig:11-population-vs-sample` illustrates this process.
In general, the process of using a sample to make a conclusion about the
broader population from which it is taken is referred to as **statistical inference**.

+++

```{figure} img/inference/population_vs_sample.png
:name: fig:11-population-vs-sample

The process of using a sample from a broader population to obtain a point estimate of a
population parameter. In this case, a sample of 10 individuals yielded 6 who own an iPhone, resulting
in an estimated population proportion of 60% iPhone owners. The actual population proportion in this example
illustration is 53.8%.
```

+++

Note that proportions are not the *only* kind of population parameter we might
be interested in. For example, suppose an undergraduate student studying at the University
of British Columbia in Canada is looking for an apartment
to rent. They need to create a budget, so they want to know something about
studio apartment rental prices in Vancouver, BC. This student might
formulate the following question:

*What is the average price-per-month of studio apartment rentals in Vancouver, Canada?*

In this case, the population consists of all studio apartment rentals in Vancouver, and the
population parameter is the *average price-per-month*. Here we used the average
as a measure of the center to describe the "typical value" of studio apartment
rental prices. But even within this one example, we could also be interested in
many other population parameters. For instance, we know that not every studio
apartment rental in Vancouver will have the same price per month. The student
might be interested in how much monthly prices vary and want to find a measure
of the rentals' spread (or variability), such as the standard deviation. Or perhaps the
student might be interested in the fraction of studio apartment rentals that
cost more than \$1000 per month. The question we want to answer will help us
determine the parameter we want to estimate. If we were somehow able to observe
the whole population of studio apartment rental offerings in Vancouver, we
could compute each of these numbers exactly; therefore, these are all
population parameters. There are many kinds of observations and population
parameters that you will run into in practice, but in this chapter, we will
focus on two settings:

1. Using categorical observations to estimate the proportion of a category
2. Using quantitative observations to estimate the average (or mean)

+++

## Sampling distributions

### Sampling distributions for proportions

```{index} Airbnb
```

We will look at an example using data from
[Inside Airbnb](http://insideairbnb.com/) {cite:p}`insideairbnb`. Airbnb is an online
marketplace for arranging vacation rentals and places to stay. The data set
contains listings for Vancouver, Canada, in September 2020. Our data
includes an ID number, neighborhood, type of room, the number of people the
rental accommodates, number of bathrooms, bedrooms, beds, and the price per
night.

```{code-cell} ipython3
import pandas as pd

airbnb = pd.read_csv("data/listings.csv")
airbnb
```

Suppose the city of Vancouver wants information about Airbnb rentals to help
plan city bylaws, and they want to know how many Airbnb places are listed as
entire homes and apartments (rather than as private or shared rooms). Therefore
they may want to estimate the true proportion of all Airbnb listings where the
room type is listed as "entire home or apartment." Of course, we usually
do not have access to the true population, but here let's imagine (for learning
purposes) that our data set represents the population of all Airbnb rental
listings in Vancouver, Canada.
We can find the proportion of listings for each room type
by using the `value_counts` function with the `normalize` parameter
as we did in previous chapters.

```{index} DataFrame; [], DataFrame; value_counts
```

```{code-cell} ipython3
airbnb["room_type"].value_counts(normalize=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("population_proportion", "{:.3f}".format(airbnb["room_type"].value_counts(normalize=True)["Entire home/apt"]))
```

We can see that the proportion of `Entire home/apt` listings in
the data set is {glue:text}`population_proportion`. This
value, {glue:text}`population_proportion`, is the population parameter. Remember, this
parameter value is usually unknown in real data analysis problems, as it is
typically not possible to make measurements for an entire population.

```{index} DataFrame; sample, seed;numpy.random.seed
```

Instead, perhaps we can approximate it with a small subset of data!
To investigate this idea, let's try randomly selecting 40 listings (*i.e.,* taking a random sample of
size 40 from our population), and computing the proportion for that sample.
We will use the `sample` method of the `DataFrame`
object to take the sample. The argument `n` of `sample` is the size of the sample to take
and since we are starting to use randomness here,
we are also setting the random seed via numpy to make the results reproducible.

```{code-cell} ipython3
import numpy as np


np.random.seed(155)

airbnb.sample(n=40)["room_type"].value_counts(normalize=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sample_1_proportion", "{:.3f}".format(airbnb.sample(n=40, random_state=155)["room_type"].value_counts(normalize=True)["Entire home/apt"]))
```

```{index} DataFrame; value_counts
```

Here we see that the proportion of entire home/apartment listings in this
random sample is {glue:text}`sample_1_proportion`. Wow&mdash;that's close to our
true population value! But remember, we computed the proportion using a random sample of size 40.
This has two consequences. First, this value is only an *estimate*, i.e., our best guess
of our population parameter using this sample.
Given that we are estimating a single value here, we often
refer to it as a **point estimate**.  Second, since the sample was random,
if we were to take *another* random sample of size 40 and compute the proportion for that sample,
we would not get the same answer:

```{code-cell} ipython3
airbnb.sample(n=40)["room_type"].value_counts(normalize=True)
```

Confirmed! We get a different value for our estimate this time.
That means that our point estimate might be unreliable. Indeed, estimates vary from sample to
sample due to **sampling variability**. But just how much
should we expect the estimates of our random samples to vary?
Or in other words, how much can we really trust our point estimate based on a single sample?

```{index} sampling distribution
```

To understand this, we will simulate many samples (much more than just two)
of size 40 from our population of listings and calculate the proportion of
entire home/apartment listings in each sample. This simulation will create
many sample proportions, which we can visualize using a histogram. The
distribution of the estimate for all possible samples of a given size (which we
commonly refer to as $n$) from a population is called
a **sampling distribution**. The sampling distribution will help us see how much we would
expect our sample proportions from this population to vary for samples of size 40.

```{index} DataFrame; sample
```

We again use the `sample` to take samples of size 40 from our
population of Airbnb listings. But this time we use a list comprehension
to repeat the operation multiple times (as we did previously in {numref}`Chapter %s <clustering>`).
In this case we repeat the operation 20,000 times to obtain 20,000 samples of size 40.
To make it clear which rows in the data frame come
which of the 20,000 samples, we also add a column called `replicate` with this information using the `assign` function,
introduced previously in {numref}`Chapter %s <wrangling>`.
The call to `concat` concatenates all the 20,000 data frames
returned from the list comprehension into a single big data frame.

```{code-cell} ipython3
samples = pd.concat([
    airbnb.sample(40).assign(replicate=n)
    for n in range(20_000)
])
samples
```

Since the column `replicate` indicates the replicate/sample number,
we can verify that we indeed seem to have 20,0000 samples
starting at sample 0 and ending at sample 19,999.

+++

Now that we have obtained the samples, we need to compute the
proportion of entire home/apartment listings in each sample.
We first group the data by the `replicate` variable&mdash;to group the
set of listings in each sample together&mdash;and then use `value_counts`
with `normalize=True` to compute the proportion in each sample.
Both the first and last few entries of the resulting data frame are printed
below to show that we end up with 20,000 point estimates, one for each of the 20,000 samples.

```{index} DataFrame;groupby, DataFrame;reset_index
```

```{code-cell} ipython3
(
    samples
    .groupby("replicate")
    ["room_type"]
    .value_counts(normalize=True)
)
```

The returned object is a series,
and as we have previously learned
we can use `reset_index` to change it to a data frame.
However,
there is one caveat here:
when we use the `value_counts` function
on a grouped series and try to `reset_index`
we will end up with two columns with the same name
and therefore get an error
(in this case, `room_type` will occur twice).
Fortunately,
there is a simple solution:
when we call `reset_index`,
we can specify the name of the new column
with the `name` parameter:

```{code-cell} ipython3
(
    samples
    .groupby("replicate")
    ["room_type"]
    .value_counts(normalize=True)
    .reset_index(name="sample_proportion")
)
```

Below we put everything together
and also filter the data frame to keep only the room types
that we are interested in.

```{code-cell} ipython3
sample_estimates = (
    samples
    .groupby("replicate")
    ["room_type"]
    .value_counts(normalize=True)
    .reset_index(name="sample_proportion")
)

sample_estimates = sample_estimates[sample_estimates["room_type"] == "Entire home/apt"]
sample_estimates
```

We can now visualize the sampling distribution of sample proportions
for samples of size 40 using a histogram in {numref}`fig:11-example-proportions7`. Keep in mind: in the real world,
we don't have access to the full population. So we
can't take many samples and can't actually construct or visualize the sampling distribution.
We have created this particular example
such that we *do* have access to the full population, which lets us visualize the
sampling distribution directly for learning purposes.

```{code-cell} ipython3
:tags: [remove-output]

sampling_distribution = alt.Chart(sample_estimates).mark_bar().encode(
    x=alt.X("sample_proportion")
        .bin(maxbins=20)
        .title("Sample proportions"),
    y=alt.Y("count()").title("Count"),
)

sampling_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-proportions7", sampling_distribution)
```

:::{glue:figure} fig:11-example-proportions7
:name: fig:11-example-proportions7

Sampling distribution of the sample proportion for sample size 40.
:::

```{code-cell} ipython3
:tags: [remove-cell]

glue("sample_proportion_center", "{:.2f}".format(sample_estimates["sample_proportion"].mean()))
glue("sample_proportion_min", "{:.2f}".format(sample_estimates["sample_proportion"].quantile(0.004)))
glue("sample_proportion_max", "{:.2f}".format(sample_estimates["sample_proportion"].quantile(0.9997)))
```

```{index} sampling distribution; shape
```

The sampling distribution in {numref}`fig:11-example-proportions7` appears
to be bell-shaped, is roughly symmetric, and has one peak. It is centered
around {glue:text}`sample_proportion_center` and the sample proportions
range from about {glue:text}`sample_proportion_min` to about
{glue:text}`sample_proportion_max`. In fact, we can
calculate the mean of the sample proportions.

```{code-cell} ipython3
sample_estimates["sample_proportion"].mean()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sample_proportion_mean", "{:.3f}".format(sample_estimates["sample_proportion"].mean()))
```

We notice that the sample proportions are centered around the population
proportion value, {glue:text}`sample_proportion_mean`! In general, the mean of
the sampling distribution should be equal to the population proportion.
This is great news because it means that the sample proportion is neither an overestimate nor an
underestimate of the population proportion.
In other words, if you were to take many samples as we did above, there is no tendency
towards over or underestimating the population proportion.
In a real data analysis setting where you just have access to your single
sample, this implies that you would suspect that your sample point estimate is
roughly equally likely to be above or below the true population proportion.

+++

### Sampling distributions for means

In the previous section, our variable of interest&mdash;`room_type`&mdash;was
*categorical*, and the population parameter was a proportion. As mentioned in
the chapter introduction, there are many choices of the population parameter
for each type of variable. What if we wanted to infer something about a
population of *quantitative* variables instead? For instance, a traveler
visiting Vancouver, Canada may wish to estimate the
population *mean* (or average) price per night of Airbnb listings. Knowing
the average could help them tell whether a particular listing is overpriced.
We can visualize the population distribution of the price per night with a histogram.

```{code-cell} ipython3
:tags: [remove-output]

population_distribution = alt.Chart(airbnb).mark_bar().encode(
    x=alt.X("price")
        .bin(maxbins=30)
        .title("Price per night (dollars)"),
    y=alt.Y("count()", title="Count"),
)

population_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means2", population_distribution)
```

:::{glue:figure} fig:11-example-means2
:name: fig:11-example-means2

Population distribution of price per night (dollars) for all Airbnb listings in Vancouver, Canada.
:::

+++

```{index} population; distribution
```

In {numref}`fig:11-example-means2`, we see that the population distribution
has one peak. It is also skewed (i.e., is not symmetric): most of the listings are
less than \$250 per night, but a small number of listings cost much more,
creating a long tail on the histogram's right side.
Along with visualizing the population, we can calculate the population mean,
the average price per night for all the Airbnb listings.

```{code-cell} ipython3
airbnb["price"].mean()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("population_mean", "{:.2f}".format(airbnb["price"].mean()))
```

```{index} population; parameter
```

The price per night of all Airbnb rentals in Vancouver, BC
is \${glue:text}`population_mean`, on average. This value is our
population parameter since we are calculating it using the population data.

```{index} DataFrame; sample
```

Now suppose we did not have access to the population data (which is usually the
case!), yet we wanted to estimate the mean price per night. We could answer
this question by taking a random sample of as many Airbnb listings as our time
and resources allow. Let's say we could do this for 40 listings. What would
such a sample look like?  Let's take advantage of the fact that we do have
access to the population data and simulate taking one random sample of 40
listings in Python, again using `sample`.

```{code-cell} ipython3
one_sample = airbnb.sample(n=40)
```

We can create a histogram to visualize the distribution of observations in the
sample ({numref}`fig:11-example-means-sample-hist`), and calculate the mean
of our sample.

```{index} altair;mark_bar
```

```{code-cell} ipython3
:tags: [remove-output]

sample_distribution = alt.Chart(one_sample).mark_bar().encode(
    x=alt.X("price")
        .bin(maxbins=30)
        .title("Price per night (dollars)"),
    y=alt.Y("count()").title("Count"),
)

sample_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means-sample-hist", sample_distribution)
```

:::{glue:figure} fig:11-example-means-sample-hist
:name: fig:11-example-means-sample-hist

Distribution of price per night (dollars) for sample of 40 Airbnb listings.
:::

```{code-cell} ipython3
one_sample["price"].mean()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("estimate_mean", "{:.2f}".format(one_sample["price"].mean()))
glue("diff_perc", "{:.1f}".format(100 * abs(1 - (one_sample["price"].mean() / airbnb["price"].mean()))))
```

The average value of the sample of size 40
is \${glue:text}`estimate_mean`.  This
number is a point estimate for the mean of the full population.
Recall that the population mean was
\${glue:text}`population_mean`. So our estimate was fairly close to
the population parameter: the mean was about
{glue:text}`diff_perc`%
off.  Note that we usually cannot compute the estimate's accuracy in practice
since we do not have access to the population parameter; if we did, we wouldn't
need to estimate it!

```{index} sampling distribution
```

Also, recall from the previous section that the point estimate can vary; if we
took another random sample from the population, our estimate's value might
change. So then, did we just get lucky with our point estimate above?  How much
does our estimate vary across different samples of size 40 in this example?
Again, since we have access to the population, we can take many samples and
plot the sampling distribution of sample means to get a sense for this variation.
In this case, we'll use the 20,000 samples of size
40 that we already stored in the `samples` variable.
First we will calculate the sample mean for each replicate
and then plot the sampling
distribution of sample means for samples of size 40.

```{code-cell} ipython3
sample_estimates = (
    samples
    .groupby("replicate")
    ["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "mean_price"})
)
sample_estimates
```

```{code-cell} ipython3
:tags: [remove-output]

sampling_distribution = alt.Chart(sample_estimates).mark_bar().encode(
    x=alt.X("mean_price")
        .bin(maxbins=30)
        .title("Sample mean price per night (dollars)"),
    y=alt.Y("count()").title("Count")
)

sampling_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means4", sampling_distribution)
```

:::{glue:figure} fig:11-example-means4
:name: fig:11-example-means4

Sampling distribution of the sample means for sample size of 40.
:::

```{code-cell} ipython3
:tags: [remove-cell]

glue("quantile_1", "{:0.0f}".format(round(sample_estimates["mean_price"].quantile(0.25), -1)))
glue("quantile_3", "{:0.0f}".format(round(sample_estimates["mean_price"].quantile(0.75), -1)))
```

```{index} sampling distribution; shape
```

In {numref}`fig:11-example-means4`, the sampling distribution of the mean
has one peak and is bell-shaped. Most of the estimates are between
about  \${glue:text}`quantile_1` and
\${glue:text}`quantile_3`; but there is
a good fraction of cases outside this range (i.e., where the point estimate was
not close to the population parameter). So it does indeed look like we were
quite lucky when we estimated the population mean with only
{glue:text}`diff_perc`% error.

```{index} sampling distribution; compared to population distribution
```

Let's visualize the population distribution, distribution of the sample, and
the sampling distribution on one plot to compare them in {numref}`fig:11-example-means5`. Comparing these three distributions, the centers
of the distributions are all around the same price (around \$150). The original
population distribution has a long right tail, and the sample distribution has
a similar shape to that of the population distribution. However, the sampling
distribution is not shaped like the population or sample distribution. Instead,
it has a bell shape, and it has a lower spread than the population or sample
distributions. The sample means vary less than the individual observations
because there will be some high values and some small values in any random
sample, which will keep the average from being too extreme.

<!---
```{r 11-example-means4.5}
sample_estimates |>
  summarize(mean_of_sample_means = mean(sample_mean))
```
Notice that the mean of the sample means is \$`r round(mean(sample_estimates$sample_mean),2)`. Recall that the population mean
was \$`r round(mean(airbnb$price),2)`.
-->

```{code-cell} ipython3
:tags: ["remove-cell"]

glue(
    "fig:11-example-means5",
    alt.vconcat(
        population_distribution.mark_bar(clip=True).encode(
            x=alt.X(
                "price",
                bin=alt.Bin(extent=[0, 660], maxbins=40),
                title="Price per night (dollars)",
                #scale=alt.Scale(domainMax=700)
            )
        ).properties(
            title="Population", height=150
        ),
        sample_distribution.encode(
            x=alt.X("price")
                .bin(extent=[0, 660], maxbins=40)
                .title("Price per night (dollars)")
        ).properties(title="Sample (n = 40)").properties(height=150),
        sampling_distribution.encode(
            x=alt.X("mean_price")
                .bin(extent=[0, 660], maxbins=40)
                .title("Price per night (dollars)")
        ).properties(
            title=alt.TitleParams(
                "Sampling distribution of the mean",
                subtitle="For 20,000 samples of size 40"
            )
        ).properties(height=150)
    ).resolve_scale(
        x="shared"
    )
)
```

:::{glue:figure} fig:11-example-means5
:name: fig:11-example-means5

Comparison of population distribution, sample distribution, and sampling distribution.
:::


+++

Given that there is quite a bit of variation in the sampling distribution of
the sample mean&mdash;i.e., the point estimate that we obtain is not very
reliable&mdash;is there any way to improve the estimate?  One way to improve a
point estimate is to take a *larger* sample. To illustrate what effect this
has, we will take many samples of size 20, 50, 100, and 500, and plot the
sampling distribution of the sample mean. We indicate the mean of the sampling
distribution with a vertical line.

```{code-cell} ipython3
:tags: ["remove-cell"]

# Plot sampling distributions for multiple sample sizes
base = alt.Chart(
    pd.concat([
        pd.concat([
            airbnb.sample(sample_size).assign(sample_size=sample_size, replicate=replicate)
            for sample_size in [20, 50, 100, 500]
        ])
        for replicate in range(20_000)
    ]).groupby(
        ["sample_size", "replicate"],
        as_index=False
    )["price"].mean(),
    height=150
)

glue(
    "fig:11-example-means7",
    alt.layer(
        base.mark_bar().encode(
            alt.X("price", bin=alt.Bin(maxbins=30)),
            alt.Y("count()")
        ),
        base.mark_rule(color="black", size=1.5, strokeDash=[6]).encode(
            x="mean(price)"
        ),
        base.mark_text(align="left", color="black", size=12, fontWeight="bold", dx=10).transform_aggregate(
            mean_price="mean(price)",
        ).transform_calculate(
            label="'Mean = ' + round(datum.mean_price * 10) / 10"
        ).encode(
            x=alt.X("mean_price:Q", title="Sample mean price per night (dollars)"),
            y=alt.value(10),
            text="label:N"
        )
    ).facet(
        alt.Facet(
            "sample_size:N",
            header=alt.Header(
                title="",
                labelFontWeight="bold",
                labelFontSize=12,
                labelPadding=3,
                labelExpr='"Sample size = " + datum.value'
            )
        ),
        columns=1,
    ).resolve_scale(
        y="independent"
    )
)
```

:::{glue:figure} fig:11-example-means7
:name: fig:11-example-means7

Comparison of sampling distributions, with mean highlighted as a vertical line.
:::

+++

```{index} sampling distribution; effect of sample size
```

Based on the visualization in {numref}`fig:11-example-means7`, three points
about the sample mean become clear:

1. The mean of the sample mean (across
   samples) is equal to the population mean. In other words, the sampling
   distribution is centered at the population mean.
2. Increasing the size of
   the sample decreases the spread (i.e., the variability) of the sampling
   distribution. Therefore, a larger sample size results in a more reliable point
   estimate of the population parameter.
3. The distribution of the sample mean is roughly bell-shaped.

```{note}
You might notice that in the `n = 20` case in {numref}`fig:11-example-means7`,
the distribution is not *quite* bell-shaped. There is a bit of skew towards the right!
You might also notice that in the `n = 50` case and larger, that skew seems to disappear.
In general, the sampling distribution&mdash;for both means and proportions&mdash;only
becomes bell-shaped *once the sample size is large enough*.
How large is "large enough?" Unfortunately, it depends entirely on the problem at hand. But
as a rule of thumb, often a sample size of at least 20 will suffice.
```

<!---
```{note}
If random samples of size $n$ are taken from a population, the sample mean
$\bar{x}$ will be approximately Normal with mean $\mu$ and standard deviation
$\frac{\sigma}{\sqrt{n}}$ as long as the sample size $n$ is large enough. $\mu$
is the population mean, $\sigma$ is the population standard deviation,
$\bar{x}$ is the sample mean, and $n$ is the sample size.
If samples are selected from a finite population as we are doing in this
chapter, we should apply a finite population correction. We multiply
$\frac{\sigma}{\sqrt{n}}$ by $\sqrt{\frac{N - n}{N - 1}}$ where $N$ is the
population size and $n$ is the sample size. If our sample size, $n$, is small
relative to the population size, this finite correction factor is less
important.
```
--->

+++

### Summary

1. A point estimate is a single value computed using a sample from a population (e.g., a mean or proportion).
2. The sampling distribution of an estimate is the distribution of the estimate for all possible samples of a fixed size from the same population.
3. The shape of the sampling distribution is usually bell-shaped with one peak and centered at the population mean or proportion.
4. The spread of the sampling distribution is related to the sample size. As the sample size increases, the spread of the sampling distribution decreases.

+++

## Bootstrapping

+++

### Overview

*Why all this emphasis on sampling distributions?*

We saw in the previous section that we could compute a **point estimate** of a
population parameter using a sample of observations from the population. And
since we constructed examples where we had access to the population, we could
evaluate how accurate the estimate was, and even get a sense of how much the
estimate would vary for different samples from the population.  But in real
data analysis settings, we usually have *just one sample* from our population
and do not have access to the population itself. Therefore we cannot construct
the sampling distribution as we did in the previous section. And as we saw, our
sample estimate's value can vary significantly from the population parameter.
So reporting the point estimate from a single sample alone may not be enough.
We also need to report some notion of *uncertainty* in the value of the point
estimate.

```{index} bootstrap, confidence interval
```

```{index} see: interval; confidence interval
```

Unfortunately, we cannot construct the exact sampling distribution without
full access to the population. However, if we could somehow *approximate* what
the sampling distribution would look like for a sample, we could
use that approximation to then report how uncertain our sample
point estimate is (as we did above with the *exact* sampling
distribution). There are several methods to accomplish this; in this book, we
will use the *bootstrap*. We will discuss **interval estimation** and
construct
**confidence intervals** using just a single sample from a population. A
confidence interval is a range of plausible values for our population parameter.

Here is the key idea. First, if you take a big enough sample, it *looks like*
the population. Notice the histograms' shapes for samples of different sizes
taken from the population in {numref}`fig:11-example-bootstrapping0`. We
see that the sample’s distribution looks like that of the population for a
large enough sample.

```{code-cell} ipython3
:tags: [remove-cell]

# plot sample distributions for n = 10, 20, 50, 100, 200 and population distribution
sample_distribution_dict = {}
for sample_n in [10, 20, 50, 100, 200]:
    sample = airbnb.sample(sample_n)
    sample_distribution_dict[f"sample_distribution_{sample_n}"] = (
        alt.Chart(sample, title=f"n = {sample_n}").mark_bar().encode(
            x=alt.X(
                "price",
                bin=alt.Bin(extent=[0, 600], step=20),
                title="Price per night (dollars)",
            ),
            y=alt.Y("count()", title="Count"),
        )
    ).properties(height=150)
# add title and standardize the x axis ticks for population histogram
population_distribution.title = "Population distribution"
population_distribution.encoding["x"]["bin"] = alt.Bin(extent=[0, 600], step=20)

glue(
    "fig:11-example-bootstrapping0",
    (
        (
            sample_distribution_dict["sample_distribution_10"]
            | sample_distribution_dict["sample_distribution_20"]
        )
        & (
            sample_distribution_dict["sample_distribution_50"]
            | sample_distribution_dict["sample_distribution_100"]
        )
        & (
            sample_distribution_dict["sample_distribution_200"]
            | population_distribution.properties(width=350, height=150)
        )
    ),
)
```

:::{glue:figure} fig:11-example-bootstrapping0
:name: fig:11-example-bootstrapping0

Comparison of samples of different sizes from the population.
:::

+++

```{index} bootstrap; distribution
```

In the previous section, we took many samples of the same size *from our
population* to get a sense of the variability of a sample estimate. But if our
sample is big enough that it looks like our population, we can pretend that our
sample *is* the population, and take more samples (with replacement) of the
same size from it instead! This very clever technique is
called **the bootstrap**.  Note that by taking many samples from our single, observed
sample, we do not obtain the true sampling distribution, but rather an
approximation that we call **the bootstrap distribution**.

```{note}
We must sample *with* replacement when using the bootstrap.
Otherwise, if we had a sample of size $n$, and obtained a sample from it of
size $n$ *without* replacement, it would just return our original sample!
```

This section will explore how to create a bootstrap distribution from a single
sample using Python.  The process is visualized in {numref}`fig:11-intro-bootstrap-image`.
For a sample of size $n$, you would do the following:

+++

1. Randomly select an observation from the original sample, which was drawn from the population.
2. Record the observation's value.
3. Replace that observation.
4. Repeat steps 1&ndash;3 (sampling *with* replacement) until you have $n$ observations, which form a bootstrap sample.
5. Calculate the bootstrap point estimate (e.g., mean, median, proportion, slope, etc.) of the $n$ observations in your bootstrap sample.
6. Repeat steps 1&ndash;5 many times to create a distribution of point estimates (the bootstrap distribution).
7. Calculate the plausible range of values around our observed point estimate.

+++

```{figure} img/inference/intro-bootstrap.jpeg
:name: fig:11-intro-bootstrap-image

Overview of the bootstrap process.
```

+++

### Bootstrapping in Python

Let’s continue working with our Airbnb example to illustrate how we might create
and use a bootstrap distribution using just a single sample from the population.
Once again, suppose we are
interested in estimating the population mean price per night of all Airbnb
listings in Vancouver, Canada, using a single sample size of 40.
Recall our point estimate was \${glue:text}`estimate_mean`. The
histogram of prices in the sample is displayed in {numref}`fig:11-bootstrapping1`.

```{code-cell} ipython3
one_sample
```

```{code-cell} ipython3
:tags: ["remove-output"]
one_sample_dist = alt.Chart(one_sample).mark_bar().encode(
    x=alt.X("price")
        .bin(maxbins=30)
        .title("Price per night (dollars)"),
    y=alt.Y("count()").title("Count"),
)

one_sample_dist
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping1", one_sample_dist)
```

:::{glue:figure} fig:11-bootstrapping1
:name: fig:11-bootstrapping1

Histogram of price per night (dollars) for one sample of size 40.
:::

+++

The histogram for the sample is skewed, with a few observations out to the right. The
mean of the sample is \${glue:text}`estimate_mean`.
Remember, in practice, we usually only have this one sample from the population. So
this sample and estimate are the only data we can work with.

```{index} bootstrap; in Python, DataFrame; sample (bootstrap)
```

We now perform steps 1&ndash;5 listed above to generate a single bootstrap
sample in Python and calculate a point estimate from that bootstrap sample. We will
continue using the `sample` function of our dataframe,
Critically, note that we now
set `frac=1` ("fraction") to indicate that we want to draw as many samples as there are rows in the dataframe
(we could also have set `n=40` but then we would need to manually keep track of how many rows there are).
Since we need to sample with replacement when bootstrapping,
we change the `replace` parameter to `True`.

```{code-cell} ipython3
:tags: ["remove-output"]

boot1 = one_sample.sample(frac=1, replace=True)
boot1_dist = alt.Chart(boot1).mark_bar().encode(
    x=alt.X("price")
        .bin(maxbins=30)
        .title("Price per night (dollars)"),
    y=alt.Y("count()", title="Count"),
)

boot1_dist
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping3", boot1_dist)
```

:::{glue:figure} fig:11-bootstrapping3
:name: fig:11-bootstrapping3

Bootstrap distribution.
:::

```{code-cell} ipython3
boot1["price"].mean()
```

Notice in {numref}`fig:11-bootstrapping3` that the histogram of our bootstrap sample
has a similar shape to the original sample histogram. Though the shapes of
the distributions are similar, they are not identical. You'll also notice that
the original sample mean and the bootstrap sample mean differ. How might that
happen? Remember that we are sampling with replacement from the original
sample, so we don't end up with the same sample values again. We are *pretending*
that our single sample is close to the population, and we are trying to
mimic drawing another sample from the population by drawing one from our original
sample.

Let's now take 20,000 bootstrap samples from the original sample (`one_sample`)
and calculate the means for
each of those replicates. Recall that this assumes that `one_sample` *looks like*
our original population; but since we do not have access to the population itself,
this is often the best we can do.
Note that here we break the list comprehension over multiple lines
so that it is easier to read.

```{code-cell} ipython3
boot20000 = pd.concat([
    one_sample.sample(frac=1, replace=True).assign(replicate=n)
    for n in range(20_000)
])
boot20000
```

Let's take a look at the histograms of the first six replicates of our bootstrap samples.

```{code-cell} ipython3
:tags: ["remove-output"]

six_bootstrap_samples = boot20000.query("replicate < 6")
six_bootstrap_fig = alt.Chart(six_bootstrap_samples, height=150).mark_bar().encode(
    x=alt.X("price")
        .bin(maxbins=20)
        .title("Price per night (dollars)"),
    y=alt.Y("count()").title("Count")
).facet(
    "replicate:N",  # Recall that `:N` converts the variable to a categorical type
    columns=2
)
six_bootstrap_fig
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping-six-bootstrap-samples", six_bootstrap_fig)
```

:::{glue:figure} fig:11-bootstrapping-six-bootstrap-samples
:name: fig:11-bootstrapping-six-bootstrap-samples

Histograms of the first six replicates of the bootstrap samples.
:::

+++

We see in {numref}`fig:11-bootstrapping-six-bootstrap-samples` how the distributions of the
bootstrap samples differ. If we calculate the sample mean for each of
these six samples, we can see that these are also different between samples.
To compute the mean for each sample,
we first group by the "replicate" which is the column containing the sample/replicate number.
Then we compute the mean of the `price` column and rename it to `mean_price`
for it to be more descriptive.
Finally we use `reset_index` to get the `replicate` values back as a column in the dataframe.

```{code-cell} ipython3
(
    six_bootstrap_samples
    .groupby("replicate")
    ["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "mean_price"})
)
```

The distributions and the means differ between the bootstrapped samples
because we are sampling *with replacement*.
If we instead would have sampled *without replacement*,
we would end up with the exact same values in the sample each time.

We will now calculate point estimates of the mean for our 20,000 bootstrap samples and
generate a bootstrap distribution of these point estimates. The bootstrap
distribution ({numref}`fig:11-bootstrapping5`) suggests how we might expect
our point estimate to behave if we take multiple samples.

```{index} DataFrame;reset_index, DataFrame;rename, DataFrame;groupby, Series;mean
```

```{code-cell} ipython3
boot20000_means = (
    boot20000
    .groupby("replicate")
    ["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "mean_price"})
)

boot20000_means
```

```{code-cell} ipython3
:tags: ["remove-output"]

boot_est_dist = alt.Chart(boot20000_means).mark_bar().encode(
    x=alt.X("mean_price")
        .bin(maxbins=20)
        .title("Sample mean price per night (dollars)"),
    y=alt.Y("count()").title("Count"),
)

boot_est_dist
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping5", boot_est_dist)
```

:::{glue:figure} fig:11-bootstrapping5
:name: fig:11-bootstrapping5

Distribution of the bootstrap sample means.
:::

+++

Let's compare the bootstrap distribution&mdash;which we construct by taking many samples from our original sample of size 40&mdash;with
the true sampling distribution&mdash;which corresponds to taking many samples from the population.

```{code-cell} ipython3
:tags: [remove-cell]

sampling_distribution.encoding.x["bin"]["extent"] = (90, 250)
bootstr6fig = alt.vconcat(
    alt.layer(
        sampling_distribution,
        alt.Chart(sample_estimates).mark_rule(color="black", size=1.5, strokeDash=[6]).encode(x="mean(mean_price)"),
        alt.Chart(sample_estimates).mark_text(color="black", size=12, align="left", dx=16, fontWeight="bold").encode(
            x="mean(mean_price)",
            y=alt.value(7),
            text=alt.value(f"Mean = {sampling_distribution['data']['mean_price'].mean().round(1)}")
        )
    ).properties(title="Sampling distribution", height=150),
    alt.layer(
        boot_est_dist,
        alt.Chart(boot20000_means).mark_rule(color="black", size=1.5, strokeDash=[6]).encode(x="mean(mean_price)"),
        alt.Chart(boot20000_means).mark_text(color="black", size=12, align="left", dx=18, fontWeight="bold").encode(
            x="mean(mean_price)",
            y=alt.value(7),
            text=alt.value(f"Mean = {boot_est_dist['data']['mean_price'].mean().round(1)}")
        )
    ).properties(title="Bootstrap distribution", height=150)
)
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping6", bootstr6fig)
```

:::{glue:figure} fig:11-bootstrapping6
:name: fig:11-bootstrapping6

Comparison of the distribution of the bootstrap sample means and sampling distribution.
:::



```{code-cell} ipython3
:tags: [remove-cell]

glue("one_sample_mean", "{:.2f}".format(one_sample["price"].mean()))
```

```{index} sampling distribution; compared to bootstrap distribution
```

There are two essential points that we can take away from
{numref}`fig:11-bootstrapping6`. First, the shape and spread of the true sampling
distribution and the bootstrap distribution are similar; the bootstrap
distribution lets us get a sense of the point estimate's variability. The
second important point is that the means of these two distributions are
slightly different. The sampling distribution is centered at
\${glue:text}`population_mean`, the population mean value. However, the bootstrap
distribution is centered at the original sample's mean price per night,
\${glue:text}`one_sample_mean`. Because we are resampling from the
original sample repeatedly, we see that the bootstrap distribution is centered
at the original sample's mean value (unlike the sampling distribution of the
sample mean, which is centered at the population parameter value).

{numref}`fig:11-bootstrapping7` summarizes the bootstrapping process.
The idea here is that we can use this distribution of bootstrap sample means to
approximate the sampling distribution of the sample means when we only have one
sample. Since the bootstrap distribution pretty well approximates the sampling
distribution spread, we can use the bootstrap spread to help us develop a
plausible range for our population parameter along with our estimate!

```{figure} img/inference/11-bootstrapping7-1.png
:name: fig:11-bootstrapping7

Summary of bootstrapping process.
```

+++

### Using the bootstrap to calculate a plausible range

```{index} confidence interval
```

Now that we have constructed our bootstrap distribution, let's use it to create
an approximate 95\% percentile bootstrap confidence interval.
A **confidence interval** is a range of plausible values for the population parameter. We will
find the range of values covering the middle 95\% of the bootstrap
distribution, giving us a 95\% confidence interval.  You may be wondering, what
does "95\% confidence" mean? If we took 100 random samples and calculated 100
95\% confidence intervals, then about 95\% of the ranges would capture the
population parameter's value.  Note there's nothing special about 95\%. We
could have used other levels, such as 90\% or 99\%. There is a balance between
our level of confidence and precision. A higher confidence level corresponds to
a wider range of the interval, and a lower confidence level corresponds to a
narrower range. Therefore the level we choose is based on what chance we are
willing to take of being wrong based on the implications of being wrong for our
application. In general, we choose confidence levels to be comfortable with our
level of uncertainty but not so strict that the interval is unhelpful. For
instance, if our decision impacts human life and the implications of being
wrong are deadly, we may want to be very confident and choose a higher
confidence level.

To calculate a 95\% percentile bootstrap confidence interval, we will do the following:

1. Arrange the observations in the bootstrap distribution in ascending order.
2. Find the value such that 2.5\% of observations fall below it (the 2.5\% percentile). Use that value as the lower bound of the interval.
3. Find the value such that 97.5\% of observations fall below it (the 97.5\% percentile). Use that value as the upper bound of the interval.

To do this in Python, we can use the `quantile` function of our DataFrame.
Quantiles are expressed in proportions rather than percentages,
so the 2.5th and 97.5th percentiles
would be the 0.025 and 0.975 quantiles, respectively.

```{index} DataFrame; [], DataFrame;quantile
```

```{index} percentile
```

```{code-cell} ipython3
ci_bounds = boot20000_means["mean_price"].quantile([0.025, 0.975])
ci_bounds
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("ci_lower", "{:.2f}".format(ci_bounds[0.025]))
glue("ci_upper", "{:.2f}".format(ci_bounds[0.975]))
```

Our interval, \${glue:text}`ci_lower` to \${glue:text}`ci_upper`, captures
the middle 95\% of the sample mean prices in the bootstrap distribution. We can
visualize the interval on our distribution in {numref}`fig:11-bootstrapping9`.

```{code-cell} ipython3
:tags: [remove-cell]
# Create the annotation for for the 2.5th percentile
rule_025 = alt.Chart().mark_rule(color="black", size=1.5, strokeDash=[6]).encode(
    x=alt.datum(ci_bounds[0.025])
).properties(
    width=500
)
text_025 = rule_025.mark_text(
    color="black",
    size=12,
    fontWeight="bold",
    dy=-160
).encode(
    text=alt.datum(f"2.5th percentile ({ci_bounds[0.025].round(1)})")
)

# Create the annotation for for the 97.5th percentile
text_975 = text_025.encode(
    x=alt.datum(ci_bounds[0.975]),
    text=alt.datum(f"97.5th percentile ({ci_bounds[0.975].round(1)})")
)
rule_975 = rule_025.encode(x=alt.datum(ci_bounds[0.975]))

# Layer the annotations on top of the distribution plot
bootstr9fig = boot_est_dist + rule_025 + text_025 + rule_975 + text_975
```

```{code-cell} ipython3
:tags: ["remove-cell"]

glue("fig:11-bootstrapping9", bootstr9fig)
```

:::{glue:figure} fig:11-bootstrapping9
:name: fig:11-bootstrapping9

Distribution of the bootstrap sample means with percentile lower and upper bounds.
:::



+++

To finish our estimation of the population parameter, we would report the point
estimate and our confidence interval's lower and upper bounds. Here the sample
mean price-per-night of 40 Airbnb listings was
\${glue:text}`one_sample_mean`, and we are 95\% "confident" that the true
population mean price-per-night for all Airbnb listings in Vancouver is between
\${glue:text}`ci_lower` and \${glue:text}`ci_upper`.
Notice that our interval does indeed contain the true
population mean value, \${glue:text}`population_mean`\! However, in
practice, we would not know whether our interval captured the population
parameter or not because we usually only have a single sample, not the entire
population. This is the best we can do when we only have one sample!

This chapter is only the beginning of the journey into statistical inference.
We can extend the concepts learned here to do much more than report point
estimates and confidence intervals, such as testing for real differences
between populations, tests for associations between variables, and so much
more. We have just scratched the surface of statistical inference; however, the
material presented here will serve as the foundation for more advanced
statistical techniques you may learn about in the future!

+++

## Exercises

Practice exercises for the material covered in this chapter
can be found in the accompanying
[worksheets repository](https://worksheets.python.datasciencebook.ca)
in the two "Statistical inference" rows.
You can launch an interactive version of each worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of each worksheet by clicking "view worksheet."
If you instead decide to download the worksheets and run them on your own machine,
make sure to follow the instructions for computer setup
found in {numref}`Chapter %s <move-to-your-own-machine>`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

## Additional resources

- Chapters 4 to 7 of *OpenIntro Statistics* {cite:p}`openintro`
  provide a good next step in learning about inference. Although it is still certainly
  an introductory text, things get a bit more mathematical here. Depending on
  your background, you may actually want to start going through Chapters 1 to 3
  first, where you will learn some fundamental concepts in probability theory.
  Although it may seem like a diversion, probability theory is *the language of
  statistics*; if you have a solid grasp of probability, more advanced statistics
  will come naturally to you!

+++

## References

```{bibliography}
:filter: docname in docnames
```
