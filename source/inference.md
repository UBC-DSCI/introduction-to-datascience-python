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

# Statistical inference

```{code-cell} ipython3
:tags: [remove-cell]

import altair as alt
import numpy as np
import pandas as pd
from sklearn.utils import resample
alt.data_transformers.disable_max_rows()

# alt.data_transformers.enable('data_server')
# alt.renderers.enable('mimetype')

from myst_nb import glue
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

* Describe real-world examples of questions that can be answered with statistical inference.
* Define common population parameters (e.g., mean, proportion, standard deviation) that are often estimated using sampled data, and estimate these from a sample.
* Define the following statistical sampling terms (population, sample, population parameter, point estimate, sampling distribution).
* Explain the difference between a population parameter and a sample point estimate.
* Use Python to draw random samples from a finite population.
* Use Python to create a sampling distribution from a finite population.
* Describe how sample size influences the sampling distribution.
* Define bootstrapping.
* Use Python to create a bootstrap distribution to approximate a sampling distribution.
* Contrast the bootstrap and sampling distributions.

+++

## Why do we need sampling? 
We often need to understand how quantities we observe in a subset
of data relate to the same quantities in the broader population. For example, suppose a
retailer is considering selling iPhone accessories, and they want to estimate
how big the market might be. Additionally, they want to strategize how they can
market their products on North American college and university campuses. This
retailer might formulate the following question:

*What proportion of all undergraduate students in North America own an iPhone?*

In the above question, we are interested in making a conclusion about *all*
undergraduate students in North America; this is referred to as the **population**. \index{population} In
general, the population is the complete collection of individuals or cases we
are interested in studying.  Further, in the above question, we are interested
in computing a quantity&mdash;the proportion of iPhone owners&mdash;based on
the entire population. This proportion is referred to as a **population parameter**. \index{population!parameter} In
general, a population parameter is a numerical characteristic of the entire
population. To compute this number in the example above, we would need to ask
every single undergraduate in North America whether they own an iPhone. In
practice, directly computing population parameters is often time-consuming and
costly, and sometimes impossible. 

A more practical approach would be to make measurements for a **sample**, i.e., a \index{sample}
subset of individuals collected from the population. We can then compute a
**sample estimate**&mdash;a numerical characteristic of the sample&mdash;that \index{sample!estimate}
estimates the population parameter. For example, suppose we randomly selected
ten undergraduate students across North America (the sample) and computed the
proportion of those students who own an iPhone (the sample estimate). In that
case, we might suspect that proportion is a reasonable estimate of the
proportion of students who own an iPhone in the entire population. 
{numref}`fig:11-population-vs-sample` illustrates this process.
In general, the process of using a sample to make a conclusion about the
broader population from which it is taken is referred to as **statistical inference**. 
\index{inference}\index{statistical inference|see{inference}}

+++

```{figure} img/population_vs_sample.png
:name: fig:11-population-vs-sample
:figclass: caption-hack

Population versus sample.
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
We will look at an example using data from 
[Inside Airbnb](http://insideairbnb.com/) {cite:p}`insideairbnb`. Airbnb \index{Airbnb} is an online
marketplace for arranging vacation rentals and places to stay. The data set
contains listings for Vancouver, Canada, in September 2020. Our data
includes an ID number, neighborhood, type of room, the number of people the
rental accommodates, number of bathrooms, bedrooms, beds, and the price per
night. 

<!--
airbnb <- read_csv("data/listings.csv") |>
  select(id, neighborhood = neighborhood_cleansed, room_type, accommodates, bathrooms = bathrooms_text, bedrooms, beds, price) |>
  mutate(price = as.numeric(str_remove(price, "[$]"))) |>
  na.omit()
airbnb <- airbnb |>
  mutate(id = 1:nrow(airbnb))
-->
\index{seed!set.seed}

```{code-cell} ipython3
import altair as alt
import pandas as pd

airbnb = pd.read_csv("data/listings.csv")
airbnb
```

Suppose the city of Vancouver wants information about Airbnb rentals to help
plan city bylaws, and they want to know how many Airbnb places are listed as
entire homes and apartments (rather than as private or shared rooms). Therefore
they may want to estimate the true proportion of all Airbnb listings where the
"type of place" is listed as "entire home or apartment." Of course, we usually
do not have access to the true population, but here let's imagine (for learning
purposes) that our data set represents the population of all Airbnb rental
listings in Vancouver, Canada. We can find the proportion of listings where
`room_type == "Entire home/apt"`.
\index{pull}\index{sum}\index{nrow}

```{code-cell} ipython3
population_summary = pd.DataFrame()
population_summary["n"] = [airbnb.query("room_type == 'Entire home/apt'")["id"].count()]
population_summary["proportion"] = population_summary["n"] / len(airbnb)

population_summary
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("population_proportion", round(population_summary["proportion"][0], 3))
```

We can see that the proportion of `Entire home/apt` listings in
the data set is {glue:}`population_proportion`. This 
value, {glue:}`population_proportion`, is the population parameter. Remember, this
parameter value is usually unknown in real data analysis problems, as it is
typically not possible to make measurements for an entire population.  

Instead, perhaps we can approximate it with a small subset of data!
To investigate this idea, let's try randomly selecting 40 listings (*i.e.,* taking a random sample of
size 40 from our population), and computing the proportion for that sample.
We will use the `sample` method of the `pandas.DataFrame`
object to take the sample. The argument `n` of `sample` is the size of the sample to take.

```{code-cell} ipython3
:tags: [remove-cell]

# Instead, perhaps we can approximate it with a small subset of data!
# To investigate this idea, let's try randomly selecting 40 listings (*i.e.,* taking a random sample of
# size 40 from our population), and computing the proportion for that sample.
# We will use the `rep_sample_n` function \index{rep\_sample\_n} from the `infer`
# package \index{infer} to take the sample. The arguments of `rep_sample_n` are (1) the data frame to
# sample from, and (2) the size of the sample to take.
```

```{code-cell} ipython3
sample_1 = airbnb.sample(n=40, random_state=12)

airbnb_sample_1 = pd.DataFrame()
airbnb_sample_1["n"] = [sample_1.query("room_type == 'Entire home/apt'")["id"].count()]
airbnb_sample_1["proportion"] = airbnb_sample_1["n"] / len(sample_1)

airbnb_sample_1
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sample_1_proportion", round(airbnb_sample_1["proportion"][0], 2))
```

Here we see that the proportion of entire home/apartment listings in this
random sample is {glue:}`sample_1_proportion`. Wow&mdash;that's close to our
true population value! But remember, we computed the proportion using a random sample of size 40.
This has two consequences. First, this value is only an *estimate*, i.e., our best guess 
of our population parameter using this sample. 
Given that we are estimating a single value here, we often
refer to it as a **point estimate**.  Second, since the sample was random,
if we were to take *another* random sample of size 40 and compute the proportion for that sample,
we would not get the same answer:

```{code-cell} ipython3
sample_2 = airbnb.sample(n=40, random_state=1234)

airbnb_sample_2 = pd.DataFrame()
airbnb_sample_2["n"] = [sample_2.query("room_type == 'Entire home/apt'")["id"].count()]
airbnb_sample_2["proportion"] = airbnb_sample_2["n"] / len(sample_2)

airbnb_sample_2
```

Confirmed! We get a different value for our estimate this time.
That means that our point estimate might be unreliable. Indeed, estimates vary from sample to
sample due to **sampling variability**. But just how much 
should we expect the estimates of our random samples to vary?
Or in other words, how much can we really trust our point estimate based on a single sample?

To understand this, we will simulate many samples (much more than just two)
of size 40 from our population of listings and calculate the proportion of
entire home/apartment listings in each sample. This simulation will create
many sample proportions, which we can visualize using a histogram. The
distribution of the estimate for all possible samples of a given size (which we
commonly refer to as $n$) from a population is called 
a **sampling distribution**. \index{sampling distribution} The sampling distribution will help us see how much we would
expect our sample proportions from this population to vary for samples of size 40. 

We again use the `sample` to take samples of size 40 from our
population of Airbnb listings. But this time we use a for loop 
to take 20,000 samples of size 40. \index{rep\_sample\_n!reps argument}
\index{rep\_sample\_n!size argument}

```{code-cell} ipython3
:tags: [remove-cell]

# We again use the `rep_sample_n` to take samples of size 40 from our
# population of Airbnb listings. But this time we set the `reps` argument to 20,000 to specify 
# that we want to take 20,000 samples of size 40. \index{rep\_sample\_n!reps argument}
# \index{rep\_sample\_n!size argument}
```

```{code-cell} ipython3
np.random.seed(1)

samples = []
for rep in range(20000):
    sample = airbnb.sample(40)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

samples
```

Notice that the column `replicate` indicates the replicate, or sample, to which
each listing belongs. Above, `pandas.DataFrame` by default shows the first and last few 
rows, so we can verify that 
we indeed created 20,000 samples (or replicates).

```{code-cell} ipython3
:tags: [remove-cell]

# Notice that the column `replicate` indicates the replicate, or sample, to which
# each listing belongs. Above, since by default R only prints the first few rows,
# it looks like all of the listings have `replicate` set to 1. But you can 
# check the last few entries using the `tail()` function to verify that 
# we indeed created 20,000 samples (or replicates).
```

Now that we have obtained the samples, we need to compute the 
proportion of entire home/apartment listings in each sample.
We first `query` the observations with room type of 'Entire home/apt';
group the data by the `replicate` variable&mdash;to group the
set of listings in each sample together&mdash;and then use `count` 
to compute the number of qualified observations in each sample; finally compute the proportion.
Both the first and last few entries of the resulting data frame are printed 
below to show that we end up with 20,000 point estimates, one for each of the 20,000 samples.

```{code-cell} ipython3
:tags: [remove-cell]

# Now that we have obtained the samples, we need to compute the 
# proportion of entire home/apartment listings in each sample.
# We first group the data by the `replicate` variable&mdash;to group the
# set of listings in each sample together&mdash;and then use `summarize` 
# to compute the proportion in each sample.
# We print both the first and last few entries of the resulting data frame
# below to show that we end up with 20,000 point estimates, one for each of the 20,000 samples.
```

```{code-cell} ipython3
# calculate the the number of observations in each sample with room_type == 'Entire home/apt'
sample_estimates = (
    samples.query("room_type == 'Entire home/apt'")
    .groupby("replicate")["room_type"]
    .count()
    .reset_index()
    .rename(columns={"room_type": "counts"})
)

# calculate the proportion
sample_estimates["sample_proportion"] = sample_estimates["counts"] / 40

# drop the count column
sample_estimates = sample_estimates.drop(columns=["counts"])

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
    x=alt.X("sample_proportion", bin=alt.Bin(maxbins=12), title="Sample proportions"),
    y=alt.Y("count()", title="Count"),
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

glue("sample_propotion_center", round(sample_estimates["sample_proportion"].mean(), 1))
glue("sample_propotion_min", round(sample_estimates["sample_proportion"].min(), 1))
glue("sample_propotion_max", round(sample_estimates["sample_proportion"].max(), 1))
```

The sampling distribution in {numref}`fig:11-example-proportions7` appears
to be bell-shaped, is roughly symmetric, and has one peak. It is centered 
around {glue:}`sample_propotion_center` and the sample proportions
range from about {glue:}`sample_propotion_min` to about
{glue:}`sample_propotion_max`. In fact, we can
calculate the mean of the sample proportions.  \index{sampling distribution!shape}

```{code-cell} ipython3
sample_estimates["sample_proportion"].mean()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("sample_proportion_mean", round(sample_estimates["sample_proportion"].mean(), 3))
```

We notice that the sample proportions are centered around the population
proportion value, {glue:}`sample_proportion_mean`! In general, the mean of
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

population_distribution = (
    alt.Chart(airbnb)
    .mark_bar()
    .encode(
        x=alt.X(
            "price", bin=alt.Bin(maxbins=30), title="Price per night (Canadian dollars)"
        ),
        y=alt.Y("count()", title="Count"),
    )
)

population_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means2", population_distribution)
```

:::{glue:figure} fig:11-example-means2
:name: fig:11-example-means2

Population distribution of price per night (Canadian dollars) for all Airbnb listings in Vancouver, Canada.
:::

+++

In {numref}`fig:11-example-means2`, we see that the population distribution \index{population!distribution}
has one peak. It is also skewed (i.e., is not symmetric): most of the listings are
less than \$250 per night, but a small number of listings cost much more, 
creating a long tail on the histogram's right side. 
Along with visualizing the population, we can calculate the population mean,
the average price per night for all the Airbnb listings.

```{code-cell} ipython3
population_parameters = airbnb["price"].mean()

population_parameters
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("population_mean", round(population_parameters, 2))
```

The price per night of all Airbnb rentals in Vancouver, BC 
is \${glue:}`population_mean`, on average. This value is our
population parameter since we are calculating it using the population data. \index{population!parameter}

Now suppose we did not have access to the population data (which is usually the
case!), yet we wanted to estimate the mean price per night. We could answer
this question by taking a random sample of as many Airbnb listings as our time
and resources allow. Let's say we could do this for 40 listings. What would
such a sample look like?  Let's take advantage of the fact that we do have
access to the population data and simulate taking one random sample of 40
listings in Python, again using `sample`. 
\index{rep\_sample\_n}

```{code-cell} ipython3
one_sample = airbnb.sample(40, random_state=1)
```

We can create a histogram to visualize the distribution of observations in the
sample ({numref}`fig:11-example-means-sample-hist`), and calculate the mean
of our sample.

```{code-cell} ipython3
:tags: [remove-output]

sample_distribution = alt.Chart(one_sample).mark_bar().encode(
    x=alt.X("price", bin=alt.Bin(maxbins=30), title="Price per night (Canadian dollars)"),
    y=alt.Y("count()", title="Count"),
)

sample_distribution
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means-sample-hist", sample_distribution)
```

:::{glue:figure} fig:11-example-means-sample-hist
:name: fig:11-example-means-sample-hist

Distribution of price per night (Canadian dollars) for sample of 40 Airbnb listings.
:::

```{code-cell} ipython3
estimates = one_sample["price"].mean()

estimates
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("estimate_mean", round(estimates, 2))
glue(
    "diff_perc",
    round(100 * abs(estimates - population_parameters) / population_parameters, 1),
)
```

The average value of the sample of size 40 
is \${glue:}`estimate_mean`.  This 
number is a point estimate for the mean of the full population.
Recall that the population mean was 
\${glue:}`population_mean`. So our estimate was fairly close to
the population parameter: the mean was about 
{glue:}`diff_perc`% 
off.  Note that we usually cannot compute the estimate's accuracy in practice
since we do not have access to the population parameter; if we did, we wouldn't
need to estimate it!

Also, recall from the previous section that the point estimate can vary; if we
took another random sample from the population, our estimate's value might
change. So then, did we just get lucky with our point estimate above?  How much
does our estimate vary across different samples of size 40 in this example?
Again, since we have access to the population, we can take many samples and
plot the sampling distribution of \index{sampling distribution} sample means for samples of size 40 to
get a sense for this variation. In this case, we'll use 20,000 samples of size
40.

```{code-cell} ipython3
np.random.seed(2)

samples = []
for rep in range(20000):
    sample = airbnb.sample(40)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

samples
```

Now we can calculate the sample mean for each replicate and plot the sampling
distribution of sample means for samples of size 40.

```{code-cell} ipython3
sample_estimates = samples.groupby("replicate")["price"].mean().reset_index().rename(
    columns={"price": "sample_mean"}
)

sample_estimates
```

```{code-cell} ipython3
:tags: [remove-output]

sampling_distribution_40 = (
    alt.Chart(sample_estimates)
    .mark_bar()
    .encode(
        x=alt.X(
            "sample_mean",
            bin=alt.Bin(maxbins=30),
            title="Sample mean price per night (Canadian dollars)",
        ),
        y=alt.Y("count()", title="Count"),
    )
)

sampling_distribution_40
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:11-example-means4", sampling_distribution_40)
```

:::{glue:figure} fig:11-example-means4
:name: fig:11-example-means4

Sampling distribution of the sample means for sample size of 40.
:::

```{code-cell} ipython3
:tags: [remove-cell]

glue("quantile_1", round(int(sample_estimates["sample_mean"].quantile(0.25)), -1))
glue("quantile_3", round(int(sample_estimates["sample_mean"].quantile(0.75)), -1))
```

In {numref}`fig:11-example-means4`, the sampling distribution of the mean
has one peak and is \index{sampling distribution!shape} bell-shaped. Most of the estimates are between 
about  \${glue:}`quantile_1` and 
\${glue:}`quantile_3`; but there are
a good fraction of cases outside this range (i.e., where the point estimate was
not close to the population parameter). So it does indeed look like we were
quite lucky when we estimated the population mean with only 
{glue:}`diff_perc`% error.

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
\index{sampling distribution!compared to population distribution}

<!---
```{r 11-example-means4.5}
sample_estimates |>
  summarize(mean_of_sample_means = mean(sample_mean))
```
Notice that the mean of the sample means is \$`r round(mean(sample_estimates$sample_mean),2)`. Recall that the population mean
was \$`r round(mean(airbnb$price),2)`. 
-->

```{code-cell} ipython3
:tags: [remove-input]

(
    (
        alt.Chart(airbnb, title="Population")
        .mark_bar(clip=True)
        .encode(
            x=alt.X(
                "price",
                bin=alt.Bin(maxbins=30),
                title="Price per night (Canadian dollars)",
                axis=alt.Axis(values=list(range(50, 601, 50))),
                scale=alt.Scale(domain=(min(airbnb["price"]), 600)),
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(width=400, height=150)
    )
    & (
        alt.Chart(one_sample, title="Sample (n = 40)")
        .mark_bar(clip=True)
        .encode(
            x=alt.X(
                "price",
                bin=alt.Bin(maxbins=30),
                title="Price per night (Canadian dollars)",
                axis=alt.Axis(values=list(range(50, 601, 50))),
                scale=alt.Scale(domain=(min(airbnb["price"]), 600)),
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(width=400, height=150)
    )
    & (
        alt.Chart(
            sample_estimates,
            title="Sampling distribution of the mean for samples of size 40",
        )
        .mark_bar(clip=True)
        .encode(
            x=alt.X(
                "sample_mean",
                bin=True,
                title="Sample mean price per night (Canadian dollars)",
                axis=alt.Axis(values=list(range(50, 601, 50))),
                scale=alt.Scale(domain=(min(airbnb["price"]), 600)),
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(width=400, height=150)
    )
)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-example-means5
:figclass: caption-hack

Comparison of population distribution, sample distribution, and sampling distribution.
```

+++

Given that there is quite a bit of variation in the sampling distribution of
the sample mean&mdash;i.e., the point estimate that we obtain is not very
reliable&mdash;is there any way to improve the estimate?  One way to improve a
point estimate is to take a *larger* sample. To illustrate what effect this
has, we will take many samples of size 20, 50, 100, and 500, and plot the
sampling distribution of the sample mean. We indicate the mean of the sampling
distribution with a red vertical line.

```{code-cell} ipython3
:tags: [remove-cell]

# Initially thought of using a loop, but Jupyter book failed to build because "cell execution
# timed out..."
# ## Sampling n = 20, 50, 100, 500
# np.random.seed(2)
# sample_dict = {}
# for sample_n in [20, 50, 100, 500]:
#     samples = []
#     for rep in range(20000):
#         sample = airbnb.sample(sample_n)
#         sample = sample.assign(replicate=rep)
#         samples.append(sample)
#     samples = pd.concat([samples[i] for i in range(len(samples))])

#     sample_dict[f"sample_estimates_{sample_n}"] = (
#         samples.groupby("replicate")["price"]
#         .mean()
#         .reset_index()
#         .rename(columns={"price": "sample_mean"})
#     )
```

```{code-cell} ipython3
:tags: [remove-cell]

sample_dict = {}

# Sampling n = 20
samples = []
for rep in range(20000):
    sample = airbnb.sample(20)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

sample_dict[f"sample_estimates_20"] = (
    samples.groupby("replicate")["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "sample_mean"})
)
```

```{code-cell} ipython3
:tags: [remove-cell]

# Sampling n = 50
samples = []
for rep in range(20000):
    sample = airbnb.sample(50)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

sample_dict[f"sample_estimates_50"] = (
    samples.groupby("replicate")["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "sample_mean"})
)
```

```{code-cell} ipython3
:tags: [remove-cell]

# Sampling n = 100
samples = []
for rep in range(20000):
    sample = airbnb.sample(100)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

sample_dict[f"sample_estimates_100"] = (
    samples.groupby("replicate")["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "sample_mean"})
)
```

```{code-cell} ipython3
:tags: [remove-cell]

# Sampling n = 500
samples = []
for rep in range(20000):
    sample = airbnb.sample(500)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

sample_dict[f"sample_estimates_500"] = (
    samples.groupby("replicate")["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "sample_mean"})
)
```

```{code-cell} ipython3
:tags: [remove-cell]

## Plot sampling distribution n = 20, 50, 100, 500
sample_plot = {}
plot_min_x = sample_dict["sample_estimates_20"]["sample_mean"].min()
plot_max_x = sample_dict["sample_estimates_20"]["sample_mean"].max()


# def max_bins(distribution):
#     if int(distribution.split("_")[-1]) >= 100:
#         return 10
#     else:
#         return 30


def text_y(distribution):
    sample_n = int(distribution.split("_")[-1])
    if sample_n == 20:
        return 2000
    elif sample_n == 50:
        return 3000
    elif sample_n == 100:
        return 4500
    else:
        return 10000


for sample_n, df in sample_dict.items():
    sample_plot[sample_n] = (
        alt.Chart(df, title=f"n = {sample_n.split('_')[-1]}")
        .mark_bar()
        .encode(
            x=alt.X(
                "sample_mean",
                title="Sample mean price per night (Canadian dollars)",
                bin=alt.Bin(extent=[80, 300], step=50/7), # maxbins=max_bins(sample_n)
                scale=alt.Scale(domain=(plot_min_x, plot_max_x)),
                axis=alt.Axis(values=list(range(80, 301, 20)))
            ),
            y=alt.Y("count()", title="Count"),
        )
    )

    sample_mean = sample_dict[sample_n]["sample_mean"].mean()
    sample_plot[sample_n] = (
        sample_plot[sample_n]
        + alt.Chart(pd.DataFrame({"x": [sample_mean]}))
        .mark_rule(color="red", size=2)
        .encode(x="x")
        + (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [plot_max_x - 20],
                        "y": [text_y(sample_n)],
                        "text": [f"mean = {round(sample_mean, 1)}"],
                    }
                )
            )
            .mark_text(dy=-5, size=15)
            .encode(x="x", y="y", text="text")
        )
    ).properties(width=350, height=250)
```

```{code-cell} ipython3
:tags: [remove-input]

(sample_plot["sample_estimates_20"] | sample_plot["sample_estimates_50"]) & (
    sample_plot["sample_estimates_100"] | sample_plot["sample_estimates_500"]
)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-example-means7
:figclass: caption-hack

Comparison of sampling distributions, with mean highlighted as a vertical red line.
```

+++

Based on the visualization in {numref}`fig:11-example-means7`, three points
about the sample mean become clear.  First, the mean of the sample mean (across
samples) is equal to the population mean. In other words, the sampling
distribution is centered at the population mean.  Second, increasing the size of
the sample decreases the spread (i.e., the variability) of the sampling
distribution. Therefore, a larger sample size results in a more reliable point
estimate of the population parameter. And third, the distribution of the sample
mean is roughly bell-shaped. \index{sampling distribution!effect of sample size}

> **Note:** You might notice that in the `n = 20` case in {numref}`fig:11-example-means7`,
> the distribution is not *quite* bell-shaped. There is a bit of skew towards the right!
> You might also notice that in the `n = 50` case and larger, that skew seems to disappear.
> In general, the sampling distribution&mdash;for both means and proportions&mdash;only 
> becomes bell-shaped *once the sample size is large enough*.
> How large is "large enough?" Unfortunately, it depends entirely on the problem at hand. But 
> as a rule of thumb, often a sample size of at least 20 will suffice.

<!--- > **Note:** If random samples of size $n$ are taken from a population, the sample mean $\bar{x}$ will be approximately Normal with mean $\mu$ and standard deviation $\frac{\sigma}{\sqrt{n}}$ as long as the sample size $n$ is large enough. $\mu$ is the population mean, $\sigma$ is the population standard deviation, $\bar{x}$ is the sample mean, and $n$ is the sample size. 
> If samples are selected from a finite population as we are doing in this chapter, we should apply a finite population correction. We multiply $\frac{\sigma}{\sqrt{n}}$ by $\sqrt{\frac{N - n}{N - 1}}$ where $N$ is the population size and $n$ is the sample size. If our sample size, $n$, is small relative to the population size, this finite correction factor is less important. 
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

Unfortunately, we cannot construct the exact sampling distribution without
full access to the population. However, if we could somehow *approximate* what 
the sampling distribution would look like for a sample, we could 
use that approximation to then report how uncertain our sample
point estimate is (as we did above with the *exact* sampling
distribution). There are several methods to accomplish this; in this book, we 
will use the \index{bootstrap} *bootstrap*. We will discuss **interval estimation** and 
construct \index{confidence interval}\index{interval|see{confidence interval}} 
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
np.random.seed(12)
for sample_n in [10, 20, 50, 100, 200]:
    sample = airbnb.sample(sample_n)
    sample_distribution_dict[f"sample_distribution_{sample_n}"] = (
        alt.Chart(sample, title=f"n = {sample_n}")
        .mark_bar()
        .encode(
            x=alt.X(
                "price",
                bin=alt.Bin(extent=[0, 600], step=20),
                title="Price per night (Canadian dollars)",
            ),
            y=alt.Y("count()", title="Count"),
        )
    ).properties(width=350, height=250)
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
            | population_distribution.properties(width=350, height=250)
        )
    ),
)
```

:::{glue:figure} fig:11-example-bootstrapping0
:name: fig:11-example-bootstrapping0

Comparison of samples of different sizes from the population.
:::

+++

In the previous section, we took many samples of the same size *from our
population* to get a sense of the variability of a sample estimate. But if our
sample is big enough that it looks like our population, we can pretend that our
sample *is* the population, and take more samples (with replacement) of the
same size from it instead! This very clever technique is 
called **the bootstrap**.  Note that by taking many samples from our single, observed
sample, we do not obtain the true sampling distribution, but rather an
approximation that we call **the bootstrap distribution**. \index{bootstrap!distribution}

\newpage

> **Note:** We must sample *with* replacement when using the bootstrap.
> Otherwise, if we had a sample of size $n$, and obtained a sample from it of
> size $n$ *without* replacement, it would just return our original sample!

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

```{figure} img/intro-bootstrap.jpeg
:name: fig:11-intro-bootstrap-image
:figclass: caption-hack

Overview of the bootstrap process.
```

+++

### Bootstrapping in Python 

Let’s continue working with our Airbnb example to illustrate how we might create
and use a bootstrap distribution using just a single sample from the population. 
Once again, suppose we are
interested in estimating the population mean price per night of all Airbnb
listings in Vancouver, Canada, using a single sample size of 40.
Recall our point estimate was \${glue:}`estimate_mean`. The
histogram of prices in the sample is displayed in {numref}`fig:11-bootstrapping1`.

```{code-cell} ipython3
:tags: [remove-cell]

pd.set_option('display.max_rows', 20)
```

```{code-cell} ipython3
one_sample
```

```{code-cell} ipython3
:tags: []

one_sample_dist = alt.Chart(one_sample).mark_bar().encode(
    x=alt.X(
        "price",
        bin=alt.Bin(maxbins=30),
        title="Price per night (Canadian dollars)",
    ),
    y=alt.Y("count()", title="Count"),
)

one_sample_dist
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping1
:figclass: caption-hack

Histogram of price per night (Canadian dollars) for one sample of size 40.
```

+++

The histogram for the sample is skewed, with a few observations out to the right. The
mean of the sample is \${glue:}`estimate_mean`.
Remember, in practice, we usually only have this one sample from the population. So
this sample and estimate are the only data we can work with.

We now perform steps 1&ndash;5 listed above to generate a single bootstrap
sample in Python and calculate a point estimate from that bootstrap sample. We will 
use the `resample` function from the `scikit-learn` package. Critically, note that we now
pass `one_sample`&mdash;our single sample of size 40&mdash;as the first argument.
And since we need to sample with replacement,
we keep the argument for `replace` to its default value of `True`.
\index{bootstrap!in R}
\index{rep\_sample\_n!bootstrap}

```{code-cell} ipython3
:tags: []

boot1 = resample(one_sample, replace=True, n_samples=40, random_state=2)
boot1_dist = alt.Chart(boot1).mark_bar().encode(
    x=alt.X(
        "price",
        bin=alt.Bin(maxbins=30),
        title="Price per night (Canadian dollars)",
    ),
    y=alt.Y("count()", title="Count"),
)

boot1_dist
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping3
:figclass: caption-hack

Bootstrap distribution.
```

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
using `resample`, and calculate the means for
each of those replicates. Recall that this assumes that `one_sample` *looks like*
our original population; but since we do not have access to the population itself,
this is often the best we can do.

```{code-cell} ipython3
np.random.seed(2)

boot20000 = []
for rep in range(20000):
    sample = resample(one_sample, replace=True, n_samples=40)
    sample = sample.assign(replicate=rep)
    boot20000.append(sample)
boot20000 = pd.concat([boot20000[i] for i in range(len(boot20000))])

boot20000
```

Let's take a look at histograms of the first six replicates of our bootstrap samples.

```{code-cell} ipython3
:tags: []

six_bootstrap_samples = boot20000.query("replicate < 6")

(
    alt.Chart(six_bootstrap_samples)
    .mark_bar()
    .encode(
        x=alt.X(
            "price",
            bin=alt.Bin(maxbins=20),
            title="Price per night (Canadian dollars)",
        ),
        y=alt.Y("count()", title="Count"),
    ).properties(width=250, height=200)
    .facet("replicate", columns=3)
)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping-six-bootstrap-samples
:figclass: caption-hack

Histograms of first six replicates of bootstrap samples.
```

+++

We see in {numref}`fig:11-bootstrapping-six-bootstrap-samples` how the
bootstrap samples differ. We can also calculate the sample mean for each of
these six replicates.

```{code-cell} ipython3
six_bootstrap_samples.groupby("replicate")["price"].mean().reset_index().rename(
    columns={"price": "mean"}
)
```

We can see that the bootstrap sample distributions and the sample means are
different. They are different because we are sampling *with replacement*. We
will now calculate point estimates for our 20,000 bootstrap samples and
generate a bootstrap distribution of our point estimates. The bootstrap
distribution ({numref}`fig:11-bootstrapping5`) suggests how we might expect
our point estimate to behave if we took another sample.

```{code-cell} ipython3
boot20000_means = boot20000.groupby("replicate")["price"].mean().reset_index().rename(
    columns={"price": "mean"}
)

boot20000_means
```

```{code-cell} ipython3
:tags: []

boot_est_dist = alt.Chart(boot20000_means).mark_bar().encode(
    x=alt.X(
        "mean",
        bin=alt.Bin(extent=[95, 245], step=5),
        title="Sample mean price per night (Canadian dollars)",
    ),
    y=alt.Y("count()", title="Count"),
)

boot_est_dist
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping5
:figclass: caption-hack

Distribution of the bootstrap sample means.
```

+++

Let's compare the bootstrap distribution&mdash;which we construct by taking many samples from our original sample of size 40&mdash;with 
the true sampling distribution&mdash;which corresponds to taking many samples from the population.

```{code-cell} ipython3
:tags: [remove-input]

samples = []
for rep in range(20000):
    sample = airbnb.sample(40)
    sample = sample.assign(replicate=rep)
    samples.append(sample)
samples = pd.concat([samples[i] for i in range(len(samples))])

sample_estimates = samples.groupby("replicate")["price"].mean().reset_index().rename(
    columns={"price": "sample_mean"}
)

plot_min_x = sample_estimates["sample_mean"].min()
plot_max_x = sample_estimates["sample_mean"].max()
sampling_mean = sample_estimates["sample_mean"].mean()
boot_sampling_mean = boot20000_means["mean"].mean()

sampling_dist = alt.Chart(sample_estimates).mark_bar().encode(
    x=alt.X(
        "sample_mean",
        bin=alt.Bin(extent=[95, 245], step=5),
        # scale=alt.Scale(domain=(plot_min_x, plot_max_x)),
        title="Sample mean price per night (Canadian dollars)",
    ),
    y=alt.Y("count()", title="Count"),
)

annotated_sampling_dist = (
    sampling_dist
    + alt.Chart(pd.DataFrame({"x": [sampling_mean]}), title="Sampling distribution")
    .mark_rule(color="red", size=2)
    .encode(x="x")
    + (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [plot_max_x - 20],
                    "y": [2000],
                    "text": [f"mean = {round(sampling_mean, 1)}"],
                }
            )
        )
        .mark_text(dy=-5, size=15)
        .encode(x="x", y="y", text="text")
    )
)

annotated_boot_est_dist = boot_est_dist + (
    alt.Chart(pd.DataFrame({"x": [boot_sampling_mean]}), title="Bootstrap distribution")
    .mark_rule(color="red", size=2)
    .encode(x="x")
    + (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [plot_max_x - 20],
                    "y": [1500],
                    "text": [f"mean = {round(boot_sampling_mean, 1)}"],
                }
            )
        )
        .mark_text(dy=-5, size=15)
        .encode(x="x", y="y", text="text")
    )
)

annotated_sampling_dist | annotated_boot_est_dist
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping6
:figclass: caption-hack

Comparison of the distribution of the bootstrap sample means and sampling distribution.
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("one_sample_mean", round(one_sample["price"].mean(), 2))
```

There are two essential points that we can take away from 
\index{sampling distribution!compared to bootstrap distribution}
{numref}`fig:11-bootstrapping6`. First, the shape and spread of the true sampling
distribution and the bootstrap distribution are similar; the bootstrap
distribution lets us get a sense of the point estimate's variability. The
second important point is that the means of these two distributions are
different. The sampling distribution is centered at 
\${glue:}`population_mean`, the population mean value. However, the bootstrap
distribution is centered at the original sample's mean price per night, 
\${glue:}`one_sample_mean`. Because we are resampling from the
original sample repeatedly, we see that the bootstrap distribution is centered
at the original sample's mean value (unlike the sampling distribution of the
sample mean, which is centered at the population parameter value). 

{numref}`fig:11-bootstrapping7` summarizes the bootstrapping process.
The idea here is that we can use this distribution of bootstrap sample means to
approximate the sampling distribution of the sample means when we only have one
sample. Since the bootstrap distribution pretty well approximates the sampling
distribution spread, we can use the bootstrap spread to help us develop a
plausible range for our population parameter along with our estimate!

```{code-cell} ipython3
:tags: [remove-cell]

!wget -O img/inference/11-bootstrapping7-1.png https://datasciencebook.ca/_main_files/figure-html/11-bootstrapping7-1.png
```

```{figure} img/inference/11-bootstrapping7-1.png
:name: fig:11-bootstrapping7
:figclass: caption-hack

Summary of bootstrapping process.
```

+++

### Using the bootstrap to calculate a plausible range  

Now that we have constructed our bootstrap distribution, let's use it to create
an approximate 95\% percentile bootstrap confidence interval. 
A **confidence interval** \index{confidence interval} is a range of plausible values for the population parameter. We will
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

+++

1. Arrange the observations in the bootstrap distribution in ascending order. 
2. Find the value such that 2.5\% of observations fall below it (the 2.5\% percentile). Use that value as the lower bound of the interval.
3. Find the value such that 97.5\% of observations fall below it (the 97.5\% percentile). Use that value as the upper bound of the interval.

\newpage

To do this in Python, we can use the `percentile()` function from the `numpy` package:
\index{quantile}
\index{pull}
\index{select}

```{code-cell} ipython3
import numpy as np
bounds = np.percentile(boot20000_means["mean"], [2.5, 97.5])

bounds
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("ci_lower", round(bounds[0], 2))
glue("ci_upper", round(bounds[1], 2))
```

Our interval, \${glue:}`ci_lower` to \${glue:}`ci_upper`, captures
the middle 95\% of the sample mean prices in the bootstrap distribution. We can
visualize the interval on our distribution in {numref}`fig:11-bootstrapping9`.

```{code-cell} ipython3
:tags: [remove-input]

boot_est_dist + (
    (
        alt.Chart(pd.DataFrame({"x": [bounds[0]]}))
        .mark_rule(color="#E69F00", size=3, strokeDash=[8, 8])
        .encode(x="x")
    )
    + (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [bounds[0] - 10],
                    "y": [1600],
                    "text": [f"2.5th percentile = {round(bounds[0], 2)}"],
                }
            )
        )
        .mark_text(dy=-5, size=12)
        .encode(x="x", y="y", text="text")
    )
) + (
    (
        alt.Chart(pd.DataFrame({"x": [bounds[1]]}))
        .mark_rule(color="#E69F00", size=3, strokeDash=[8, 8])
        .encode(x="x")
    )
    + (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [bounds[1]],
                    "y": [1600],
                    "text": [f"97.5th percentile = {round(bounds[1], 2)}"],
                }
            )
        )
        .mark_text(dy=-5, size=12)
        .encode(x="x", y="y", text="text")
    )
)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:11-bootstrapping9
:figclass: caption-hack

Distribution of the bootstrap sample means with percentile lower and upper bounds.
```

+++

To finish our estimation of the population parameter, we would report the point
estimate and our confidence interval's lower and upper bounds. Here the sample
mean price-per-night of 40 Airbnb listings was 
\${glue:}`one_sample_mean`, and we are 95\% "confident" that the true
population mean price-per-night for all Airbnb listings in Vancouver is between
\$({glue:}`ci_lower`, {glue:}`ci_upper`).
Notice that our interval does indeed contain the true
population mean value, \${glue:}`population_mean`\! However, in
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
[worksheets repository](https://github.com/UBC-DSCI/data-science-a-first-intro-worksheets#readme)
in the two "Statistical inference" rows.
You can launch an interactive version of each worksheet in your browser by clicking the "launch binder" button.
You can also preview a non-interactive version of each worksheet by clicking "view worksheet."
If you instead decide to download the worksheets and run them on your own machine,
make sure to follow the instructions for computer setup
found in Chapter {ref}`move-to-your-own-machine`. This will ensure that the automated feedback
and guidance that the worksheets provide will function as intended.

+++

## Additional resources

- Chapters 7 to 10 of *Modern Dive* {cite:p}`moderndive` provide a great
  next step in learning about inference. In particular, Chapters 7 and 8 cover
  sampling and bootstrapping using `tidyverse` and `infer` in a slightly more
  in-depth manner than the present chapter. Chapters 9 and 10 take the next step
  beyond the scope of this chapter and begin to provide some of the initial
  mathematical underpinnings of inference and more advanced applications of the
  concept of inference in testing hypotheses and performing regression. This
  material offers a great starting point for getting more into the technical side
  of statistics.
- Chapters 4 to 7 of *OpenIntro Statistics* {cite:p}`openintro`
  provide a good next step after *Modern Dive*. Although it is still certainly
  an introductory text, things get a bit more mathematical here. Depending on
  your background, you may actually want to start going through Chapters 1 to 3
  first, where you will learn some fundamental concepts in probability theory.
  Although it may seem like a diversion, probability theory is *the language of
  statistics*; if you have a solid grasp of probability, more advanced statistics
  will come naturally to you!

+++

## References

```{bibliography}
```
