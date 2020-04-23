# Decision Analysis using Weighted Product Model

As a friendly Data Scientist, coworkers always come to me with different problems to get some help. People ask me to create a model, a formula, a chart, something...to solve some quantitative problem.

In the past six months, a recurring problem appeared for different people: how to decide for the best project, the best item, the best place among multiple alternatives, based on multiple criteria?

People may think I pick, without hesitation, some amazing Artificial Intelligence algorithm, a Deep Learning classifier, or even an Hill-climbing Ensemble Selection with Bootstrap Sampling. However, as Robert Browning said "Well, less is more, Lucrezia".

A Weighted Product Model (WPM) is a simple and popular technique to solve Multi-Criteria Decision Analysis problems. It basically consists in multiply all attributes values to get an score. The higher, the better.

As an example, look at the following table:

|Projects / Features|A|B|C|D|
|--|--|--|--|--|
|Project 1|30|17|5|20|
|Project 2|27|8|14|10|
|Project 3|8|25|13|36|

A WPM score for the first line is:

Project 1 = 30 x 17 x 5 x 20 = 51,000

A more formal definition is:

![WPM Formula](images/wpm.png)

where:

- *i* : index
- *n* : number of features
- *v<sub>i</sub>* : value of the i-th feature
- *w<sub>i</sub>* : weight of the i-th feature

Thus, using the score provided by the WPM, the preferences are ranked like this

|Projects / Features|A|B|C|D|WPM
|--|--|--|--|--|--|
|Project 3|8|25|13|36|93,600|
|Project 1|30|17|5|20|51,000|
|Project 2|27|8|14|10|30,240|

if the weights are the same for each feature (*w<sub>i</sub> = 1*).

Weight Product Model is a very easy technique. You may use any programming language (even a spreadsheet). To show how to do it in Python, let's take another example:

|    |   bedrooms |   bathrooms |   area |   hoa |   parking |   year |   floor | exposure   | elevator   |   price |
|---:|-----------:|------------:|-------:|------:|----------:|-------:|--------:|:-----------|:-----------|--------:|
|  0 |          1 |           1 |    750 |   546 |         1 |   1951 |       4 | W          | Yes        |  175,000 |
|  1 |          1 |           1 |    700 |   230 |         0 |   1895 |       2 | W          | No         |  199,000 |
|  2 |          3 |           2 |   1600 |   150 |         0 |   2020 |       3 | E          | No         |  279,000 |
|  3 |          1 |           1 |    985 |   424 |         1 |   1892 |       3 | W          | Yes        |  210,000 |
|  4 |          2 |           2 |   1200 |   973 |         1 |   1965 |       8 | E          | Yes        |  209,900 |

The table above corresponds to apartments for sale in Chicago, IL and the data was obtained from a popular real state web site. The features are: number of **bedrooms**, number of **bathrooms** , total **area**, homeowners association fee (**hoa**), number of **parking** slots, **year** of the building, apartment's **floor**, **exposure** to the sun, existence of **elevator**, and **price**.

Even if WPM is very simple to use, there are some pitfalls. First I'll show you how to do it right, then I'll show you what you shouldn't do.

Let's import some libraries:

```{python}
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```
Then, I create a pandas DataFrame:

```{python}
raw = pd.read_csv('apartments.csv')
```

WPM uses only numerical features. So, I convert *exposure* and *elevator* to numeric:

```{python}
# keep original data safe
df = raw.copy()

# now exposure to West and to East are, respectively, 0 and 1
df['exposure'] = df.apply(lambda r: 1 if r['exposure'] == 'E' else 0, axis=1)

# if an building has an elevator, it will receive 1; 0 otherwise
df['elevator'] = df.apply(lambda r: 1 if r['elevator'] == 'Yes' else 0, axis=1)
```

Now, the table looks like this:

|    |   bedrooms |   bathrooms |   area |   hoa |   parking |   year |   floor |   exposure |   elevator |   price |
|---:|-----------:|------------:|-------:|------:|----------:|-------:|--------:|-----------:|-----------:|--------:|
|  0 |          1 |           1 |    750 |   546 |         1 |   1951 |       4 |          0 |          1 |  175000 |
|  1 |          1 |           1 |    700 |   230 |         0 |   1895 |       2 |          0 |          0 |  199000 |
|  2 |          3 |           2 |   1600 |   150 |         0 |   2020 |       3 |          1 |          0 |  279000 |
|  3 |          1 |           1 |    985 |   424 |         1 |   1892 |       3 |          0 |          1 |  210000 |
|  4 |          2 |           2 |   1200 |   973 |         1 |   1965 |       8 |          1 |          1 |  209900 |


Next, I scale all features to same range:

```{python}
# create a scaler with range [10, 100]
mms = MinMaxScaler(feature_range=(10, 100)) 

# apply scaler
df = pd.DataFrame(mms.fit_transform(df), columns=df.columns)
```

Now, every value is between 10 and 100.

|    |   bedrooms |   bathrooms |   area |      hoa |   parking |     year |   floor |   exposure |   elevator |    price |
|---:|-----------:|------------:|-------:|---------:|----------:|---------:|--------:|-----------:|-----------:|---------:|
|  0 |         10 |          10 |   15   |  53.305  |       100 |  51.4844 |      40 |         10 |        100 |  10      |
|  1 |         10 |          10 |   10   |  18.7485 |        10 |  12.1094 |      10 |         10 |         10 |  30.7692 |
|  2 |        100 |         100 |  100   |  10      |        10 | 100      |      25 |        100 |         10 | 100      |
|  3 |         10 |          10 |   38.5 |  39.9635 |       100 |  10      |      25 |         10 |        100 |  40.2885 |
|  4 |         55 |         100 |   60   | 100      |       100 |  61.3281 |     100 |        100 |        100 |  40.2019 |

It's time to define the weights:

```{python}
weights = {'bedrooms'     :   1.,
           'bathrooms'     :  1.,
           'area'          :  1.,
           'hoa'           : -1.,
           'parking'       :  1.,
           'year'          :  1.,
           'floor'         : -1.,
           'price'         : -1.,
           'exposure'      :  1.,
           'elevator'      :  1.}
```

The idea here is to make negative the features aren't benefits. If price goes up, WPM Score goes down.

Let's define the WPM function:

```{python}
def wpm(option, weights):
    # initial value
    value = 1

    # iterate over the features
    for column in option.keys():

        try:
            value *= option[column] ** weights[column]

        # a caution if some feature is forgotten
        except KeyError:
            pass

    return value
```

Then, we apply the function over the rows:

```{python}
df['wpm'] = df.apply(lambda r: wpm(r, weights), axis=1)

# merge the wpm score with the original table to guarantee
# a better interpretation of the results
pd.merge(raw, df['wpm'], left_index=True, right_index=True) \
  .sort_values(by='wpm', ascending=False)
```

|    |   bedrooms |   bathrooms |   area |   hoa |   parking |   year |   floor | exposure   | elevator   |   price |              wpm |
|---:|-----------:|------------:|-------:|------:|----------:|-------:|--------:|:-----------|:-----------|--------:|-----------------:|
|  4 |          2 |           2 |   1200 |   973 |         1 |   1965 |       8 | E          | Yes        |  209900 |      5.03416e+07 |
|  2 |          3 |           2 |   1600 |   150 |         0 |   2020 |       3 | E          | No         |  279000 |      4e+07       |
|  0 |          1 |           1 |    750 |   546 |         1 |   1951 |       4 | W          | Yes        |  175000 | 362192           |
|  3 |          1 |           1 |    985 |   424 |         1 |   1892 |       3 | W          | Yes        |  210000 |  95648           |
|  1 |          1 |           1 |    700 |   230 |         0 |   1895 |       2 | W          | No         |  199000 |   2099.13        |

The WPM Score ranked the apartments in a way all features are equally important. But, if an affordable HOA is more important than the number of bathrooms? Different features importances can be achieved by changes on their weights:

```{python}
weights = {'bedrooms'     :   1.,
           'bathrooms'     :  1.,
           'area'          :  1.,
           'hoa'           : -3.,
           'parking'      :   5.,
           'year'          :  1.,
           'floor'         : -1.,
           'price'         : -5.,
           'exposure_east' :  1.,
           'has_elevator'  :  5.
}
```
These weight values were purely arbitrary and they usually depend on the judgement of the decision maker. She may define a different scale. Here I used a Likert Scale, because it is very simple to use.

Now, parking slots, price, and elevator are very important; HOA is somewhat important, and the rest keeps the same.

|    |   bedrooms |   bathrooms |   area |   hoa |   parking |   year |   floor | exposure   | elevator   |   price |            wpm |
|---:|-----------:|------------:|-------:|------:|----------:|-------:|--------:|:-----------|:-----------|--------:|---------------:|
|  0 |          1 |           1 |    750 |   546 |         1 |   1951 |       4 | W          | Yes        |  175000 | 1274.69        |
|  4 |          2 |           2 |   1200 |   973 |         1 |   1965 |       8 | E          | Yes        |  209900 |   19.2726      |
|  3 |          1 |           1 |    985 |   424 |         1 |   1892 |       3 | W          | Yes        |  210000 |    2.27313     |
|  2 |          3 |           2 |   1600 |   150 |         0 |   2020 |       3 | E          | No         |  279000 |    0.04        |
|  1 |          1 |           1 |    700 |   230 |         0 |   1895 |       2 | W          | No         |  199000 |    0.000666255 |

The WPM Score has now a totally different scale, but it doesn't matter, because only the ranking is important.