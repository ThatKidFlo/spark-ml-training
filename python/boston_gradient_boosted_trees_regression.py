import os

import numpy as np
import pandas as pd
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error

"""
1. CRIM: per capita crime rate by town
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS: proportion of non-retail business acres per town
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. NOX: nitric oxides concentration (parts per 10 million)
6. RM: average number of rooms per dwelling
7. AGE: proportion of owner-occupied units built prior to 1940
8. DIS: weighted distances to five Boston employment centres
9. RAD: index of accessibility to radial highways
10. TAX: full-value property-tax rate per $10,000
11. PTRATIO: pupil-teacher ratio by town
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13. LSTAT: % lower status of the population
14. MEDV: Median value of owner-occupied homes in $1000's
"""

df = pd.read_csv('boston_training.csv')
print(df.columns)
df_y = df.iloc[:, -1]  # survived or not
df_x = df.iloc[:, :-1]

os.environ["SPARK_HOME"] = "/Users/alexsisu/programs/spark-1.6.0"
conf = SparkConf().setAppName("myapp").setMaster("local")
sc = SparkContext(conf=conf)

xx = df_x.to_records(index=False)
yy = df_y.values

all_data = np.array(zip(yy, xx))
sss = ShuffleSplit(len(all_data) - 1, test_size=0.20, random_state=1234)

for train_indexes, test_indexes in sss:
    lparr = []
    test_lp_arr = []
    sample_data = all_data[train_indexes]
    test_data = all_data[test_indexes]

    for medianvalue, record in sample_data:
        lp = LabeledPoint(medianvalue, tuple(record))
        lparr.append(lp)

    for medianvalue, record in test_data:
        lp = LabeledPoint(medianvalue, tuple(record))
        test_lp_arr.append(lp)

    training_data = sc.parallelize(lparr).cache()
    test_data_rdd = sc.parallelize(test_lp_arr).cache()

    regression_model = GradientBoostedTrees.trainRegressor(training_data, categoricalFeaturesInfo={}, numIterations=10,maxDepth=10)
    result = regression_model.predict(test_data_rdd.map(lambda x: x.features))
    print regression_model
    print regression_model.toDebugString()
    print "==============================="
    predicted_data = result.collect()
    actual_data = test_data_rdd.map(lambda x: float(x.label)).collect()

    print mean_absolute_error(actual_data, predicted_data)
    break
