import numpy as np

import pandas as pd
from mllib.evaluation import RegressionMetrics
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('kin8nm.csv')
print(df.columns)
df_y = df.iloc[:, -1]
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

    for target, record in sample_data:
        lp = LabeledPoint(target, tuple(record))
        lparr.append(lp)

    for target, record in test_data:
        lp = LabeledPoint(target, tuple(record))
        test_lp_arr.append(lp)

    training_data = sc.parallelize(lparr).cache()
    test_data_rdd = sc.parallelize(test_lp_arr).cache()

    regression_model = RandomForest.trainRegressor(training_data, categoricalFeaturesInfo={},
                                                   numTrees=3, featureSubsetStrategy="auto",
                                                   impurity='variance', maxDepth=5, maxBins=32)
    result = regression_model.predict(test_data_rdd.map(lambda x: x.features))
    print regression_model
    print regression_model.toDebugString()
    print "==============================="
    predicted_data = result.collect()
    actual_data = test_data_rdd.map(lambda x: float(x.label)).collect()

    print mean_absolute_error(actual_data, predicted_data)
    gt = test_data_rdd.map(lambda x: float(x.label))
    metrics = RegressionMetrics(gt.zip(result))
    print mean_absolute_error(actual_data, predicted_data)

    print metrics.meanAbsoluteError
    print metrics.meanSquaredError
    break
