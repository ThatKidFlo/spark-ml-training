import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pyspark.mllib.tree import RandomForest


def df_cleaner(df):
    return df


df = pd.read_csv('german_credit.csv')



df = df_cleaner(df)
print(df.columns)
df_y = df.iloc[:, 0]
df_x = df.iloc[:, 1:]
import os

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

    classificationModel = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},
                                                       numTrees=3)
    result = classificationModel.predict(test_data_rdd.map(lambda x: x.features))
    print classificationModel
    print classificationModel.toDebugString()
    print "==============================="
    predicted_data = result.collect()
    actual_data = test_data_rdd.map(lambda x: float(x.label)).collect()

    print mean_absolute_error(actual_data, predicted_data)
    print accuracy_score(actual_data,predicted_data)
    print(classificationModel)

    #for p in predicted_data:
    #    print p
    break
