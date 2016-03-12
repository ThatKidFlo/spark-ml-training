import pandas as pd
import numpy as np
from mllib.evaluation import RegressionMetrics
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
import os

def f(x):
    """The function to predict."""
    return 5+ x*np.sin(x)


x = np.arange(0,10,0.01)
y = f(x)

print(x)
print(y)

os.environ["SPARK_HOME"] = "/Users/alexsisu/programs/spark-1.6.0"
conf = SparkConf().setAppName("myapp").setMaster("local")
sc = SparkContext(conf=conf)


input_data = []
for (xx,yy) in zip(x,y):
    lp = LabeledPoint(xx,[yy])
    input_data.append(lp)

training_data = sc.parallelize(input_data).cache()
test_data_rdd = sc.parallelize(input_data).cache()


classificationModel = GradientBoostedTrees.trainRegressor(training_data,categoricalFeaturesInfo={}, numIterations=100, maxDepth=10)
result = classificationModel.predict(test_data_rdd.map(lambda x: x.features))

print classificationModel
print classificationModel.toDebugString()
print "==============================="
predicted_data = result.collect()
print(predicted_data)

zippedResult = test_data_rdd.map(lambda x: x.label).zip(result)


metrics = RegressionMetrics(zippedResult)

print(metrics.meanAbsoluteError)
print(metrics.meanSquaredError)
print(metrics.rootMeanSquaredError)
print(metrics.explainedVariance)

for p in zippedResult.collect():
    print p

