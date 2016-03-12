from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import numpy as np
import pandas as pd
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error
import os

os.environ["SPARK_HOME"] = "/Users/alexsisu/programs/spark-1.6.0"
conf = SparkConf().setAppName("myapp").setMaster("local")
sc = SparkContext(conf=conf)

data = MLUtils.loadLibSVMFile(sc, 'sample_libsvm_data.txt')
(trainingData, testData) = data.randomSplit([0.7, 0.3])

model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

# Save and load model
# model.save(sc, "target/tmp/myRandomForestClassificationModel")
# sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
