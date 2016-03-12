import os

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

os.environ["SPARK_HOME"] = "/Users/alexsisu/programs/spark-1.6.0"
conf = SparkConf().setAppName("myapp").setMaster("local")
sc = SparkContext(conf=conf)
data = MLUtils.loadLibSVMFile(sc, 'sample_linear_regression_data.txt')
(trainingData, testData) = data.randomSplit([0.7, 0.3])

model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / \
          float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest model:')
print(model.toDebugString())
