import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by alexsisu on 10/03/16.
  */
object IrisDTClassification extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorBostonData")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)

  def prepareData(): RDD[LabeledPoint] = {
    val allIrisContent = scala.io.Source.fromFile("resources/iris.csv").mkString
    val allLines = allIrisContent.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val allDataset = sc.parallelize(recordsOnly).map(line => new IrisEntry(line))

    val trainingDataInitial: RDD[LabeledPoint] = allDataset.map(record => record.toLabeledPoint)
    trainingDataInitial
  }

  val trainingData: RDD[LabeledPoint] = prepareData()

  val split = trainingData.randomSplit(List(0.8, 0.2).toArray)
  val trainingSet = split(0)
  val testSet = split(1)


  val numClasses = 3
  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 5
  val featureSubsetStrategy = "auto"
  val impurity = "entropy"
  val maxDepth = 5
  val maxBins = 32

  val model: DecisionTreeModel = DecisionTree.trainClassifier(trainingSet, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

  val testData = testSet.map(labeledPoint => labeledPoint.features)
  val gt = testSet.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData)

  val unmatchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ != 0d).count()
  val matchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ == 0d).count()

  val predictionAndLabels = result.zip(gt)

  println(model.toDebugString)
  println(unmatchedPredictions)
  println(matchedPredictions)

  predictionAndLabels.collect.foreach(println)



  /*
  val metrics = new MulticlassMetrics(predictionAndLabels)

  println("Confusion matrix:")
  println(metrics.confusionMatrix)

  // Overall Statistics
  val precision = metrics.precision
  val recall = metrics.recall // same as true positive rate
  val f1Score = metrics.fMeasure
  println("Summary Statistics")
  println(s"Precision = $precision")
  println(s"Recall = $recall")
  println(s"F1 Score = $f1Score")

  // Precision by label
  val labels = metrics.labels
  labels.foreach { l =>
    println(s"Precision($l) = " + metrics.precision(l))
  }

  // Recall by label
  labels.foreach { l =>
    println(s"Recall($l) = " + metrics.recall(l))
  }

  // False positive rate by label
  labels.foreach { l =>
    println(s"FPR($l) = " + metrics.falsePositiveRate(l))
  }

  // F-measure by label
  labels.foreach { l =>
    println(s"F1-Score($l) = " + metrics.fMeasure(l))
  }

  // Weighted stats
  println(s"Weighted precision: ${metrics.weightedPrecision}")
  println(s"Weighted recall: ${metrics.weightedRecall}")
  println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
  println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}") */
}