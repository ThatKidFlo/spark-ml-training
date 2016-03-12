import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.loss.LogLoss
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.NonFatal

/**
  * Created by alexsisu on 18/12/15.
  */
object TitanicGBTClassification extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("GradientBoostedTreeExamples")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)

  def parseDouble(s: String, default: Double = 0): Double = try {
    s.toDouble
  } catch {
    case NonFatal(_) => default
  }

  def parseSex(s: String): Double = (if (s == "male") 1d else 0d)

  def prepareData(): RDD[LabeledPoint] = {
    val allTitanicTrainingContent = scala.io.Source.fromFile("resources/train.csv").mkString
    val allLines = allTitanicTrainingContent.split("\n")
    val recordsOnly = allLines.slice(1, 892)
    val allDataset = sc.parallelize(recordsOnly).map(line => new TitanicEntry(line))

    val trainingData: RDD[LabeledPoint] = allDataset.map(record => record.toLabeledPoint)
    trainingData.collect.foreach(println)
    trainingData.cache()
    trainingData

  }

  //DATA PREPARATION
  val trainingData: RDD[LabeledPoint] = prepareData()

  //MODEL CONFIGURATION
  val customStrategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 30, numClasses = 2)
  val boostingStrategy = new BoostingStrategy(customStrategy, LogLoss, 10, 0.1, 1e-5)

  //TRAINING
  val model: GradientBoostedTreesModel = GradientBoostedTrees.train(trainingData, boostingStrategy)

  val testData = trainingData.map(labeledPoint => labeledPoint.features)
  val gt = trainingData.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData)

  val unmatchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ != 0d).count()
  val matchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ == 0d).count()
  println(matchedPredictions)
  println(unmatchedPredictions)
  println(matchedPredictions * 1.0 / ((unmatchedPredictions + matchedPredictions) * 1.0))

  val predictionAndLabels = result.zip(gt)

  // Instantiate metrics object
  val metrics = new BinaryClassificationMetrics(predictionAndLabels)

  // Precision by threshold
  val precision = metrics.precisionByThreshold
  precision.foreach { case (t, p) =>
    println(s"Threshold: $t, Precision: $p")
  }

  // Recall by threshold
  val recall = metrics.recallByThreshold
  recall.foreach { case (t, r) =>
    println(s"Threshold: $t, Recall: $r")
  }

  // Precision-Recall Curve
  val PRC = metrics.pr

  // F-measure
  val f1Score = metrics.fMeasureByThreshold
  f1Score.foreach { case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 1")
  }

  val beta = 0.5
  val fScore = metrics.fMeasureByThreshold(beta)
  f1Score.foreach { case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 0.5")
  }

  // AUPRC
  val auPRC = metrics.areaUnderPR
  println("Area under precision-recall curve = " + auPRC)

  // Compute thresholds used in ROC and PR curves
  val thresholds = precision.map(_._1)

  // ROC Curve
  val roc = metrics.roc

  // AUROC
  val auROC = metrics.areaUnderROC
  println("Area under ROC = " + auROC)
}

