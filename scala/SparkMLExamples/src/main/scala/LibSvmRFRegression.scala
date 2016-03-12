import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 18/12/15.
  */
object LibSvmRFRegression extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressionExamplke")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)
  val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "resources/sample_linear_regression_data.txt")
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))


  //MODEL CONFIGURATION
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 100 // Use more in practice.
  val featureSubsetStrategy = "auto" // Let the algorithm choose.
  val impurity = "variance"
  val maxDepth = 10
  val maxBins = 32

  val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)



  val gt = testData.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData.map(l => l.features))

  val unmatchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ != 0d).count()
  val matchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ == 0d).count()

  val toBeEvaluated = gt.zip(result)
  val metrics = new RegressionMetrics(toBeEvaluated)


  // Squared error
  println(s"MSE = ${metrics.meanSquaredError}")
  println(s"RMSE = ${metrics.rootMeanSquaredError}")

  // R-squared
  println(s"R-squared = ${metrics.r2}")

  // Mean absolute error
  println(s"MAE = ${metrics.meanAbsoluteError}")

  // Explained variance
  println(s"Explained variance = ${metrics.explainedVariance}")


  val res = gt.zip(result).collect
  res.foreach(p=> p._1 +"    "+p._2)
  Console.in.readLine()

}

