import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 10/03/16.
  */
object CalHousingRegressor extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorKin8Data")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)

  def prepareData(): RDD[LabeledPoint] = {
    val allMovement = scala.io.Source.fromFile("resources/cal_housing.csv").mkString
    val allLines = allMovement.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val trainingDataInitial: RDD[LabeledPoint] = sc.parallelize(recordsOnly).map { case line =>
      val arr = line.split(",").map(token=>token.trim.toDouble)
        new LabeledPoint(arr.last.asInstanceOf[Int],Vectors.dense(arr.slice(0,arr.size-1)))
    }

    trainingDataInitial
  }

  val data: RDD[LabeledPoint] = prepareData()

  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 30 // Use more in practice.
  val featureSubsetStrategy = "auto" // Let the algorithm choose.
  val impurity = "variance"
  val maxDepth = 30
  val maxBins = 32

  val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  val gt = testData.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData.map(l => l.features))

  //  val res = gt.zip(result).collect

  //  res.foreach(println)

  // Instantiate metrics object
  val metrics = new RegressionMetrics(gt.zip(result))

  // Squared error
  println(s"MSE = ${metrics.meanSquaredError}")
  println(s"RMSE = ${metrics.rootMeanSquaredError}")

  // R-squared
  println(s"R-squared = ${metrics.r2}")

  // Mean absolute error
  println(s"MAE = ${metrics.meanAbsoluteError}")

  // Explained variance
  println(s"Explained variance = ${metrics.explainedVariance}")
}