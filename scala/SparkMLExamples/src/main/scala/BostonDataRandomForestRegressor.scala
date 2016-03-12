import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 18/12/15.
  */
object BostonDataRandomForestRegressor extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorBostonData")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)
  sc.setLogLevel("ERROR") // TODO: remove

  def prepareData(): RDD[LabeledPoint] = {
    val allBostonContent = scala.io.Source.fromFile("resources/boston_training.csv").mkString
    val allLines = allBostonContent.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val allDataset = sc.parallelize(recordsOnly).map(line => new BostonEntry(line))

    val trainingDataInitial: RDD[LabeledPoint] = allDataset.map(record => record.toLabeledPoint)
    //trainingDataInitial.collect.foreach(println)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingDataInitial.map(x => x.features))

    val trainingData = trainingDataInitial.map(x => (x.label, scaler.transform(x.features))).map(x => LabeledPoint(x._1, x._2))

    trainingData.collect.foreach(println)
    trainingData.cache()
    trainingData
  }

  val data: RDD[LabeledPoint] = prepareData()

  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 100 // Use more in practice.
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

