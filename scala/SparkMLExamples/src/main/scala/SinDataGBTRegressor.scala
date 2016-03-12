import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 18/12/15.
  */
object SinDataGBTRegressor extends App {

  def generateData(x: Double) = x * scala.math.sin(x) + 5

  def retrieveSinData(): List[LabeledPoint] = {
    val xx = (1 to 1000).map(_.toDouble).map(_ / 100.0)
    val yy = xx.map(generateData)
    yy.zip(xx).map(x => x._1.toString + "," + x._2.toString).foreach(println)
    val collection = yy.zip(xx).map(x => {
      val labelValue = x._1
      val s = Vectors.dense(x._2)
      LabeledPoint(labelValue, s)
    })
    collection.toList

  }

  val sparkConf = new SparkConf()
  sparkConf.setAppName("GradientBoostedTreeExamples")
  sparkConf.setMaster("local")


  val sc = new SparkContext(sparkConf)
  val data: RDD[LabeledPoint] = sc.parallelize(retrieveSinData())
  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))


  //MODEL CONFIGURATION
  val boostingStrategy: BoostingStrategy = BoostingStrategy.defaultParams("Regression")
  boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
  boostingStrategy.treeStrategy.maxDepth = 5
  //  Empty categoricalFeaturesInfo indicates all features are continuous.
  boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

  val model: GradientBoostedTreesModel = GradientBoostedTrees.train(trainingData, boostingStrategy)


  val gt = testData.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData.map(l => l.features))

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

  println(model.toDebugString)

}

