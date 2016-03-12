import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.{SparkConf, SparkContext}

/*** Created by alexsisu on 10/03/16.
  */
object IrisClustering extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("RandomForestRegressorBostonData")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)

  def prepareData(): RDD[org.apache.spark.mllib.linalg.Vector] = {
    val allIrisContent = scala.io.Source.fromFile("resources/iris.csv").mkString
    val allLines = allIrisContent.split("\n")
    val recordsOnly = allLines.slice(1, allLines.length)
    val allDataset = sc.parallelize(recordsOnly).map(line => new IrisEntry(line).toLabeledPoint().features)
    allDataset
  }

  val trainingData: RDD[org.apache.spark.mllib.linalg.Vector] = prepareData()
  val numClusters = 4
  val numIterations = 20
  val clusters = KMeans.train(trainingData, numClusters, numIterations)

  // Evaluate clustering by computing Within Set Sum of Squared Errors
  val WSSSE = clusters.computeCost(trainingData)
  println(clusters.predict(Vectors.dense(5.1,3.5,1.4,0.2)))
  println(clusters.predict(Vectors.dense(4.6,3.1,1.5,0.2)))
  println(clusters.predict(Vectors.dense(6.2,2.2,4.5,1.5)))
  println(clusters.predict(Vectors.dense(6.4,3.2,5.3,2.3)))
  println("Within Set Sum of Squared Errors = " + WSSSE)

}