import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.NonFatal


object TitanicRFClassificationCrossValidation extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("localApp")
  sparkConf.setMaster("local")
  val sc = new SparkContext(sparkConf)

  def parseDouble(s: String, default: Double = 0): Double = try {
    s.toDouble
  } catch {
    case NonFatal(_) => default
  }

  def parseSex(s: String): Double = (if (s == "male") 1d else 0d)

  def prepareData(): (DataFrame, DataFrame) = {
    val allTitanicTrainingContent: String = scala.io.Source.fromFile("resources/train.csv").mkString
    val allLines: Array[String] = allTitanicTrainingContent.split("\n")
    val recordsOnly: Array[String] = allLines.slice(1, 892)
    val allDataset: RDD[LabeledPoint] = sc.parallelize(recordsOnly).map(line => new TitanicEntry(line).toLabeledPoint())
    val sqlContext: SQLContext = new SQLContext(sc)
    import sqlContext.implicits._
    val trainSet: DataFrame = allDataset.toDF()
    val testSet: DataFrame = trainSet.select("features")
    (trainSet, testSet)
  }

  val (trainingData, testData) = prepareData()

  println(trainingData.columns)

  val split: Array[DataFrame] = trainingData.randomSplit(List(0.8, 0.2).toArray)
  val trainingSet: DataFrame = split(0)
  val testSet: DataFrame = split(1)


  val randomForest: RandomForestClassifier = new RandomForestClassifier().
    setImpurity("gini").
    setFeatureSubsetStrategy("auto").
    setNumTrees(4).
    setMaxDepth(10).
    setMaxBins(10).setLabelCol("indexedLabel").setFeaturesCol("features")

  val labelIndexer: StringIndexerModel = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(trainingSet)

  val pipeline: Pipeline = new Pipeline().setStages(Array(labelIndexer, randomForest))
  val paramGrid: Array[ParamMap] = new ParamGridBuilder().
    addGrid(randomForest.maxBins, Array(25, 28, 31))
    .addGrid(randomForest.maxDepth, Array(4, 6, 8))
    .addGrid(randomForest.impurity, Array("entropy", "gini")).build()

  val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")

  val cv: CrossValidator = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(4)


  val cvModel: CrossValidatorModel = cv.fit(trainingSet)


  val pred3: DataFrame = cvModel.transform(testData)
  pred3.collect().foreach(println)
  pred3.columns.foreach(println)
  println(cvModel.explainParams())


}
