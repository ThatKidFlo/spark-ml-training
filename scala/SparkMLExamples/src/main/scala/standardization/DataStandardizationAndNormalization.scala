package standardization

import org.apache.spark.ml.feature.{MinMaxScaler, Normalizer, StandardScaler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 09/03/16.
  */
object DataStandardizationAndNormalization extends App {

  def getTraininData() = {
    import sqlContext.implicits._
    val df = sc.parallelize(List(1d,2d,3d,4d,5f)).map(x=> (Vectors.dense(x,x+1,x+2,x+3,x+4,x+5),x+1,x+math.sin(x)))
    df.toDF("col1","col2","col3")
  }

  val sparkConf = new SparkConf()
  sparkConf.setAppName("GradientBoostedTreeExamples")
  sparkConf.setMaster("local")

  val sc = new SparkContext(sparkConf)
  val sqlContext = new SQLContext(sc)
 val df = getTraininData()


  var scaler = new StandardScaler()
    .setInputCol("col1")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false)

  // Compute summary statistics by fitting the StandardScaler.
  var scalerModel = scaler.fit(df)

  // Normalize each feature to have unit standard deviation.
  var scaledData = scalerModel.transform(df)
  scaledData.show()
  scaledData.map(row=>row(3)).collect.foreach(println)

  scaler = new StandardScaler()
    .setInputCol("col1")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(true)

  // Compute summary statistics by fitting the StandardScaler.
  scalerModel = scaler.fit(df)

  // Normalize each feature to have unit standard deviation.
  scaledData = scalerModel.transform(df)
  scaledData.show()
  scaledData.map(row=>row(3)).collect.foreach(println)

  val normalizer = new Normalizer()
    .setInputCol("col1")
    .setOutputCol("normFeatures")
    .setP(1.0)

  // Compute summary statistics by fitting the StandardScaler.
  println("===================== NORMALIZED DATA=====================")
  scaledData  = normalizer.transform(df)

  //scaledData.show()
  scaledData.map(row=>row(3)).collect.foreach(println)

  println("==========================================")



  val minMaxScaler = new MinMaxScaler()
    .setInputCol("col1")
    .setOutputCol("scaledFeatures")

  // Compute summary statistics by fitting the StandardScaler.
  val minMaxModel = scaler.fit(df)

  // Normalize each feature to have unit standard deviation.
  scaledData = minMaxModel.transform(df)
  //println("==========================================")
  //scaledData.show()
  println("==========================================")
  scaledData.map(row=>row(3)).collect.foreach(println)



}
