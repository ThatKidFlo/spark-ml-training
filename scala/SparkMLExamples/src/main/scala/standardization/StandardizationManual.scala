package standardization

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.functions.stddev
import org.apache.spark.sql.functions.mean

/**
  * Created by alexsisu on 10/03/16.
  */

/**
  * ZScore Normalization
  */
object StandardizationManual extends App {

  val sparkConf = new SparkConf()
  sparkConf.setAppName("localApp")
  sparkConf.setMaster("local")
  val sc = new SparkContext(sparkConf)
  val sqlContext: SQLContext = new SQLContext(sc)

  import sqlContext.implicits._

  val list = List(14.23, 13.2, 13.16, 14.37, 13.24, 14.2, 14.39, 14.06, 14.83, 13.86,
    14.1, 14.12, 13.75, 14.75, 14.38, 13.63, 14.3, 13.83, 14.19, 13.64)

  val rdd = sc.parallelize(list).map(x => (x, x))
  val df = rdd.toDF("col1", "col2")


  val selectedCollumn = df.select("col1").cache()
  val meanVal  = selectedCollumn.agg(mean($"col1"))
  val stdevVal = selectedCollumn.agg(stddev($"col1"))

  val m: Double =meanVal.collect()(0)(0).asInstanceOf[Double]
  val std: Double = stdevVal.collect()(0)(0).asInstanceOf[Double]
  println(m)
  println(std)

  val columnIndex = df.columns.indexOf("col1")


  val normalizedDF: RDD[Row] = df.map {
    case row=> Row.merge(row,Row((row(columnIndex).toString.toDouble-m)/std))
  }

  val normalizedColumn = normalizedDF.collect.map(row=>row(2))
 // normalizedColumn.foreach(println)

  println(normalizedColumn.mkString(","))

  val total = normalizedColumn.foldLeft[Double](0.0)((x,y)=>x+(y.toString.toDouble)*(y.toString.toDouble))

  var sum = 0d
  for(x<-normalizedColumn) {
    sum += (x.toString.toDouble * x.toString.toDouble)
  }

  println(sum)
  println(total)





}
