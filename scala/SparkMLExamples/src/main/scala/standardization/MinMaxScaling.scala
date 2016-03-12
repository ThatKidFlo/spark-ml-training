package standardization

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{min,max}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by alexsisu on 10/03/16.
  */

/**
  * ZScore Normalization
  */
object MinMaxScaling extends App {

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
  val minVal  = selectedCollumn.agg(min($"col1"))
  val maxVal = selectedCollumn.agg(max($"col1"))

  val m: Double =minVal.collect()(0)(0).asInstanceOf[Double]
  val std: Double = maxVal.collect()(0)(0).asInstanceOf[Double]

  val res = df.describe("col1")
  res.collect().foreach(println)
  df.schema
  println(df.schema)


  def colIndex(df:DataFrame,colName:String) = {
    df.schema.fields.foreach(field=> field.dataType)
  }






}
