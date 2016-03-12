import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.Map
import scala.util.control.NonFatal


class TitanicEntry(line: String) {
  var features: Map[String, String] = {
    val record = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)
    if (record.size == 12) {
      val Array(passengerId, survived, pclass, name, sex, age, sibSp, parch, ticket, fare, cabin, embarked) = record

      Map("passengerId" -> passengerId, "survived" -> survived, "pclass" -> pclass,
        "name" -> name, "sex" -> sex, "age" -> age, "sibSp" -> sibSp, "parch" -> parch,
        "ticket" -> ticket, "fare" -> fare, "cabin" -> cabin, "embarked" -> embarked)
    } else if (record.size == 11) {
      val Array(passengerId, pclass, name, sex, age, sibSp, parch, ticket, fare, cabin, embarked) = record
      Map("passengerId" -> passengerId, "pclass" -> pclass,
        "name" -> name, "sex" -> sex, "age" -> age, "sibSp" -> sibSp, "parch" -> parch,
        "ticket" -> ticket, "fare" -> fare, "cabin" -> cabin, "embarked" -> embarked)
    } else {
      Map[String, String]()
    }
  }

  def toDfLine() = {
    ( features("survived"),
      features("passengerId"),
      features("pclass"),
      features("name"),
      features("sex"),
      features("age"),
      features("sibSp"),
      features("parch"),
      features("ticket"),
      features("fare")
      )
  }

  def parseDouble(s: String, default: Double = 0): Double = try {
    s.toDouble
  } catch {
    case NonFatal(_) => default
  }

  def parseEmbarked(s: String): Double = {
    if (s.toLowerCase.trim.equals("s")) return 0d
    if (s.toLowerCase.trim.equals("c")) return 1d
    return 2d
  }

  def parseSex(s: String): Double = (if (s == "male") 1d else 0d)

  def toVector(): Vector = {
    return Vectors.dense(
      parseDouble(features("pclass")),
      parseSex(features("sex")),
      parseDouble(features("age")),
      parseDouble(features("sibSp")),
      parseDouble(features("parch")),
      parseEmbarked(features("embarked")),
      parseDouble(features("fare"))
    )
  }

  def toLabeledPoint(): LabeledPoint = {
    return LabeledPoint(parseDouble(features("survived")), toVector())
    //return LabeledPoint(parseDouble(features("pclass"))-1d, toVector())
  }
}