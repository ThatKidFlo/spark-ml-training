import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.immutable.Map
import scala.util.control.NonFatal

/**
  * Created by alexsisu on 10/03/16.
  */
class IrisEntry(line: String) {
  var features: Map[String, String] = {
    val record = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)
    val Array(sepalLength, sepalWidth, petalLength, petalWidth, species) = record
    Map("sepalLength" -> sepalLength, "sepalWidth" -> sepalWidth, "petalLength" -> petalLength,
      "petalWidth" -> petalWidth, "species" -> species)
  }

  def parseDouble(s: String, default: Double = 0): Double = try {
    s.toDouble
  } catch {
    case NonFatal(_) => default
  }

  def toVector(): Vector = {
    return Vectors.dense(
      parseDouble(features("sepalLength")),
      parseDouble(features("sepalWidth")),
      parseDouble(features("petalLength")),
      parseDouble(features("petalWidth"))
    )
  }

  def parseSpecies(name: String):Int = {
    if ("Iris-setosa".equals(name)) 0
    else if ("Iris-versicolor".equals(name)) 1
    else 2
  }

  def toLabeledPoint(): LabeledPoint = {
    return LabeledPoint(parseSpecies(features("species")), toVector())
  }
}