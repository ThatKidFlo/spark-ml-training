import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.read

import scala.collection.mutable

case class NewsArticle(date : String, title : String, byline : String, fulltext : String)

object LDA extends App {
    /*
      At the moment the LDA is configured for training on the Salary Prediction dataset,
      you may want to use other configurations for the datasets
     */

    // Init the Spark Context
    val conf = new SparkConf().setAppName("LDA").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val path = "resources/"

    // Build RDD from Tweets dataset, using tweet body as the document
    val tweets_rdd = sc.textFile(path + "tweets.csv")

    // Build RDD from Salaries dataset, using Job Title as the document
    // TODO: You can download a more bigger train dataset from from https://www.kaggle.com/c/job-salary-prediction/data ~400MB
    val salaries_rdd = sc.textFile(path + "Valid_rev1.csv")

    // Build RDD from wikinews dataset, using news title as the document
    val news_rdd = sc.textFile(path + "wikinews.json")
    val news_json = news_rdd.map(record => {
      implicit val formats = DefaultFormats
      read[NewsArticle](record)
    })


    val verbs = sc.textFile(path + "english_verbs.csv").map(_.split(",").toList).reduce(_ ::: _)
    val stopwords = sc.textFile(path + "english-stopwords.csv").collect().toList

    val tweetsBody = tweets_rdd.map(x => x.split(",")(4))
    val salaryTitle = salaries_rdd.map(x => x.split(",")(1))
    val wikinews_title = news_json.map(_.title)

    // TODO: Here use either tweetsBody, salaryTitle or wikinews_title to change LDA input dataset
    val tokenized: RDD[Array[String]] = salaryTitle.map(_.toLowerCase.split("\\s")).map(
      _.filter(_.length > 3)
        .filter(!verbs.contains(_))
        .filter(!stopwords.contains(_))
        .filter(_.forall( x => { x == '#' || java.lang.Character.isLetter(x) }))
    )
    val termCounts: Array[(String, Long)] = tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val noOfLeastImportantWordsToRemove = 0 // Use 0 when training on salaries, we only use titles and already filtered them above, we should not remove other words
    val vocabArray: Array[String] = termCounts.takeRight(termCounts.size - noOfLeastImportantWordsToRemove).map(_._1)

    // Build the input to pass to the LDA
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] = tokenized.zipWithIndex.map {
      case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
    }

    // Set LDA parameters, at the moment its configured for Salary Prediction dataset
    val numTopics = 10
    val maxIterations = 200
    val ldaModel = new LDA().setK(numTopics).setMaxIterations(maxIterations).run(documents)
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }
}
