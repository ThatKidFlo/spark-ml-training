import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by alexsisu on 08/03/16.
 */
public class TitanicRFClassificationJava {


    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestClassifierTitanicData");
        sparkConf.setMaster("local");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        JavaRDD<LabeledPoint> data = prepareData(sc);
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];


        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        int numTrees = 10; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        int maxDepth = 4;
        int maxBins = 32;
        int seed = 12345;

        RandomForestModel model = RandomForest.trainClassifier(trainingData, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        JavaRDD<Double> gt = testData.map(x -> x.label());
        JavaRDD<Double> result = model.predict(testData.map(l -> l.features()));

        JavaPairRDD<Double, Double> tuple2RDD = gt.zip(result);//null; //result.zip(gt).map(x -> new Tuple2<Object, Object>(x._1(), x._2()));
        Object res = tuple2RDD.map(x -> new Tuple2<>(x._1, x._2)).rdd();
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics((RDD<Tuple2<Object, Object>>) res);


        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.toArray());

// Recall by threshold
        JavaRDD<Tuple2<Object, Object>> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.toArray());

// F Score by threshold
        JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.toArray());

        JavaRDD<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.toArray());

// Precision-recall curve
        JavaRDD<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.toArray());

// Thresholds
        JavaRDD<Double> thresholds = precision.map(
                new Function<Tuple2<Object, Object>, Double>() {
                    public Double call(Tuple2<Object, Object> t) {
                        return new Double(t._1().toString());
                    }
                }
        );

// ROC Curve
        JavaRDD<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.toArray());

// AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

// AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());


    }

    public static JavaRDD<LabeledPoint> prepareData(JavaSparkContext sc) {
        JavaRDD<String> allTitanicContent = sc.textFile("resources/train.csv");

        String header = allTitanicContent.first();

        JavaRDD<String> all = allTitanicContent.filter(s -> !s.equals(header));

        JavaRDD<TitanicEntryJava> allDataset = all.map(x -> new TitanicEntryJava(x));
        JavaRDD<LabeledPoint> trainingDataInitial = allDataset.map(x -> x.toLabeledPoint());

        JavaRDD<LabeledPoint> trainingData = trainingDataInitial.map(x -> new Tuple2<>(x.label(), x.features())).map(x -> new LabeledPoint(x._1(), x._2()));

        trainingData.cache();
        return trainingData;
    }
}
