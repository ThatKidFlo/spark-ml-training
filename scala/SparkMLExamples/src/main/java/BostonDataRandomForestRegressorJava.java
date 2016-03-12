import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
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
public class BostonDataRandomForestRegressorJava {
    public static JavaSparkContext sc;

    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestRegressorBostonData");
        sparkConf.setMaster("local");

        sc = new JavaSparkContext(sparkConf);

        JavaRDD<LabeledPoint> data = prepareData();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];


        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        int numTrees = 100; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "variance";
        int maxDepth = 4;
        int maxBins = 32;
        int seed = 12345;

        RandomForestModel model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        JavaRDD<Double> gt = testData.map(x -> x.label());
        JavaRDD<Double> result = model.predict(testData.map(l -> l.features()));

        JavaPairRDD<Double, Double> tuple2RDD = gt.zip(result);//null; //result.zip(gt).map(x -> new Tuple2<Object, Object>(x._1(), x._2()));
        Object res = tuple2RDD.map(x -> new Tuple2<Double, Double>(x._1, x._2)).rdd();
        RegressionMetrics metrics = new RegressionMetrics((RDD<Tuple2<Object, Object>>) res);

        System.out.println("MSE = " + metrics.meanSquaredError());
        System.out.println("RMSE = " + metrics.rootMeanSquaredError());

        System.out.println("R-squared = " + metrics.r2());

        System.out.println("MAE = " + metrics.meanAbsoluteError());

        System.out.println("Explained variance = " + metrics.explainedVariance());
    }

    public static JavaRDD<LabeledPoint> prepareData() {
        JavaRDD<String> allBostonContent = sc.textFile("resources/boston_training.csv");

        String header = allBostonContent.first();

        JavaRDD<String> all = allBostonContent.filter(s -> !s.equals(header));

        JavaRDD<BostonEntryJava> allDataset = all.map(x -> new BostonEntryJava(x));
        JavaRDD<LabeledPoint> trainingDataInitial = allDataset.map(x -> x.toLabeledPoint());

//        StandardScaler scaler = new StandardScaler().setWithMean(true).setWithStd(true).fit(trainingDataInitial);

        JavaRDD<LabeledPoint> trainingData = trainingDataInitial.map(x -> new Tuple2<>(x.label(), x.features())).map(x -> new LabeledPoint(x._1(), x._2()));

        trainingData.cache();
        return trainingData;
    }
}
