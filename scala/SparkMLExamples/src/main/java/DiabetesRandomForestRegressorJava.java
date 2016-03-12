import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by alexsisu on 08/03/16.
 */
public class DiabetesRandomForestRegressorJava {
    public static JavaSparkContext sc;

    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestRegressorDiabetesData");
        sparkConf.setMaster("local");

        sc = new JavaSparkContext(sparkConf);

        JavaRDD<LabeledPoint> data = prepareData();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];


        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        int numTrees = 30; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "variance";
        int maxDepth = 30;
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
        JavaRDD<String> allContent = sc.textFile("resources/diabetes.csv");

        String header = allContent.first();

        JavaRDD<String> all = allContent.filter(s -> !s.equals(header));

        JavaRDD<DiabetesEntry> allDataset = all.map(x -> new DiabetesEntry(x));
        JavaRDD<LabeledPoint> trainingDataInitial = allDataset.map(x -> x.toLabeledPoint());
        JavaRDD<LabeledPoint> trainingData = trainingDataInitial.map(x -> new Tuple2<>(x.label(), x.features())).map(x -> new LabeledPoint(x._1(), x._2()));

        trainingData.cache();
        return trainingData;
    }

    public static class DiabetesEntry {
        private final String line;

        public DiabetesEntry(String line) {
            this.line = line;
        }

        public LabeledPoint toLabeledPoint() {
            String[] tokens = this.line.split(",");
            ArrayList<Double> vals = new ArrayList<Double>();
            for (int i = 0; i < tokens.length-1; i++) {
                vals.add(Double.parseDouble(tokens[i].trim()));
            }
            Double[] arrType = new Double[0];
            Double[] valArr = vals.toArray(arrType);
            double[] valArrPrimitives = ArrayUtils.toPrimitive(valArr);
            Double label= Double.parseDouble(tokens[tokens.length-1].trim());
            return new LabeledPoint(label.intValue(), Vectors.dense(valArrPrimitives));
        }
    }
}
