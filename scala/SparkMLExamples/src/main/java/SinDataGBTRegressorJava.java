import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by alexsisu on 09/03/16.
 */
public class SinDataGBTRegressorJava {
    public static Double generateData(Double x) {
        return x * Math.sin(x) + 5;
    }

    public static List<LabeledPoint> retrieveSinData() {
        List<LabeledPoint> ll = new ArrayList<LabeledPoint>();
        for (double xx = 0; xx < 10.0; xx = xx + 0.01) {
            ll.add(new LabeledPoint(xx, Vectors.dense(generateData(xx))));
        }
        return ll;

    }

    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestRegressorSinData");
        sparkConf.setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<LabeledPoint> data = sc.parallelize(retrieveSinData());

        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
        boostingStrategy.setNumIterations(10); // Note: Use more iterations in practice.
        boostingStrategy.getTreeStrategy().setMaxDepth(5);
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        final GradientBoostedTreesModel model =  GradientBoostedTrees.train(trainingData, boostingStrategy);

        JavaRDD<Double> gt = trainingData.map(x -> x.label());
        JavaRDD<Double> result = model.predict(trainingData.map(l -> l.features()));

        JavaPairRDD<Double,Double> tuple2RDD = gt.zip(result);//null; //result.zip(gt).map(x -> new Tuple2<Object, Object>(x._1(), x._2()));
        Object res = tuple2RDD.map(x-> new Tuple2<Double,Double>(x._1,x._2)).rdd();
        RegressionMetrics metrics = new RegressionMetrics((RDD<Tuple2<Object, Object>>) res);

        System.out.println("MSE = " + metrics.meanSquaredError());
        System.out.println("RMSE = " + metrics.rootMeanSquaredError());

        System.out.println("R-squared = " + metrics.r2());

        System.out.println("MAE = " + metrics.meanAbsoluteError());

        System.out.println("Explained variance = " + metrics.explainedVariance());

    }
}
