import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
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
public class IrisRFClassificationJava {


    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestRegressorIrisData");
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

        RandomForestModel model = RandomForest.trainClassifier(trainingData, 3, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        JavaRDD<Double> gt = testData.map(x -> x.label());
        JavaRDD<Double> result = model.predict(testData.map(l -> l.features()));

        JavaPairRDD<Double, Double> tuple2RDD = gt.zip(result);//null; //result.zip(gt).map(x -> new Tuple2<Object, Object>(x._1(), x._2()));
        Object res = tuple2RDD.map(x -> new Tuple2<>(x._1, x._2)).rdd();

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics((RDD<Tuple2<Object, Object>>) res);

// Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);

// Overall statistics
        System.out.println("Precision = " + metrics.precision());
        System.out.println("Recall = " + metrics.recall());
        System.out.println("F1 Score = " + metrics.fMeasure());

// Stats by labels
        for (int i = 0; i < metrics.labels().length; i++) {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision
                    (metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics
                    .labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure
                    (metrics.labels()[i]));
        }

//Weighted stats
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
        System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
        System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
        System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());


    }

    public static JavaRDD<LabeledPoint> prepareData(JavaSparkContext sc) {
        JavaRDD<String> allTitanicContent = sc.textFile("resources/iris.csv");

        String header = allTitanicContent.first();

        JavaRDD<String> all = allTitanicContent.filter(s -> !s.equals(header));

        JavaRDD<IrisEntryJava> allDataset = all.map(x -> new IrisEntryJava(x));
        JavaRDD<LabeledPoint> trainingDataInitial = allDataset.map(x -> x.toLabeledPoint());

        JavaRDD<LabeledPoint> trainingData = trainingDataInitial.map(x -> new Tuple2<>(x.label(), x.features())).map(x -> new LabeledPoint(x._1(), x._2()));

        trainingData.cache();
        return trainingData;
    }

    public static class IrisEntryJava {
        private final String line;

        public IrisEntryJava(String line) {
            this.line = line;
        }

        public int parseSpecies(String name) {
            if ("Iris-setosa".equals(name)) return 0;
            else if ("Iris-versicolor".equals(name)) return 1;
            else return 2;
        }

        public LabeledPoint toLabeledPoint() {
            String[] tokens = this.line.split(",");
            ArrayList<Double> vals = new ArrayList<Double>();
            for (int i = 0; i < tokens.length - 1; i++) {
                vals.add(Double.parseDouble(tokens[i].trim()));
            }
            Double[] arrType = new Double[0];
            Double[] valArr = vals.toArray(arrType);
            double[] valArrPrimitives = ArrayUtils.toPrimitive(valArr);
            int label = parseSpecies(tokens[tokens.length - 1].trim());
            return new LabeledPoint(label, Vectors.dense(valArrPrimitives));
        }
    }
}
