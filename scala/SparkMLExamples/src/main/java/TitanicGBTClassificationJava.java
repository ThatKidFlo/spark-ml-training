import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;


public class TitanicGBTClassificationJava {
    public static JavaSparkContext sc;

    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestClassificationTitanicData");
        sparkConf.setMaster("local");

        sc = new JavaSparkContext(sparkConf);

        JavaRDD<LabeledPoint> data = prepareData();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];


        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(3); // Note: Use more iterations in practice.
        boostingStrategy.getTreeStrategy().setNumClasses(2);
        boostingStrategy.getTreeStrategy().setMaxDepth(5);
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);


        GradientBoostedTreesModel model = GradientBoostedTrees.train(trainingData, boostingStrategy);

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

    public static JavaRDD<LabeledPoint> prepareData() {
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
