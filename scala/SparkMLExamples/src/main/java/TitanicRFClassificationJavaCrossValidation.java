import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import scala.collection.immutable.$colon$colon;
import scala.collection.immutable.List$;

/**
 * Created by alexsisu on 09/03/16.
 */
public class TitanicRFClassificationJavaCrossValidation {
    public static void main(String[] argv) {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("RandomForestClassificationTitanicData");
        sparkConf.setMaster("local");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);

        JavaRDD<LabeledPoint> data = TitanicRFClassificationJava.prepareData(sc);
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});

        JavaRDD<LabeledPoint> trainingSetRdd = splits[0];
        JavaRDD<LabeledPoint> testSetRdd = splits[1];

        DataFrame trainingSet = sqlContext.createDataFrame(trainingSetRdd, LabeledPoint.class);
        DataFrame testSet = sqlContext.createDataFrame(testSetRdd, LabeledPoint.class);


        RandomForestClassifier randomForest = new RandomForestClassifier().
                setImpurity("gini").
                setFeatureSubsetStrategy("auto").
                setNumTrees(4).
                setMaxDepth(10).
                setMaxBins(10).setLabelCol("indexedLabel").setFeaturesCol("features");

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(trainingSet);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{labelIndexer, randomForest});
        scala.collection.immutable.List impurities = list("entropy", "gini");

        ParamMap[] paramGrid = new ParamGridBuilder().
                addGrid(randomForest.maxBins(), new int[]{25, 28, 31})
                .addGrid(randomForest.maxDepth(), new int[]{4, 6, 8})
                .addGrid(randomForest.impurity(), impurities).build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(4);
        CrossValidatorModel cvModel = cv.fit(trainingSet);
        System.out.println(cvModel.explainParams());

        Model model = cvModel.bestModel();
        System.out.println(cvModel.avgMetrics());
        DataFrame predictions = model.transform(testSet);

        for (Row x : predictions.collect()) {
            System.out.println(x);
        }

    }

    public static <T> scala.collection.immutable.List<T> list(T... ts) {
        scala.collection.immutable.List<T> result = List$.MODULE$.empty();
        for (int i = ts.length; i > 0; i--) {
            result = new $colon$colon(ts[i - 1], result);
        }
        return result;
    }
}
