import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BostonEntryJava {
    Map<String, String> features;

    public BostonEntryJava(String line) {
        String[] record = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1);

        features = new HashMap<>(14);

        features.put("crim", record[0]);
        features.put("zn", record[1]);
        features.put("indus", record[2]);
        features.put("chas", record[3]);
        features.put("nox", record[4]);
        features.put("rm", record[5]);
        features.put("age", record[6]);
        features.put("dis", record[7]);
        features.put("rad", record[8]);
        features.put("tax", record[9]);
        features.put("ptratio", record[10]);
        features.put("b", record[11]);
        features.put("lstat", record[12]);
        features.put("medv", record[13]);
    }

    public Double parseDouble(String s) {
        try {
            return Double.parseDouble(s);
        } catch (NumberFormatException e) {
            return 0d;
        }
    }

    public Vector toVector() {
        return Vectors.dense(
                parseDouble(features.get("crim")),
                parseDouble(features.get("zn")),
                parseDouble(features.get("indus")),
                parseDouble(features.get("chas")),
                parseDouble(features.get("nox")),
                parseDouble(features.get("rm")),
                parseDouble(features.get("age")),
                parseDouble(features.get("dis")),
                parseDouble(features.get("rad")),
                parseDouble(features.get("tax")),
                parseDouble(features.get("ptratio")),
                parseDouble(features.get("b")),
                parseDouble(features.get("lstat"))
        );
    }

    public LabeledPoint toLabeledPoint() {
        return new LabeledPoint(parseDouble(features.get("medv")), toVector());
    }
}
