import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.HashMap;
import java.util.Map;

public class TitanicEntryJava {
    Map<String, String> features;

    //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    public TitanicEntryJava(String line) {
        String[] record = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1);

        features = new HashMap<>(14);

        features.put("PassengerId", record[0]);
        features.put("Survived", record[1]);
        features.put("Pclass", record[2]);
        features.put("Name", record[3]);
        features.put("Sex", record[4]);
        features.put("Age", record[5]);
        features.put("SibSp", record[6]);
        features.put("Parch", record[7]);
        features.put("Ticket", record[8]);
        features.put("Fare", record[9]);
        features.put("Cabin", record[10]);
        features.put("Embarked", record[11]);
    }

    public Double parseDouble(String s) {
        try {
            return Double.parseDouble(s);
        } catch (NumberFormatException e) {
            return 0d;
        }
    }

    public Double parseEmbarked(String s) {
        if (s.toLowerCase().trim().equals("s")) return 0d;
        if (s.toLowerCase().trim().equals("c")) return 1d;
        return 2d;
    }

    public Double parseSex(String s) {
        if ("male".equals(s)) return 1d;
        return 0d;
    }

    public Vector toVector() {
        return Vectors.dense(
                parseDouble(features.get("Pclass")),
                parseSex(features.get("Sex")),
                parseDouble(features.get("Age")),
                parseDouble(features.get("SibSp")),
                parseDouble(features.get("Parch")),
                parseEmbarked(features.get("Embarked")),
                parseDouble(features.get("Fare")));
    }

    public LabeledPoint toLabeledPoint() {
        return new LabeledPoint(parseDouble(features.get("Survived")), toVector());
    }
}
