package ednel.eda.individual;

import java.util.ArrayList;

public class PredictionsSizeContainer {
    protected int numberOfRules;
    protected ArrayList<String> lines;
    public PredictionsSizeContainer(int numberOfRules, ArrayList<String> lines) {
        this.numberOfRules = numberOfRules;
        this.lines = lines;
    }

    public int getNumberOfRules() {
        return numberOfRules;
    }

    public ArrayList<String> getLines() {
        return lines;
    }
}
