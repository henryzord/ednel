package ednel.eda.individual;

/**
 * Fitness of an individual.
 *
 * Might be the fitness using a holdout procedure, or using an internal n-fold cross-validation procedure; this class
 * is ignorant to this fact.
 */
public class Fitness {
    private int size;
    private double learnQuality;
    private double valQuality;

    Fitness() {
        this.size = 0;
        this.learnQuality = 0;
        this.valQuality = 0;
    }

    Fitness(int size, double learnQuality) {
        this(size, learnQuality, 0);
    }

    Fitness(int size, double learnQuality, double valQuality) {
        this.size = size;
        this.learnQuality = learnQuality;
        this.valQuality = valQuality;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public void setLearnQuality(double learnQuality) {
        this.learnQuality = learnQuality;
    }

    public void setValQuality(double valQuality) {
        this.valQuality = valQuality;
    }

    public int getSize() {
        return size;
    }

    public double getLearnQuality() {
        return learnQuality;
    }

    public double getValQuality() {
        return valQuality;
    }

    @Override
    public String toString() {
        return String.format(
                "LearnQuality: %01.4f ValQuality: %01.4f Size: %03d",
                this.getLearnQuality(), this.getValQuality(), this.getSize()
        );
    }
}
