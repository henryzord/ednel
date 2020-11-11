package ednel.eda.individual;

/**
 * Fitness of an individual.
 *
 * Might be the fitness using a holdout procedure, or using an internal n-fold cross-validation procedure; this class
 * is ignorant to this fact.
 */
public class Fitness {
    private int size;
    private double quality;

    Fitness() {
        this.size = 0;
        this.quality = 0;
    }

    Fitness(int size, double quality) {
        this.size = size;
        this.quality = quality;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public void setQuality(double quality) {
        this.quality = quality;
    }

    public int getSize() {
        return size;
    }

    public double getQuality() {
        return quality;
    }

    @Override
    public String toString() {
        return String.format("Quality: %01.4f Size: %03d", this.getQuality(), this.getSize());
    }
}
