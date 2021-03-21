package ednel.eda.individual;

/**
 * Fitness of an individual.
 *
 * Might be the fitness using a holdout procedure, or using an internal n-fold cross-validation procedure; this class
 * is ignorant to this fact.
 */
public class Fitness {
    private Integer size;
    private Double learnQuality;
    private Double valQuality;

    Fitness(Integer size, Double learnQuality, Double valQuality) {
        this.size = size == null? 0 : size;
        this.learnQuality = learnQuality == null? 0 : learnQuality;
        this.valQuality = valQuality == null? 0 : valQuality;
    }

    public void setSize(Integer size) {
        this.size = size;
    }

    public void setLearnQuality(Double learnQuality) {
        this.learnQuality = learnQuality;
    }

    public void setValQuality(Double valQuality) {
        this.valQuality = valQuality;
    }

    public Integer getSize() {
        return size;
    }

    public Double getLearnQuality() {
        return learnQuality;
    }

    public Double getValQuality() {
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
