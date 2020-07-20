package dn.variables;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Locale;

public class ShadowMultivariateNormalDistribution extends Shadowvalue {
    protected double[] means;
    protected double[][] covMatrix;

    protected double a_min;
    protected double a_max;
    protected double scale_init;

    protected MultivariateNormalDistribution mv;

    private String mean_str;
    private String cov_matrix_str;

    public ShadowMultivariateNormalDistribution(
        MersenneTwister mt, double[][] data, double a_min, double a_max, double scale_init) throws NoSuchMethodException {
        super(null, null);

        this.a_min = a_min;
        this.a_max = a_max;
        this.scale_init = scale_init;

        this.mean_str = "[";
        this.means = new double [data.length];

        for(int i = 0; i < data.length; i++) {
            DescriptiveStatistics rS = new DescriptiveStatistics(data[i]);
            means[i] = rS.getMean();
            this.mean_str += String.format(Locale.US, "%01.6f", means[i]) + ",";
        }
        this.mean_str = this.mean_str.substring(0, this.mean_str.lastIndexOf(",")) + "]";

        this.cov_matrix_str = "[";
        this.covMatrix = new double[data.length][data.length];
        Covariance cov = new Covariance();
        for(int i = 0; i < data.length; i++) {
            this.cov_matrix_str += "[";
            for(int j = 0; j < data.length; j++) {
                covMatrix[i][j] = cov.covariance(data[i], data[j]);
                this.cov_matrix_str += String.format(Locale.US, "%01.6f", covMatrix[i][j]) + ",";
            }
            this.cov_matrix_str = this.cov_matrix_str.substring(0, this.cov_matrix_str.lastIndexOf(",")) + "]";
        }
        this.cov_matrix_str = this.cov_matrix_str.substring(0, this.cov_matrix_str.lastIndexOf(",")) + "]";

        this.mv = new MultivariateNormalDistribution(mt, means, covMatrix);

        this.method = this.mv.getClass().getMethod("sample");
        this.obj = this.mv;
    }

    @Override
    public String getValue() {
        return null;
        // TODo change later! test this!
//        Double clipped = Math.max(this.a_min, Math.min(this.a_max, this.mv.sample()));
//        return String.valueOf(clipped);
    }

    @Override
    public String toString() {
        return String.format(
            Locale.US,
            "(means=%s,covMatrix=%s,a_min=%01.6f,a_max=%01.6f,scale_init=%01.6f)",
            this.mean_str, this.cov_matrix_str, this.a_min, this.a_max, this.scale_init
        );
    }

}
