package dn.variables;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;

public class ShadowNormalDistribution extends Shadowvalue {

    protected double loc;
    protected double scale;
    protected double a_min;
    protected double a_max;
    protected double scale_init;

    protected NormalDistribution dist;

    public ShadowNormalDistribution(MersenneTwister mt, HashMap<String, Double> normalProperties) throws Exception {
        super(null, null);

        this.loc = normalProperties.get("loc");
        this.scale = normalProperties.get("scale");
        this.a_min = normalProperties.get("a_min");
        this.a_max = normalProperties.get("a_max");
        this.scale_init = normalProperties.get("scale_init");

        this.dist = new NormalDistribution(mt, loc, scale);

        this.method = this.dist.getClass().getMethod("sample");
        this.obj = this.dist;
    }

    public ShadowNormalDistribution(MersenneTwister mt, double loc, double scale, double a_min, double a_max, double scale_init) throws Exception {
        super(null, null);

        this.loc = loc;
        this.scale = scale;
        this.a_min = a_min;
        this.a_max = a_max;
        this.scale_init = scale_init;

        this.dist = new NormalDistribution(mt, loc, scale);

        this.method = this.dist.getClass().getMethod("sample");
        this.obj = this.dist;
    }

    @Override
    public String getValue() {
        Double clipped = Math.max(this.a_min, Math.min(this.a_max, this.dist.sample()));
        return String.valueOf(clipped);
    }

    @Override
    public String toString() {
        return String.format(
            Locale.US,
            "(loc=%01.6f,scale=%01.6f,a_min=%01.6f,a_max=%01.6f,scale_init=%01.6f)",
            this.loc, this.scale, this.a_min, this.a_max, this.scale_init
        );
    }

    public double getLoc() {
        return loc;
    }

    public double getScale() {
        return scale;
    }

    public double getA_min() {
        return a_min;
    }

    public double getA_max() {
        return a_max;
    }

    public double getScale_init() {
        return scale_init;
    }
}
