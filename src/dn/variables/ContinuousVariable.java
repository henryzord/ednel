package dn.variables;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class ContinuousVariable extends AbstractVariable {

    protected HashMap<String, HashMap<String, Float>> normalProperties;

    public ContinuousVariable(String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table, ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt) throws Exception {
        super(name, parents, table, values, probabilities, mt);

        normalProperties = new HashMap<>(values.size());
        for(int i = 0; i < values.size(); i++) {
            String[] property = values.get(i).replaceAll("[\\(\\)\"]", "").split(",");

            HashMap<String, Float> thisProperty = new HashMap<>(property.length);

            for(int j = 0; j < property.length; j++) {
                String[] pair = property[j].split("=");
                thisProperty.put(pair[0], Float.valueOf(pair[1]));
            }
            normalProperties.put(values.get(i), thisProperty);
        }
    }

    @Override
    public String[] unconditionalSampling(int sample_size) throws Exception {
        throw new Exception("not implemented yet!");
    }

    @Override
    public String conditionalSampling(HashMap<String, String> lastStart) throws Exception {
        String value = super.conditionalSampling(lastStart);
        if(value != null) {
            HashMap<String, Float> thisNormal = normalProperties.get(value);
            NormalDistribution nd = new NormalDistribution(this.mt, thisNormal.get("loc"), thisNormal.get("scale"));
            double sampled = Math.max(
                    thisNormal.get("a_min"),
                    Math.min(thisNormal.get("a_max"), nd.sample())
            );
            return String.valueOf(sampled);
        }
        return null;
    }
}
