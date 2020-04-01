package dn.variables;

import java.lang.reflect.Method;

public class ShadowMultivariateNormalDistribution extends Shadowvalue {
    protected double[] means;
    protected double[][] covMatrix;

    /**
     * method must not have any parameters.
     *
     * @param method Method to call when this class getValue method is called.
     * @param obj    Object used to call method.
     */
    public ShadowMultivariateNormalDistribution(Method method, Object obj) {
        super(method, obj);
    }
}
