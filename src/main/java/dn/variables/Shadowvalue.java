package dn.variables;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.lang.reflect.Method;

public class Shadowvalue implements Comparable<String> {
    protected Method method;
    protected Object obj;

    /**
     * method must not have any parameters.
     * @param method Method to call when this class getValue method is called.
     * @param obj Object used to call method.
     */
    public Shadowvalue(Method method, Object obj) {
        this.method = method;
        this.obj = obj;
    }

    public String getValue() {
        if(obj != null) {
            try {
                return String.valueOf(method.invoke(obj));
            } catch(Exception e) {
                // TODO double check!
                System.out.println("Deu erro aqui!!!");
                return null;
            }
        }
        return null;
    }

    @Override
    public int compareTo(String o) {
        return this.getValue().compareTo(o);
    }

    @Override
    public String toString() {
        if(obj == null) {
            return null;
        }
        return obj.toString();
    }
}
