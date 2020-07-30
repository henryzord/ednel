package ednel.network.variables;

import java.lang.reflect.Method;
import java.util.Objects;

/**
 * A shadow value is a samplabe value from a probabilistic distribution, stored in an AbstractVariable.
 * To access the sampled value from a Shadowvalue object, one must invoke method getValue. This will unpack the value.
 *
 * This class is necessary since continuous variables (that encode normal distributions in this EDA) may have a
 * shadow value (i.e. the normal distribution mean, standard deviation) and a sampled value (i.e. the actual value sampled
 * from that distribution).
 */
public class Shadowvalue implements Comparable<String> {
    protected Method method;
    protected Object obj;

    /**
     * method must not have any parameters.
     *
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
                System.err.println("Deu erro aqui!!!");  // TODO double check!
                return null;
            }
        }
        return null;
    }

    @Override
    public String toString() {
        return String.valueOf(obj);
    }

    @Override
    public int compareTo(String o) {
        return this.toString().compareTo(o);
    }

    @Override
    public boolean equals(Object o) {
        return this.toString().equals(String.valueOf(o));
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.toString());
    }
}
