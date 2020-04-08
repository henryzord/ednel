package dn.variables;

import java.lang.reflect.Method;
import java.util.Objects;

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
        if(String.valueOf(obj).equals("null")) {
            return null;
        }
        try {
            return String.valueOf(method.invoke(obj, null));
        } catch(Exception e) {
            // TODO double check!
            System.out.println("Deu erro aqui!!!");
            return null;
        }
    }

    @Override
    public int compareTo(String o) {
        return this.toString().compareTo(o);
    }

    @Override
    public String toString() {
        return String.valueOf(obj);
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
