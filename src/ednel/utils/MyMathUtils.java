package ednel.utils;

public class MyMathUtils {
    /**
     * Computes n!
     * @param n An integer
     * @return n!
     */
    public static long factorial(int n) {
        if(n < 0) {
            return n;
        }
        if(n == 0) {
            return 1;
        }
        long res = 1;
        while(n > 1) {
            res *= n;
            n -= 1;
        }
        return res;
    }

    /**
     * Computes the log factorial of a number.
     *
     * Adapted from
     * https://github.com/haifengl/smile/blob/1826b2f0fd9ba57ec0956792f00a419e950c850f/math/src/main/java/smile/math/MathEx.java#L447
     * @param n Number to have its log factorial computed
     * @return The log factorial of a number
     */
    public static double lfactorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException(String.format("n has to be non-negative: %d", n));
        }

        double f = 0.0;
        for (int i = 2; i <= n; i++) {
            f += Math.log(i);
        }

        return f;
    }
}
