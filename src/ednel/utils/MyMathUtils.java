package ednel.utils;

public class Math {
    /**
     * Computes n!
     * @param n An integer
     * @return n!
     */
    public static int factorial(int n) {
        if(n == 0) {
            return 1;
        }
        int res = 1;
        while(n > 1) {
            res *= n;
            n -= 1;
        }
        return res;
    }
}
