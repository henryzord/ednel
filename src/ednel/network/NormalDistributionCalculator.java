/**
 * This class was adapted from
 *
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package ednel.network;

/**
 * This class calculates Gaussian probability density and cumulative
 * distribution functions.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NormalDistributionCalculator {

    /**
     * Parameters for the cumulative density function for normal distributions.
     *
     * The idea comes from
     * Zelen, Marvin; Severo, Norman C. (1964). Probability Functions (chapter 26).
     * Handbook of mathematical functions with formulas, graphs, and mathematical tables,
     * by Abramowitz, M.; and Stegun, I. A.: National Bureau of Standards. New York, NY:
     * Dover. ISBN 978-0-486-61272-0.
     *
     * The specific function is Algorithm 26.2.17
     *
     */
    static double[] b = new double[]{0.2316419, 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429};
    // 1/2*pi factor.
    public static final double normFact = 1 / Math.sqrt(2 * Math.PI);

    /**
     * @param x Double value.
     * @return The probability density function of a normal x, assuming zero
     * mean and unit variance.
     */
    private static double pdf(double x) {
        return normFact * Math.exp(-x * x / 2);
    }

    /**
     *
     * @param x Double value
     * @param mean Mean value.
     * @param sigma Standard deviation.
     * @return Gaussian probability density of the passed value for the given
     * mean and standard deviation.
     */
    public static double pdf(double x, double mean, double sigma) {
        return (1 / sigma) * pdf((x - mean) / sigma);
    }

    /**
     * Implements a method for approximating the cumulative
     * distribution function for a normal distribution, assuming zero
     * mean and unit variance, as seen in
     *
     * Zelen, Marvin; Severo, Norman C. (1964). Probability Functions (chapter 26).
     * Handbook of mathematical functions with formulas, graphs, and mathematical tables,
     * by Abramowitz, M.; and Stegun, I. A.: National Bureau of Standards. New York, NY:
     * Dover. ISBN 978-0-486-61272-0.
     *
     * The specific function is Algorithm 26.2.17
     *
     * @param x Double value.
     * @return Cumulative distribution function for the normal distribution,
     * upper bounded by the passed value x.
     */
    private static double cdf(double x) {
        double t = 1 / (1 + b[0] * x);
        double tDeg = t;

        double result = 0;
        for(int i = 1; i < b.length; i++) {
            result += b[i] * tDeg;
            tDeg *= t;
        }
        result = 1 - pdf(x) * result;
        return result;
    }

    /**
     * Implements a method for approximating the cumulative
     * distribution function for a normal distribution, as seen in
     *
     * Zelen, Marvin; Severo, Norman C. (1964). Probability Functions (chapter 26).
     * Handbook of mathematical functions with formulas, graphs, and mathematical tables,
     * by Abramowitz, M.; and Stegun, I. A.: National Bureau of Standards. New York, NY:
     * Dover. ISBN 978-0-486-61272-0.
     *
     * The specific function is Algorithm 26.2.17
     *
     * @param x Double value.
     * @param mean Mean value.
     * @return Cumulative distribution function for the normal distribution,
     * upper bounded by the passed value x.
     */
    public static double cdf(double x, double mean, double stDev) {
        return cdf((x - mean) / stDev);
    }
}