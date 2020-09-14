package ednel.eda.individual;

public class NoAggregationPolicyException extends Exception {
    public NoAggregationPolicyException() {
    }

    public NoAggregationPolicyException(String message) {
        super(message);
    }

    public NoAggregationPolicyException(String message, Throwable cause) {
        super(message, cause);
    }

    public NoAggregationPolicyException(Throwable cause) {
        super(cause);
    }

    public NoAggregationPolicyException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
