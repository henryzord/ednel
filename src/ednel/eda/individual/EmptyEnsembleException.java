package ednel.eda.individual;

public class EmptyEnsembleException extends Exception {
    public EmptyEnsembleException() {
    }

    public EmptyEnsembleException(String message) {
        super(message);
    }

    public EmptyEnsembleException(String message, Throwable cause) {
        super(message, cause);
    }

    public EmptyEnsembleException(Throwable cause) {
        super(cause);
    }

    public EmptyEnsembleException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
