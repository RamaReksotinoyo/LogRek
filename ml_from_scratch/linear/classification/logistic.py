import numpy as np
from typing import Optional, Union


def _compute_cost_function(y: np.ndarray, y_pred: np.ndarray, coef_: np.ndarray, alpha: float, penalty: Optional[str] = None) -> float:
    # Extract number of samples
    m = len(y)

    # Calculate the log-loss
    log_loss = 0.0
    for i in range(m):
        log_loss += y[i] * np.log(y_pred[i]) + (1 - y[i]) * np.log(1 - y_pred[i])

    log_loss *= -(1 / m)

    if penalty is None:
        return log_loss
    elif penalty == 'L1':
        # Add L1 regularization term
        l1_penalty = alpha * np.sum(np.abs(coef_))
        cost = log_loss + l1_penalty

        return cost

class LogisticRegression:

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100_000,
        tol: float = 1e-4,
        alpha: float = 0.01,  # Regularization strength (L1 penalty)
        penalty: Optional[str] = None
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.penalty = penalty

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'LogisticRegression':
        """
        Fit the model according to the given training data
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features
        
        y : array-like of shape (n_samples,)
            Target vector relative to X
        
        Returns
        --------
        self : Fitted estimator
        """
        # Update the data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract the data & feature size
        self.n_samples, self.n_features = X.shape

        # Initialize the parameters
        self.coef_ = np.zeros(self.n_features)
        self.intercept_ = 0.0

        if self.penalty == 'L1':
            # Tune the parameters with L1 regularization
            for i in range(self.max_iter):
                # Make a new prediction
                y_pred = self.predict_proba(X)

                # Calculate the gradient
                grad_coef_ = -(y - y_pred).dot(X) / self.n_samples
                grad_intercept_ = -(y - y_pred).dot(np.ones(self.n_samples)) / self.n_samples

                # Add L1 regularization to the gradient (only for coef_, not intercept_)
                grad_coef_ += self.alpha * np.sign(self.coef_)

                # Update parameters
                self.coef_ -= self.learning_rate * grad_coef_
                self.intercept_ -= self.learning_rate * grad_intercept_

                # Calculate the cost function with L1 regularization (optional)
                cost = _compute_cost_function(y, y_pred, self.coef_, self.alpha, penalty=self.penalty)
                if cost < self.tol:
                    break
        else:
            # Tune the parameters without regularization
            for i in range(self.max_iter):
                # Make a new prediction
                y_pred = self.predict_proba(X)

                # Calculate the gradient
                grad_coef_ = -(y - y_pred).dot(X) / self.n_samples
                grad_intercept_ = -(y - y_pred).dot(np.ones(self.n_samples)) / self.n_samples

                # Update parameters
                self.coef_ -= self.learning_rate * grad_coef_
                self.intercept_ -= self.learning_rate * grad_intercept_

                # Break the iteration if the gradient is small
                grad_stack_ = np.hstack((grad_coef_, grad_intercept_))  # stack the gradient
                if all(np.abs(grad_stack_) < self.tol):
                    break

        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        proba : array-like of shape (n_samples,)
            Probability estimates for the input samples.
        """

        # Calculate the log odds
        logits = np.dot(X, self.coef_) + self.intercept_

        # Calculate the probability using sigmoid function
        proba = 1. / (1 + np.exp(-logits))

        return proba
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        class_labels : array-like of shape (n_samples,)
            Class labels for the input samples.
        """
        return (self.predict_proba(X) > 0.5).astype(int)
