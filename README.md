# Logistic Regression From Scratch

    Proyek kecerdasan buatan ini mengimplementasikan algoritma *logistic regression*. Kode program ditulis *from scratch* / tanpa menggunakan pustaka pembelajaran mesin terkenal saat ini, seperti *Scikit-learn*. 

    *Logistic Regression* adalah metode statistik yang sering digunakan untuk kasus klasifikasi. Meskipun namanya mengandung kata *"regression"*, metode ini digunakan untuk memprediksi probabilitas suatu kelas atau kejadian, seperti dalam masalah klasifikasi biner.

    Komponen pembelajaran:
    1. Paramameters:
        - Bobot (koefisien)
        - Bias (*intercept*)
    2. Tujuan Utama:
        - Meminimalkan *negative log-likelihood*
    3. Hyperparameters:
        - *Learning rate*
        - *Maximum Iteration*
        - *Tolerance*
        - Jenis penalti yang digunakan (Jika dibutuhkan)
        - Alpha (Jika dibutuhkan)

    Pseudocode untuk melatih model:
    ```
    FUNCTION fit(X, y):
        # X: fitur input (n_samples, n_features)
        # y: target label (n_samples,)
        
        # Convert X and y to numpy arrays (if they aren't already)
        X = CONVERT_TO_ARRAY(X)
        y = CONVERT_TO_ARRAY(y)
        
        # Extract number of samples and features
        n_samples, n_features = SHAPE(X)

        # Initialize coefficients and intercept
        coef_ = ARRAY_OF_ZEROS(n_features)  # Koefisien awal diinisialisasi ke 0
        intercept_ = 0.0                    # Intercept diinisialisasi ke 0

        # Check if L1 regularization is used
        IF penalty == 'L1':
            FOR i FROM 1 TO max_iter:
                # Calculate predicted probabilities (sigmoid of logits)
                y_pred = PREDICT_PROBA(X, coef_, intercept_)

                # Calculate gradient of coefficients and intercept
                grad_coef_ = -(y - y_pred) DOT X / n_samples
                grad_intercept_ = -(y - y_pred) DOT ONES(n_samples) / n_samples

                # Add L1 regularization term to gradient of coefficients
                grad_coef_ += alpha * SIGN(coef_)

                # Update coefficients and intercept
                coef_ -= learning_rate * grad_coef_
                intercept_ -= learning_rate * grad_intercept_

                # Calculate cost (negative log-likelihood with L1 regularization)
                cost = COMPUTE_COST_FUNCTION(y, y_pred, coef_, alpha, penalty='L1')

                # Check for convergence
                IF cost < tol:
                    BREAK

        ELSE:
            # No regularization case (standard logistic regression)
            FOR i FROM 1 TO max_iter:
                # Calculate predicted probabilities (sigmoid of logits)
                y_pred = PREDICT_PROBA(X, coef_, intercept_)

                # Calculate gradient of coefficients and intercept
                grad_coef_ = -(y - y_pred) DOT X / n_samples
                grad_intercept_ = -(y - y_pred) DOT ONES(n_samples) / n_samples

                # Update coefficients and intercept
                coef_ -= learning_rate * grad_coef_
                intercept_ -= learning_rate * grad_intercept_

                # Check for convergence
                grad_stack_ = STACK(grad_coef_, grad_intercept_)
                IF ALL(ABS(grad_stack_) < tol):
                    BREAK

        RETURN self  # Return the trained model

    ```


    Referensi:
        - [L1 regularisasi untuk *logistik regression*](https://www.youtube.com/watch?v=_aGWjt7GKBE&t=1228s)