# Formal Verification Framework for Fairness and Robustness

This document provides a high-level explanation of how the framework works, focusing on the general logic and formal verification aspects.

---

## General Workflow of the Framework

The framework evaluates the fairness and robustness of a Decision Tree model using formal verification techniques. Here’s how it works:

---

### 1. Loading and Training the Model

- **Step**: Load the dataset and train a Decision Tree model.
- **Purpose**: The model is trained on the dataset to make predictions.
- **Output**: A trained Decision Tree model (`clf`).

---

### 2. Extracting Z3 Constraints

- **Step**: Extract the constraints of the trained Decision Tree into a format compatible with the Z3 solver.
- **Purpose**: The Z3 solver requires the model's logic to be expressed as constraints.
- **Output**: A set of Z3 constraints (`tree_cons`) representing the Decision Tree.

---

### 3. Evaluating the Model

- **Step**: Evaluate the model's performance and check for violations.
- **Purpose**: Determine if the model's predictions are reliable by checking for fairness and robustness violations.
- **Output**: Predictions, accuracy, precision, time taken, and violations.

#### **Key Logic**:

1. **Generate Predictions**:
   - For the **Constrained Model**, use the Z3 solver to generate predictions and check for violations.
   - For the **Original Model**, use the trained model to generate predictions without checking for violations.

2. **Check for Violations**:
   - **Fairness Check**: Flip the sensitive attribute (e.g., gender) and verify if the prediction changes.
   - **Robustness Check**: Perturb non-sensitive features within a small delta and verify if the prediction changes.

3. **Adjust Predictions for Violations**:
   - If a sample has a violation, flip its prediction to treat it as a misclassification.

4. **Calculate Metrics**:
   - Calculate accuracy and precision using the adjusted predictions.

---

### 4. Printing the Results

- **Step**: Print the classification report, accuracy, precision, time taken, and violation rate.
- **Purpose**: Provide a clear and comprehensive evaluation of the model's performance and reliability.

---

## Formal Verification Logic

1. **Fairness Check**:
   - Flip the sensitive attribute (e.g., gender) and verify if the prediction changes.
   - If it does, the model is considered unfair for that sample.

2. **Robustness Check**:
   - Perturb non-sensitive features within a small delta and verify if the prediction changes.
   - If it does, the model is considered unstable for that sample.

3. **Verified Accuracy**:
   - A prediction is only considered correct if:
     - The prediction matches the true label.
     - No violations (fairness or robustness) are found in the neighborhood of the sample.

---

## Example Output

When you run `comparison.py`, you will see output similar to the following:

```plaintext
Classification Report for Constrained Model:
              precision    recall  f1-score   support

           0       0.85      0.95      0.90      4533
           1       0.77      0.48      0.59      1500

    accuracy                           0.83      6033
   macro avg       0.81      0.71      0.74      6033
weighted avg       0.83      0.83      0.82      6033

Constrained Model Performance:
Accuracy: 0.8339
Precision: 0.7666
Time for constrained model predictions: 21.1299 seconds
Violation Rate: 0.0104

Classification Report for Original Model:
              precision    recall  f1-score   support

           0       0.85      0.95      0.90      4533
           1       0.77      0.48      0.59      1500

    accuracy                           0.83      6033
   macro avg       0.81      0.71      0.74      6033
weighted avg       0.83      0.83      0.82      6033

Original Model Performance:
Accuracy: 0.8339
Precision: 0.7666
Time for original model predictions: 0.0009 seconds
```

---

## Interpretation of Results

- **Classification Report**: Provides a detailed breakdown of the model's performance for each class, including precision, recall, and F1-score.
- **Accuracy**: The fraction of correctly classified samples.
- **Precision**: The fraction of true positives among all positive predictions.
- **Time Taken**: The time taken for predictions is significantly higher for the constrained model due to the use of the Z3 solver.
- **Violation Rate**: The fraction of samples with violations, indicating the model's reliability.

---

This explanation provides a high-level overview of how the framework works, focusing on the general logic and formal verification aspects.