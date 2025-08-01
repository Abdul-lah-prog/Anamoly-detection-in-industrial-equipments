**Project Title:**
Anomaly Detection in Industrial Equipment using Machine Learning and FastAPI Deployment

**Project Overview:**
This project aims to detect anomalies in industrial equipment using multiple machine learning models, offering a smart solution for predictive maintenance. Real-time anomaly prediction is enabled via a lightweight FastAPI-based deployment.

**Data Collection & Preprocessing:**
The dataset contains sensor readings and operational signals from industrial machinery. The preprocessing steps included:

Label Encoding & One-Hot Encoding for categorical data.

Standardization using StandardScaler to normalize feature scales.

Dimensionality Reduction using Linear Discriminant Analysis (LDA) to improve class separation and reduce computational cost.

Train-test split to assess generalization performance.

**Machine Learning Models Applied:**
Five machine learning classifiers were trained and evaluated:

K-Nearest Neighbors (KNN) – distance-based classification.

Naive Bayes – probabilistic approach.

Decision Tree – interpretable rule-based learning.

Random Forest – an ensemble of decision trees, improving robustness and accuracy.

Linear Discriminant Analysis (LDA) – also used as a classifier, not just for dimensionality reduction.

Each model was evaluated using accuracy, and results were visualized in a bar chart for comparison. Random Forest and Naive Bayes achieved the highest performance, demonstrating strong generalization and reliability.

**Visualization:**
Accuracy scores for each model were plotted using a bar chart (accuracies.png), aiding model selection based on performance.

**Web Deployment with FastAPI:**
To operationalize the solution:

A FastAPI app was developed with a /predict endpoint.

The best-performing model (Naive Bayes) and the corresponding scaler were serialized using joblib.

The API accepts POST requests with structured sensor input data and returns a classification result (Anomaly/Normal).

**Example Input:**
json
Copy
Edit
{
  "feature1": 14.2,
  "feature2": 1.07,
  "feature3": 8.9,
  "feature4": 2.3
}
**Output:**
json
Copy
Edit
{
  "prediction": 1,
  "label": "Anomaly"
}
FastAPI’s built-in Swagger UI (http://127.0.0.1:8000/docs) provides a user-friendly interface for real-time testing and integration.

**Conclusion:**
This project demonstrates an end-to-end pipeline for industrial anomaly detection using classical machine learning models and deploying them via FastAPI for real-time inference. It offers a scalable, interpretable, and deployable solution for smart manufacturing systems to minimize downtime and increase operational efficiency.
