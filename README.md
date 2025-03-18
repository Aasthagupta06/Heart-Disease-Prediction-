# 🩺 **Heart Disease Prediction Using Machine Learning and Deep Learning**

## 📌 Problem Statement
Heart disease is a leading cause of death worldwide. Early detection through predictive modeling can significantly improve patient outcomes. This project aims to build a robust machine learning and deep learning pipeline to predict heart disease using patient health data. By conducting exploratory data analysis (EDA) and applying various algorithms, we extract insights that can assist medical professionals in making informed decisions.

---

## 🎯 **Objectives**
- Perform exploratory data analysis (EDA) to understand and clean the dataset.
- Identify patterns in categorical and numerical variables.
- Apply data preprocessing techniques including feature engineering.
- Train and compare multiple machine learning models.
- Implement a deep learning model using TensorFlow.
- Evaluate models using classification metrics.

---

## 📂 **Dataset**
The dataset contains patient health records with the following key features:
- **Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol**  
- **Fasting Blood Sugar, ECG Results, Max Heart Rate, Exercise Induced Angina**  
- **ST Depression, Slope of ST Segment, Number of Major Vessels, Thalassemia**  
- **Target Variable:** Presence or absence of heart disease (1 or 0)  

---

## 🛠 **Tools and Libraries Used**
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM  
- **Deep Learning:** TensorFlow, Keras  
- **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

---

## 🔍 **Exploratory Data Analysis (EDA)**
- Analyzed categorical and numerical columns separately.
- Cleaned and transformed data by handling missing values.
- Applied feature engineering to improve model performance.
- Identified the most important features affecting heart disease prediction.

---

## 🧹 **Data Preprocessing and Feature Engineering**
- Converted categorical variables using **label encoding** and **one-hot encoding**.
- Standardized numerical features using **StandardScaler**.
- Engineered new features to improve predictive accuracy.

---

## 🧑‍💻 **Model Training and Evaluation**
- Trained multiple machine learning models:
  - **GaussianNB, Decision Tree, Random Forest, K-Nearest Neighbors (KNN)**  
  - **Logistic Regression, AdaBoost, XGBoost, LightGBM**  
- Developed a **Deep Learning Model** using TensorFlow and Keras.
- Evaluated models based on performance metrics such as:
  - **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**.
- Compared models to determine the most effective approach.

---

## 🧪 **Model Comparison**

| **Model**                 | **Accuracy** | **Precision** | **Recall** | **F1-score** | **Support** |
|----------------------------|--------------|---------------|------------|-------------|-------------|
| **GaussianNB**             | **0.9074**   | **0.9105**    | **0.9074** | **0.9086**  | **54**     |
| **DecisionTreeClassifier** | 0.6667       | 0.6914        | 0.6667    | 0.6708      | 54         |
| **KNeighborsClassifier**   | 0.7778       | 0.7764        | 0.7778    | 0.7724      | 54         |
| **RandomForestClassifier** | 0.7963       | 0.7984        | 0.7963    | 0.7897      | 54         |
| **LogisticRegression**     | 0.8889       | 0.8889        | 0.8889    | 0.8889      | 54         |
| **AdaBoostClassifier**     | 0.8333       | 0.8324        | 0.8333    | 0.8325      | 54         |
| **XGBClassifier**          | 0.8148       | 0.8158        | 0.8148    | 0.8104      | 54         |
| **LGBMClassifier**         | 0.8333       | 0.8333        | 0.8333    | 0.8306      | 54         |
| **NeuralNetwork**          | 0.8519       | 0.8519        | 0.8519    | 0.8519      | 54         |

---

## 📊 **Key Insights**
- **GaussianNB** emerged as the best-performing model with an accuracy of **90.74%** and an F1-score of **90.86%**.  
- **Logistic Regression** also performed well, achieving **88.89%** accuracy.  
- The **Neural Network** demonstrated reliable results, with an accuracy of **85.19%**.  
- **Decision Tree Classifier** had the lowest accuracy, likely due to overfitting.  

---

## 🧑‍⚕️ **Conclusion**
- The **GaussianNB** model provides the highest accuracy for predicting heart disease in this dataset.  
- Neural networks could further improve with hyperparameter tuning.  
- Feature selection and dimensionality reduction could enhance overall performance.  
- These insights can aid medical professionals in early detection of heart disease.  

---

## 🚀 **Future Enhancements**
- Fine-tune hyperparameters for improved model performance.
- Experiment with different deep learning architectures.
- Perform feature selection to enhance model efficiency.

---

## 🤝 **Acknowledgments**
Thanks to the open-source contributors for providing the dataset and tools that made this project possible.

