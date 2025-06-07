🩺 Diabetes Prediction Project
Predicting diabetes early can help people take timely action, get the right treatment, and avoid complications. In this project, I built a machine learning model that estimates whether someone is likely to have diabetes based on their health data.
The goal? To make a simple tool that can assist in early-stage diagnosis and awareness.

📊 What the Model Does
The model takes in a few health-related inputs like:
♦️BMI (Body Mass Index)
♦️Blood Pressure
♦️Cholesterol♦
♦️Age
…and other important medical features
Based on this, it predicts whether a person is diabetic or not.

🛠️ Tools & Libraries Used
To make this work, I used:
♦️Python (the brain behind it all)
♦️Pandas and NumPy for handling and cleaning the data
♦️Matplotlib for visualizing patterns and insights
♦️Scikit-learn for building and evaluating the machine learning models

🤖 Machine Learning Approach
I tried out three different models:
♦️Logistic Regression
♦️Decision Tree
♦️Random Forest

You might wonder, "Why use three?"
Well, I wanted to compare how each model performed. By testing them side-by-side, I could choose the one that gives the best results.
To further improve the predictions, I also did Hyperparameter Tuning — basically fine-tuning the models for better accuracy.

✅ Results
After training and testing, Logistic Regression turned out to be the best performer with an accuracy of 82%. So I picked that as the final model for predictions.

📁 Dataset
I used the Pima Indians Diabetes Dataset from Kaggle. It’s widely used for diabetes prediction and contains data from over 700 female patients of Pima Indian heritage.
