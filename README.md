# CodeAlpha_IrisDataClassification
A ML model to classify Iris flowers based on measurements using Scikit-learn.


The project uses the physical measurements of Iris flowers—like petal and sepal length and width—to predict which species the flower belongs to: Setosa, Versicolor, or Virginica.

📂 **Step 1: Dataset Preparation**

I started by uploading the dataset in CSV format. This dataset contains labeled measurements of 150 flowers across three species.



import pandas as pd

df = pd.read_csv("Iris.csv")

print(df.head())

Here, I loaded the data using Pandas and printed the first few rows to see what the data looks like.


🧹 **Step 2: Data Preprocessing**
I cleaned the data by removing the 'Id' column, since it doesn’t help with prediction. Then, I used Label Encoding to convert the species names into numbers so the model can understand them.



from sklearn.preprocessing import LabelEncoder

df = df.drop("Id", axis=1)

label_encoder = LabelEncoder()

df['Species'] = label_encoder.fit_transform(df['Species'])



🔀 **Step 3: Splitting the Dataset**
Next, I split the data into training and testing sets using an 80-20 split. This helps evaluate how well the model performs on unseen data.



from sklearn.model_selection import train_test_split

X = df.drop("Species", axis=1)

y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


🤖 **Step 4: Model Training**
I used a Random Forest Classifier from Scikit-learn to train the model. It’s a simple yet powerful ensemble method that works well for classification tasks.



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)


📊 **Step 5: Model Evaluation**
Once the model was trained, I evaluated it using accuracy score, a classification report, and a confusion matrix.



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

The accuracy was quite high, and the classification report showed good precision and recall across all three classes.


📈 **Step 6: Visualization**
To make it more visual, I plotted the confusion matrix using Seaborn.



import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

This helps us see how many predictions were correct or misclassified.


**Summary:**


This project focuses on building a machine learning model to classify Iris flower species—Setosa, Versicolor, and Virginica—based on their petal and sepal measurements. Using Python and libraries like Pandas, Scikit-learn, and Seaborn, I:

Loaded and cleaned the Iris dataset.

Preprocessed the data by encoding the species labels.

Split the dataset into training and testing sets.

Trained a Random Forest Classifier to learn the patterns.

Evaluated the model's accuracy using metrics and a confusion matrix.

The model achieved high accuracy and demonstrates a basic example of supervised classification in machine learning. This project helped me strengthen my understanding of data preprocessing, model training, and evaluation.


