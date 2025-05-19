import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

file_path = "original_text_data.txt"
data = pd.read_csv(file_path, sep="\t", encoding="utf-16le")
clean_data = data.dropna()

# TODO: Make a model that will predict if a student will graduate or drop out

# LabelEncoder object
le = LabelEncoder()
# Transforms the categorical variable 'Target' into numerical values (0, 2)
clean_data['Target'] = le.fit_transform(clean_data['Target'])

X = clean_data.drop('Target', axis=1)
y = clean_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
# MLP works better with scaled data:
# StandardScaler object
scaler = StandardScaler()
# Scaling of X_train and X_test variables
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Defining the model
model = MLPClassifier(hidden_layer_sizes=(33, 100, 1), max_iter=300, random_state=42)
# Fitting data to model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print()

def show_menu():
    print("1. Read the labelled text data file, display the first 5 lines")
    print("2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)")
    print("3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)")
    print("4. Train on 80% of labeled data, display progress graph")
    print("5. Classify the unlabeled data, output training report and confusion matrix")
    print("6. Exit the program")


def choice1(file):
    for i in range(1, 6):
        print(file.readline())