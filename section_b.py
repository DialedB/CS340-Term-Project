import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Path of the txt file containing our dataset
file_path = "original_text_data.txt"

# Dataset that consists of data in our tab delimited txt file
data = pd.read_csv(file_path, sep="\t", encoding="utf-16le")

# Fixes any 'broken' floats (ex. 13.428.571.428.571.400)
def fix_broken_floats(df):
    float_cols = df.select_dtypes(include='object').columns
    pattern = re.compile(r"(\d+\.\d{1,4})")
    for col in float_cols:
        df[col] = df[col].apply(lambda x: round(float(pattern.search(str(x)).group(1)), 2)
                                if isinstance(x, str) and pattern.search(x) else x)
    return df

data = fix_broken_floats(data)

# Clean dataset (dropped any missing values)
clean_data = data.dropna()

# Keep only rows where Target is 'Graduate' or 'Dropout'
clean_data = clean_data[clean_data['Target'].isin(['Graduate', 'Dropout'])]

# Convert Target to binary (Dropout=0, Graduate=1)
clean_data['Target'] = clean_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Dataset X containing all fields except 'Target'
X = clean_data.drop('Target', axis=1)
# Dataset y containing only the field 'Target'
y = clean_data['Target']

# Train/test split with 80% training size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Global variables
hidden_layers = (100,)  # 1 hidden layer with 100 neurons (default)
learning_rate = 'adaptive'  # learning rate setting (default)
model = None
loss_curve = None

# Function used to display the menu choices
def show_menu():
    print("1. Read the labelled text data file, display the first 5 lines")
    print("2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)")
    print("3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)")
    print("4. Train on 80% of labeled data, display progress graph")
    print("5. Classify the unlabeled data, output training report and confusion matrix")
    print("6. Exit the program")

# Function that reads the text file and displays first 5 lines
def choice1(file):
    df = pd.read_csv(file, sep="\t", encoding="utf-16le")
    df = fix_broken_floats(df)
    print("\nFirst 5 rows of the dataset:\n")
    print(df.head())

def choice2():
    global hidden_layers
    user_input = input("Enter hidden layer sizes separated by '-' (e.g. 6-12-2): ").strip()
    try:
        layers = tuple(int(x) for x in user_input.split('-') if x.isdigit())
        if len(layers) == 0:
            print("No valid layers entered, keeping previous setting:", hidden_layers)
        else:
            hidden_layers = layers
            print("Hidden layers set to:", hidden_layers)
    except Exception as e:
        print("Invalid input format. Please enter numbers separated by '-'. Error:", e)

def choice3():
    global learning_rate
    user_input = input("Enter learning rate (0.001 - 0.5), or press ENTER for 'adaptive': ").strip()
    if user_input == '':
        learning_rate = 'adaptive'
    else:
        try:
            lr = float(user_input)
            if 0.001 <= lr <= 0.5:
                learning_rate = lr
            else:
                print("Learning rate out of range. Using default:", learning_rate)
        except ValueError:
            print("Invalid input. Using default:", learning_rate)
    print("Learning rate set to:", learning_rate)

def choice4():
    global model, loss_curve
    print("Training model...")
    model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                          learning_rate_init=learning_rate if isinstance(learning_rate, float) else 0.001,
                          learning_rate=learning_rate if isinstance(learning_rate, str) else 'constant',
                          max_iter=300, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    loss_curve = model.loss_curve_
    plt.plot(loss_curve)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def choice5():
    if model is None:
        print("Please train the model first (option 4).")
        return
    print("Evaluating on test data...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fail', 'Graduate']))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
