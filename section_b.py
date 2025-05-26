import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path of the txt file containing our dataset
file_path = "original_text_data.txt"

# Dataset that consists of data in out tab delimited txt file
data = pd.read_csv(file_path, sep="\t", encoding="utf-16le")

# Clean dataset (dropped any missing values)
clean_data = data.dropna()

# LabelEncoder object
le = LabelEncoder()

# Transforms the categorical field 'Target' from string to 0 and 1
clean_data.loc[:, 'Target'] = le.fit_transform(clean_data['Target'])

# Dataset X containing all fields except 'Target'
X = clean_data.drop('Target', axis=1)
# Dataset y containing only the field 'Target'
y = clean_data['Target']

# Train/test split with 80% training size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Global variable which will store user input
hidden_layers = (100,)  # 1 hidden layer with 100 neurons (default)
learning_rate = 'adaptive'  # learning rate setting (default)

# Global variable model which will later hold our trained AI model
model = None
# Global variable for our model's loss curve
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
    for i in range(1,5):
        print(df[i])

def choice2():
    global hidden_layers
    user_input = input("Enter hidden layer sizes separated by '-' (e.g. 6-12-2): ").strip()
    try:
        # Convert string to tuple of ints
        layers = tuple(int(x) for x in user_input.split('-') if x.isdigit())
        if len(layers) == 0:
            print("No valid layers entered, keeping previous setting:", hidden_layers)
        else:
            hidden_layers = layers
            print("Hidden layers set to:", hidden_layers)
    except Exception as e:
        print("Invalid input format. Please enter numbers separated by '-'. Error:", e)