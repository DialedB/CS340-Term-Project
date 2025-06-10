"""
    Section B of the CS340 Term Project
    Created by Dusan Boljevic on 02/06/2025
"""
import re # Allows use of Regex
import pandas as pd # Allows for easier data manipulation
import numpy as np # Allows numerical operations with arrays
import os # Allows interacting with the operating system
import zipfile # Allows for reading and creating zip files
import time # Allows for measuring training time
import matplotlib.pyplot as plt # Allows visual data plotting
from sklearn.metrics import confusion_matrix, classification_report # Allows for calculating and displaying the confusion matrix and classification report
from sklearn.model_selection import train_test_split # Allows for the dataset to be split into training and testing sets
from sklearn.neural_network import MLPClassifier # Allows for use of Multi-Layer Perceptor model
from sklearn.preprocessing import StandardScaler # Allows for scaling the data, resulting in better results for the model
import warnings # Allows for managing non-fatal warning messages

# Ignore non_fatal warning messages
warnings.filterwarnings('ignore')

# Path of the txt file containing our dataset
file_path = "original_text_data.txt"

# Global variables for storing results of the experiments
experiment_results = [] # Stores all experiment results for a collective graph
best_model = None # Stores the best performing model produced by the experimentation
best_config = None # Stores the best configuration produced by the experimentation
fastest_model = None # Stores the fastest performing model produced by the experimentation
fastest_config = None # Stores the fastest configuration produced by the experimentation
current_model = None # Stores the model that is currently being used by the program
current_X_test = None # Stores currently used test features
current_y_test = None # Stores currently used test labels
current_scaler = None # Stores currently used scaler for predictions

# Experimental configurations
topologies = [(100,), (100, 50), (100, 50, 25)] # 3 different ANN topologies
learning_rates = [0.001, 0.1, 'adaptive'] # Small fixed learning rate, large fixed learning rate and an adaptive learning rate
data_splits = [(0.5, False), (0.8, True)] # 50/50 split without randomization and 80/20 split with randomization

# Dataset that consists of data in our tab delimited txt file
try:
    data = pd.read_csv(file_path, sep="\t", encoding="utf-16le")
except FileNotFoundError: # If there is no "original_text_data.txt" file
    print("File not found")
    data = None


# Function that fixes any 'broken' floats (ex. 13.428.571.428.571.400) by extracting the first valid float pattern from string
def fix_broken_floats(df):
    float_cols = df.select_dtypes(include='object').columns # Get columns with the object data type
    pattern = re.compile(r"(\d+\.\d{1,4})") # Regex pattern to match float numbers
    for col in float_cols: # Iterates through each column in float_cols
        # Applies the lambda function to fix the 'broken' floats in each cell
        df[col] = df[col].apply(lambda x: round(float(pattern.search(str(x)).group(1)), 2)
                                if isinstance(x, str) and pattern.search(x) else x)
    return df # Returns the data frame with fixed float values


# Function that prepares and cleans the dataset for training
def prepare_data():
    # In case the .txt wasn't loaded
    if data is None:
        return None, None
    # Applies the fix_broken_floats function to clean up the data
    cleaned_data = fix_broken_floats(data.copy())
    # Cleans dataset by dropping any missing values (just in case)
    clean_data = cleaned_data.dropna()
    # Keeps only the rows where Target is equal to 'Graduate' or 'Dropout'
    clean_data = clean_data[clean_data['Target'].isin(['Graduate', 'Dropout'])]
    # Converts the Target column to binary
    clean_data['Target'] = clean_data['Target'].map({'Graduate': 1, 'Dropout': 0})
    # Dataset X containing all fields except 'Target'
    X = clean_data.drop('Target', axis=1)
    # Dataset y containing only the field 'Target'
    y = clean_data['Target']
    # Returns the X and y datasets
    return X, y


# Function used to display the menu choices as specified
def show_menu():
    print("\n" + "=" * 50)
    print("MULTI-LAYER PERCEPTRON STUDENT DROPOUT PREDICTION SYSTEM")
    print("=" * 50)
    print("1. Read the labelled text data file, display the first 5 lines")
    print("2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)")
    print("3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)")
    print("4. Train on 80% of labeled data, display progress graph")
    print("5. Classify the unlabeled data, output training report and confusion matrix")
    print("6. Predict if student will drop out or graduate")
    print("7. Exit the program")
    print("=" * 50)


# Function that reads the text file and displays first 5 lines
def choice1 ():
    try:
        # Reads the dataset with utf-16le encoding to avoid unreadable values
        df  = pd.read_csv(file_path, sep="\t", encoding="utf-16le")
        # Applies float fixing function to clean the data
        df = fix_broken_floats(df)
        # Prints the first 5 rows of the dataset
        print("\nFirst 5 rows of the dataset:")
        print(df.head(5))
    except Exception as e:
        print(f"Error reading file: {e}")


# Function that asks the user for size of hidden layers of the model's MLP topology and updates the global topologies list with the user input
def choice2():
    # Calls the topologies global variable
    global topologies
    # Outputs the topologies currently in the list
    print("\nCurrent available topologies:")
    for i, topology in enumerate(topologies):
        print(f"{i + 1}. {topology}")
    # Gets user input for custom topology
    user_input = input("\nEnter hidden layer sizes separated by '-' (e.g. 100-50-25): ").strip()
    # Try block to catch invalid inputs
    try:
        # Parses the input string into tuple of integers
        layers = tuple(int(x.strip()) for x in user_input.split('-') if x.strip().isdigit())
        # If no valid layers are entered by the user
        if len(layers) == 0:
            print("No valid layers entered, keeping current topologies")
        else:
            # Adds the new topology to the list if it's not already there
            if layers not in topologies:
                topologies.append(layers)
                print(f"Added new topology: {layers}")
            else:
                print(f"{layers} topology already exists")
            # Outputs the topologies in the list after user input
            print("Current topologies: ", topologies)
    # Warns the user if their input format doesn't fit the x-x-x-x format specified
    except Exception as e:
        print(f"Invalid input format: {e}")


# Function that asks the user for the learning rate and updates the learning_rates global with user's input
def choice3():
    # Calls the learning_rates global variable
    global learning_rates
    # Outputs the learning rates currently in the list
    print(f"\nCurrent learning rates: {learning_rates}")
    # Gets user input for the learning rate
    user_input = input("Enter learning rate (0.001 - 0.5), or press ENTER to keep current: ").strip()
    # If user presses the enter key without input
    if user_input == "":
        print("Keeping current learning rate options")
    # If the user inputs a valid value
    else:
        # Try block to catch invalid inputs
        try:
            # Converts the user input to float
            lr = float(user_input)
            # Validates the range
            if 0.001 <= lr <= 0.5:
                # Adds the new learning rate to the list if it's not already there
                if lr not in learning_rates:
                    learning_rates.append(lr)
                    print(f"Added new learning rate: {lr}")
                else:
                    print(f"{lr} learning rate already exists")
            # Warns the user if they input a value outside the range
            else:
                    print("Input out of range (0.001 - 0.5)")
        # Warns the user if their input is invalid (string in this case)
        except ValueError:
            print("Invalid input. Please enter a number.")
    # Outputs the learning rates in the list after the user input
    print("\nCurrent learning rates: ", learning_rates)


# Function that runs a single experiment with given parameters and returns model, loss curve, training time and test performance
def run_one_experiment(topology, learning_rate, train_size, randomize, X, y):
    # Outputs the specifications of the MLP model for this experiment
    print("\nStarting experiment with the following: ")
    print(f"Topology: {topology}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train size: {train_size}")
    print(f"Randomization: {randomize}")
    # Creates the train-test split based on the parameters (randomize = False/True)
    if randomize:
        # Uses random state for consistent randomization
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    else:
        # Uses stratify to maintain class distribution without randomizing
        split_idx = int(len(X) * train_size)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # Scales the features for better ANN performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Configures the learning rate parameters for the MLPClassifier
    if isinstance(learning_rate, str): # If the learning rate is a string then it's 'adaptive'
        lr_init = 0.001
        lr_schedule = 'adaptive'
    else: # If the learning rade is fixed
        lr_init = learning_rate
        lr_schedule = 'constant'
    # Creates and configures the MLP model
    model = MLPClassifier(
        hidden_layer_sizes=topology,
        learning_rate_init=lr_init,
        learning_rate=lr_schedule,
        max_iter=500,
        random_state=42,
        verbose=False,
        early_stopping=True, # Stops early if the validation score doesn't improve
        validation_fraction=0.1 # Uses 10% of training data for validation
    )
    # Trains the model and measures training time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    # Makes predictions on the test set
    y_pred = model.predict(X_test_scaled)
    # Calculates the test accuracy of the model
    test_accuracy = model.score(X_test_scaled, y_test)
    # Returns all relevant information about the experiment
    return {
        'model': model,
        'scaler': scaler,
        'loss_curve': model.loss_curve_,
        'training_time': training_time,
        'test_accuracy': test_accuracy,
        'topology': topology,
        'learning_rate': learning_rate,
        'train_size': train_size,
        'randomize': randomize,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train': X_train,
        'y_train': y_train
    }


# Trains models with all parameter combinations and displays the progress graphs
def choice4():
    # Calls the global variables needed
    global experiment_results, best_model, best_config, fastest_model, fastest_config
    global current_model, current_X_test, current_y_test, current_scaler
    # Calls the prepare data function to get X and y from the main dataset
    X, y = prepare_data()
    # If the data is never loaded this lets the user know, in case they skipped right to option 4
    if X is None:
        print("No data provided. Exiting...")
        return
    # Lets the user know experiments are starting and how many will be conducted
    print("Starting training experiments")
    print(f"Total experiments to run: {len(topologies) * len(learning_rates) * len(data_splits)}")
    # Clears previous experiment results
    experiment_results = []
    best_accuracy = 0
    fastest_time = float('inf')
    # Creates a directory for storing produced graphs
    os.makedirs('graphs', exist_ok = True)
    # Counter
    experiment_count = 0
    # Run all combinations of experiments
    for topology in topologies: # For every topology in topologies
        for learning_rate in learning_rates: # For every learning_rate in learning_rates
            for train_size, randomize in data_splits: # For every data split and randomization combination
                # Increase counter by 1 for every experiment conducted
                experiment_count += 1
                # Output the number of the experiment currently running
                print(f"Experiment: {experiment_count}")
                # Runs the experiment
                result = run_one_experiment(topology, learning_rate, train_size, randomize, X, y)
                # Adds the results of said experiment to experiment_results
                experiment_results.append(result)
                # Tracks best performing model (highest accuracy)
                if result['test_accuracy'] > best_accuracy:
                    best_accuracy = result['test_accuracy']
                    best_model = result['model']
                    best_config = result
                # Tracks fastest model (shortest training time)
                if result['training_time'] < fastest_time:
                    fastest_time = result['training_time']
                    fastest_model = result['model']
                    fastest_config = result
                # Creates and saves individual training curve graphs
                plt.figure(figsize = (10, 6))
                plt.plot(result['loss_curve'], linewidth = 2)
                plt.title(f'Training loss curve\nTopology: {topology}, LR: {learning_rate}, Train Size: {train_size}, Randomize: {randomize}'
                          f'\nAccuracy: {result["test_accuracy"]:.3f}, Time: {result["training_time"]:.2f}s')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.grid(True, alpha = 0.3)
                # Creates the file name for saving the graph
                lr_str = str(learning_rate).replace('.', '_')
                topology_str = '_'.join(map(str, topology))
                filename = f'exp_{experiment_count}_topo_{topology_str}_lr_{lr_str}_split_{train_size}_rand_{randomize}.png'
                # Saves the files in the graphs directory
                plt.savefig(f"graphs/{filename}", dpi  =300, bbox_inches='tight')
                # Displays the created graph
                plt.show()
                # Outputs the accuracy and training time of the model
                print(f"Accuracy: {result['test_accuracy']:.3f}, Training time: {result['training_time']:.2f}s")
    # Set the best model as current for meny options 5 and 6
    current_model = best_model
    current_X_test = best_config['X_test']
    current_y_test = best_config['y_test']
    current_scaler = best_config['scaler']
    # Creates the collective graph showing all loss curves
    create_collective_graph()
    # Creates weight change visualization for best and fastest models
    create_weight_change_graphs()
    # Generates required output files
    generate_output_files()
    # Creates zip file with all graphs()
    create_graph_zip()
    # Outputs the training summary for the experiment
    print(f"\n{'=' * 50}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 50}")
    print(f"Best model accuracy: {best_accuracy:.3f}")
    print(f"Best model config: Topology={best_config['topology']}, LR={best_config['learning_rate']}")
    print(f"Fastest model time: {fastest_time:.2f}s")
    print(f"Fastest model config: Topology={fastest_config['topology']}, LR={fastest_config['learning_rate']}")
    print(f"All graphs saved in 'graphs' directory")


# Function that creates a collective graph showing all 18 error curves with legend
def create_collective_graph():
    # Sets the figure size for the plot
    plt.figure(figsize = (10, 6))
    # Creates a list of colors that will be used for displaying each line on the collective graph
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiment_results)))
    # For each experiment result a plot is created along with its label
    for i, result in enumerate(experiment_results):
        lr_label = f'LR: {result["learning_rate"]}'
        split_label = f'Split: {result["train_size"]}'
        randomize_label = f'Randomize: {result["randomize"]}'
        topology_label = f'Topology: {result["topology"]}'
        # Label for the result consists of the labels created for each feature
        label = f'{topology_label} - {lr_label} - {split_label} - {randomize_label}'
        # Plots the loss curve for each result
        plt.plot(
            result['loss_curve'],
            color=colors[i],
            linewidth=1.5,
            alpha=0.8,
            label=label
        )
    # Creates a collective graph that consists of the above created singular plots
    plt.title('Collective Training Loss Curves - All 18 Experiments', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Iterations', fontsize = 10)
    plt.ylabel('Loss', fontsize = 10)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 8)
    plt.tight_layout()
    # Saves the plot as a file in the graphs directory
    plt.savefig('graphs/collective_error_curves.png', dpi = 300, bbox_inches='tight')
    plt.show()


# Function that creates weight change visualization graphs for best and fastest models
def create_weight_change_graphs():
    # Sets the figure size for the plot
    plt.figure(figsize = (12, 5))
    # Creates the first subplot for the best performing model
    plt.subplot(1, 2, 1)
    # Gets the loss curve from the best_config
    loss_curve = best_config['loss_curve']
    # Since MLPClassifier doesn't have a function for calculating weight changes
    # we approximate them using the formula: |loss(t+1) - loss(t)| where t is the current iteration number
    weight_changes = np.abs(np.diff(loss_curve))
    # Creates the plot for the best performing model, first subplot
    plt.plot(weight_changes, color = 'blue', linewidth = 2)
    plt.title('Weight Changes - Best Model')
    plt.xlabel('Iterations')
    plt.ylabel('Weight Change')
    plt.grid(True, alpha=0.3)
    # Creates the second subplot for the fastest performing model
    plt.subplot(1, 2, 2)
    # Gets the loss curve from the fastest_config
    loss_curve_fast = fastest_config['loss_curve']
    # Again we use the formula: |loss(t+1) - loss(t)| where t is the current iteration number
    weight_changes_fast = np.abs(np.diff(loss_curve_fast))
    # Creates the plot for the fastest model, second subplot
    plt.plot(weight_changes_fast, color = 'red', linewidth = 2)
    plt.title('Weight Changes - Fastest Model')
    plt.xlabel('Iterations')
    plt.ylabel('Weight Change')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Saves the plot as a file in the graphs directory
    plt.savefig('graphs/weight_change_comparison.png', dpi = 300, bbox_inches='tight')
    plt.show()


# Function that generates the required output files
def generate_output_files():
    # If no experiments have been performed yet there is no output data to create the files
    if not experiment_results:
        print("No results found. Exiting...")
        return
    # The best_result variable is equal to the configuration of the most accurate model
    best_result = best_config
    # The training data consists of the X_train and y_train data
    training_data = pd.concat([best_result['X_train'], best_result['y_train']], axis = 1)
    # A .txt file is then created to store the training data
    training_data.to_csv('training_data.txt', sep = "/t", index = False)
    # A .txt file is created to store the unlabeled test data
    best_result['X_test'].to_csv('testing_data_unlabeled.txt', sep = "/t", index = False)
    # The labeled testing data consists of the X_test and y_test data
    testing_labeled = pd.concat([best_result['X_test'], best_result['y_test']], axis = 1)
    # A.txt file is then created to store the labeled test data
    testing_labeled.to_csv('testing_data_labeled.txt', sep = "/t", index = False)
    # The output_data dataframe consists of the predictions and actual values
    output_data = pd.DataFrame({
        'Predicted_Label': best_result['y_pred'],
        'Actual_Label': best_result['y_test'].values,
        'Predicted_Class': ['Graduate' if p == 1 else 'Dropout' for p in best_result['y_pred']],
        'Actual_Class': ['Graduate' if a == 1 else 'Dropout' for a in best_result['y_test'].values]
    })
    # The output_data dataframe is then exported into a .txt file
    output_data.to_csv('output_data.txt', sep = "/t", index = False)
    # Notifies the user of which files were created
    print("Generated output files: training_data.txt, testing_data_unlabeled.txt, testing_data_labeled.txt, output_data.txt")


# Function that creates a zip file containing all the generated graphs
def create_graph_zip():
    # Calls the ZipFile function that opens a zip file in the write mode
    with zipfile.ZipFile('sectionB_graphs.zip', 'w') as zipf:
        # Looks for the graphs directory
        for root, dirs, files in os.walk('graphs'):
            # For each file in the graphs directory
            for file in files:
                # If the file is a .png file, meaning it's a plot
                if file.endswith('.png'):
                    # Adds the file to the path so it can be added to the zip file
                    filepath = os.path.join(root, file)
                    # Writes the filepath into the zip file
                    zipf.write(filepath)
    # Notifies the user that the zip file was created
    print("Generated zip file: sectionB_graphs.zip")

# Function that classifies the unlabeled data, outputs training report and confusion matrix
def choice5():
    # Warns the user to train the model first, before trying to get a training report and confusion matrix
    if current_model is None:
        print("Please train the model first (option 4)")
        return
    # Lets the user know the process is starting
    print("Evaluating model performance...")
    # Scales the test data
    X_test_scaled = current_scaler.transform(current_X_test)
    # Uses the model to get a prediction using the scales test data
    y_pred = current_model.predict(X_test_scaled)
    # Outputs the classification report
    print("Classification report:")
    print(classification_report(current_y_test, y_pred, target_names = ['Graduate', 'Dropout']))
    # Outputs the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(current_y_test, y_pred))
    # The score function lets us get the model's accuracy
    accuracy = current_model.score(X_test_scaled, current_y_test)
    # Outputs the overall accuracy of the model
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy * 100:.2f}%)")


# Function that lets the model predict if a student holding the user inputted index is going to Drop Out or Graduate
def choice6():
    # Warns the user to train the model first, before trying to test it
    if current_model is None:
        print("Please train the model first (option 4)")
    # Try block to catch invalid inputs
    try:
        # Sets the max index to the length of the testing set
        max_index = len(current_X_test) - 1
        # User input asking for an index between 0 and the max index
        index = int(input(f"Enter index of a student (0 to {max_index}): "))
        # Validates input
        if 0 <= index <= max_index:
            # Gets the student features at the user input index
            student_features = current_X_test.iloc[[index]]
            # Scales the data for better ANN performance
            student_features_scaled = current_scaler.transform(student_features)
            # Gets the actual target value at the user input index
            actual = current_y_test.iloc[index]
            # Predicts the target value
            prediction = current_model.predict(student_features_scaled)[0]
            # Calculates the prediction probability (confidence)
            prediction_probability = current_model.prediction_proba(student_features_scaled)[0]
            # Sets the labels back to strings as they were converted to binary values
            predicted_label = "Graduate" if prediction == 1 else "Dropout"
            actual_label = "Graduate" if actual == 1 else "Dropout"
            # Outputs the result of the prediction
            print(f"PREDICTION RESULTS FOR STUDENT AT INDEX {index}:")
            print(f"Prediction: {predicted_label}")
            print(f"Actual outcome: {actual_label}")
            print(f"Prediction confidence: {max(prediction_probability):.3f}")
            print(f"Probabilities - Dropout: {prediction_probability[0]:.3f}, Graduate: {prediction_probability[1]:.3f}")
            # Checks if prediction is correct and notifies the user)
            if prediction == actual:
                print("Correct!")
            else:
                print("Incorrect!")
        # Warns the user if they input an index outside the scope
        else:
            print("Please enter a valid index.")
    # Warns the user to enter a valid value
    except ValueError:
        print("Invalid input. Please enter a number.")
    # Warns the user of any other error
    except Exception as e:
        print(f"An error occurred: {e}")


# Main function that runs the above created functions as needed
def main():
    # Again, if no data is available, the project can't run
    if data is None:
        print("No data found. Exiting...")
        return
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block that looks for invalid input allowing for error catching
        try:
            # Calls the function show_menu() from py
            show_menu()
            # Asks user for input
            x = int(input("Enter a choice (1-7): "))
        # Prints the warning if user input is not a number
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 7.")
            continue  # Goes back to the menu
        # Another try block for error catching
        try:
            # Checks for user input and runs the appropriate choice
            if x == 1:
                # Runs the choice1 function
                choice1()
            elif x == 2:
                # Runs the choice2 function
                choice2()
            elif x == 3:
                # Runs the choice3 function
                choice3()
            elif x == 4:
                # Runs the choice4 function
                choice4()
            elif x == 5:
                # Runs the choice5 function
                choice5()
            elif x == 6:
                # Runs the choice6 function
                choice6()
            elif x == 7:
                print("Exiting...")
                # Since choice 7 is 'Exit the program',
                # it sets the choice to hold the value 'False' in order to break the while loop
                choice = False
                break
            else:
                # Warns user to input a number in the given range
                print("Invalid choice. Please select a number from 1 to 7.")
        # Print the warning if an exception happens
        except Exception as e:
            print("An error occurred during execution:", e)
# Runs the main function directly
if __name__ == "__main__":
    main()