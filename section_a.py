import pandas as pd

# Function used to display the menu choices
def show_menu():
    print("Welcome to the Baseball Statistics Analyzer!")
    print("This program analyzes Red Sox player statistics.\n")
    print("1. Read & display stats for the 12 players with the most HRs")
    print("2. Display players with home runs above a certain threshold, in alphabetical order")
    print("3. Calculate HR/G & SO/G, save into a new file along with original stats, display")
    print("4. Sort by a field indicated by the user (Name, Games, Home Runs, Strikes)")
    print("5. Exit the program")


# Function that reads and displays stats for select players
def choice1(file):
    df = pd.read_csv(file, sep=";", encoding="utf-16le")
    df.columns = ['Name', 'Games', 'Home Runs', 'Strikes']
    df.sort_values(by='Home Runs', ascending=False, inplace=True)
    print(df.head(12))


# Function that displays in alphabetical order players that satisfy a certain threshold
def choice2(file, x):
    df = pd.read_csv(file, sep=";", encoding="utf-16le")
    df.columns = ['Name', 'Games', 'Home Runs', 'Strikes']
    filtered_df = df[df['Home Runs'] >= x]
    filtered_df_sorted = filtered_df.sort_values(by='Name', ascending=True)  # Add this line
    print(filtered_df_sorted)


# Function that calculates HR/G and SO/G, saves in file and displays the data
def choice3(file):
    df = pd.read_csv(file, sep=";", encoding="utf-16le")
    df.columns = ['Name', 'Games', 'Home Runs', 'Strikes']

    df['HR/G'] = (df['Home Runs'] / df['Games']).round(2)
    df['SO/G'] = (df['Strikes'] / df['Games']).round(2)

    output_file = 'partA_output_data.txt'
    df.to_csv(output_file, sep=";", encoding="utf-16le", index=False)

    print(df)


# Function that allows for custom sorting by user input
def choice4(file, field):
    try:
        df = pd.read_csv(file, sep=";", encoding="utf-16le")
        df.columns = ['Name', 'Games', 'Home Runs', 'Strikes']

        if field not in df.columns:
            raise ValueError(f"'{field}' is not a valid column. Choose from: {list(df.columns)}")

        df.sort_values(by=field, ascending=False, inplace=True)
        print(df)
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
    except ValueError as ve:
        print(f"Invalid input: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

def main():
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block looks for invalid input and allows for error trapping
        try:
            # Calls the show menu function from py
            show_menu()
            # Asks for user input
            x = int(input("Enter a choice: "))
            # Checks for user input section_and runs the appropriate menu choice
            if x == 1:
                # Opens the needed file in order to pass it in the function
                with open("boljevic_red_sox.txt", 'r') as f:
                    choice1(f)
                print()
            elif x == 2:
                # Asks for additional user input because it is needed for this menu choice
                y = int(input("Enter a threshold: "))
                # Opens the needed file in order to pass it in the function
                with open("boljevic_red_sox.txt", 'r') as f:
                    choice2(f, y)
                print()
            elif x == 3:
                # Opens the needed file in order to pass it in the function
                with open("boljevic_red_sox.txt", 'r') as f:
                    choice3(f)
                print()
            elif x == 4:
                # Asks for additional user input because it is needed for this menu choice
                z = input("Enter a sorting field: ")
                # Opens the needed file in order to pass it in the function
                with open("boljevic_red_sox.txt", 'r') as f:
                    choice4(f, z)
                print()
            elif x == 5:
                print("Exiting...")
                # Since choice 5 is 'Exit the program',
                # it sets the choice to hold the value 'False' in order to break the while loop
                choice = False
            # Prints the warning if input is outside the scope
            else:
                print("Invalid choice. Please select a number from 1 to 5.")
        # Prints the warning if the user input is not a number
        except ValueError:
            print("Invalid input. Please enter a number.")
        # Prints the error and exits program if user interrupts the execution of the
        # program using a keyboard action (ex. Ctrl+C)
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting...")
            break


if __name__ == "__main__":
    main()