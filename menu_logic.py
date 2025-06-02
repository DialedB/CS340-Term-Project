import section_a
import section_b

# Section A of Term Project
def a():
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block looks for invalid input and allows for error trapping
        try:
            # Calls the show menu function from section_a.py
            section_a.show_menu()
            # Asks for user input
            x = int(input("Enter a choice: "))
            # Checks for user input section_and runs the appropriate menu choice
            if x == 1:
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    section_a.choice1(f)
            elif x == 2:
                # Asks for additional user input because it is needed for this menu choice
                y = int(input("Enter a threshold: "))
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    section_a.choice2(f, y)
            elif x == 3:
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    section_a.choice3(f)
                # Opens the needed file in order to pass it in the function
                with open("mlbdata_updated.csv", 'r') as f1:
                    section_a.choice1(f1)
            elif x == 4:
                # Asks for additional user input because it is needed for this menu choice
                z = input("Enter a sorting field: ")
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    section_a.choice4(f, z)
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

# Section B of Term Project
def b():
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block that looks for invalid input allowing for error catching
        try:
            # Calls the function show_menu() from section_b.py
            section_b.show_menu()
            # Asks user for input
            x = int(input("Enter a choice: "))
        # Prints the warning if user input is not a number
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue  # Goes back to the menu
        # Another try block for error catching
        try:
            # Checks for user input and runs the appropriate choice
            if x == 1:
                # Opens the needed file in order to pass it in the function
                with open("original_text_data.txt", "r", encoding="utf-16le") as f:
                    # Runs the choice1 function from section_b.py
                    section_b.choice1(f)
            elif x == 2:
                # Runs the choice2 function from section_b.py
                section_b.choice2()
            elif x == 3:
                # Runs the choice3 function from section_b.py
                section_b.choice3()
            elif x == 4:
                # Runs the choice4 function from section_b.py
                section_b.choice4()
            elif x == 5:
                # Runs the choice5 function from section_b.py
                section_b.choice5()
            elif x == 6:
                # Runs the choice6 function from section_b.py
                section_b.choice6()
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