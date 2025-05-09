import menu

# Section A of Term Project
def section_a():
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block looks for invalid input and allows for error trapping
        try:
            # Calls the showMenu function from menu.py
            menu.show_menu_a()
            # Asks for user input
            x = int(input("Enter a choice: "))
            # Checks for user input and runs the appropriate menu choice
            if x == 1:
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    menu.a_choice1(f)
            elif x == 2:
                # Asks for additional user input because it is needed for this menu choice
                y = int(input("Enter a threshold: "))
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    menu.a_choice2(f,y)
            elif x == 3:
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    menu.a_choice3(f)
                # Opens the needed file in order to pass it in the function
                with open("mlbdata_updated.csv", 'r') as f1:
                    menu.a_choice1(f1)
            elif x == 4:
                # Asks for additional user input because it is needed for this menu choice
                z = input("Enter a sorting field: ")
                # Opens the needed file in order to pass it in the function
                with open("mlbdata.csv", 'r') as f:
                    menu.a_choice4(f, z)
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
def section_b():
    # Allows for use of while loop
    choice = True
    # Runs while choice holds the value 'True'
    while choice:
        # Try block looks for invalid input and allows for error trapping
        try:
            menu.show_menu_b()

            x = int(input("Enter a choice: "))

            # TODO: Add the 6 choice in if loops
        # Prints the warning if the user input is not a number
        except ValueError:
            print("Invalid input. Please enter a number.")
        # Prints the error and exits program if user interrupts the execution of the
        # program using a keyboard action (ex. Ctrl+C)
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting...")
            break