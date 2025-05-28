import menu_logic

def main():
    print("Welcome to the CS340 Term Project\n")

    while True:
        print("\nMain Menu:")
        print("1. Section A menu")
        print("2. Section B menu")
        print("0. Exit the program")

        user_input = input("Select an option (0-2): ").strip()

        if user_input == '1':
            menu_logic.a()  # Runs Section A menu
        elif user_input == '2':
            menu_logic.b()  # Runs Section B menu
        elif user_input == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid input. Please enter 0, 1, or 2.")

if __name__ == "__main__":
    main()
