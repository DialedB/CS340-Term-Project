import menu_logic

def main():
    print("Welcome to the CS340 Term Project\n")
    x = int(input("Input 1 for the section A menu or 2 for the section B menu: "))
    if x == 1:
        menu_logic.a()
    elif x == 2:
        menu_logic.b()
    else:
        print("Invalid input")


if __name__ == "__main__":
    main()