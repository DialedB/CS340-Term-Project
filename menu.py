import csv

def show_menu_a():
    print("1. Read & display stats for the 12 players with the most HRs")
    print("2. Display players with home runs above a certain threshold, in alphabetical order")
    print("3. Calculate HR/G & SO/G, save into a new file along with original stats, display")
    print("4. Sort by a field indicated by the user")
    print("5. Exit the program")


# TODO: Fix output format
def a_choice1(file):
    for i in range(1,13):
        print(file.readline())


def a_choice2(file, x):
    players = []
    for line in file:
        parts = line.strip().split(';')  # Adjust delimiter if needed
        name = parts[0].strip()
        try:
            home_runs = int(parts[2].strip())
            if home_runs > x:
                players.append(name)
        except (IndexError, ValueError):
            continue  # Skip lines with invalid data
    print("Players: ", players)
    for name in sorted(players):
        print(name)


# TODO: Fix output format
def a_choice3(file):
    players = []

    for line in file:
        parts = line.strip().split(';')
        try:
            name = parts[0].strip()
            games = int(parts[1].strip())
            home_runs = int(parts[2].strip())
            strikes = int(parts[3].strip())

            if games == 0:
                continue  # Avoid division by zero

            player_data = [
                name,
                games,
                home_runs,
                strikes,
                home_runs / games,
                strikes / games
            ]

            players.append(player_data)
        except (IndexError, ValueError):
            continue

    # Save to CSV without header
    with open("mlbdata_updated.csv", mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(players)


# TODO: Custom sorting by user input
def a_choice4(file, field):
    players = []


def show_menu_b():
    print("1. Read the labelled text data file, display the first 5 lines")
    print("2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)")
    print("3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)")
    print("4. Train on 80% of labeled data, display progress graph")
    print("5. Classify the unlabeled data, output training report and confusion matrix")
    print("6. Exit the program")