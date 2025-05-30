import pandas as pd

# Function used to display the menu choices
def show_menu():
    print("1. Read & display stats for the 12 players with the most HRs")
    print("2. Display players with home runs above a certain threshold, in alphabetical order")
    print("3. Calculate HR/G & SO/G, save into a new file along with original stats, display")
    print("4. Sort by a field indicated by the user")
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
    print(filtered_df)


# Function that calculates HR/G and SO/G, saves in file and displays the data
def choice3(file):
    df = pd.read_csv(file, sep=";", encoding="utf-16le")
    df.columns = ['Name', 'Games', 'Home Runs', 'Strikes']

    df['HR/G'] = df['Home Runs'] / df['Games']
    df['SO/G'] = df['Strikes'] / df['Games']

    output_file = 'mlbdata_updated'
    df.to_csv(output_file, index=False)

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