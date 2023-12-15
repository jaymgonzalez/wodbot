import sqlite3, json, random
from collections import defaultdict

def create_jsonl_file():
    # Connect to your SQLite database
    conn = sqlite3.connect('wodbot.db')  # Replace with your database file
    cursor = conn.cursor()

    # Query to select all records
    query = "SELECT title, wod, date FROM tetuan_wod_data"

    # Execute the query
    cursor.execute(query)

    # Group data by date
    data_by_date = defaultdict(list)
    for title, wod, date in cursor.fetchall():
        data_by_date[date].append({'title': title, 'wod': wod})

    # Open a JSONL file to write the grouped records
    with open('output.jsonl', 'w') as file:
        for date, workouts in data_by_date.items():
            # Create the JSON object with 'workout' key
            json_object = json.dumps({'workout': workouts})
            # Write the JSON object to the file
            file.write(json_object + '\n')

    # Close the database connection
    conn.close()

def split_data(jsonl_file, train_ratio=0.8):
    with open(jsonl_file, 'r') as file:
        data = file.readlines()

    # Shuffle the data to ensure randomness
    random.shuffle(data)

    # Split the data
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    validation_data = data[split_index:]

    # Write the training data to a file
    with open('train.jsonl', 'w') as file:
        file.writelines(train_data)

    # Write the validation data to a file
    with open('validation.jsonl', 'w') as file:
        file.writelines(validation_data)


if __name__ == '__main__':

    split_data('output.jsonl')