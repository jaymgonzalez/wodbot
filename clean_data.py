import requests
import sqlite3
    

def get_parsed_wod(cursor):

    url = 'http://localhost:11434/api/generate'

    for id, wod in cursor.fetchall():
        
        data = {
            "model": "mistral",
            "prompt": f"You are an expert personal trainer looking at workout routines. Your task is to clean up the data removing anything not related with the routing. If you find the pattern \R:\d+'{1,2}\ add it as such. {wod}",
            "stream": False
        }

        response = requests.post(url, json=data).json()
        parsed_wod = response.get('response')

        print(f'{parsed_wod} \n with ID:{id}')

        if parsed_wod:
            update_query = "UPDATE tetuan_wod_data SET parsed_wod = ? WHERE id = ?"
            cursor.execute(update_query, (parsed_wod, id))
            conn.commit()
        else:
            print(f"No parsed_wod for id {id}")



if __name__ == '__main__':
    # Connect to SQLite DB
    conn = sqlite3.connect('wodbot.db')  # Replace with your database file
    cursor = conn.cursor()

    # Create a new column to store the parsed_wod if it doesn't exist
    # cursor.execute("ALTER TABLE tetuan_wod_data ADD COLUMN parsed_wod TEXT")

    query = "SELECT id, wod FROM tetuan_wod_data"
    # query = "SELECT id, wod FROM tetuan_wod_data WHERE parsed_wod = ''"
    cursor.execute(query)
    get_parsed_wod(cursor)

    conn.close()