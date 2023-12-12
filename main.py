from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from bs4 import BeautifulSoup as bs
import sqlite3

def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wod_data (
            id INTEGER PRIMARY KEY,
            wod TEXT,
            date TEXT,
            type TEXT
        )
    ''')
    conn.commit()
    return conn

def store_data(conn, wod, date, type):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO wod_data (wod, date, type) VALUES (?, ?, ?)', (wod, date, type))
    conn.commit()

def scrape(conn, url):
    driver = uc.Chrome(headless=True)

    try:
        while True:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#main .wod')))

            soup = bs(driver.page_source, 'html.parser')

            wod = soup.select_one('#main .wod').get_text(strip=True)
            title_parts = soup.title.text.split('-')
            date, type = title_parts[0].strip(), title_parts[1].strip() if len(title_parts) > 1 else 'Unknown'

            store_data(conn, wod, date, type)

            prev_page = soup.select_one('nav .nav-previous a')
            if prev_page['href'] == 'https://www.crossfitinvictus.com/wod/january-1-2015-performance-and-fitness/':
                break

            url = prev_page['href']
            print(url)
    
    except Exception as e:
        print(e)
    
    finally:
        print(url)
        driver.quit()



def main():
    db_name = 'wodbot.db'
    url = 'https://www.crossfitinvictus.com/wod/june-26-2021-fitness/'  

    conn = init_db(db_name)
    scrape(conn, url)
    conn.close()

if __name__ == '__main__':
    main()

