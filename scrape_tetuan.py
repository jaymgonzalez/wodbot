from datetime import datetime, timedelta
from bs4 import BeautifulSoup as bs
import sqlite3
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def reverse_date_range(start_date, end_date):
    date_format = "%Y-%m-%d"
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    while start >= end:
        print(start.strftime(date_format))
        start -= timedelta(days=1)

def store_data(conn, title, wod, date):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO tetuan_wod_data (title, wod, date) VALUES (?, ?, ?)', (title, wod, date))
    conn.commit()
    
def fetch_url(driver, date):
    date_str = date.strftime('%d/%m/%Y').replace('/', '%2F')
    url = f'https://crosshero.com/dashboard/admin/wods?program_id=57c7f19ecc34d5001218102d?&date={date_str}'
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, date.strftime('%Y-%m-%d'))))

    select_element = driver.find_element(By.ID, 'program_id')
    select_object = Select(select_element)
    select_object.select_by_value('57c7f19ecc34d5001218102d')

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, date.strftime('%Y-%m-%d'))))
    
    return driver


def scrape_data_for_date(conn, driver, date):
    soup = bs(driver.page_source, 'html.parser')
    data_container_id = date.strftime('%Y-%m-%d')
    data_container = soup.find(id=data_container_id)

    if not data_container:
        page = fetch_url(driver, date)
        scrape_data_for_date(conn, page, start)
        return

    wod = data_container.find(class_='today-wod-components')
    titles = wod.find_all('h4')
    wods = wod.find_all(class_='trix-content')

    if len(titles) + 1 == len(wods):
        for title, wod in zip(titles, wods[1:]):
            title = title.get_text(strip=True)
            wod = wod.get_text(strip=True)
            store_data(conn, title, wod, date)

    if len(titles) == len(wods) and 'Ningún workout encontrado' not in wod:
        for title, wod in zip(titles, wods):
            title = title.get_text(strip=True)
            wod = wod.get_text(strip=True)
            store_data(conn, title, wod, date)


def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tetuan_wod_data (
            id INTEGER PRIMARY KEY,
            title TEXT,
            wod TEXT,
            date TEXT
        )
    ''')
    conn.commit()
    return conn



if __name__ == '__main__':
    driver = uc.Chrome(headless=False)
    conn = init_db('wodbot.db')
    now = '2023-12-29'
    end_date = '2021-11-01'

    
    date_format = "%Y-%m-%d"
    start = datetime.strptime(now, date_format)
    end = datetime.strptime(end_date, date_format)
    page = fetch_url(driver, start)
    
    while start >= end:
        print(start.strftime(date_format))
        scrape_data_for_date(conn, page, start)
        start -= timedelta(days=1)

    driver.quit()
    conn.close()
