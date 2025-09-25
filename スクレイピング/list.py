from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)
ids_dates = []
for page in range(1, 6):
    url = f"https://min-repo.com/tag/%E3%83%9E%E3%83%AB%E3%83%8F%E3%83%B3%E5%85%A5%E9%96%93%E5%BA%97/?page={page}"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        match = re.search(r'/(\d{7,8})/', href)
        if match:
            page_id = match.group(1)
            date_txt = a_tag.get_text(strip=True)
            #月の指定
            if date_txt.startswith("9/"):
                ids_dates.append((page_id, date_txt))
driver.quit()
print(ids_dates)
