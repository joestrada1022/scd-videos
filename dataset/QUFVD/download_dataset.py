import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    URL = "https://www.dropbox.com/sh/nb543na9qq0wlaz/AAAc5N8ecjawk2KlVF8kfkrya?dl=0"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    print(' ')
