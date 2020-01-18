import requests 
from bs4 import BeautifulSoup 
import csv 
  
URL = "http://www.naa.gov.in/news.php"
r = requests.get(URL) 
soup = BeautifulSoup(r.content) 

mydivs = soup.findAll("div", {"class": "col-*-*"})
mydivs = str(mydivs)
mydivs = mydivs.split('"</div>, <div class="col-*-*">"')
for i in mydivs:
	print(i)

quotes=[]  # a list to store quotes 
  
