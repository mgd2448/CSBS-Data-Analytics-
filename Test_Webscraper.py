# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:42:06 2022

@author: thepw
"""
import os, json
from serpapi import GoogleSearch


params = {
  "q": "community bank fintech partnership",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "num": "100",
  "start": "10",
  "api_key": "2e00b45413073077c84125108ec6cbb25efc535c06e37d910fa1bc7c0fc17bf2"
}

search = GoogleSearch(params)
results = search.get_dict()
#print(results)
#prints full JSON response from the first page
for result in results["organic_results"]:
    print(json.dumps(result['link'], indent=2))  
    
    # want a title and a link?
    # print(result['title'])
    # print(result['link'])
    
    


