# -*- coding: UTF-8 -*-

import sys
import time
import urllib
import requests
import numpy as np
from bs4 import BeautifulSoup
import openpyxl
import re
import json

# Some User Agents
hds = [{'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}, \
       {
           'User-Agent': 'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'}, \
       {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'}]



def get_bacteria_pathway(bac):
    ls = []
    url='https://www.genome.jp/dbget-bin/www_bget?pathway+'+bac+'01100' # For Test
    try:
        req = urllib.request.Request(url, headers=hds[np.random.randint(0, len(hds))])
        source_code = urllib.request.urlopen(req).read()
        plain_text = str(source_code)
    except (urllib.request.HTTPError, urllib.request.URLError) as e:
        print(e)
    soup = BeautifulSoup(plain_text,"html.parser")
    try:
        module = soup.find_all('td',attrs={'class':'td30'})[2]
        for ah in module.find_all('a'):
            href = ah.attrs['href']
            try:
                ms = re.split(bac+'_',href)[1]
                ls.append(ms)
            except:
                continue
    except:
        print("not found")

    return ls

def redirect_xlsx(inputfile):
    wb = openpyxl.load_workbook(inputfile)
    # bac_wiki = wb['wiki']
    bac_all = wb['bac-bac']
    new_sheet = wb.create_sheet()
    ls = []

    for start in range(len(list(bac_all.rows))):
        rows = [item.value for item in list(bac_all.rows)[start]]
        if rows[1] is not None:
            print(rows)
            bac = rows[2]
            ls = get_bacteria_pathway(bac)
            rows = rows + ls
        new_sheet.append(rows)
    wb.save('bacteria_01.xlsx')


def getAllModuleOfMap01100():
    url = 'https://www.kegg.jp/dbget-bin/www_bget?pathway+map01100'  # For Test
    try:
        req = urllib.request.Request(url, headers=hds[np.random.randint(0, len(hds))])
        source_code = urllib.request.urlopen(req).read()
        plain_text = str(source_code)
    except (urllib.request.HTTPError, urllib.request.URLError) as e:
        print(e)
    soup = BeautifulSoup(plain_text, "html.parser")
    for a in soup.find_all('a'):
        # print(a)
        try:
            href = a['href']
            # print(href)
            module = re.split(r'/kegg-bin/show_module\?',href)[1]
            getJson(module)
            # with open('module.txt','a+',encoding='utf8') as f:
            #     f.write(module+'\n')
        except:
            continue
    # f.close()

def getJson(line):
    # f = open('module.txt','r',encoding='utf8')
    # for line in f.readlines():
        url = 'https://www.kegg.jp/dbget-bin/www_bget?' +  line# For Test
        try:
            req = urllib.request.Request(url, headers=hds[np.random.randint(0, len(hds))])
            source_code = urllib.request.urlopen(req).read()
            plain_text = str(source_code)
        except (urllib.request.HTTPError, urllib.request.URLError) as e:
            print(e)
        soup = BeautifulSoup(plain_text, "html.parser")

        reaction = soup.find_all('td',attrs={'class':'td31'})[3]
        compound = soup.find_all('td',attrs={'class':'td30'})[4]
        dic = {}
        dic['name'] = line
        # getReactions
        dic['reactions'] = []
        for rline in reaction.find_all('div',attrs={'style':'float:left'}):
            # print("reation:")
            if rline.get_text() is not None:
                # print(rline.get_text())
                reactions = rline.get_text().encode('utf-8').decode('unicode_escape')
                # if re.match("\u00c2",reactions) or  re.match("\u00a0",reactions) is not None:
                #     reactions = re.sub('\u00c2\u00a0',"",reactions)
                dic['reactions'].append(reactions)
        # getCompounds
        dic['compounds'] = []
        for cline in reaction.find_all('div',attrs={'style':'margin-left:16.8px'}):
            # print("compound:")
            if cline.get_text() is not None:
                # print(cline.get_text())
                dic['compounds'].append(cline.get_text())
        # dic['reaction']

        for iline in compound.find_all('table',attrs={'style':'border:0;border-collapse:collapse;'}):
            # print("compoundsinfo:")
            if iline.get_text() is not None:
                # print(iline.get_text())
                # reactions = rline.get_text().encode('utf-8').decode('unicode_escape')
                # dic['reactions'].append(reactions)
                with open('compoundsinfo.txt', 'a', encoding='utf8') as ci:
                    ci.write(iline.get_text())
                    ci.write('\n')

        if len(dic['reactions']) == len(dic['compounds']):
            print("module: %s has finished"%line)
            with open('module.json','a',encoding='utf8') as m:
                json.dump(dic,m)
                m.write('\n')



        # print("reaction:",reaction)
        # print("compound:",compound)


if __name__ == '__main__':
    inputfile = 'bacteria_01.xlsx'
    redirect_xlsx(inputfile)
    # getAllModuleOfMap01100()
    # getJson()