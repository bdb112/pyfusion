#!/usr/bin/env python
""" Extract headings and links from the H-1 wiki, typically a 'year' file 
     to create an overview wiki page in a faster loading and more compact 
     form e.g. http://h1ds.anu.edu.au/wiki/Day/MHD_new
    This is especially useful with wiki_day_db to do allow queries with summary_db
    
"""
from future.standard_library import install_aliases
import sys
from bs4 import BeautifulSoup
import time as tm
import numpy as np
import calendar

install_aliases()
from urllib.request import urlopen, Request


def datelike(url, txt, minscore=5):
    """ return true if the score for likelihood that txt contains the date of the url
    if minscore is None return the actual score
    """
    txt = txt.lower()
    tst = tm.strptime(url.split('Day/')[-1], '%Y/%m/%d')
    score = 0
    if calendar.day_name[tst.tm_wday].lower() in txt:
        score += 5
    elif tm.strftime('%a', tst).lower() in txt:
        score += 2
    if calendar.month_name[tst.tm_mon].lower() in txt:
        score += 5
    elif tm.strftime('%b', tst).lower() in txt:
        score += 2
    day = str(tst.tm_mday)
    if len(day) == 2 and day in txt:
        score += 2
    elif len(day) == 1 and day in txt:
        score += 1
    if str(tst.tm_year) in txt:
        score += 3
    if minscore is None:
        return(score)
    else:
        return(score > minscore)


def get_links(url='http://h1svr.anu.edu.au/wiki/Day/2010', debug=1, exceptions=()):     # ?action=raw'
    """ return a list of links on the page 
    if page not found, and exceptions == Exception , return None quietly
    if no links, return []
    """
    try:
        conn = urlopen(url)
    except exceptions as reason:
        return None
    
    html = conn.read()

    # specifying lxml is recommended by the author (by a message when it was omitted)
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all('a')

    linklist = []
    for tag in links:
        link = tag.get('href',None)
        if link is not None:
            linklist.append([link, tag.get_text()])
            if debug: print(linklist[-1])
    if debug > 2:  # integrate debug_ here instead
        1/0
    return(linklist)

if __name__ == '__main__':

    from collections import OrderedDict

    URL = sys.argv[1] if len(sys.argv)>1 else 'http://h1svr.anu.edu.au/wiki/Day/2010' 

    try:
        if oldurl == URL:
            links = oldlinks
        else:
            raise LookupError
    except:
        links = get_links(URL)

    oldlinks, oldurl = links, URL

    daylinks = [link for link in links if '/wiki' in link[0] and '?' not in link[0] and len(link[0])==len('/wiki/Day/2010/12/17')]
    # The page urls (days) for this period (year)
    pages = np.unique([url for url,txt in daylinks])
    # for each day find all the links, and take the first that isn't a date 
    pagedict = OrderedDict()
    for page in pages:
        matches = [link for link in links if link[0] == page]
        for url, txt in matches:
            if datelike(url, txt):
                continue
            else:
                title = txt
                break
        else:
            title = None
        if title is not None: 
            pagedict.update({url: title})
            print(url + ': ' + title)

    import json
    pagedict = pagedict
    json.dump(pagedict, open('wikipagedict_{y}.json'.format(y=URL[-4:]),'w'))

"""
cp -rp $MDSPLUS_DIR/trees /tmp/
export main_path=/tmp/trees/
"""
