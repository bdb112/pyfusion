""" modelled after howtoREST.py - Note: see archiveDB.py for ideas (better ideas?)
on the W7X virtual desktop, urllib wont work (no attribute 'request') unless get_shot_list update is run first (2015 version of urllib.py)
"""
import urllib, json, datetime, calendar

# base_url = 'http://archive-webapi.ipp-hgw.mpg.de'
# datastream = '/Test/raw/W7X/QSB_Bolometry/BoloSignal_DATASTREAM/0/HBCm_U30_01'

fmt = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.{CDS}/DataModuleDesc.{DMD}_DATASTREAM/{ch}/Channel_{ch}'            # /scaled/_signal.json?from={shot_f}&upto={shot_t}

# Diagnostic:W7X_L57_LP01_I
params = dict(CDS=82,DMD=181,ch=1)
base_url =''
datastream = fmt.format(**params)

d_start = datetime.datetime(2015, 11, 1)
d_stop = datetime.datetime(2018, 3, 10)
my_start = int(round(calendar.timegm(d_start.timetuple())*1000) + d_start.microsecond)*1000000
my_stop = int(round(calendar.timegm(d_stop.timetuple())*1000) + d_stop.microsecond)*1000000

filter_query = '?filterstart=' + str(my_start) + '&filterstop=' + str(my_stop)
address = base_url + datastream + filter_query

request = urllib.request.Request(url=address, headers={'Accept':'application/json'})
response = json.loads(urllib.request.urlopen(request).read().decode('utf-8'))
seg_list = response['_links']['children']
while 'next' in response['_links']:
    address = response['_links']['next']['href']
    request = urllib.request.Request(url=address, headers={'Accept':'application/json'})
    response = json.loads(urllib.request.urlopen(request).read().decode('utf-8'))
    seg_list.extend(response['_links']['children'])

json.dump(dict(seg_list=seg_list), file('seg_list.json','w'))
