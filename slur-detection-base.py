import pandas as pd
import requests
from warcio.archiveiterator import ArchiveIterator
import re
import random


def get_file(url):
    return requests.get(url, stream=True)


def get_record_with_header(warc_file, header, value):
    for record in ArchiveIterator(warc_file.raw):
        if record.rec_headers.get_header(header) == value:
            return record


def get_WET_text(record):
    return record.content_stream().read().decode('utf-8').lower()


prefix_url = 'https://data.commoncrawl.org/'
warc_match = re.compile('warc')
wet_sub = 'wet'
ext_sub = 'warc.wet'

slurs_file = "Data/slur_subset.csv"
f_len = sum(1 for line in open(slurs_file)) - 1
sample_size = round(f_len/5)
skip = sorted(random.sample(range(1, f_len+1), f_len-sample_size))
slurs = pd.read_csv(slurs_file, skiprows=skip)['word'].tolist()

reader = pd.read_csv('Data/b75c614d-bb12-4083-9664-3ea424f6d4ce.csv', usecols=[0], chunksize=1000)
for df in reader:
    for index, row in df.iterrows():
        # warc_url = prefix_url + row[0]
        # warc_file = get_file(warc_url)

        wet_url = prefix_url + re.sub(warc_match, ext_sub, re.sub(warc_match, wet_sub, row[0], count=1))
        wet_file = get_file(wet_url)

        # Get WARC and associated WET record
        # warc_record = get_record_with_header(
        #     warc_file,
        #     header='WARC-Type',
        #     value='response'
        # )
        # wet_record = get_record_with_header(
        #     wet_file,
        #     header='WARC-Refers-To',
        #     value=warc_record.rec_headers.get_header('WARC-Record-ID')
        # )

        # Get English WET record and print details
        wet_record = get_record_with_header(
            wet_file,
            header='WARC-Identified-Content-Language',
            value="eng"
        )
        # print(wet_record.rec_headers.get_header('WARC-Target-URI'))
        # print(wet_record.content_stream().read().decode('utf-8') + '\n')

        text = get_WET_text(wet_record)
        if any(term in text for term in slurs):
            print(wet_record.rec_headers.get_header('WARC-Target-URI'))
            print(text)