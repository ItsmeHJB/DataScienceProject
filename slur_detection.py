import pandas as pd
import requests
from warcio.archiveiterator import ArchiveIterator
import re
import random
import time


def get_file(url):
    return requests.get(url, stream=True)


def wrapper(gen):
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except Exception as e:
            print(e)


def get_record_with_header(warc_file, header, value):
    iterobject = wrapper(ArchiveIterator(warc_file.raw))
    for record in iterobject:
        if record.rec_headers.get_header(header) == value:
            return record


def get_WET_text(record):
    if record:
        # Reads text, decodes it using utf-8, makes it lowercase
        return record.content_stream().read().decode('utf-8').lower()
    else:
        return None


prefix_url = 'https://data.commoncrawl.org/'
warc_match = re.compile('warc')
wet_sub = 'wet'
ext_sub = 'warc.wet'
slurs_file = "Data/slur_subset.csv"

read_all = True
if read_all:
    slurs = pd.read_csv(slurs_file)['word'].tolist()
else:
    f_len = sum(1 for line in open(slurs_file)) - 1
    sample_size = round(f_len / 5)
    skip = sorted(random.sample(range(1, f_len + 1), f_len - sample_size))
    slurs = pd.read_csv(slurs_file, skiprows=skip)['word'].tolist()

cols = ['word', 'wet-url']
matches = pd.DataFrame(columns=cols)
filename = "matches2.csv"

df = pd.read_csv('Data/198da78a-2deb-4fc7-b516-3c9363188510.csv')
processed = 0
found = 0
start = time.perf_counter()

for index, row in df.iterrows():
    # Use to skip to certain row index
    if processed <= 0:
        processed += 1
        continue
    if processed % 1000 == 0:
        print("processed: " + str(processed))
        print(f"time: {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}")
        print("found: " + str(found) + "\n")

    wet_url = prefix_url + re.sub(warc_match, ext_sub, re.sub(warc_match, wet_sub, row[0], count=1))
    wet_file = get_file(wet_url)

    # Get English WET record and print details
    wet_record = get_record_with_header(
        wet_file,
        header='WARC-Identified-Content-Language',
        value="eng"
    )

    text = get_WET_text(wet_record)
    if text is None:
        processed += 1
        continue
    else:
        # Basic contains check - issues with offensive terms list and context
        for term in slurs:
            if re.search(r'\b' + term + r'\b', text):
                data = [term, wet_url]
                matches.loc[len(matches.index)] = data
                matches.to_csv(filename, mode='a', index=False, header=False)
                matches = matches[0:0]
                # print(wet_record.rec_headers.get_header('WARC-Target-URI'))
                # print("Term: " + term)
                # print(text[:25]+"\n")
                found += 1
    # if matches.shape[0] >= 5:
    #     matches.to_csv(filename, mode='a', index=False, header=False)
    #     matches = matches[0:0]
    processed += 1

matches.to_csv(filename, mode='a', index=False, header=False)
print("Done :)")
