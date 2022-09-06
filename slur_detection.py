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
slurs_file = "Data/slur_subset_low.csv"
large_slurs_file = "Data/slur_subset_high.csv"

read_all = True
if read_all:
    slurs = pd.read_csv(slurs_file)['word'].tolist()
    large_slurs = pd.read_csv(large_slurs_file)['word'].tolist()
else:
    f_len = sum(1 for line in open(slurs_file)) - 1
    sample_size = round(f_len / 5)
    skip = sorted(random.sample(range(1, f_len + 1), f_len - sample_size))
    slurs = pd.read_csv(slurs_file, skiprows=skip)['word'].tolist()

    f_len = sum(1 for line in open(large_slurs_file)) - 1
    sample_size = round(f_len / 5)
    skip = sorted(random.sample(range(1, f_len + 1), f_len - sample_size))
    large_slurs = pd.read_csv(large_slurs_file, skiprows=skip)['word'].tolist()

cols = ['word', 'text']
matches = pd.DataFrame(columns=cols)
version = "4"
small_filename = "matches_short"+version+".csv"
large_filename = "matches_large"+version+".csv"

df = pd.read_csv('Data/wet.paths', sep=':-\s*', names=['wet-url'], engine="python")
processed = 0
found_small = 0
found_large = 0
start = time.perf_counter()
# 60 for v3, 250 for v4
number_of_processses = 200

# 10 for v3, 25 for v4
random.seed(25)
for index, row in df.iterrows():
    # Use to skip to certain row index
    if processed < 166:
        processed += 1
        continue
    if processed % 1 == 0:
        print("processed: " + str(processed))
        print(f"time: {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}")
        print("small found: " + str(found_small))
        print("large found: " + str(found_large) + "\n")

    wet_url = prefix_url + row[0]
    wet_file = get_file(wet_url)

    iterobject = wrapper(ArchiveIterator(wet_file.raw))
    for record in iterobject:
        if record.rec_headers.get_header('WARC-Identified-Content-Language') == 'eng':
            if random.random() <= 0.20: # 0.25 for v3, 0.2 for v4
                # Get text, decode, lowercase, remove newline and commas
                text = record.content_stream().read().decode('utf-8').lower().replace('\n', '.').replace(',', ' ')
                if text is None:
                    processed += 1
                    continue
                else:
                    # Basic contains check - issues with offensive terms list and context
                    for term in slurs:
                        if re.search(r'\b' + term + r'\b', text):
                            data = [term, text]
                            matches.loc[len(matches.index)] = data
                            matches.to_csv(small_filename, mode='a', index=False, header=False)
                            matches = matches[0:0]
                            found_small += 1
                    for term in large_slurs:
                        if re.search(r'\b' + term + r'\b', text):
                            data = [term, text]
                            matches.loc[len(matches.index)] = data
                            matches.to_csv(large_filename, mode='a', index=False, header=False)
                            matches = matches[0:0]
                            found_large += 1

    processed += 1
    if processed == number_of_processses:
        break

print("Done :)")
