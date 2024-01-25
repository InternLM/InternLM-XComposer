import os
import re
import json
import requests
import urllib.request
from multiprocessing.pool import ThreadPool


def download_image(url, path):
    if url == '':
        print('url is empty')
        return False

    try:
        urllib.request.urlopen(url)
        urllib.request.urlretrieve(url, path)
        return True
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    print(f"{url} download failed")
    return False


def download_image_thread(url_list, folder, index, num_processes, Async=True):
    pool = ThreadPool(processes=num_processes)
    thread_list = []
    os.makedirs(folder, exist_ok=True)
    for i in range(len(url_list)):
        path = os.path.join(folder, f'temp_{index}_{i}.png')
        if Async:
            out = pool.apply_async(func=download_image, args=(url_list[i], path))
        else:
            out = pool.apply(func=download_image, args=(url_list[i], path))
        thread_list.append(out)

    pool.close()
    pool.join()


