import os
import re
import json
import requests
import urllib.request
from multiprocessing.pool import ThreadPool

headers = {
    'Accept': 'text/plain, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://www.baidu.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
}
session = requests.session()
session.headers = headers


ACE_CONTROL_CHARACTER = "\x00-\x1f\x80-\xa0\xad\u1680\u180E\u2000-\u200f\u2028\u2029\u202F\u205F\uFEFF\uFFF9-\uFFFC\u2066\u2067\u2068\u202A\u202B\u202D\u202E\u202C\u2069"
NON_PRINTING_CHARS_RE = re.compile(
    f"[{ACE_CONTROL_CHARACTER + ''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)

def remove_non_printing_char(text: str) -> str:
    return NON_PRINTING_CHARS_RE.sub("", text)


def search_imgs(kwd: str, num: int) -> list:
    """
    kwd 搜索关键字
    num 返回图片数，最大30
    """
    session.get("https://www.baidu.com/")
    for retry in range(3):
        try:
            # proxy = {'http': 'http://***', 'https': 'http://***'}
            proxy = None
            params = {
                'tn': 'baiduimage',
                'ipn': 'r',
                'ct': '201326592',
                'cl': '2',
                'lm': '-1',
                'st': '-1',
                'fm': 'result',
                'fr': '',
                'sf': '1',
                'fmq': '1695091273229_R',
                'pv': '',
                'ic': '',
                'nc': '1',
                'z': '',
                'hd': '',
                'latest': '',
                'copyright': '',
                'se': '1',
                'showtab': '0',
                'fb': '0',
                'width': '',
                'height': '',
                'face': '0',
                'istype': '2',
                'dyTabStr': 'MTEsMCwxLDYsMyw0LDUsMiw4LDcsOQ==',
                'ie': 'utf-8',
                'ctd': '1695091273229^00_2543X643',
                'sid': '',
                'word': kwd,
            }

            response = session.get('https://image.baidu.com/search/index', params=params, proxies=proxy, timeout=15)
            response.raise_for_status()
            # response.encoding = response.apparent_encoding
            json_string = re.findall(
                r"\'imgData\', (.*)\s\);", response.text)[0]
            json_string = remove_non_printing_char(json_string)
            json_string = json_string.replace('\\\'', '\'')
            res_dic = json.loads(json_string)["data"]
            res_list = []
            for j in res_dic:
                if j.get("hoverURL", ""):
                    res_list.append(j.get("hoverURL"))
                    continue
                if j.get("middleURL", ""):
                    res_list.append(j.get("middleURL"))
                    continue
            return res_list[:num]
        except Exception as e:
            print(e)
    return []


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




if __name__ == '__main__':
    print(search_imgs("足球", 10))
