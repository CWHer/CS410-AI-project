# collect data of top players
#   for IL

import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import requests
from tqdm import tqdm

# example
# http://www.jidiai.cn:3389/api/v1/evaluationsbypage?userid=1566&envid=6&page=1


INFO_URL = "http://www.jidiai.cn:3389/api/v1/evaluationsbypage?userid={}&envid={}&page={}"
RECORD_URL = "http://www.jidiai.cn:3389/api/v1/replay?taskid={}"
ENV = 6
# USERS = [1566, 240]
USERS = [240]
INIT_PAGE = 1
HEADERS = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
}


def getURL(url, headers=HEADERS, timeout=4):
    for _ in range(20):
        try:
            r = requests.get(
                url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                # print(r.json())
                return r.json()
        except Exception as e:
            print(e)
    raise RuntimeError()


if __name__ == "__main__":
    for user in USERS:
        url = INFO_URL.format(user, ENV, INIT_PAGE)
        total_pages = getURL(url)["num"]

        tasks = []
        with tqdm(total=total_pages) as pbar:
            with ProcessPoolExecutor(max_workers=10) as executor:
                urls = [INFO_URL.format(user, ENV, i)
                        for i in range(total_pages)]
                for result in executor.map(getURL, urls):
                    info = result["tasks"]
                    for task in info:
                        tasks.append(task["id"])
                    pbar.update()

        data_buffer = []
        with tqdm(total=total_pages) as pbar:
            with ProcessPoolExecutor(max_workers=10) as executor:
                urls = [RECORD_URL.format(task) for task in tasks]
                for result in executor.map(getURL, urls):
                    data = result["data"]
                    data_buffer.append(data)
                    pbar.update()

        # tasks = []
        # for i in tqdm(range(total_pages)):
        #     url = INFO_URL.format(user, ENV, i)
        #     info = getURL(url)["tasks"]
        #     # print(info)
        #     for task in info:
        #         tasks.append(task["id"])

        # data_buffer = []
        # for task in tqdm(tasks):
        #     url = RECORD_URL.format(task)
        #     data = getURL(url)["data"]
        #     data_buffer.append(data)

        with open(f"{user}_log.pkl", "wb") as f:
            pickle.dump(data_buffer, f)
