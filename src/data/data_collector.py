import sys

import logging
from typing import Dict
import requests
from tqdm import tqdm

DATA_URL = "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/:id"
ID_START = 125
ID_END = 125
COOKIES = { "sessionid": "ujhl1fmy1imwcd92pwelmfpmddst54k6" }

logger = logging.getLogger(__name__)

def download_file(url: str, filename: str, cookies: Dict[str, str]) -> str:
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    logger.info(f"Downloading {url}")
        
    r = requests.get(url, stream=True, cookies=cookies)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

def get_data() -> None:
    for id in range(ID_START, ID_END + 1):
        
        destination_url = DATA_URL.replace(":id", str(id))
        file_name = download_file(destination_url, "data/raw/{}.zip".format(id), COOKIES)
    