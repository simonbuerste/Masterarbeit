import wget

# url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
# wget.download(url, 'C:/Users/st158084/tensorflow_datasets/wget/ILSVRC2012_img_train.tar')

import requests

from pathlib import Path
from tqdm import tqdm

URL = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'

DOWNLOAD_FOLDER = Path('C:/Users/st158084/tensorflow_datasets/wget/')
"""pathlib.Path: Points to the target directory of downloads."""


def downloader(resume_byte_pos: int = None):
    """Download url in ``URLS[position]`` to disk with possible resumption.
    Parameters
    ----------
    position: int
        Position of url.
    resume_byte_pos: int
        Position of byte from where to resume the download
    """
    # Get size of file
    url = URL
    r = requests.head(url)
    file_size = int(r.headers.get('content-length', 0))

    # Append information to resume download at specific byte position
    # to header
    resume_header = ({'Range': f'bytes={resume_byte_pos}-'}
                     if resume_byte_pos else None)

    # Establish connection
    r = requests.get(url, stream=True, headers=resume_header)

    # Set configuration
    block_size = 1024
    initial_pos = resume_byte_pos if resume_byte_pos else 0
    mode = 'ab' if resume_byte_pos else 'wb'
    file = DOWNLOAD_FOLDER / url.split('/')[-1]

    with open(file, mode) as f:
        with tqdm(total=file_size, unit='B',
                  unit_scale=True, unit_divisor=1024,
                  desc=file.name, initial=initial_pos,
                  ascii=True, miniters=1) as pbar:
            for chunk in r.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file():
    """Execute the correct download operation.
    Depending on the size of the file online and offline, resume the
    download if the file offline is smaller than online.
    Parameters
    ----------
    position: int
        Position of url.
    """
    # Establish connection to header of file
    url = URL
    r = requests.head(url)

    # Get filesize of online and offline file
    file_size_online = int(r.headers.get('content-length', 0))
    file = DOWNLOAD_FOLDER / url.split('/')[-1]

    if file.exists():
        file_size_offline = file.stat().st_size

        if file_size_online != file_size_offline:
            downloader(file_size_offline)
        else:
            pass
    else:
        downloader()


if __name__ == '__main__':
    download_file()
