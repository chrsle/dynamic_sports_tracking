#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import argparse
import logging
from tqdm import tqdm

def download_video(url, filename):
    """
    Downloads a video from a direct URL.
    """
    try:
        # Send a HTTP request to the URL of the video
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception if the GET request was unsuccessful

        # Get the total size of the video in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Open the file in write and binary mode
        with open(filename, 'wb') as video, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                size = video.write(chunk)
                bar.update(size)

        logging.info(f"Video downloaded successfully: {filename}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise SystemExit(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Downloader")
    parser.add_argument("url", help="URL of the video to download")
    parser.add_argument("filename", help="Filename to save the video as")
    args = parser.parse_args()

    download_video(args.url, args.filename)

