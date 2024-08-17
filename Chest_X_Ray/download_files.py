#!/usr/bin/env python3
import requests
import tarfile
from io import BytesIO
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def create_image_directory():
    """Create directory to store images if it doesn't already exist."""
    os.makedirs('xray_images', exist_ok=True)

def download_and_stream_unpack(link, idx):
    """Download and unpack a .tar.gz file from a given link."""
    print(f'Starting download and unpacking of file {idx + 1}...')

    # Start the streamed download
    with requests.get(link, stream=True) as r:
        r.raise_for_status()  # check status code, raise exception
        
        # Create a tarfile object from the streamed content
        file_like_object = BytesIO(r.raw.read())
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            tar.extractall(path='xray_images')
            
    print(f'Completed download and unpacking of file {idx + 1}.')

def download_all_images(links):
    """Download and unpack all image .tar.gz files."""
    for idx, link in enumerate(links):
        download_and_stream_unpack(link, idx)

def find_latest_file(directory):
    """Find the latest downloaded file in the given directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def download_csv_with_selenium(url, download_directory, filename):
    """Download the CSV file using Selenium and save it with a specific name."""
    # Set up the Chrome options
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_directory,  # Set default download directory
        "download.prompt_for_download": False,  # Disable the download prompt
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless")  # Run in headless mode to not open a browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url) # Navigate to the page

        time.sleep(5)  # wait 5 sec for page to load

        # Find the download button using XPath and click it
        download_button = driver.find_element(By.XPATH, "//button[@data-testid='download-button']")
        download_button.click()

        time.sleep(20) # Wait for the download to complete

        # Find the latest downloaded file in the directory
        downloaded_file = find_latest_file(download_directory)
        if downloaded_file and os.path.exists(downloaded_file):
            new_file_path = os.path.join(download_directory, filename) 
            os.rename(downloaded_file, new_file_path) # Rename the file
            print(f"Download completed successfully and saved as {filename}.")
        else:
            print("File not found after download. Check the download directory.")
    except Exception as e: # print error if exception
        print(f"Error during download: {e}")
    finally:
        driver.quit() # Close the WebDriver

def main():
    # Create the directory for images
    create_image_directory()

    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    # Download and unpack all image files
    download_all_images(links)

    # Download supporting CSV file
    csv_url = 'https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468'
    download_directory = os.getcwd()  # Get the current working directory (root)
    filename = 'ChestXray-NIHCC.csv'
    download_csv_with_selenium(csv_url, download_directory, filename)
    
    
if __name__ == "__main__":
    main()
