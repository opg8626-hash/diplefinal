import requests
import zipfile
import os
import urllib3
urllib3.disable_warnings()

url = "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip"
print(f"Downloading {url}...")
response = requests.get(url, verify=False, stream=True)
with open("weedsgalore.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print("Download complete. Extracting...")

with zipfile.ZipFile("weedsgalore.zip", 'r') as zip_ref:
    zip_ref.extractall("weedsgalore_extracted")
print("Extraction complete.")
