import gdown
import os

# URL of your Google Drive file
url = 'https://drive.google.com/uc?id=1T9q_pieH8-UjmdMDM3_08wnU4j-VSg6J'

# Desired filename
output = 'lr0.005_epoch_100.pth'

# Remove the file if it already exists
if os.path.exists(output):
    os.remove(output)

# Download the file from Google Drive
gdown.download(url, output, quiet=False)

print(f"File '{output}' downloaded successfully.")
