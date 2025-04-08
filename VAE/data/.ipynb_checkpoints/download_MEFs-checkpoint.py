import gdown
import numpy as np

# URL to your Google Drive file
url = 'https://drive.google.com/uc?id=1PYKLA1CNDpcbJqvaapXNofkN6tBDAknU'

# Path to save the downloaded file
output = 'MEFs.npy'

# Download the file
gdown.download(url, output, quiet=False)

# Load and check the file
data = np.load(output)
print(data)
