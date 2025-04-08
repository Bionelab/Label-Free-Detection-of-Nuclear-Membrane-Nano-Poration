import os
import zipfile
import subprocess
import sys
sys.path.append("PREP/")


def install_gdown():
    """Install gdown if not already installed."""
    try:
        import gdown
        print("gdown is already installed")
        return True
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True

def main():
    # Make sure gdown is installed
    install_gdown()
    import gdown
    
    # URLs to download
    urls = [
        "https://drive.google.com/file/d/1qDKEBWBVPDGOfsg8TlU5K6Thbpz6WkQG/view?usp=drive_link",
        "https://drive.google.com/file/d/1MK4KLtfPs_1gslTPQRVJQda3zGI95GYB/view?usp=drive_link",
        "https://drive.google.com/file/d/1LgSg5amuxZtzO42QHaNd0TY9c4JpysEk/view?usp=drive_link"
    ]
    
    # Output directory - current working directory
    output_dir = os.getcwd()
    
    for i, url in enumerate(urls):
        try:
            print(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
            
            # Extract file ID from Google Drive URL
            file_id = url.split('/d/')[1].split('/')[0]
            print(f"Extracted file ID: {file_id}")
            
            # Download file using gdown
            zip_filename = f"download_{i+1}.zip"
            print(f"Downloading to {zip_filename}...")
            gdown.download(id=file_id, output=zip_filename, quiet=False)
            
            # Check if the file was downloaded and has content
            if os.path.exists(zip_filename) and os.path.getsize(zip_filename) > 0:
                # Verify it's a valid zip file
                try:
                    # Extract the zip file
                    print(f"Extracting {zip_filename}...")
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    print(f"Successfully extracted {zip_filename}")
                    
                    # Remove the zip file
                    os.remove(zip_filename)
                    print(f"Removed {zip_filename}")
                    
                except zipfile.BadZipFile:
                    print(f"Error: {zip_filename} is not a valid zip file. Trying alternate method...")
                    # Try with the URL method instead
                    alt_zip_filename = f"download_{i+1}_alt.zip"
                    gdown.download(url=url, output=alt_zip_filename, quiet=False, fuzzy=True)
                    
                    # Try to extract the new file
                    try:
                        print(f"Extracting {alt_zip_filename}...")
                        with zipfile.ZipFile(alt_zip_filename, 'r') as zip_ref:
                            zip_ref.extractall(output_dir)
                        print(f"Successfully extracted {alt_zip_filename}")
                        
                        # Remove the zip file
                        os.remove(alt_zip_filename)
                        print(f"Removed {alt_zip_filename}")
                        
                        # Also remove the original failed download
                        if os.path.exists(zip_filename):
                            os.remove(zip_filename)
                    except zipfile.BadZipFile:
                        print(f"Error: Both download methods failed for {url}")
            else:
                print(f"Error: File {zip_filename} was not downloaded correctly")
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    print("\nAll files have been processed.")

if __name__ == "__main__":
    main()