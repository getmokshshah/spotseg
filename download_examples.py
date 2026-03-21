"""
Download example images for SpotSeg demo.
Uses high-quality public domain / Unsplash images.
"""

import os
import urllib.request


EXAMPLE_URLS = {
    "park_dog.jpg": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=640&q=80",
    "city_street.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640&q=80",
    "kitchen.jpg": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&q=80",
}


def download_examples(output_dir: str = "examples"):
    """Download example images if they don't already exist."""
    os.makedirs(output_dir, exist_ok=True)

    for filename, url in EXAMPLE_URLS.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            continue
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"  Warning: Could not download {filename}: {e}")


if __name__ == "__main__":
    download_examples()
