import kagglehub

# Download latest version
path = kagglehub.dataset_download("au1206/20-newsgroup-original")

print("Path to dataset files:", path)