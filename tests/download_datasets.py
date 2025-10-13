from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("likaixin/ScreenSpot-v2-variants", cache_dir="ScreenSpot-v2-variants/")
ds = load_dataset("likaixin/ScreenSpot-Pro",cache_dir="Screenspot-pro/")


