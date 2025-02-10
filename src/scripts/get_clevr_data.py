from datasets import load_dataset

# cache_dir = "/apdcephfs_gy2/share_302735770/stephenruan/data/"
cache_dir = "/apdcephfs_sh8/share_301266059/stephenruan/data/"
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("leonardPKU/clevr_cogen_a_train", cache_dir=cache_dir)