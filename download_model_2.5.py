from modelscope import snapshot_download

# 指定下载目录到当前文件夹下的 models 目录
# model_dir = snapshot_download(
#     'Qwen/Qwen2.5-VL-3B-Instruct',
#     cache_dir='models',
#     revision='master'
# )
# print(f"模型已下载到: {model_dir}")


model_dir = snapshot_download(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    cache_dir='models',
    revision='master'
)
print(f"模型已下载到: {model_dir}")