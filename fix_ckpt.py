import torch

# 指向你刚才报错的那个权重文件
ckpt_path = './weights/parseq.ckpt'

print(f"正在修复权重文件: {ckpt_path} ...")
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 核心逻辑：如果缺少版本号，我们就给它强行加上一个
if 'pytorch-lightning_version' not in checkpoint:
    # 填入一个经典的旧版本号，绕过新版本的兼容性检查
    checkpoint['pytorch-lightning_version'] = '1.6.0'

    # 顺便检查一下 state_dict 是否在正确的位置
    if 'state_dict' not in checkpoint:
        print("检测到纯权重文件，正在包装为 Lightning 格式...")
        checkpoint = {'state_dict': checkpoint, 'pytorch-lightning_version': '1.6.0'}

torch.save(checkpoint, ckpt_path)
print("修复完成！现在你可以重新运行 read.py 了。")