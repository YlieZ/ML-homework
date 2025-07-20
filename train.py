import random
import torch
from visdom import Visdom
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from tqdm import tqdm

from models.ViT import VisionTransformer

import os

weights_dir = 'weights'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

seed = 42
depth = 6
n_heads = 8
n_classes = 10
n_epoches = 20
batch_size = 16
show_size = 8
# historical_weights = 'VIT_depth6_8heads_10classes_epoch4_acc63.26%.pth'
historical_weights = None

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Use {device}')

trans = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])
untrans = T.Compose([
    T.Normalize([-.485/.229, -.456/.224, -.406/.225],
                [1/.229, 1/.224, 1/.225]),
])

train_set = CIFAR10(root='data', train=True, download=True, transform=trans)
val_set = CIFAR10(root='data', train=False, download=True, transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=0)

model = VisionTransformer(img_size=32,
                          patch_size=2,
                          n_classes=n_classes,
                          depth=depth,
                          n_heads=n_heads,
                          embed_dim=512,
                          mlp_ratio=1
                          ).to(device)

# 续训
if historical_weights is not None:
    model.load_state_dict(torch.load(f'weights/{historical_weights}'))
    historical_epoch = int(historical_weights.split('_')[4][5:])
else:
    historical_epoch = 0

optim = Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
criterion = nn.CrossEntropyLoss()
viz = Visdom()
viz.line([0], [0], win='Train Loss', opts=dict(title='Train Loss'))
viz.line([0], [0], win='Val ACC', opts=dict(title='Val ACC'))


for epoch in range(historical_epoch+1, n_epoches+1):

    model.train()                           # 训练模式。这会启用 Dropout 等
    total_loss = 0.

    for index, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch{epoch} Train')):
        data = data.to(device)              # 数据
        target = target.to(device)          # 标签
        scores = model(data)                # 前向传播过程
        loss = criterion(scores, target)    # 输入本轮的 前向结果和标签 计算损失（交叉熵）
        optim.zero_grad()                   # 清除上一步中计算的梯度
        loss.backward()                     # 反向传播
        optim.step()                        # 更新参数(Adam)
        total_loss += loss.item()           # 累计损失并将当前 epoch 的每个batch的平均损失 (total_loss / (index+1)) 发送到名为 'Train Loss' 的窗口中进行可视化，帮助你监控训练过程是否顺利。
        viz.line([total_loss / (index+1)],
                 [(epoch-1) * len(train_loader) + index + 1],
                 win='Train Loss',
                 update='append')

    model.eval()                            # 评估模式。这会关闭 Dropout 等只在训练时使用的层，确保评估结果的稳定性和一致性。
    n_correct = 0

    with torch.no_grad():                   # 验证阶段不要计算梯度。因为在验证阶段不需要反向传播。
        for data, target in tqdm(val_loader, desc=f'Epoch{epoch} Val'): #验证集 val_loader
            data = data.to(device)          
            target = target.to(device)

            scores = model(data)            # 前向传播
            pred = scores.argmax(dim=1)     # 前向结果scores中评分最大的元素代表了模型预测的类别

            n_correct += torch.eq(pred, target).sum().item()        # 得出和标签相同的预测结果的个数

            # 一次取batchsize=16的一个batch的前show_size个预测结果和实际图像，并可视化
            pred_labels = [classes[int(idx)]                        
                           for idx in pred[:show_size].cpu().numpy().flatten()]
            pred_str = ''
            for label in pred_labels:
                pred_str += label + '<br><br>'

            viz.images(untrans(data[:show_size]), nrow=1, win='Predict Image',
                       opts=dict(title='Predict Image'))
            viz.text(pred_str, win='Predict', opts=dict(title='Predict'))
            # caption_str = '<br>'.join(pred_labels)
            # # 
            # viz.images(
            #     untrans(data[:show_size]),
            #     nrow=1,
            #     win='Live Predictions',  # 使用一个统一的窗口ID
            #     opts=dict(
            #         title='Live Predictions', # 设置窗口标题
            #         caption=caption_str       # 将预测结果作为图片下方的说明文字
            #     )
            # )


    acc = n_correct / len(val_set)          # 在整个验证集上的准确率
    viz.line([acc], [epoch],                # 当前 epoch 的验证准确率 acc 绘制到名为 'Val ACC' 的图表中
            win='Val ACC', update='append')

    torch.save(model.state_dict(            # 保存本轮训练出的模型参数
    ), f'weights/VIT_depth{depth}_{n_heads}heads_{n_classes}classes_epoch{epoch}_acc{acc:.2%}.pth')
    