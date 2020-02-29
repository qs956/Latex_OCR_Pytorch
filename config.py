#数据路径
data_name = 'small_test'  # 模型名称,仅在保存的时候用到
vocab_path = './data/small/vocab.json'
train_set_path = './data/small/train.json'
val_set_path = './data/small/val.json'


# 模型参数
emb_dim = 80  # 词嵌入维数80
attention_dim = 256  # attention 层维度 256
decoder_dim = 512  # decoder维度 128
dropout = 0.5


# 训练参数
start_epoch = 0
epochs = 100  # 不触发早停机制时候最大迭代次数
epochs_since_improvement = 0  # 用于跟踪在验证集上分数没有提高的迭代次数
batch_size = 2 #训练解批大小
test_batch_size = 6 #验证集批大小
encoder_lr = 1e-4  # 学习率
decoder_lr = 4e-4  # 学习率
grad_clip = 5.  # 梯度裁剪阈值
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_score = 0.  # 目前最好的 score 
print_freq = 5  # 状态的批次打印间隔
checkpoint = None  # checkpoint文件目录(用于断点继续训练)
save_freq = 2 #保存的间隔