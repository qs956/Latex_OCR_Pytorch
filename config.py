#数据路径
data_name = 'small_test'  # 模型名称,仅在保存的时候用到
vocab_path = 'vocab.json'

# 模型参数
emb_dim = 30  # 词嵌入维数80
attention_dim = 8  # attention 层维度 256
decoder_dim = 4  # decoder维度 128
dropout = 0.5

'''
如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
'''
# 训练参数
start_epoch = 0
epochs = 100  # 不触发早停机制时候最大迭代次数
epochs_since_improvement = 0  # 用于跟踪在验证集上 BLEU分数没有提高的迭代次数
batch_size = 2 #训练解批大小
test_batch_size = 6 #验证集批大小
encoder_lr = 1e-4  # 学习率
decoder_lr = 4e-4  # 学习率
grad_clip = 5.  # 梯度裁剪阈值
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_score = 0.  # 目前最好的 score 
print_freq = 1  # 状态的批次打印间隔
checkpoint = None  # checkpoint文件目录(用于断点继续训练)
save_freq = 2 #保存的间隔