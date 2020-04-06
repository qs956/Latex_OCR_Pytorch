Latex_OCR_Pytorch

主要是这个版本的Pytorch实现:

[LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)

感谢@LinXueyuanStdio 的工作以及指导.本项目与上述项目思路一致，但在实现上修改了一些地方:

* 数据集的重新定义,但使用原有类似的预处理方式
* 代码简化，目前仅保留主要部分，命令行控制等在后续补充
* 内存优化，相对较少的内存需求，支持较大批量的训练。但批大小一样的情况下实测速度提高不大
* 使用Checkpoint特性，在编码过程中出现OOM则自动进行分段计算
* 在训练时候采用贪婪策略，Beam Search仅在推断时候采用
* Scheduled Sampling策略

Follow these paper:

1. [Show, Attend and Tell(Kelvin Xu...)](https://arxiv.org/abs/1502.03044)
2. [Harvard's paper and dataset](http://lstm.seas.harvard.edu/latex/)

Follow these tutorial:

1. [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html).
2. [a PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

## 环境
1. Python >= 3.6

2. Pytorch >= 1.2

## 数据
使用[LinXueyuanStdio/Data-for-LaTeX_OCR](https://github.com/LinXueyuanStdio/Data-for-LaTeX_OCR) 数据集,原仓库较大,后续提供打包下载.

已包括上述仓库中small数据集
印刷体数据全集[百度云](https://pan.baidu.com/s/1xIsgHDhVu85L8cGdqqG7kw ) 提取码：tapj [Google Drive](https://drive.google.com/open?id=1THp_O7uwavcjsnQXsxx_JPvYn9-gml7T)
自己划分的混合CROHME2011,2012数据集[Google Drive](https://drive.google.com/open?id=1KgpAzA7k8ayjPTstin6M8ykGsW8GR9bu)


## 训练模型
在自己划分CROHME2011,2012数据集上使用以下参数的训练模型[Google Drive](https://drive.google.com/open?id=1_geqm9a86TJKK9RpZ39d9X5655s4NXa9)
emb_dim = 30 
attention_dim = 128
decoder_dim = 128
后续补充模型测试结果以及colab

## 数据格式

数据集文件生成参考[utils.py](./model/utils.py)的get_latex_ocrdata

数据集文件json格式,包括训练集文件,验证集文件,字典文件.

字典格式:

python字典(符号——编号)的json储存

数据集格式:

```
​```shell
训练/验证数据集
├── file_name1 图片文件名 str
│   ├── img_path:文件路径(到文件名,含后缀) str
│   ├── size:图片尺寸 [长,宽] list
│   ├── caption:图片代表的公式,各个符号之间必须要空格分隔 str
│   └── caption_len:len(caption.split()) int
|   ...
eg:
{
"0.png":
    {
    "img_path":"./mydata/0.png",
    "size":[442,62],
    "caption":"\frac { a + b } { 2 }",
    "caption_len":9,
    }
"2.png":...
}

​```
```

图片预处理

参考dataloader/data_turn主要进行以下操作

1. 灰度化
2. 裁剪公式部分
3. 上下左右各padding 8个像素
4. `[可选]`下采样


## To do

- [ ]  推断部分
- [ ] Attention层的可视化
- [x] 预训练模型
- [x] 打包的训练数据
- [ ] perplexity指标