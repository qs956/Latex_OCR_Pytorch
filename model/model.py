import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,stride=1,padding=1)
        self.maxpool1 = nn.MaxPool2d(2,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(2,stride=1,padding=1)
        
        self.conv3 = nn.Conv2d(128,256,3,stride=1,padding=1)
        
        self.conv4 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.maxpool3 = nn.MaxPool2d((2,1),stride=(2,1),padding=(1,0))
        
        self.conv5 = nn.Conv2d(256,512,3,stride=1,padding=1)
        self.maxpool4 = nn.MaxPool2d((1,2),stride=(1,2),padding=(0,1))
        
        self.conv6 = nn.Conv2d(512,512,3)
    def forward(self,img):
        layer1 = self.conv1(img)
        layer1 = self.maxpool1(layer1)
        layer1 = F.relu(layer1)
        
        layer2 = self.conv2(layer1)
        layer2 = self.maxpool2(layer2)
        layer2 = F.relu(layer2)
        
        layer3 = self.conv3(layer2)
        layer3 = F.relu(layer3)
        
        layer4 = self.conv4(layer3)
        layer4 = self.maxpool3(layer4)
        layer4 = F.relu(layer4)
        
        layer5 = self.conv5(layer4)
        layer5 = self.maxpool4(layer5)
        layer5 = F.relu(layer5)
        
        layer6 = self.conv6(layer5)
        layer6 = F.relu(layer6)

        #位置嵌入
        layer7 = layer6.permute(0,2,3,1)
        layer7 = self.add_timing_signal_nd(layer7)
        layer7 = layer7.permute(0,3,1,2)

        # layer7 = layer7.contiguous()
        return layer7
#修改自:
# https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py
    def add_timing_signal_nd(self, x, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a difft
        frequency and phase in one of the positional dimensions.

        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.

        The use of relative position is possible because sin(a+b) and cos(a+b) can
        be experessed in terms of b, sin(a) and cos(a).

        x is a Tensor with n "positional" dimensions, e.g. one dimension for a
        sequence or two dimensions for an image

        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels // (n * 2). For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.

        Args:
            x: a Tensor with shape [batch, d1 ... dn, channels]
            min_timescale: a float
            max_timescale: a float

        Returns:
            a Tensor the same shape as x.

        """
        static_shape = list(x.shape) # [2, 512, 50, 120]
        num_dims = len(static_shape) - 2  # 2
        channels = x.shape[-1]  # 512 
        num_timescales = channels // (num_dims * 2)  # 512 // (2*2) = 128
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.FloatTensor([i for i in range(num_timescales)]) * -log_timescale_increment)  # len == 128
        for dim in range(num_dims):  # dim == 0; 1
            length = x.shape[dim + 1]  # 要跳过前两个维度
            position = torch.arange(length).float()  # len == 50
            scaled_time = torch.reshape(position,(-1,1)) * torch.reshape(inv_timescales,(1,-1))
            #[50,1] x [1,128] = [50,128]
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1).to(device)  # [50, 256]
            prepad = dim * 2 * num_timescales  # 0; 256
            postpad = channels - (dim + 1) * 2 * num_timescales  # 512-(1;2)*2*128 = 256; 0
            signal = F.pad(signal, (prepad,postpad,0,0))  # [50, 512]
            for _ in range(1 + dim):  # 1; 2
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):  # 1, 0
                signal = signal.unsqueeze(-2)
            x += signal  # [1, 14, 1, 512]; [1, 1, 14, 512]
        return x

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim).to(device)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):
        """
        根据编码器的图片输出初始化解码器中LSTM层状态
        :param encoder_out: 编码器的输出 (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoder的输出 (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: caption的编码张量,不是字符串！ (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)#这里和普通的resnet输出的不同，resnet是最后一个维度是C
        vocab_size = self.vocab_size

        # 把特征图展平作为上下文向量
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        # print('sort_ind',sort_ind,'encoder_out',encoder_out.shape,'encoder_captions',encoded_captions.shape)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初始化LSTM状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 我们一旦生成了<end>就已经完成了解码
        # 因此需要解码的长度实际是 lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # 新建两个张量用于存放 word predicion scores and alphas
        global device
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # 在每一个时间步根据解码器的前一个状态以及经过attention加权后的encoder输出进行解码
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind