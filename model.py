import math
import torch
from torch import nn

class NewGenerator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(NewGenerator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        block9 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.block9 = nn.Sequential(*block9)

        self.block10 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.attention1 = OptimizedSelfAttention(64)
        self.attention2 = OptimizedSelfAttention(64)
        self.attention3 = OptimizedSelfAttention(64)


    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block2 = self.attention1(block2)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block4 = self.attention2(block4)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block6 = self.attention3(block6)
        block7 = self.block7(block6)
        block9 = self.block9(block7+block1)
        block10 = self.block10(block9)

        return (torch.tanh(block10) + 1) / 2




class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.transformer = Encoder(512, image_size,256, 16, 16)

        self.post_process = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        conv_output = self.conv_net(x)

        transformer_output = self.transformer(conv_output)

        post_process_output = self.post_process(conv_output)

        # 可以根据需要调整合并方式，这里简单相加
        combined_output = post_process_output.view(batch_size) + transformer_output

        return self.tanh(combined_output)




class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim =-1)


    def forward(self, x):
        batch_size, channels, height, width = x.size()

        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        chunk_size = 64
        num_chunks = (height * width + chunk_size - 1) // chunk_size
        attention_chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, height * width)
            q_chunk = q[:, start:end, :]
            attn_chunk = torch.bmm(q_chunk, k)
            attention_chunks.append(attn_chunk)

        attention = self.softmax(torch.cat(attention_chunks, dim=1))

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out



class OptimizedSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(OptimizedSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # 线性自注意力计算
        k = torch.softmax(k, dim=-1)
        # 交换 k 和 v 的乘法顺序
        kv = torch.bmm(k, v.permute(0, 2, 1))
        attn_output = torch.bmm(q, kv)

        attn_output = attn_output.view(batch_size, channels, height, width)
        out = self.gamma * attn_output + x
        return out




# class OldGenerator(nn.Module):
#     def __init__(self, scale_factor):
#         upsample_block_num = int(math.log(scale_factor, 2))
#
#         super(OldGenerator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#
#         block9 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         self.block9 = nn.Sequential(*block9)
#
#         self.block10 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block9 = self.block9(block7+block1)
#         block10 = self.block10(block9)
#
#         return (torch.tanh(block10) + 1) / 2


#
# class OldGenerator(nn.Module):
#     def __init__(self, scale_factor):
#         #upsample_block_num = int(math.log(scale_factor, 2))
#
#         super(OldGenerator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         # block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         block8 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#         self.block8 = nn.Sequential(*block8)
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block8 = self.block8(block1 + block7)
#
#         return (torch.tanh(block8) + 1) / 2

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2

class MutlHeadAttention(nn.Module):
    def __init__(self,embed_dim,head_num,head_dim):
        super(MutlHeadAttention,self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.head_num = head_num

        self.values = nn.Linear(head_dim,head_dim)
        self.keys = nn.Linear(head_dim,head_dim)
        self.queries = nn.Linear(head_dim,head_dim)
        self.fc = nn.Linear(head_dim*head_num, embed_dim)

    def forward(self, value, key, query,mask = None):
        batch_size = value.size()[0]
        seq_len = value.size()[1]
        value, key, query = (value.reshape(batch_size, seq_len, self.head_num, self.head_dim),
                             key.reshape(batch_size, seq_len, self.head_num, self.head_dim),
                             query.reshape(batch_size, seq_len, self.head_num, self.head_dim)
                             )

        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)

        enegy = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            enegy = enegy.masked_fill(mask == 0, -1e9)

        attention = torch.matmul(torch.softmax(enegy/self.head_dim**(0.5), dim=-1),value).reshape(batch_size, seq_len, -1)
        return self.fc(attention)


class PositionalEncoding(nn.Module):
    def __init__(self,seq_len,dim_size):
        super(PositionalEncoding,self).__init__()
        po = torch.zeros(seq_len,dim_size)
        position = torch.arange(0,seq_len).unsqueeze(1)

        div = torch.exp(-(torch.arange(0,dim_size,2).float()/dim_size)*math.log(10000))

        po[:,0::2] = torch.sin(position*div)

        po[:,1::2] = torch.cos(position*div)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        po = po.unsqueeze(0).to(device)
        self.register_buffer('po',po)

    def forward(self,x):
        return x+self.po


class FeedForward(nn.Module):
    def __init__(self,input_size,output_size):
        super(FeedForward,self).__init__()
        self.fc1 = nn.Linear(input_size,256)
        self.fc2 = nn.Linear(256,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.fc1(x)
        x= self.relu(x)
        x= self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, head_num,head_dim):
        super(EncoderLayer,self).__init__()
        self.attention = MutlHeadAttention(embed_dim,head_num,head_dim)
        self.normalization = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, embed_dim)

    def forward(self,x):
        x = self.attention(x,x,x)
        x = self.normalization(x)
        x = self.feedforward(x)
        return x


class Encoder(nn.Module):
    def __init__(self,in_channel,image_size,embed_dim, head_num,head_dim):
        super(Encoder,self).__init__()
        self.num_patches = (image_size // 4)**2
        self.embedding = PatchEmbedding(image_size, 4, in_channel, embed_dim)
        self.encode_layer = nn.Sequential(
            EncoderLayer(embed_dim,head_num,head_dim),
            EncoderLayer(embed_dim, head_num, head_dim)
        )
        self.linear1 = nn.Linear(embed_dim,1)
        self.relu1 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.embedding(x)
        x = self.encode_layer(x)
        x = self.relu1(self.linear1(x).squeeze(2))
        x = x.mean(dim = -1)
        return x


#残差模块： 经过俩个卷积层和批归一化层，进行特征提取
#用PReLU激活函数连接
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


#亚像素卷积网络
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale) #像素重排模块，图像大小上采样为原来的平方倍,仅对像素进行重拍，输入channel 是r的平方被
        self.prelu = nn.PReLU() #参数化的线性整流激活函数： 和一般ReLU的区别，在于处理负数：
                                #PReLU ： max(0,x)+a*min(0,x) : 在输入为负数时，多了一个可学习的参数a

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x








