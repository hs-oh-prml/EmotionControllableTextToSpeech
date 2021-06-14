
import torch
import torch.nn as nn

class SERModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        # conv block
        # self.conv2Dblock = nn.Sequential(
            # # 1. conv block
            # nn.Conv2d(in_channels=1,
            #            out_channels=16,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.3),
            # # 2. conv block
            # nn.Conv2d(in_channels=16,
            #            out_channels=32,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # nn.Dropout(p=0.3),
            # 3. conv block
            # nn.Conv2d(in_channels=32,
            #            out_channels=32,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # nn.Dropout(p=0.3),
            # 4. conv block
            # nn.Conv2d(in_channels=32,
            #            out_channels=32,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # nn.Dropout(p=0.3)
        # )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=40, nhead=2, dim_feedforward=256, dropout=0.2, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)

        # Linear softmax layer
        self.out_linear = nn.Linear(40, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # conv embedding
        # conv_embedding = self.conv2Dblock(x) #(b,channel,freq,time)
        # conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension
        # print(conv_embedding.shape)

        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        # concatenate
        # complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        complete_embedding = transf_embedding
        # print("complete_embedding:", complete_embedding.shape)
        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)

        # print(output_softmax.shape)
        return output_logits, output_softmax
    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
        