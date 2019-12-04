# Class RCNN definition
import torch
import torch.nn as nn
import utils.config as cfg

class BinLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BinLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, imgH, imgW, n_hidden = 2, n_class=cfg.n_class, init_weights=True):
        super(CRNN, self).__init__()
        self.cfg = [64, 'M22', 128, 'M22', 256, 256, 'M12', 512, 'bc', 512, 'bc', 'M12', 512]
        self.out_channels = [64, 128, 256, 256, 512, 512, 512]  # output channels of each conv layers
        self.layers = self.make_layers(self.cfg)
        self.rnn = nn.Sequential(
            BinLSTM(512, n_hidden, n_hidden),
            BinLSTM(n_hidden, n_hidden, n_class))
        
    def make_layers(self, cfg):
        layers = []
        in_channels = 1  # in this example, channel is 1
        layer_count = -1  # count conv layers
        for v in cfg:
            if v == 'M22':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'M12':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1))]
            elif v == 'bc':
                layers += [nn.BatchNorm2d(self.out_channels[layer_count]), nn.ReLU(inplace=True)]
            else:
                if layer_count == 5:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=2, padding=0)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                layer_count += 1
                in_channels = v
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        b,c,h,w = x.size()
        # print(b,c,h,w)  100,512,1,26
        # convert it to   26,100,512
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # squeeze second dimension --> 100 512 26
        x = x.permute(2, 0, 1)  # 26 100 512
        # input to the lstm MUST be 3D: 
        # The first axis is the sequence itself, the second indexes mini-batch size
        x = self.rnn(x)

        return x