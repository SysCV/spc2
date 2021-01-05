import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
from utils.util import weights_init


class convLSTMCell(nn.Module):
    def __init__(self, in_channels, feature_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(convLSTMCell, self).__init__()
        self.feature_channels = feature_channels
        self.conv = nn.Conv2d(in_channels + feature_channels, 4 * feature_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.apply(weights_init)
    
    def forward(self, x, hidden_states):
        hx, cx = hidden_states
        combined = torch.cat([x, hx], dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.feature_channels, dim=1)#it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * cx + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c


class convLSTM(nn.Module):
    def __init__(self, args, channel=256):
        super(convLSTM, self).__init__()
        self.args = args
        self.channel = channel
        self.lstm0 = convLSTMCell(258, self.channel, kernel_size=5, padding=2)
        self.lstm1 = convLSTMCell(258, self.channel, kernel_size=3, padding=1)
        self.lstm2 = convLSTMCell(258, self.channel, kernel_size=3, padding=1)
        self.lstm3 = convLSTMCell(258, self.channel, kernel_size=1, padding=0)
        self.lstm4 = convLSTMCell(258, self.channel, kernel_size=1, padding=0)

        # self.identifies = [nn.Conv2d(258, self.channel, kernel_size=1, stride=1) for _ in range(5)]
        # self.stairs = [nn.Conv2d(self.channel, self.channel, kernel_size=1, stride=1) for _ in range(5)]
        self.identify = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)
        self.stair = nn.Conv2d(self.channel, self.channel, kernel_size=1, stride=1)
        # self.identify0 = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)
        # self.identify1 = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)
        # self.identify2 = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)
        # self.identify3 = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)
        # self.identify4 = nn.Conv2d(258, self.channel, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x[0][0].shape[0]
        h, w = self.args.frame_height, self.args.frame_width
        
        # The last two dimensions should be: frame_height / 8, frame_width / 8
        hx0 = Variable(torch.zeros(batch_size, self.channel, int(h/8), int(w/8)))
        cx0 = Variable(torch.zeros(batch_size, self.channel, int(h/8), int(w/8)))
        # The last two dimensions should be: frame_height / 16, frame_width / 16
        hx1 = Variable(torch.zeros(batch_size, self.channel, int(h/16), int(w/16)))
        cx1 = Variable(torch.zeros(batch_size, self.channel, int(h/16), int(w/16)))
        # The last two dimensions should be: frame_height / 32, frame_width / 32
        hx2 = Variable(torch.zeros(batch_size, self.channel, int(h/32), int(w/32)))
        cx2 = Variable(torch.zeros(batch_size, self.channel, int(h/32), int(w/32)))
        # The last two dimensions should be: frame_height / 64, frame_width / 64
        hx3 = Variable(torch.zeros(batch_size, self.channel, int(h/64), int(w/64)))
        cx3 = Variable(torch.zeros(batch_size, self.channel, int(h/64), int(w/64)))
        # The last two dimensions should be: frame_height / 128, frame_width / 128
        hx4 = Variable(torch.zeros(batch_size, self.channel, int(h/128), int(w/128)))
        cx4 = Variable(torch.zeros(batch_size, self.channel, int(h/128), int(w/128)))
        
        '''
        # The last two dimensions should be: frame_height / 4, frame_width / 4
        hx0 = torch.zeros(batch_size, self.channel, int(h/4), int(w/4))
        cx0 = torch.zeros(batch_size, self.channel, int(h/4), int(w/4))
        # The last two dimensions should be: frame_height / 8, frame_width / 8
        hx1 = torch.zeros(batch_size, self.channel, int(h/8), int(w/8))
        cx1 = torch.zeros(batch_size, self.channel, int(h/8), int(w/8))
        # The last two dimensions should be: frame_height / 16, frame_width / 16
        hx2 = torch.zeros(batch_size, self.channel, int(h/16), int(w/16))
        cx2 = torch.zeros(batch_size, self.channel, int(h/16), int(w/16))
        # The last two dimensions should be: frame_height / 32, frame_width / 32
        hx3 = torch.zeros(batch_size, self.channel, int(h/32), int(w/32))
        cx3 = torch.zeros(batch_size, self.channel, int(h/32), int(w/32))
        # The last two dimensions should be: frame_height / 64, frame_width / 64
        hx4 = torch.zeros(batch_size, self.channel, int(h/64), int(w/64))
        cx4 = torch.zeros(batch_size, self.channel, int(h/64), int(w/64))
        '''
        if torch.cuda.is_available():
            hx0 = hx0.cuda()
            cx0 = cx0.cuda()
            hx1 = hx1.cuda()
            cx1 = cx1.cuda()
            hx2 = hx2.cuda()
            cx2 = cx2.cuda()
            hx3 = hx3.cuda()
            cx3 = cx3.cuda()
            hx4 = hx4.cuda()
            cx4 = cx4.cuda()
        

        for step in range(len(x)):
            hx0, cx0 = self.lstm0(x[step][0], (hx0, cx0))
            # cx0 = self.stair(self.identify(x[step][0]) + cx0)
            # cx0 = self.stair(cx0)
            hx1, cx1 = self.lstm1(x[step][1], (hx1, cx1))
            # cx1 = self.stair(self.identify(x[step][1]) + cx1)
            # cx1 = self.stair(cx1)
            hx2, cx2 = self.lstm2(x[step][2], (hx2, cx2))
            # cx2 = self.stair(self.identify(x[step][2]) + cx2)
            # cx2 = self.stair(cx2)
            hx3, cx3 = self.lstm3(x[step][3], (hx3, cx3))
            # cx3 = self.stair(self.identify(x[step][3]) + cx3)
            # cx3 = self.stair(cx3)
            hx4, cx4 = self.lstm4(x[step][4], (hx4, cx4))
            # cx4 = self.stair(self.identify(x[step][4]) + cx4)
            # cx4 = self.stair(cx4)

        return [hx0, hx1, hx2, hx3, hx4]
