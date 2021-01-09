# from ..utils.transform import resize, random_crop, random_flip, center_crop, meshgrid, box_iou, box_nms, change_box_order
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn.init as init


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    # import pdb; pdb.set_trace()
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out, inplace=True)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.conv50 = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1)
        # self.conv60 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer10 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer20 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer30 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7


def FPN50():
    return FPN(Bottleneck, [3,4,6,3])


class Header3D(nn.Module):
    # header to predict 3d-related instance information
    def __init__(self, num_classes=1, bins=4, num_anchors=9):
        super(Header3D, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bins = bins
        # confidence group: negative, ignore + bins
        self.confidence_head = self._make_head(self.num_anchors * bins)
        self.residual_head = self._make_head(self.num_anchors * bins * 2)
        self.dimension_head = self._make_head(self.num_anchors * 3)
        self.center_head = self._make_head(self.num_anchors * 3) # (x,y,depth) of center points

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, fm):
        confidence_pred = self.confidence_head(fm)
        residual_pred = self.residual_head(fm)
        dimension_pred = self.dimension_head(fm)
        center_pred = self.center_head(fm)
        return confidence_pred, residual_pred, dimension_pred, center_pred


class RetinaNet_Header(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes=1):
        super(RetinaNet_Header, self).__init__()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.coll_head = self._make_head(self.num_anchors * 2) # coll / non-coll prob
        self.bins = 4
        self.thrd_head = Header3D(self.num_classes, self.bins, self.num_anchors)

    def forward(self, fms):
        loc_preds = []
        cls_preds = []
        coll_preds = []
        conf_preds = []
        residual_preds = []
        dim_preds = []
        center_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            coll_pred = self.coll_head(fm)
            conf_pred, residual_pred, dim_pred, center_pred = self.thrd_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, 4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            coll_pred = coll_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, 2) 
            conf_pred = conf_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, self.bins)
            residual_pred = residual_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, self.bins*2)
            dim_pred = dim_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, 3)
            center_pred = center_pred.permute(0,2,3,1).contiguous().view(fm.size(0), -1, 3)
      
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            coll_preds.append(coll_pred)
            residual_preds.append(residual_pred)
            conf_preds.append(conf_pred)
            dim_preds.append(dim_pred)
            center_preds.append(center_pred)
        
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1), torch.cat(coll_preds, 1), torch.cat(residual_preds, 1), torch.cat(conf_preds, 1), torch.cat(dim_preds, 1), torch.cat(center_preds, 1)

    def get_cen(self):
        # get the center coordinates of target anchors
        pass 

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()



