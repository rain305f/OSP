import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from torch.autograd import Variable
# from utils import cuda
from kmeans_pytorch import kmeans


logger = logging.getLogger(__name__)

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
    

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)



class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0,gamma=0.8,input_channels=3):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.gamma = gamma
        self.conv1 = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3]*2, block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3]*2, momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.grads_dict = {}
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    
    
    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std
    
    '''
    input:
        id_feats : (bz,128)
        ood_feat_bank:(bz,128)
    output:
        debias_id_feats : (bz,128)
        gamma=1, debias_id_feats 与 ood_feats正交
    '''
    def debias_ood_classwise( self, id_feats, ood_feats_bank , mode = 'random' , gamma=0.8 , args=None , ood_centers=None, label=None):
        ood_feats = torch.zeros(0,128).to(args.device)
        tot_class_ptr =[0]*args.tot_class
        with torch.no_grad():
            if label is not None:
                for index in range(label.shape[0]): 
                    if ood_feats_bank[ label[index].item() ].shape[0] == 0:
                        for jj in range(args.tot_class):
                            if ood_feats_bank[jj].shape[0] != 0:
                                break
                        if jj < args.tot_class-1:
                            idx = torch.randint(0, ood_feats_bank[jj].shape[0] , (1,) ) # 随机选择一个index
                            ood_feats = torch.cat((ood_feats,ood_feats_bank[jj][idx] ) ,dim=0)
                        else:
                            ood_feats = torch.cat((ood_feats,id_feats[index].clone().detach().unsqueeze(dim=0) ) ,dim=0) # 如果bank为空，对自己作正交分解
                    else:
                        idx = torch.randint( 0, ood_feats_bank[ label[index].item() ].shape[0] , (1,) ) # 随机选择一个index
                        ood_feats = torch.cat((ood_feats,ood_feats_bank[label[index].item()][idx] ) ,dim=0)
            else:
                logits = self.fc(id_feats)
                label = torch.max(logits,dim=-1)[1] # 先过一层fc，得伪标签
                for index in range(label.shape[0]): 
                    if ood_feats_bank[ label[index].item() ].shape[0] == 0:
                        for jj in range(args.tot_class):
                            if ood_feats_bank[jj].shape[0] != 0:
                                break
                        if jj < args.tot_class-1:
                            idx = torch.randint(0, ood_feats_bank[jj].shape[0] , (1,) ) # 随机选择一个index
                            ood_feats = torch.cat((ood_feats,ood_feats_bank[jj][idx] ) ,dim=0)
                        else:
                            ood_feats = torch.cat((ood_feats,id_feats[index].clone().detach().unsqueeze(dim=0) ) ,dim=0) # 如果bank为空，对自己作正交分解
                    else:
                        idx = torch.randint(0, ood_feats_bank[ label[index].item() ].shape[0] , (1,) ) # 随机选择一个index
                        ood_feats = torch.cat((ood_feats , ood_feats_bank[label[index].item()][idx] ) ,dim=0)
                        

        id_mm =  [ ( n *( m@n.t()/torch.norm(n, p=2).pow(2)) ).unsqueeze(dim=0) for (m,n) in zip(id_feats,ood_feats)]

        id_feats_plus = torch.cat(id_mm,dim=0) #(bz,128)
        debias_id_feats = id_feats - gamma*id_feats_plus  #(bz,128)    
        
        return debias_id_feats


    '''
    input:
        id_feats : (bz,128)
        ood_feat_bank:(bz,128)
    output:
        debias_id_feats : (bz,128)
        gamma=1, debias_id_feats 与 ood_feats正交
    '''
    def debias_ood(self, id_feats , ood_feats_bank, mode = 'random',gamma=0.8, args=None ,ood_centers=None):
        if mode == 'random':
            bank_len = ood_feats_bank.shape[0]
            id_len = id_feats.shape[0]
            perm = torch.randperm(bank_len)[:id_len]
            ood_feats = ood_feats_bank[perm]

        elif mode == 'sim':
            id_feats_norm = F.normalize(id_feats,p=2.0, dim=1)
            ood_feats_norm =  F.normalize(ood_feats_bank,p=2.0, dim=1)
            sim_matrix = id_feats_norm.mm(ood_feats_norm.t()) # (id_num,ood_num)
            max_sim , perm = torch.max(sim_matrix,dim=1)
            ood_feats = ood_feats_bank[perm]
            
        elif mode == 'dissim':
            id_feats_norm = F.normalize(id_feats,p=2.0, dim=1)
            ood_feats_norm =  F.normalize(ood_feats_bank,p=2.0, dim=1)
            sim_matrix = id_feats_norm.mm(ood_feats_norm.t()) # (id_num,ood_num)
            max_sim , perm = torch.min(sim_matrix,dim=1)
            ood_feats = ood_feats_bank[perm]

    
        elif mode =='proto':
            id_feats_norm = F.normalize(id_feats,p=2.0, dim=1)
            ood_feats_norm =  F.normalize(ood_centers,p=2.0, dim=1)
            sim_matrix = id_feats_norm.mm(ood_feats_norm.t()) # (id_num,ood_num)
            max_sim , perm = torch.max(sim_matrix,dim=1)
            ood_feats = ood_feats_bank[perm]      
  


        elif mode =='proto_dissim':
            id_feats_norm = F.normalize(id_feats,p=2.0, dim=1)
            ood_feats_norm =  F.normalize(ood_centers,p=2.0, dim=1)
            sim_matrix = id_feats_norm.mm(ood_feats_norm.t()) # (id_num,ood_num)
            max_sim , perm = torch.min(sim_matrix,dim=1)
            ood_feats = ood_feats_bank[perm]     
    
    
        # cal debiased
        id_mm =  [ ( n *( m@n.t()/torch.norm(n, p=2).pow(2)) ).unsqueeze(dim=0) for (m,n) in zip(id_feats,ood_feats)]

        id_feats_plus = torch.cat(id_mm,dim=0) #(bz,128)
        debias_id_feats = id_feats - gamma*id_feats_plus  #(bz,128)    

        return debias_id_feats


    def forward_ood(self, x, penci=0.9):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels*2)
        encoding  = out[:,:self.channels] 
        threshold = torch.quantile(encoding, penci)
        encoding = encoding.clip(max=threshold)
        return self.fc(encoding) , encoding

    def forward(self, x, output_feats=False, reparametrize =True , ood_feats_bank = None ,mode='random',args=None, ood_centers=None,label=None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels*2)
              
        mu  = out[:,:self.channels] 
        std = F.softplus(out[:,self.channels:]-5,beta=1)
        if reparametrize : 
            encoding = self.reparametrize_n(mu, std)
        else:
            encoding = mu

        if ood_feats_bank is not None:
            if mode == 'classwise':
                encoding = self.debias_ood_classwise(encoding, ood_feats_bank, mode, args=args, ood_centers=ood_centers, label=label, gamma=self.gamma)
            else:
                encoding = self.debias_ood(encoding, ood_feats_bank, mode, args=args, ood_centers=ood_centers, gamma=self.gamma)
        if output_feats:
            return self.fc(encoding), encoding , std
               
        return self.fc(encoding)
            


def build_wideresnet(depth, widen_factor, dropout, num_classes,gamma=0.1,input_channels=3):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                     gamma=gamma,input_channels=input_channels)
