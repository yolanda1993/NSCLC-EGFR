import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint



class Muti_Task_Seg(nn.Module):
    def __init__(self):
        super().__init__()
        # 假设Seg_net在PyTorch中的实现同样接收PET和CT数据的列表作为输入，并返回相应输出
        self.Seg_net = seg_net()
        # 假设DenseNet在PyTorch中的实现接收PET、CT和Seg_pred作为输入，并返回相应输出
        self.DenseNet1 = DenseNet()
        self.DenseNet2 = DenseNet()
        # 假设classifier在PyTorch中的实现接收对应的特征以及Clinic数据，返回Survival_pred
        self.classifier1 = classifier(num_node=64, clinic_size=19)
        self.classifier2 = two_classifier(num_node=64,Text_size=19)

    def forward(self, pet, ct, clinic, text):
        """
        pet: 对应PET图像数据，形状应为(batch_size, *vol_size, self.pet_channel)
        ct: 对应CT图像数据，形状应为(batch_size, *vol_size, self.ct_channel)
        clinic: 对应临床数据，形状应为(batch_size, Clinic_size)
        """
        # 通过Seg_net处理PET和CT数据
        seg_pred1, x_out1, x_out2, x_out3, x_out4, x_out5 = self.Seg_net(pet, ct)
        # 通过DenseNet处理PET、CT和seg_pred数据
        x_out6, x_out7, x_out8 = self.DenseNet1([pet, ct, seg_pred1])
        out6, out7, out8 = self.DenseNet2([pet, ct, seg_pred1])
        x_in1 = [x_out1, x_out2, x_out3, x_out4, x_out5]
        x_in2 = [x_out6, x_out7, x_out8]
        x_in3 = [out6, out7, out8]
        EGFR_Pred = self.classifier2(x_in1, x_in2, text, clinic)
        # 通过classifier得到Survival_pred
        survival_pred = self.classifier1(x_in1, x_in3, text, clinic)

        return seg_pred1, EGFR_Pred, survival_pred


class two_classifier(nn.Module):
    def __init__(self, num_node, Text_size, droprate=0.5):
        super().__init__()
        self.Dense1 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(124,num_node),
            nn.ReLU()
        )

        self.Dense2 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(112,num_node),
            nn.ReLU()
        )

        self.Dense3 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(2*num_node+Text_size,1),
            nn.Sigmoid()
        )


    def forward(self, x_in1, x_in2, text, clinic):
        """
        x_in1: 对应原代码中的x_in1列表，包含多个特征张量
        x_in2: 对应原代码中的x_in2列表，包含多个特征张量
        clinic: 对应临床数据张量
        """
        # 处理x_in1
        x_concat1 = []
        for feat in x_in1:
            # 使用自适应平均池化模拟GlobalAveragePooling3D，这里假设输入是三维张量情况
            x = F.adaptive_avg_pool3d(feat, (1, 1, 1))
            x = x.view(feat.size(0), -1)  # 展平
            x_concat1.append(x)
        x_concat1 = torch.cat(x_concat1, dim=1)

        # 处理x_in2
        x_concat2 = []
        for feat in x_in2:
            x = F.adaptive_avg_pool3d(feat, (1, 1, 1))
            x = x.view(feat.size(0), -1)
            x_concat2.append(x)
        x_concat2 = torch.cat(x_concat2, dim=1)

        # 应用Dropout和全连接层处理x_concat1
        x_1 = self.Dense1(x_concat1)
        x_2 = self.Dense2(x_concat2)
        x = torch.cat([x_1,x_2,text,clinic],dim=1)
        EGFR_pred = self.Dense3(x)

        return EGFR_pred

class classifier(nn.Module):
    def __init__(self, num_node, clinic_size,droprate=0.5):
        super().__init__()
        self.Dense1 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(124,num_node),
            nn.ReLU()
        )

        self.Dense2 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(112,num_node),
            nn.ReLU()
        )

        self.Dense3 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(2*num_node+clinic_size,64)
        )

        self.Dense4 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(64,1)
        )

    def forward(self, x_in1, x_in2, text, clinic):
        """
        x_in1: 对应原代码中的x_in1列表，包含多个特征张量
        x_in2: 对应原代码中的x_in2列表，包含多个特征张量
        clinic: 对应临床数据张量
        """
        # 处理x_in1
        x_concat1 = []
        for feat in x_in1:
            # 使用自适应平均池化模拟GlobalAveragePooling3D，这里假设输入是三维张量情况
            x = F.adaptive_avg_pool3d(feat, (1, 1, 1))
            x = x.view(feat.size(0), -1)  # 展平
            x_concat1.append(x)
        x_concat1 = torch.cat(x_concat1, dim=1)

        # 处理x_in2
        x_concat2 = []
        for feat in x_in2:
            x = F.adaptive_avg_pool3d(feat, (1, 1, 1))
            x = x.view(feat.size(0), -1)
            x_concat2.append(x)
        x_concat2 = torch.cat(x_concat2, dim=1)

        # 应用Dropout和全连接层处理x_concat1
        x_1 = self.Dense1(x_concat1)
        x_2 = self.Dense2(x_concat2)
        x = torch.cat([x_1,x_2,text,clinic],dim=1)
        x_3 = self.Dense3(x)
        survival_pred = self.Dense4(x_3)

        return survival_pred


class dense_factor(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm3d(input_channels),
            nn.ReLU(),
            nn.Conv3d(input_channels,64,(1,1,1),padding="same"),
            nn.Dropout(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64,16,(3,3,3),padding="same"),
            nn.Dropout(0.05)
        )

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return x

class DenseBlock(nn.Module):
    def __init__(self, numlayers, input_channels):
        super(DenseBlock, self).__init__()
        self.numlayers = numlayers
        self.layers = nn.ModuleList()
        self.in_channels = input_channels

        for _ in range(numlayers):
            dense_factor_module = dense_factor(self.in_channels)
            self.layers.append(dense_factor_module)
            self.in_channels += 16

    def forward(self, inputs):
        concatenated_inputs = inputs
        for layer in self.layers:
            x = layer(concatenated_inputs)
            concatenated_inputs = torch.cat([concatenated_inputs, x], dim=1)
        return concatenated_inputs



class DenseNet(nn.Module):
    def __init__(self,droprate=0.05):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3,24,(3,3,3),stride=(2,2,2)),
            nn.Dropout(droprate),
            nn.MaxPool3d((3,3,3),(2,2,2))
        )

        self.batch1 = nn.Sequential(
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.batch2 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.batch3 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.block1 = DenseBlock(4,24)

        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(88),
            nn.ReLU(),
            nn.Conv3d(88,16,(1,1,1),padding="same"),
            nn.Dropout(droprate)
        )

        self.block2 = DenseBlock(8,16)

        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(144),
            nn.ReLU(),
            nn.Conv3d(144, 32, (1, 1, 1), padding="same"),
            nn.Dropout(droprate)
        )

        self.block3 = DenseBlock(16, 32)

        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(288),
            nn.ReLU(),
            nn.Conv3d(288, 64, (1, 1, 1), padding="same"),
            nn.Dropout(droprate)
        )


    def forward(self,inputs):
        x_in = torch.cat(inputs,dim=1)

        x = self.conv1(x_in)
        x = self.block1(x)
        x = self.conv2(x)


        x_out1 = self.batch1(x)
        # x_out1 = nn.BatchNorm3d(16).to("cuda")(x)
        # x_out1 = nn.ReLU().to("cuda")(x_out1)

        x = nn.AvgPool3d((2,2,2),stride=(2,2,2))(x)
        x = self.block2(x)
        x = self.conv3(x)

        x_out2 = self.batch2(x)
        # x_out2 = nn.BatchNorm3d(32).to("cuda")(x)
        # x_out2 = nn.ReLU().to("cuda")(x_out2)

        x = nn.AvgPool3d((2,2,2),stride=(2,2,2))(x)

        x = self.block3(x)
        x = self.conv4(x)

        x_out3 = self.batch3(x)
        # x_out3 = nn.BatchNorm3d(64).to("cuda")(x)
        # x_out3 = nn.ReLU().to("cuda")(x_out3)

        return x_out1,x_out2,x_out3

class seg_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv3d(2,8,(1,1,1),stride=(1,1,1),padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.module2 = nn.Sequential(
            nn.Conv3d(2,8,(3,3,3),stride=(1,1,1),padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.module3 = nn.Sequential(
            nn.Conv3d(8, 8, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.module4 = nn.Sequential(
            nn.Conv3d(8, 16, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module5 = nn.Sequential(
            nn.Conv3d(8, 16, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module6 = nn.Sequential(
            nn.Conv3d(16, 16, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module7 = nn.Sequential(
            nn.Conv3d(16, 32, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module8 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module9 = nn.Sequential(
            nn.Conv3d(32, 32, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module10 = nn.Sequential(
            nn.Conv3d(32, 64, (1,1,1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module11 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module12 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module13 = nn.Sequential(
            nn.Conv3d(64, 128, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.module14 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.module15 = nn.Sequential(
            nn.Conv3d(128, 128, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.module16 = nn.Sequential(
            nn.Conv3d(192, 64, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module17 = nn.Sequential(
            nn.Conv3d(192, 64, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module18 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.module19 = nn.Sequential(
            nn.Conv3d(96, 32, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module20 = nn.Sequential(
            nn.Conv3d(96, 32, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module21 = nn.Sequential(
            nn.Conv3d(32, 32, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.module22 = nn.Sequential(
            nn.Conv3d(48, 16, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module23 = nn.Sequential(
            nn.Conv3d(48, 16, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module24 = nn.Sequential(
            nn.Conv3d(16, 16, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.module25 = nn.Sequential(
            nn.Conv3d(24, 8, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.module26 = nn.Sequential(
            nn.Conv3d(24, 8, (3, 3, 3), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.module27 = nn.Sequential(
            nn.Conv3d(8, 8, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.Seg_out = nn.Sequential(
            nn.Conv3d(8, 2,(1, 1, 1),stride=(1,1,1),padding="same"),
            nn.ReLU()
        )
        self.out1 = nn.Sequential(
            nn.Conv3d(8, 4, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(4),
            nn.ReLU()
        )
        self.out2 = nn.Sequential(
            nn.Conv3d(16, 8, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.out3 = nn.Sequential(
            nn.Conv3d(32, 16, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.out4 = nn.Sequential(
            nn.Conv3d(64, 32, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.out5 = nn.Sequential(
            nn.Conv3d(128, 64, (1, 1, 1), stride=(1,1,1), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool3d(kernel_size = 2)
        self.up_sample = nn.Upsample(scale_factor = 2)

    def forward(self, pet, ct):
        x_in = torch.cat([pet, ct],dim=1)

        res_1 = self.module1(x_in)
        x = self.module2(x_in)
        res_2 = x+res_1
        x = self.module3(res_2)
        x_1 = x+res_2

        x_in = self.max_pool(x_1)

        res_1 = self.module4(x_in)
        x = self.module5(x_in)
        res_2 = x+res_1
        x = self.module6(res_2)
        x_2 = x+res_2

        x_in = self.max_pool(x_2)

        res_1 = self.module7(x_in)
        x = self.module8(x_in)
        res_2 = x + res_1
        x = self.module9(res_2)
        res_3 = x+res_2
        x = self.module9(res_3)
        x_3 = x + res_3

        x_in = self.max_pool(x_3)

        res_1 = self.module10(x_in)
        x = self.module11(x_in)
        res_2 = x + res_1
        x = self.module12(res_2)
        res_3 = x + res_2
        x = self.module12(res_3)
        x_4 = x + res_3

        x_in = self.max_pool(x_4)
        res_1 = self.module13(x_in)
        x = self.module14(x_in)
        res_2 = x + res_1
        x = self.module15(res_2)
        res_3 = x + res_2
        x = self.module15(res_3)
        res_4 = x + res_3
        x = self.module15(res_4)
        x_5 = x + res_4

        #Upsample
        x = self.up_sample(x_5)
        x_in = torch.cat([x,x_4], dim=1)

        res_1 = self.module16(x_in)
        x = self.module17(x_in)
        res_2 = x + res_1
        x = self.module18(res_2)
        res_3 = x + res_2
        x = self.module18(res_3)
        x = x + res_3

        x = self.up_sample(x)
        x_in = torch.cat([x,x_3], dim=1)

        res_1 = self.module19(x_in)
        x = self.module20(x_in)
        res_2 = x + res_1
        x = self.module21(res_2)
        res_3 = x + res_2
        x = self.module21(res_3)
        x = x + res_3

        x = self.up_sample(x)
        x_in = torch.cat([x, x_2], dim=1)

        res_1 = self.module22(x_in)
        x = self.module23(x_in)
        res_2 = x + res_1
        x = self.module24(res_2)
        x = x + res_2

        x = self.up_sample(x)
        x_in = torch.cat([x, x_1], dim=1)

        res_1 = self.module25(x_in)
        x = self.module26(x_in)
        res_2 = x + res_1
        x = self.module27(res_2)
        x = x + res_2

        x = self.Seg_out(x)
        Seg_pred = x[:, 0:1, ...]
        x_out1 = self.out1(x_1)
        x_out2 = self.out2(x_2)
        x_out3 = self.out3(x_3)
        x_out4 = self.out4(x_4)
        x_out5 = self.out5(x_5)

        return Seg_pred, x_out1, x_out2, x_out3, x_out4, x_out5























