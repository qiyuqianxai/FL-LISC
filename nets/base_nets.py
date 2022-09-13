from torchsummary import summary
import torch
from torch import nn
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.student_nets import StuNet

class base_net(nn.Module):
    def __init__(self, isc_model, channel_model):
        super(base_net, self).__init__()
        self.isc_model = isc_model
        self.ch_model = channel_model

    def forward(self,x):
        encoding = self.isc_model(x)
        encoding_with_noise = self.ch_model(encoding)
        decoding = self.isc_model(x, encoding_with_noise)
        return encoding,encoding_with_noise,decoding

if __name__ == '__main__':
    stu_model = StuNet("resnet18")
    mentor_model = MentorNet()
    channel_model = channel_net()

    tst_model = base_net(mentor_model,channel_model)
    summary(tst_model,(3,224,224),device="cpu")

    tst_model = base_net(stu_model,channel_model)
    summary(tst_model,(3,224,224),device="cpu")