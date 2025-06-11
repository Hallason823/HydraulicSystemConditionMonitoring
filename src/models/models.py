from torch import nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Dropout(p=0.0),
                                     nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(negative_slope=0.3)
                                     )

        self.decoder = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Dropout(p=0.0),
                                     nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(negative_slope=0.4)
                                     )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(p=0.0),
                                  nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  )
        
        self.fc = nn.Sequential(nn.Linear(120, 30),
                                 nn.LeakyReLU(negative_slope=0.1) 
                                 )
        
    def forward(self, input):
        output = self.cnn(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output