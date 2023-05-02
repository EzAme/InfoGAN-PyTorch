import torch
from torch import nn

class autoencoder(nn.Module):
    def __init__(self, shape):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 512, 7,1,3),  
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(512, 128, 5,1,2),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 8, 3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 128, 3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 512, 5,1,2),  
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 1, 7,1,3),
        )

        def layer_size_out(size, kernel_size = 3, stride = 3, padding = 0):
                return (size - kernel_size + 2 * padding) // stride  + 1
        
        self.encoder_size = layer_size_out(layer_size_out(layer_size_out(layer_size_out(shape,3,3,1),2,2),3,3,1),2,2)
        print(self.encoder_size)
        
    def forward_encoder(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return torch.flatten(x,start_dim = 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 512, 3,1,1),  
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(512, 128, 3,1,1),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 8, 3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),#4608 2048
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x=torch.flatten(x,1)
        x = self.classifier(x)
        return x


            #         nn.Conv2d(1, 128, 7,1,3),  
            # nn.ReLU(),
            # nn.MaxPool2d(2),  
            # nn.Conv2d(128, 32, 3,1,1),  
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 32, 3,1,1),
            # nn.ReLU(),
            # nn.MaxPool2d(2)

            #             nn.Conv2d(1, 512, 3,1,1),  
            # nn.ReLU(),
            # nn.MaxPool2d(2),  
            # nn.Conv2d(512, 128, 3,1,1),  
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 8, 3,1,1),
            # nn.ReLU(),
            # nn.MaxPool2d(2)

        #             self.decoder = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(8, 128, 5,1,2),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(128, 512, 5,1,2),  
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(512, 1, 5,1,2),
        # )