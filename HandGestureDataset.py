# %%
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

# %%
class HandGestureDataSet(Dataset):
    def __init__(self, root, train = True,transform = None, ):
        self.root= root
        self.transform = transform
        self.train = train
        self.data = self.create_dictionary_of_images()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image = read_image(self.data.iloc[idx]['Location'])
        if self.transform:
            image = self.transform(image)
        return image ,self.data.iloc[idx]['Class'] 

    def create_dictionary_of_images(self):
        df =pd.DataFrame(columns=['Class','Location'])
        basedir = self.root
        if self.train:
            path = os.listdir(basedir)[:8]
        else:
            path = os.listdir(basedir)[8:]
        for folder in path:
            for Class in os.listdir(basedir+folder+'/'):
                for location in os.listdir(basedir+folder+'/'+Class+'/'):
                    df = df._append({'Class':int(Class[:2])-1,'Location':basedir+folder+'/'+Class+'/'+location},ignore_index=True)
        # df = df.sample(frac = 1) 
        return df


# %%
if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = "leapGestRecog/"
    train = HandGestureDataSet(root = path, train = True, 
        transform = torch.nn.Sequential(
                T.RandomRotation(45),
                T.CenterCrop(350)))
    validation = HandGestureDataSet(root = path, train = False)
    plt.title('Label = %s' % train[1][1])
    plt.subplot(2,1,1)
    plt.imshow((train[1][0]).squeeze().numpy())
    plt.title('Label = %s '%train[1][1])
    plt.subplot(2,1,2)
    plt.imshow((validation[1][0]).squeeze().numpy())
    plt.title('Label = %s '%validation[1][1])
    plt.show()
    print(len(train))




# %%
