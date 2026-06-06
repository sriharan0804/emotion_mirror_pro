import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class FER2013Dataset(Dataset):
    def __init__(self , csv_path , usage='Training', transform = None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['usage'] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self , idx):
        emotion = int(self.data.iloc[idx]['emotion'])
        pixels = self.data.iloc[idx]['pixels']
        image = np.array(pixels.split() , dtype = np.float32)
        image = torch.tensor(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(emotion , dtype = torch.long)
        return image , label



