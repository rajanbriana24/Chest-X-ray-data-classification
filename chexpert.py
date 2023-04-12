import torch 
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from cuda import cuda, nvrtc
import cv2
import torchvision.transforms as tfs
import pandas as pd
from torch.utils.data import DataLoader



def set_all_seeds(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_dir = '../../../mnt/storage/CheXpert-v1.0-small/'

#image augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
    transforms.Grayscale(num_output_channels=3)
])

feature_data = pd.read_csv(data_dir+'/valid.csv')
feature_data

class CheXpert(Dataset):

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 image_size=320,
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transforms=None,
                 train_cols=['Edema', 'Atelectasis','Cardiomegaly','Consolidation','Pleural Effusion','No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices'],

                 mode='train'):


        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=True)
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices']: # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]


        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 14 classes
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms

        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self.targets = self.df[train_cols].values[:, class_index].tolist()
        else:
            self.targets = self.df[train_cols].values.tolist()

        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    except:
                        if len(self.value_counts_dict[class_key]) == 1 :
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0 # no postive samples
                            else:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1 # no negative samples

                    imratio_list.append(imratio)
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        print ()
                        #print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list


    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train' :
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]])
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        return image, label

train_dataset = CheXpert(csv_path=data_dir+'train.csv', image_root_path=data_dir, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)


valid_dataset = CheXpert(csv_path=data_dir+'valid.csv', image_root_path=data_dir, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

#Setting up the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#defining loss function
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(device)

# REPRODUCIBILITY
torch.manual_seed(1)
np.random.seed(1)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#defining the model
from torchvision.models import densenet121
model = densenet121(pretrained=False, num_classes=14)
model = model.to(device)


for param in model.parameters():
    param.requires_grad = True

lr = 1e-4
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

import torchmetrics
best_val_auc =0

for epoch in range(10):
    for idx, data in enumerate(train_loader):
      train_data, train_labels = data
      train_data, train_labels = train_data.to(device), train_labels.to(device)
      y_pred = model(train_data)
     
      loss = loss_fn(y_pred, train_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
     # validation
      if idx % 400 == 0:
         model.eval()
         with torch.no_grad():
              test_pred = []
              test_true = []
             
              for jdx, data in enumerate(val_loader):
                  test_data, test_labels = data
                  test_data = test_data.to(device)
                  y_pred = model(test_data)
                  sigmoid = torch.nn.Sigmoid()
                  y_pred = sigmoid(y_pred)
                  

                  test_pred.append(y_pred.cpu().detach().numpy())
                  
                  test_true.append(test_labels.numpy())
                  
              test_true = np.concatenate(test_true)
              

              test_pred = np.concatenate(test_pred)
              
              val_auc_mean =  roc_auc_score(test_true.flatten(), test_pred.flatten())
              model.train()

              if best_val_auc < val_auc_mean:
                 best_val_auc = val_auc_mean
                 torch.save(model.state_dict(), 'ce_pretrained_model_10epochs.pth')

              print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))
