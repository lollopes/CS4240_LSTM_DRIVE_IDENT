import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset


###################################### CUDA FUNCTIONS ##################################################

if torch.cuda.is_available():
    device = "cuda:0"
    print("gpu available")

#clear cash 
torch.cuda.empty_cache()


# to move data to target device 
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# to move dataloaders into device
class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)



####################################### CUSTOM DATASET #################################################


import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset

class DatasetDrivers(Dataset):
    def __init__(self, file_path='dataset.csv', classes_to_drop=['Class', 
                                                                 'PathOrder', 
                                                                 'Time(s)', 
                                                                 'Filtered_Accelerator_Pedal_value',
                                                                 'Inhibition_of_engine_fuel_cut_off',
                                                                 'Fuel_Pressure',
                                                                 'Torque_scaling_factor(standardization)',
                                                                 'Glow_plug_control_request'], window_size=16, normalize=True, normalize_method='mean_std'):
        print("Loading")
        self._window_size=window_size
        self._data=pd.read_csv(file_path)
        self._data.sort_values(by="Class", inplace=True)
        print("Removing extra columns, initial data size is:")
        print(len(self._data))
        n_per_class=[]
        for class_uniq in list(self._data['Class'].unique()):
            tot_number=sum(self._data['Class']==class_uniq)
            n_per_class.append(tot_number-tot_number%window_size)
            to_drop=tot_number%window_size
            index_to_start_removing=self._data[self._data['Class']==class_uniq].index[0]
            #import pdb; pdb.set_trace()
            self._data.drop(self._data.index[index_to_start_removing:index_to_start_removing+to_drop],inplace=True)
        print("After removing, the size becomes:")
        print(len(self._data))
        self._data.reset_index(inplace=True)
        #import pdb; pdb.set_trace()

        index_starting_class=[]
        #index_starting_class.append(0)
        for class_uniq in list(self._data['Class'].unique()):
            index_starting_class.append(self._data[self._data['Class']==class_uniq].index[0])
        #import pdb; pdb.set_trace()
        index_starting_class.append(len(self._data))
        sequences=[]
        for i in range(len(index_starting_class)):
            if i!=len(index_starting_class)-1:#index_starting_class[-1]:
                ranges=np.arange(index_starting_class[i], index_starting_class[i+1])
            
            for j in range(0,len(ranges),int(self._window_size/2)):
                if len(ranges[j:j+self._window_size])==16:
                    sequences.append(ranges[j:j+self._window_size])
        self._sequences=sequences
        
        print("Drop columns: ")
        print(classes_to_drop)
        self._labels=self._data['Class'].values
        self._data.drop(classes_to_drop, inplace=True, axis=1)
        if normalize:
            for col in self._data.columns:
                if normalize_method=='min_max':
                    self._data[col]=(self._data[col]-self._data[col].min())/(self._data[col].max()-self._data[col].min())
                elif normalize_method=="mean_std":
                    self._data[col]=(self._data[col]-self._data[col].mean())/(self._data[col].std())
        X=np.empty((len(sequences), self._window_size, len(self._data.columns)))
        y=[]#np.empty((len(sequences), ))
        for n_row, sequence in enumerate(sequences):
            X[n_row,:,:]=self._data.iloc[sequence]
            y.append(self._labels[sequence[0]])
        assert len(y)==len(X)
        self._X=X
        targets = preprocessing.LabelEncoder().fit_transform(y)
        targets = torch.as_tensor(targets)
        #targets_tensor = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10)
        self._y=targets
    
    def __len__(self):
        return len(self._X)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self._X[index,:,:]), self._y[index]


# Istance of the dataset 
a=DatasetDrivers()




#################################### DATA LOADERS ################################################

train_size = int(0.85 * len(a))
val_size = int(0.05 * len(a))
test_size = len(a)-train_size-val_size#int(0.10 * len(a))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(a, [train_size,val_size, test_size])

batch_size = 32
n_workers=0

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=n_workers)
 
validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=n_workers)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=n_workers)


# Loading dataloaders into the gpu 
train_loader = DeviceDataLoader(train_loader,device)
validation_loader = DeviceDataLoader(validation_loader,device)
test_loader = DeviceDataLoader(test_loader,device)


#################################### NETWORK ##########################################################
torch.autograd.set_detect_anomaly(True)

class NEURAL(torch.nn.Module):
    
    def __init__(self, batch_size, window_size, num_features):
        super(NEURAL, self).__init__()
        self.lstm1 = torch.nn.LSTM(num_features, 160,batch_first = True)
        self.lstm2 = torch.nn.LSTM(160, 200, 1)    
        # I need to get the output of last hidden layer 
        # What size is it DONT KNOW
        self.fc = torch.nn.Linear(16, 10)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):

        lstm1_out, (h_t1, c_t1) = self.lstm1(x) 
        #print("shape of lstm1 is: ",lstm1_out.shape)
        #print("shape of lstm1 h is: ",h_t1.shape)        
        #print("shape of lstm1 c is: ",c_t1.shape)   
        
        lstm2_out, (h_t2, c_t2) = self.lstm2(lstm1_out)      
        #print("shape of lstm2 is: ",lstm2_out.shape)
        #print("shape of lstm2 h is: ",h_t2.shape)        
        #print("shape of lstm2 c is: ",c_t2.shape)
        #print("shape of lstm2 c is: ",h_t2[:,:,-1].shape)
              
        fc_out = self.fc(lstm2_out[:,:,-1])
        out = self.sigmoid(fc_out)

        return out
    
    
inputs, classes = next(iter(train_loader)) 

model = NEURAL(inputs.shape[0],inputs.shape[1],inputs.shape[2])
to_device(model,device)

learning_rate = 1e-4
num_epochs = 500

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.5 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

################################## TRAIN LOOP ###############################################################

def train_one_epoch():
    running_loss = 0

    for i, (data, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(data)
        # Compute the loss and its gradients
        loss = criterion(outputs,labels)
        loss.backward()
        
        # Compute the loss and its gradients
        optimizer.step()
 
        running_loss += loss.item()
        #print(i)
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

epoch_number = 0

for epoch in range(num_epochs):


    model.train(True)
    avg_loss = train_one_epoch()
    torch.cuda.empty_cache()
    scheduler.step()

    epoch_number += 1

