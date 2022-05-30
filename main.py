import pandas as pd
import torch
from sklearn import preprocessing
import numpy as np
from torch.autograd import Variable
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
torch.set_printoptions(threshold=10_000)

# Dropping useless features and separate labels from data
data = df.drop(columns=['Class','PathOrder','Time(s)'])
label = df.Class.to_numpy()
print("the dataset has shape: ", data.shape)
print("the labels has shape: ", label.shape)
print("nan count in data", data.isnull().values.any())

# Normalize each column(feature) of the dataframe 

# A buffer to fill with the normalized values of the dataframe 
data_buffer = pd.DataFrame().reindex_like(data)

for colname, colval in data.iteritems():
    #get min and max values of the current column 
    col_min = data[colname].min() 
    col_max = data[colname].max() 
        
    #check if I there is a division by zero in denominator 
    if (col_max - col_min > 0):
        # I can do the normalization 
        data_buffer[colname] = (data[colname] - col_min)/(col_max-col_min)
    else:
        # Set the normalized value to 0 
        data_buffer[colname] = 0

print("nan count in data_buffer", data_buffer.isnull().values.any())

# Transform the normalized dataframe into a torch tensor 
data_norm_tensor = torch.tensor(data_buffer.values)
print("data tensor are of type: ", type(data_norm_tensor), "shape: ", data_norm_tensor.size())


#One-Hot Encode the label column of dataframe and keep it as a pytorch vector    
targets = preprocessing.LabelEncoder().fit_transform(label)
targets = torch.as_tensor(targets)
label_encoded_tensor = torch.nn.functional.one_hot(targets, num_classes = 10)
print("encoded label are of type: ", type(label_encoded_tensor), "shape: ", label_encoded_tensor.size())

# At this point I have the normalized dataframe and the encoded labels in a tensor format 
# I need to create a windowed tensor with window_size 

window_size = 16 
step = int(window_size//2)
size = len(range(0,len(data_buffer)-step+1,step))-1
print(size)

data_norm_tensor_win = torch.empty(size,window_size,data.shape[1], dtype=torch.float32)
label_encoded_tensor_win = torch.empty(size,window_size,10, dtype=torch.float32)


counter = 0
for i in range(0,len(data_buffer)-step,step):
    if(counter<size):
        data_norm_tensor_win[counter] = data_norm_tensor[i:i+window_size][:]
        label_encoded_tensor_win[counter] = label_encoded_tensor[i:i+window_size][:]
        counter=counter+1      

#data_norm_tensor_win = data_norm_tensor_win[:]    
    
print("data normalized and windowed are of type: ", type(data_norm_tensor_win), "shape: ", data_norm_tensor_win.size())    
print("label normalized and windowed are of type: ", type(label_encoded_tensor_win), "shape: ", label_encoded_tensor_win.size())    



# I now have the pytorch tensors windowed to feed in the network 
# data_norm_tensor_win -> data
# label_encoded_tensor_win -> label


torch.nan_to_num(data_norm_tensor_win)
print(torch.isnan(data_norm_tensor_win).any())
torch.isnan(data_norm_tensor_win)
print(" d",torch.isnan(data_norm_tensor_win.view(-1)).sum().item()==0)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
dataset = CustomDataset(data_norm_tensor_win,label_encoded_tensor_win)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 64


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


torch.autograd.set_detect_anomaly(True)
class NEURAL(torch.nn.Module):
    
    def __init__(self):
        super(NEURAL, self).__init__()
        self.lstm1 = torch.nn.LSTM(51, 160, 1,dropout = 0.5,batch_first = False)
        self.lstm2 = torch.nn.LSTM(160, 200, 1,dropout = 0.5)
        self.fc = torch.nn.Linear(200, 10)
        self.sigmoid = torch.nn.Sigmoid()
        self.logsoftmax=torch.nn.LogSoftmax()
       
        

    def forward(self, x):
        #print(x.size(0))
        x = torch.nan_to_num(x,nan=0.0)
        #print(x.shape)
        if(x.isnan().any()): 
            print("found a nan")
            x = torch.nan_to_num(x,nan=0.0)
            
        #print("is nan x", x.isnan().any() ) 
        h_t1 = Variable(torch.zeros(1, x.size()[1], 160))
        c_t1 = Variable(torch.zeros(1, x.size()[1], 160))
        h_t2 = Variable(torch.zeros(1, x.size()[1], 200))
        c_t2 = Variable(torch.zeros(1, x.size()[1], 200))
        
        h1, (h_t1, c_t1) = self.lstm1(x, (h_t1, c_t1))
        
    
        #print("is nan h1", h1.isnan().any() ) 
        
        h2, (h_t2, _) = self.lstm2(h1, (h_t2, c_t2))
        
        #print("is nan h2", h2.isnan().any() ) 
        
        
        # Propagate input through LSTM


        h3 = self.fc(h2)
        #print("is nan h3", h3.isnan().any() ) 
        h4 = self.sigmoid(h3)
        #print("is nan h4", h4.isnan().any() ) 
        
        return h4
    
model = NEURAL()
print(model)

learning_rate = 1e-4
num_epochs = 500

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        
        data = data.requires_grad_()
       # print("data",data)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(data)
        #print("outputs",outputs.shape)
        #print("labels",labels.shape)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        #loss = torch.nn.NLLLoss()(torch.log(outputs), labels)  
        optimizer.zero_grad()

        # Getting gradients w.r.t. parameters
        loss.backward()


        # Updating parameters
        optimizer.step()

        train_losses.append(loss)
        
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
