#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn.functional as F
import torch.nn as nn

class LSTM_FF(nn.Module):
    def __init__(self):
        super(LSTM_FF, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        #self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        #self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        #out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        #out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out
    
class LSTM_FF_dropout(nn.Module):
    def __init__(self, inp_size = 4):
        super(LSTM_FF_dropout, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=inp_size, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out

class LSTM_MLP_no_recurrent(nn.Module):
    def __init__(self):
        super(LSTM_MLP_no_recurrent, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        #self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        #self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        #out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        #out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        #out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out

class LSTM_MLP(nn.Module):
    def __init__(self, batch_size):
        super(LSTM_MLP, self).__init__()
        self.batch_size = batch_size
        
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        #self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        #self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

        # Initialize initial hidden states
        #self.h_01 = torch.zeros(1, 100)  # Initial hidden state for LSTM 1
        #self.c_01 = torch.zeros(1, 100)  # Initial cell state for LSTM 1
        #self.h_02 = torch.zeros(1, 50)  # Initial hidden state for LSTM 2
        #self.c_02 = torch.zeros(1, 50)  # Initial cell state for LSTM 2

        self.h_01 = torch.zeros(1, batch_size, 100)  # Initial hidden state for LSTM 1
        self.c_01 = torch.zeros(1, batch_size, 100)  # Initial cell state for LSTM 1
        self.h_02 = torch.zeros(1, batch_size, 50)  # Initial hidden state for LSTM 2
        self.c_02 = torch.zeros(1, batch_size, 50)  # Initial cell state for LSTM 2

    def forward(self, x, hidden=None):
        #if hidden is None:
            #hidden1 = (self.h_01.to(x.device).detach(), self.c_01.to(x.device).detach())
            #hidden2 = (self.h_02.to(x.device).detach(), self.c_02.to(x.device).detach())
        #else:
            #hidden1 = (hidden[0][0].detach(), hidden[0][1].detach())
            #hidden2 = (hidden[1][0].detach(), hidden[1][1].detach())

        if hidden is None:
            # LSTM layer 1
            out, hidden1 = self.lstm1(x)
            # LSTM layer 2
            out, hidden2 = self.lstm2(out)
        else:
            hidden1 = (hidden[0][0].detach(), hidden[0][1].detach())
            hidden2 = (hidden[1][0].detach(), hidden[1][1].detach())

            out, hidden1 = self.lstm1(x, hidden1)
            out, hidden2 = self.lstm2(out, hidden2)
        
        # Dropout after the second LSTM
        #out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        #out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out, (hidden1, hidden2)
    
class LSTM_MLP_norm(nn.Module):
    def __init__(self, batch_size):
        super(LSTM_MLP_norm, self).__init__()
        self.batch_size = batch_size
        
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

        # Initialize initial hidden states
        #self.h_01 = torch.zeros(1, 100)  # Initial hidden state for LSTM 1
        #self.c_01 = torch.zeros(1, 100)  # Initial cell state for LSTM 1
        #self.h_02 = torch.zeros(1, 50)  # Initial hidden state for LSTM 2
        #self.c_02 = torch.zeros(1, 50)  # Initial cell state for LSTM 2

        self.h_01 = torch.zeros(1, batch_size, 100)  # Initial hidden state for LSTM 1
        self.c_01 = torch.zeros(1, batch_size, 100)  # Initial cell state for LSTM 1
        self.h_02 = torch.zeros(1, batch_size, 50)  # Initial hidden state for LSTM 2
        self.c_02 = torch.zeros(1, batch_size, 50)  # Initial cell state for LSTM 2

    def forward(self, x, hidden=None):
        #if hidden is None:
            #hidden1 = (self.h_01.to(x.device).detach(), self.c_01.to(x.device).detach())
            #hidden2 = (self.h_02.to(x.device).detach(), self.c_02.to(x.device).detach())
        #else:
            #hidden1 = (hidden[0][0].detach(), hidden[0][1].detach())
            #hidden2 = (hidden[1][0].detach(), hidden[1][1].detach())

        if hidden is None:
            # LSTM layer 1
            out, hidden1 = self.lstm1(x)
            # LSTM layer 2
            out, hidden2 = self.lstm2(out)
        else:
            hidden1 = (hidden[0][0].detach(), hidden[0][1].detach())
            hidden2 = (hidden[1][0].detach(), hidden[1][1].detach())

            out, hidden1 = self.lstm1(x, hidden1)
            out, hidden2 = self.lstm2(out, hidden2)
        
        # Dropout after the second LSTM
        out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out, (hidden1, hidden2)

class LSTM_MLP_old(nn.Module):
    def __init__(self):
        super(LSTM_MLP_old, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        #self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        #self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x, hidden=None):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        #out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        #out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out, hidden
    
class LSTM_FF_dropout_3(nn.Module):
    def __init__(self):
        super(LSTM_FF_dropout_3, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out
    
class LSTM_MLP_old_norm(nn.Module):
    def __init__(self, inp_size = 4):
        super(LSTM_MLP_old_norm, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=inp_size, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x, hidden=None):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)
        
        return out, hidden

class LSTM_MLP_old_2_norm(nn.Module):
    def __init__(self):
        super(LSTM_MLP_old_2_norm, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=100, batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(50, 25)
        self.relu1 = nn.ReLU()
        
        self.dropout_dense = nn.Dropout(0.25)
        
        self.output_layer = nn.Linear(25, 1)

    def forward(self, x, hidden=None):
        
        # LSTM layer 1
        out, _ = self.lstm1(x)
        # LSTM layer 2
        out, _ = self.lstm2(out)
        
        # Dropout after the second LSTM
        out = self.dropout_lstm(out)
        
        # Reshape for the fully connected layers
        out = out[:, -1, :]  # Take the last time step's output

        
        # Second dense layer
        out = self.fc1(out)
        out = self.relu1(out)
        
        # Dropout between the two dense layers
        out = self.dropout_dense(out)
        
        # Output layer for regression
        out = self.output_layer(out)

        #out = self.output_layer(out[:, -1])
        
        return out, hidden


class MLP4(nn.Module):
    def __init__(self, batch_norm=False, dropout_var=False, dropout_prob=0.25):
        super(MLP4, self).__init__()
        self.batch_norm = batch_norm
        self.dropout_var = dropout_var
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 1)

        # Batch normalization layers
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(100)
            self.bn2 = nn.BatchNorm1d(50)
            self.bn3 = nn.BatchNorm1d(25)

        self.relu = nn.ReLU()
        if self.dropout_var:
            self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        # Fully connected layers with ReLU activations and optional dropout
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc4(x)

        return x

class MLP8(nn.Module):
    def __init__(self, batch_norm=False, dropout_var=False, dropout_prob=0.25):
        super(MLP8, self).__init__()
        self.batch_norm = batch_norm
        self.dropout_var = dropout_var
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 40)
        self.fc6 = nn.Linear(40, 30)
        self.fc7 = nn.Linear(30, 20)
        self.fc8 = nn.Linear(20, 1)

        # Batch normalization layers
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
            self.bn5 = nn.BatchNorm1d(40)
            self.bn6 = nn.BatchNorm1d(30)
            self.bn7 = nn.BatchNorm1d(20)

        self.relu = nn.ReLU()
        if self.dropout_var:
            self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x, hidden=None):
        # Fully connected layers with ReLU activations and optional dropout
        #print(x.shape)
        
        x = torch.squeeze(x, dim=1)
                
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc5(x)
        if self.batch_norm:
            x = self.bn5(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc6(x)
        if self.batch_norm:
            x = self.bn6(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc7(x)
        if self.batch_norm:
            x = self.bn7(x)
        x = self.relu(x)
        if self.dropout_var:
            x = self.dropout(x)

        x = self.fc8(x)

        return x, hidden
