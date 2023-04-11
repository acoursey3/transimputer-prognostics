import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch

class CMAPSSTrainDataset(Dataset):
    def __init__(self, dataset_no=1):
        ragged_data = []
        self.scaler = None
        X_train, y_train = self.load_split_datasets(dataset_no)
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                ragged_data.append([np.array(X_train[i][j]), np.array(y_train[i][j])])
                
        del X_train
        sequenced_data = self.sequence_data(ragged_data)
        del ragged_data
        
        self.pad_len = 542 # max of any sequence from any dataset
        
        self.data = []
        for X, y in sequenced_data:
            padded_X = np.pad(X, ((0, self.pad_len-X.shape[0]), (0,0)))
            self.data.append([padded_X, y])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensors, rul = self.data[idx]
        return torch.from_numpy(sensors), rul
    
    def sequence_data(self, ragged_data):
        sequenced_data = []
        for X, y in ragged_data:
            prev_rul = y[0]
            i = 0
            for rul in y:
                if prev_rul == 0:
                    break
                prev_rul = rul
                sequenced_data.append([X[:i], rul])
                i += 1

        return sequenced_data
    
    def get_scaler(self):
        return self.scaler
    
    def load_split_datasets(self, dataset_no):
        dataPath = '../CMAPSSData'
        id_col = ['id']
        cycle_col = ['cycle']
        setting_cols = ['setting1', 'setting2', 'setting3']
        sensor_cols = ['sensor' + str(i) for i in range(1, 22)]
        rul_col = ['RUL']
        all_cols = id_col + cycle_col + setting_cols + sensor_cols + rul_col
        
                # This section is to load data
        def loadData(fileName):
            data = pd.read_csv(fileName, sep=" ", header=None)
            data.drop([26, 27], axis = 1, inplace=True)
            data.columns = id_col + cycle_col + setting_cols +sensor_cols
            return data
        
                # load train RUL also returns the max cycle, and this max cycle is also the life cylce
        def addTrainRul(data, decrease_threshold=None):
            lifeCycles = {mcId: data[data['id']==mcId]['cycle'].max() for mcId in data['id'].unique()}
            if decrease_threshold == None: decrease_threshold = 1
            ruls = [lifeCycles[row[0]] - decrease_threshold if row[1] < decrease_threshold else lifeCycles[row[0]] - row[1] for row in data.values]
            data['RUL'] = ruls
            return lifeCycles

        # use this last one only, return the data as well as the max life cycles
        def loadTrainData(setNumber, decrease_threshold=None):
            fileName = dataPath + '/train_FD00' + str(setNumber) + '.txt'
            data = loadData(fileName)
            lifeCycles = addTrainRul(data, decrease_threshold)
            return data, lifeCycles
        
        decrease_threshold = None
        
        train_datasets, train_lifecycles = [], []
        
        self.scaler = MinMaxScaler()
        setNumber = dataset_no
        train, trainLifeCycles = loadTrainData(setNumber, decrease_threshold)
        target = train['RUL'].copy()
        transformed_train = self.scaler.fit_transform(train)
        train = pd.DataFrame(transformed_train, columns=train.columns, index=train.index)
        train['RUL'] = target
        train_datasets.append(train)
        train_lifecycles.append(trainLifeCycles)
        
        X_train = []
        y_train = []
        
        split_data, ruls = self.split_by_id(train_datasets[0], dataset_no)
        X_train.append(split_data)
        y_train.append(ruls)
            
        return X_train, y_train
        
    def split_by_id(self, dataset, dataset_no=1):
        split_data = []
        ruls = []
        for id_no in np.unique(dataset['id']):
            split = dataset.groupby('id').get_group(id_no).copy()
            split['dataset_no'] = dataset_no
            rul = split['RUL']
            split = split.drop('RUL', axis=1)
            split_data.append(split)
            ruls.append(rul)

        return split_data, ruls
    
    
class CMAPSSTestDataset(Dataset):
    def __init__(self, size, dataset_no=1, scaler=None):
        ragged_data = []
        self.scaler = scaler
        X_test, y_test = self.load_split_datasets(dataset_no)
        for i in range(len(X_test)):
            for j in range(len(X_test[i])):
                ragged_data.append([np.array(X_test[i][j]), np.array(y_test[i][j])])
                
        del X_test
        sequenced_data = self.sequence_data(ragged_data)
        del ragged_data
        
        self.pad_len = size
        
        self.data = []
        for X, y in sequenced_data:
            padded_X = np.pad(X, ((0, self.pad_len-X.shape[0]), (0,0)))
            self.data.append([padded_X, y])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensors, rul = self.data[idx]
        return torch.from_numpy(sensors), rul
    
    def sequence_data(self, ragged_data):
        sequenced_data = []
        for X, y in ragged_data:
            prev_rul = y[0]
            i = 0
            for rul in y:
                if prev_rul == 0:
                    break
                prev_rul = rul
                sequenced_data.append([X[:i], rul])
                i += 1

        return sequenced_data
    
    def load_split_datasets(self, setNumber):
        dataPath = '../CMAPSSData'
        id_col = ['id']
        cycle_col = ['cycle']
        setting_cols = ['setting1', 'setting2', 'setting3']
        sensor_cols = ['sensor' + str(i) for i in range(1, 22)]
        rul_col = ['RUL']
        all_cols = id_col + cycle_col + setting_cols + sensor_cols + rul_col
        
                # This section is to load data
        def loadData(fileName):
            data = pd.read_csv(fileName, sep=" ", header=None)
            data.drop([26, 27], axis = 1, inplace=True)
            data.columns = id_col + cycle_col + setting_cols +sensor_cols
            return data
        
        decrease_threshold = None
        
        def loadTestRul(fileName):
            data = pd.read_csv(fileName, sep = " ", header=None)
            data.drop([1], axis=1, inplace=True)
            data.columns = ['RUL']
            return data
        def addTestRul(data, rulData, decrease_threshold=None):
            testRuls = {i+1: rulData.iloc[i, 0] for i in range(len(rulData))}
            lifeCycles = {mcId: data[data['id']==mcId]['cycle'].max() + testRuls[mcId] for mcId in data['id'].unique()}
            if decrease_threshold == None: decrease_threshold = 1
            ruls = [lifeCycles[row[0]] - decrease_threshold if row[1] < decrease_threshold else lifeCycles[row[0]] - row[1] for row in data.values]
            data['RUL'] = ruls
            return lifeCycles
        # Use this last one only => return data as well as the max life cycles for each machine
        def loadTestData(setNumber, decrease_threshold=None):
            data = loadData(dataPath + '/test_FD00' +str(setNumber)+'.txt')
            rulData = loadTestRul(dataPath + '/RUL_FD00' + str(setNumber)+'.txt')
            lifeCycles = addTestRul(data, rulData, decrease_threshold)
            return data, lifeCycles
        
        test_datasets, test_lifecycles = [], []

        test, testLifeCycles = loadTestData(setNumber, decrease_threshold)
        target = test['RUL'].copy()
        transformed_test = self.scaler.transform(test)
        test = pd.DataFrame(transformed_test, columns=test.columns, index=test.index)
        test['RUL'] = target
        test_datasets.append(test)
        test_lifecycles.append(testLifeCycles)
        
        X_test = []
        y_test = []

        split_data, ruls = self.split_by_id(test_datasets[0], setNumber)
        X_test.append(split_data)
        y_test.append(ruls)
            
        return X_test, y_test
        
    def split_by_id(self, dataset, dataset_no=1):
        split_data = []
        ruls = []
        for id_no in np.unique(dataset['id']):
            split = dataset.groupby('id').get_group(id_no).copy()
            split['dataset_no'] = dataset_no
            rul = split['RUL']
            split = split.drop('RUL', axis=1)
            split_data.append(split)
            ruls.append(rul)

        return split_data, ruls
    
def get_dataloaders(batch, dataset_no=1):
    traindata = CMAPSSTrainDataset(dataset_no)
    trainloader = DataLoader(traindata, batch_size=batch, shuffle=True)
    size = next(enumerate(trainloader))[1][0].shape[1]
    
    testdata = CMAPSSTestDataset(size, dataset_no, traindata.get_scaler())
    testloader = DataLoader(testdata, batch_size=batch, shuffle=False)

    return trainloader, testloader

def get_new_dataloaders(batch, dataset_no=1):
    traindata = CMAPSSTrainDataset(dataset_no)
    trainloader = DataLoader(traindata, batch_size=batch, shuffle=True)
    
    train_size = int(0.8 * len(traindata))
    test_size = len(traindata) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(traindata, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return trainloader, testloader