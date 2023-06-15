import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
from pickle import load
import h5py

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
    
def get_cmapss_dataloaders(batch, dataset_no=1):
    traindata = CMAPSSTrainDataset(dataset_no)
    trainloader = DataLoader(traindata, batch_size=batch, shuffle=True)
    size = next(enumerate(trainloader))[1][0].shape[1]
    
    testdata = CMAPSSTestDataset(size, dataset_no, traindata.get_scaler())
    testloader = DataLoader(testdata, batch_size=batch, shuffle=False)

    return trainloader, testloader

def get_cmapss_dataloaders_new(batch, dataset_no=1):
    traindata = CMAPSSTrainDataset(dataset_no)
    trainloader = DataLoader(traindata, batch_size=batch, shuffle=True)
    
    train_size = int(0.8 * len(traindata))
    test_size = len(traindata) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(traindata, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return trainloader, testloader

class NCMAPSSTrainDataset(Dataset):
    def __init__(self, ds_no, timesteps=10):
        self.ds_no = ds_no
        self.fileloc = self.get_fileloc(ds_no)
        self.timesteps = timesteps
        self.scaler = self.get_scaler(ds_no)
        
    
    def __getitem__(self, index):
        start = index - self.timesteps + 1
        if start < 0:
            start = 0
        indices = list(range(start, index+1))
        
        X_train, unit, A_dev, Y_dev = self.get_data(indices, index)
        X_train = self.scaler.transform(X_train)
        X_train[:,42] = A_dev[:,0]
        n_pad = self.timesteps - X_train.shape[0]
        X_train = np.pad(X_train, ((n_pad, 0),(0,0)), mode='constant')
        
        for i, row in enumerate(X_train):
            curr_unit = row[42] # ensure column 42 contains the unit number
            if curr_unit != unit and not np.all(row==0):
                X_train[i] = np.zeros_like(row)
        
        return X_train, Y_dev
    
    def get_data(self, indices, index):
        with h5py.File(self.fileloc, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_dev')[indices])
            X_s_dev = np.array(hdf.get('X_s_dev')[indices])
            X_v_dev = np.array(hdf.get('X_v_dev')[indices])
            T_dev = np.array(hdf.get('T_dev')[indices])
            Y_dev = np.array(hdf.get('Y_dev')[index])
            A_dev = np.array(hdf.get('A_dev')[indices])
            
            unit = A_dev[-1:, 0]
            
        X_train = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, A_dev), axis=1)
        
        return X_train, unit, A_dev, Y_dev
    
    def __len__(self):
        lengths = {
            1: 4906636,
            2: 5263447,
            3: 5571277,
            4: 6377452,
            5: 4350606,
            6: 4257209,
            7: 4350176,
            8: 4885389,
            9: 4299918
        }
        
        return lengths[self.ds_no]
    
    def get_fileloc(self, ds_no):
        locations = {
            1: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS01-005.h5',
            2: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS02-006.h5',
            3: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS03-012.h5',
            4: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS04.h5',
            5: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS05.h5',
            6: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS06.h5',
            7: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS07.h5',
            8: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS08a-009.h5',
            9: '/data/courseac/N-CMAPSS/data_set/N-CMAPSS_DS08c-008.h5',
        }
        
        return locations[ds_no]
    
    def get_scaler(self, ds_no):
        return load(open('./scalers/scaler' + str(ds_no) + '.pkl', 'rb'))
    
class NCMAPSSTestDataset(NCMAPSSTrainDataset):
    def __init__(self, ds_no, timesteps=10):
        super().__init__(ds_no,timesteps)
    
    def get_data(self, indices, index):
        with h5py.File(self.fileloc, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_test')[indices])
            X_s_dev = np.array(hdf.get('X_s_test')[indices])
            X_v_dev = np.array(hdf.get('X_v_test')[indices])
            T_dev = np.array(hdf.get('T_test')[indices])
            Y_dev = np.array(hdf.get('Y_test')[index])
            A_dev = np.array(hdf.get('A_test')[indices])
            
            unit = A_dev[-1:, 0]
            
        X_train = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, A_dev), axis=1)
        
        return X_train, unit, A_dev, Y_dev
    
    def __len__(self):
        lengths = {
            1: 2735232,
            2: 1253743,
            3: 4251560,
            4: 3602561,
            5: 2562046,
            6: 2522447,
            7: 2869786,
            8: 3722997,
            9: 2117819
        }
        
        return lengths[self.ds_no]

class SubsampledNCMAPSSTrainDataset(Dataset):
    def __init__(self, ds_no, timesteps=10):
        self.ds_no = ds_no
        self.fileloc = self.get_fileloc(ds_no)
        self.timesteps = timesteps
        self.scaler = self.get_scaler(ds_no)
        
    
    def __getitem__(self, index):
        start = index - self.timesteps + 1
        if start < 0:
            start = 0
        indices = list(range(start, index+1))
        
        X_train, unit, A_dev, Y_dev = self.get_data(indices, index)
        X_train = self.scaler.transform(X_train)
        X_train[:,42] = A_dev[:,0]
        n_pad = self.timesteps - X_train.shape[0]
        X_train = np.pad(X_train, ((n_pad, 0),(0,0)), mode='constant')
        
        for i, row in enumerate(X_train):
            curr_unit = row[42] # ensure column 42 contains the unit number
            if curr_unit != unit and not np.all(row==0):
                X_train[i] = np.zeros_like(row)
        
        return X_train, Y_dev
    
    def get_data(self, indices, index):
        with h5py.File(self.fileloc, 'r') as hdf:
            X_train = np.array(hdf.get('X_train')[indices])
            Y_dev = np.array(hdf.get('y_train')[index])
            A_dev = X_train[:, -4:]
            
            unit = A_dev[-1:, 0]
        
        return X_train, unit, A_dev, Y_dev
    
    def __len__(self):
        lengths = {
            1: 490663,
            2: 526344,
            3: 557127,
            4: 637745,
            5: 435060,
            6: 425720,
            7: 435017,
            8: 488538,
            9: 429991
        }
        
        return lengths[self.ds_no]
    
    def get_fileloc(self, ds_no):
        locations = {
            1: '/data/courseac/N-CMAPSS/subsampled/ds1.h5',
            2: '/data/courseac/N-CMAPSS/subsampled/ds2.h5',
            3: '/data/courseac/N-CMAPSS/subsampled/ds3.h5',
            4: '/data/courseac/N-CMAPSS/subsampled/ds4.h5',
            5: '/data/courseac/N-CMAPSS/subsampled/ds5.h5',
            6: '/data/courseac/N-CMAPSS/subsampled/ds6.h5',
            7: '/data/courseac/N-CMAPSS/subsampled/ds7.h5',
            8: '/data/courseac/N-CMAPSS/subsampled/ds8a.h5',
            9: '/data/courseac/N-CMAPSS/subsampled/ds8c.h5',
        }
        
        return locations[ds_no]
    
    def get_scaler(self, ds_no):
        return load(open('./scalers/scaler' + str(ds_no) + '.pkl', 'rb'))
    
class SubsampledNCMAPSSTestDataset(SubsampledNCMAPSSTrainDataset):
    def __init__(self, ds_no, timesteps=10):
        super().__init__(ds_no,timesteps)
    
    def get_data(self, indices, index):
        with h5py.File(self.fileloc, 'r') as hdf:
            X_test = np.array(hdf.get('X_test')[indices])
            Y_test = np.array(hdf.get('y_test')[index])
            A_test = X_test[:, -4:]
            
            unit = A_test[-1:, 0]
        
        return X_test, unit, A_test, Y_test
    
    def __len__(self):
        lengths = {
            1: 273523,
            2: 125374,
            3: 425156,
            4: 360256,
            5: 256204,
            6: 252244,
            7: 286978,
            8: 372299,
            9: 211781
        }
        
        return lengths[self.ds_no]
    
def get_ncmapss_dataloaders(ds_no, n_timesteps, batch, workers=1, subsampled=True):
    if not subsampled:
        traindata = NCMAPSSTrainDataset(ds_no, timesteps=n_timesteps)
        trainloader = DataLoader(traindata, batch_size=batch, shuffle=True, num_workers=workers)

        testdata = NCMAPSSTestDataset(ds_no, timesteps=n_timesteps)
        testloader = DataLoader(testdata, batch_size=batch, shuffle=False, num_workers=workers)
    else: 
        traindata = SubsampledNCMAPSSTrainDataset(ds_no, timesteps=n_timesteps)
        trainloader = DataLoader(traindata, batch_size=batch, shuffle=True, num_workers=workers)

        testdata = SubsampledNCMAPSSTestDataset(ds_no, timesteps=n_timesteps)
        testloader = DataLoader(testdata, batch_size=batch, shuffle=False, num_workers=workers) 

    return trainloader, testloader