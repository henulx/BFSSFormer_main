import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import cls_BFSSFormer_IP.get_cls_map
import time
import cls_BFSSFormer_IP.BFSSFormer

def loadData():
    #load the datasets after Band grouping fusion
    data = sio.loadmat('..\BFdata\img2.mat')['img2']
    #no Band grouping fusion
    # BFdata = sio.loadmat('..\BFdata\Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('..\BFdata\Indian_pines_gt.mat')['indian_pines_gt']

    return data, labels

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels = patchesLabels-1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

# BATCH_SIZE_TRAIN = 64 #for ksc
BATCH_SIZE_TRAIN = 64 #for ip
# BATCH_SIZE_TRAIN = 128 #for ho
# BATCH_SIZE_TRAIN = 128 #LK
# BATCH_SIZE_TRAIN = 64 #SSFTT

def create_data_loader():
    # Classes
    # class_num = 15  #numbersamples=15029 HO
    class_num = 16   #numbersamples=10249 IP
    # class_num = 9 # numbersamples=204542 LK
    # class_num = 13 #numbersamples=5211 ksc
    # loading datasets
    X, y = loadData()
    # test ratio for all four datasets
    test_ratio = 0.9
    # patch size
    patch_size = 15 #IP
    # patch_size = 25  # KSC
    # patch_size = 11 #LK
    # patch_size = 13 # SSFTT
    # patch_size = 11 # HO
    pca_components = 30

    print('Hyperspectral BFdata shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create BFdata cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test BFdata ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # change the size of Xtrain and Ytrain to support keras
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # transpose for pytorch
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # make train_loader and test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

def train(train_loader, epochs):

    # use GPU to train the BFSSFormer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # put the net on GPU
    net = cls_BFSSFormer_IP.BFSSFormer.BFSSFormernet().to(device)
    # Loss Function During Training
    criterion = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # all four datasets
    # start training
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss+loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # test the BFSSFormer
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):
    # target_names = ['Scrub','Willow swamp','Cabbage plam hammock','Cabbage palm/oak', 'Slash pine','Oak/broadleaf hammock','Hardwood swamp',
    #                       'Graminoid marsh','Spartina marsh','Cattail marsh','Salt marsh','Muld flats','Water']
    # target_names = ['Corn','Cotton','Sesame','Broad-leaf soybean', 'Narrow-leaf soybean','Rice','Water',
    #                     'Roads and houses','Mixed weed']
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                    'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    # target_names = ['Healthy-grass','Stressed-grass','Synthetic-grass','Trees','Soil','Water','Residential','Commercial',
    #                 'Road','Highway','Railway','Parking-Lot 1','Parking-Lot 2','Tennis-Court','Running-Track']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    # net, device = train(train_loader, epochs=150)#ho
    # net, device = train(train_loader, epochs=100)  # SSFTT
    # net, device = train(train_loader, epochs=100) #LK
    # net, device = train(train_loader, epochs=150)  # ksc
    net, device = train(train_loader, epochs=150)#ip
    print('cliqueNet parameters:', sum(param.numel() for param in net.parameters()))
    # save params
    torch.save(net.state_dict(), '..\cls_BFSSFormer_IP\cls_params\BFSSFormer_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # Evaluation Metrics
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    Total_time = Training_Time + Test_time
    file_name = "..\cls_result\classification_report.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    cls_BFSSFormer_IP.get_cls_map.get_cls_map(net, device, all_data_loader, y_all)




