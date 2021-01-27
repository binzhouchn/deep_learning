#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch
import os
import torch.nn as nn
import scipy.io as sio
import random
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting for Data
NUM_FEEDBACK_BITS = 510
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
# Parameters Setting for Training
BATCH_SIZE = 64
EPOCHS = 202
LEARNING_RATE = 2e-3
PRINT_RREQ = 1000
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 42
seed_everything(SEED)

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
split = int(data.shape[0] * 0.8)
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS)
autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 0.1
for epoch in range(EPOCHS):
    autoencoderModel.train()
    if epoch%11==10:
        optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.87
    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = autoencoderModel(autoencoderInput)
        loss = criterion(autoencoderInput, autoencoderOutput)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % PRINT_RREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
    # Model Evaluating
    autoencoderModel.eval()
    totalLoss = 0
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            autoencoderInput = autoencoderInput.cuda()
            autoencoderOutput = autoencoderModel(autoencoderInput)
            totalLoss += torch.sum(NMSE_cuda(autoencoderInput, autoencoderOutput)).item()
        averageLoss = totalLoss / len(test_dataset)
        print('val averageLoss: ', averageLoss)
        if averageLoss < bestLoss:
            # Model saving
            # Encoder Saving
            torch.save({'state_dict': autoencoderModel.encoder.state_dict(), }, './modelSubmit/encoder.pth.tar')
            # Decoder Saving
            torch.save({'state_dict': autoencoderModel.decoder.state_dict(), }, './modelSubmit/decoder.pth.tar')
            print("Model saved")
            bestLoss = averageLoss
#=======================================================================================================================
#=======================================================================================================================