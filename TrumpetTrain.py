import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from MeDIT.DataAugmentor import random_2d_augment
from T4T.Utility.Data import *
from T4T.Utility.Loss import FocalLoss, DiceLoss

from SY.CAD.path_config import *
from SY.CAD.Model.TrumpetNet import TrumpetNetWithROI
from ZYH.CAD.Train.CheckPoint import MyEarlyStopping

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def BinaryPred(prediction):
    one = torch.ones_like((prediction))
    zero = torch.zeros_like(prediction)
    binary_pred = torch.where(prediction > 0.5, one, zero)
    return binary_pred

def LoadTVData():
    spliter = DataSpliter()
    train_list, val_list = spliter.LoadName(train_name), spliter.LoadName(val_name)
    train_dataset = DataManager(random_2d_augment, sub_list=train_list)
    val_dataset = DataManager(random_2d_augment, sub_list=val_list)

    ###########################################################
    train_dataset.AddOne(Image2D(t2_folder, shape=(192, 192)))
    train_dataset.AddOne(Image2D(dwi_folder, shape=(192, 192)))
    train_dataset.AddOne(Image2D(adc_folder, shape=(192, 192)))
    train_dataset.AddOne(Image2D(prostate_roi_folder, shape=(192, 192), is_roi=True))
    train_dataset.AddOne(Image2D(pca_roi_folder, shape=(192, 192), is_roi=True), is_input=False)
    train_dataset.AddOne(Image2D(pirads_folder, shape=(192, 192), is_roi=True), is_input=False)

    val_dataset.AddOne(Image2D(t2_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(dwi_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(adc_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(prostate_roi_folder, shape=(192, 192), is_roi=True))
    val_dataset.AddOne(Image2D(pca_roi_folder, shape=(192, 192), is_roi=True), is_input=False)
    val_dataset.AddOne(Image2D(pirads_folder, shape=(192, 192), is_roi=True), is_input=False)

    ###########################################################
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, validation_loader

def MyTrain():
    from torchsample.modules import ModuleTrainer
    from torchsample.callbacks import History, CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from T4T.Utility.Initial import HeWeightInit

    if not os.path.isdir(store_folder):
        os.mkdir(store_folder)

    train_loader, val_loader = LoadTVData()
    model = TrumpetNetWithROI(in_planes=3, planes=32, stride=1, encode_num=3).to(device)
    model.to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss1 = nn.BCELoss()
    loss2 = FocalLoss()

    lr = ReduceLROnPlateau(patience=10, factor=0.5)
    history = History(model)
    csv_logger = CSVLogger(os.path.join(store_folder, 'train_log.csv'))
    check_point = ModelCheckpoint(directory=store_folder, filename='{epoch}_{loss}.pth', monitor='val_loss',
                                  save_best_only=True)
    early_stop = EarlyStopping(patience=50)

    trainer = ModuleTrainer(model)
    trainer.compile(loss=[loss1, loss2], optimizer=optimizer,
                    callbacks=[check_point, history, csv_logger, lr, early_stop])
    trainer.fit_loader(train_loader, val_loader=val_loader, num_epoch=10000, cuda_device=0)

def MyTest():
    from MeDIT.Visualization import Imshow3DArray, FlattenImages, MergeImageWithRoi
    from MeDIT.Normalize import Normalize01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = TrumpetNetWithROI(in_planes=3, planes=32, stride=1, encode_num=3).to(device)
    net.load_state_dict(torch.load(r'd:\model\ProstateTumorDetection\T2AdcDwiRoi-PcaPirads-3Slices-20200628\029_0.063329.pth')['state_dict'])
    net.eval()
    net.to(device)

    spliter = DataSpliter()
    val_list = spliter.LoadName(val_name)
    val_dataset = DataManager(random_2d_augment, sub_list=val_list)

    val_dataset.AddOne(Image2D(t2_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(dwi_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(adc_folder, shape=(192, 192)))
    val_dataset.AddOne(Image2D(prostate_roi_folder, shape=(192, 192), is_roi=True))
    val_dataset.AddOne(Image2D(pca_roi_folder, shape=(192, 192), is_roi=True), is_input=False)
    val_dataset.AddOne(Image2D(pirads_folder, shape=(192, 192), is_roi=True), is_input=False)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    with torch.no_grad():
        for data, label in val_loader:
            t2, adc, dwi, prostate_roi = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            pca_label, pirads_label = label[0].data.numpy().astype(int), \
                                      label[1].data.numpy().astype(int)
            pca_pred, pirads_pred = net(t2, adc, dwi, prostate_roi)

            t2 = t2.cpu().data.numpy()
            adc = adc.cpu().data.numpy()
            prostate_roi = prostate_roi.cpu().data.numpy().astype(int)
            pca_pred = pca_pred.cpu().data.numpy()
            pirads_pred = pirads_pred.cpu().data.numpy()

            for index in range(t2.shape[0]):
                plt.subplot(231)
                data = MergeImageWithRoi(Normalize01(t2[index, 1, ...]),
                                         roi=[prostate_roi[index, 1, ...],
                                              pca_label[index, 0, ...]])
                plt.imshow(data)

                plt.subplot(232)
                plt.imshow(Normalize01(adc[index, 1, ...]), cmap='gray')

                plt.subplot(233)
                show_pca_label = pca_label[index, 0, ...]
                plt.imshow(show_pca_label, cmap='gray')
                plt.title('PCa Label')

                plt.subplot(234)
                show_pca_pred = pca_pred[index, 0, ...]
                plt.imshow(show_pca_pred, cmap='gray', vmax=1., vmin=0.)
                if pca_label[index, 0, ...].sum() > 0:
                    pred = show_pca_pred[show_pca_label == 1].max()
                else:
                    pred =show_pca_pred.max()
                plt.title('PCa Predict: {:.4f}-{:.4f}'.format(pred, show_pca_pred.max()))

                plt.subplot(235)
                plt.imshow(np.argmax(pirads_label[index, ...], axis=0), vmin=0, vmax=3, cmap='jet')
                plt.colorbar()

                plt.subplot(236)
                plt.imshow(np.argmax(pirads_pred[index, ...], axis=0), vmin=0, vmax=3, cmap='jet')
                plt.colorbar()

                plt.show()



def Train():

    if not os.path.isdir(store_folder):
        os.mkdir(store_folder)

    train_loader, validation_loader = LoadTVData()
    train_loss = 0.0
    valid_loss = 0.0

    model = TrumpetNetWithROI(in_planes=3, planes=32, stride=1, encode_num=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # criterion = FocalLoss()
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    early_stopping = MyEarlyStopping(patience=50, verbose=True, store_name=os.path.join(store_folder, '{:3d}-{:.6f}.pt'))
    writer = SummaryWriter(log_dir=store_folder, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        train_samples, val_samples = 0, 0

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, prostate = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device)
            pca, pirads = outputs[0].to(device), outputs[1].to(device)

            pca_pred, pirads_pred = model(t2, dwi, adc, prostate)

            loss = criterion(pca_pred, pca)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' %(epoch + 1, 1000, i + 1, train_loss / 10))
                train_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc, prostate = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(
                    device)
                pca = outputs.to(device)

                input_list = [t2, dwi, adc]

                pca_pred = model(input_list, prostate)

                loss = criterion(pca_pred, pca)

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f' % (epoch + 1, 1000, i + 1, valid_loss / 10))
                    valid_loss = 0.0

        train_loss = np.mean(train_loss_list) / len(train_loss_list)

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, evaluation=min,
                       store_key=[epoch + 1, valid_loss])

        if early_stopping.early_stop:
            print("Early stopping")
            break

def Test():
    from Metric import Dice

    # test_loader = LoadTestData()
    train_loader, validation_loader = LoadTVData(is_test=True)

    model = TrumpetNetWithROI(in_planes=3, planes=64, stride=1, encode_num=3).to(device)
    model.load_state_dict(torch.load(model_path))

    dice = Dice()

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [validation_loader]

    # with torch.no_grad():
    model.eval()
    for name_num, loader in enumerate(loader_list):
        pca_true_list, pca_pred_list = [], []
        dice_list  = []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, prostate = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device)
            pca = outputs.to(device)

            input_list = [t2, dwi, adc]

            pca_pred = model(input_list, prostate)

            binary_pca_pred = BinaryPred(pca_pred).cpu().detach()
            pca_true = pca.cpu()
            pca_pred_list.extend(binary_pca_pred)
            pca_true_list.extend(pca_true)

            plt.subplot(231)
            plt.title('t2')
            plt.imshow(np.squeeze(t2[0, 1, ...].cpu().numpy()), cmap='gray')
            plt.axis('off')

            plt.subplot(232)
            plt.title('dwi')
            plt.imshow(np.squeeze(dwi[0, 1, ...].cpu().numpy()), cmap='gray')
            plt.axis('off')

            plt.subplot(233)
            plt.title('adc')
            plt.imshow(np.squeeze(adc[0, 1, ...].cpu().numpy()), cmap='gray')
            plt.axis('off')

            plt.subplot(234)
            plt.title('true')
            plt.imshow(np.squeeze(t2[0, 1, ...].cpu().numpy()), cmap='gray')
            plt.contour(np.squeeze(prostate[0, 1, ...].cpu().numpy()), colors='y')
            plt.contour(np.squeeze(pca.cpu().numpy()), colors='r')
            plt.axis('off')

            plt.subplot(235)
            plt.title('prediction')
            plt.imshow(np.squeeze(pca_pred.cpu().detach().numpy()), cmap='gray')
            plt.axis('off')

            plt.subplot(236)
            plt.title('prediction')
            plt.imshow(np.squeeze(t2[0, 1, ...].cpu().numpy()), cmap='gray')
            plt.contour(np.squeeze(pca_pred.cpu().detach().numpy()), colors='r')
            plt.axis('off')

            plt.show()

            # plt.subplot(133)
            # plt.title('t2')
            # plt.imshow(np.squeeze(t2), cmap='gray')
            # plt.contour(np.squeeze(prostate), colors='y')
            # plt.contour(np.squeeze(cancer), colors='r')
            # plt.axis('off')



        # for index in range(len(pca_true_list)):
        #     dice_list.append(dice(pca_true_list[index], pca_pred_list[index]).numpy())
        # print('average dice is', sum(dice_list)/len(dice_list))
        # plt.hist(dice_list)
        # plt.title('Dice of Pca in' + name_list[name_num])
        # plt.show()

def ShowPicture():
    train_loader, validation_loader = LoadTVData(is_test=True)
    test_loader = LoadTestData()
    loader_list = [train_loader, validation_loader, test_loader]
    loader_name = ['train', 'validation', 'test']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = TrumpetNetWithROI(in_planes=3, planes=64, stride=1, encode_num=3).to(device)
    model.load_state_dict(torch.load(model_path))
    ece_pre_list = []
    ece_list = []
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    for name, loader in enumerate(loader_list):
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.squeeze(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.FloatTensor).to(device)

            class_out, _ = model(inputs)
            class_out = torch.squeeze(class_out, dim=1)
            class_out_sigmoid = class_out.sigmoid()

            ece_pre_list.append(class_out_sigmoid.cpu().detach().numpy()[0])
            ece_list.append(ece.cpu().numpy()[0])

        # for index in range(len(ece_list)):
        #     if ece_list[index] == 0.0:
        #         if ece_pre_list[index] > thresholds:
        #             FP += 1
        #         if ece_pre_list[index] < thresholds:
        #             FN += 1
        #     elif ece_list[index] == 1.0:
        #         if ece_pre_list[index] > thresholds:
        #             TP += 1
        #         if ece_pre_list[index] < thresholds:
        #             TN += 1
        # print(TP, TN, FP, FN)
        plt.suptitle(loader_name[name])
        plt.subplot(121)
        plt.title('ECE label')
        plt.hist(ece_list)
        plt.subplot(122)
        plt.title('ECE prediction')
        plt.hist(ece_pre_list)
        plt.show()

def FeatureMap():
    model = TrumpetNetWithROI(in_planes=3, planes=64, stride=1, encode_num=3).to(device)
    model.load_state_dict(torch.load(model_path))

    adc = np.load(os.path.join(train_adc_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    dwi = np.load(os.path.join(train_dwi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    t2 = np.load(os.path.join(train_t2_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    prostate = np.load(os.path.join(train_prostate_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    cancer = np.load(os.path.join(train_roi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))

    inputs = np.concatenate([t2, dwi, adc], axis=0)
    inputs = inputs[np.newaxis, :]
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(device)

    x, feature_map = model(inputs)

    plt.subplot(331)
    plt.title('t2')
    plt.imshow(np.squeeze(t2), cmap='gray')
    plt.contour(np.squeeze(prostate), colors='y')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(332)
    plt.title('dwi')
    plt.imshow(np.squeeze(dwi), cmap='gray')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(333)
    plt.title('adc')
    plt.imshow(np.squeeze(adc), cmap='gray')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(334)
    plt.title('conv1')
    plt.imshow(feature_map['conv1'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(335)
    plt.title('layer1')
    plt.imshow(feature_map['layer1'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(336)
    plt.title('layer2')
    plt.imshow(feature_map['layer2'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(337)
    plt.title('layer3')
    plt.imshow(feature_map['layer3'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(338)
    plt.title('layer4')
    plt.imshow(feature_map['layer4'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.show()



if __name__ == '__main__':
    # MyTrain()
    MyTest()

    # Test()
    # ShowPicture()
    # FeatureMap()
