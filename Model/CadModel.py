import os

import matplotlib.pyplot as plt

from CNNModel.Training.Loss import FocalLoss
from CNNModel.Model.TrumpetNet import TrumpetNetWithROI
from CNNModel.Utility.SaveAndLoad import SaveModel, SaveHistory
from MeDIT.DataAugmentor import random_2d_augment
from CNNModel.Training.SequenceGenerator import ImageInImageOut2D

if __name__ == '__main__':
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    from keras.utils.vis_utils import plot_model
    train_folder = r'w:\CNNFormatData\PCaDetection\T2_ADC_DWI1500_3slices-recheck_adding\training'
    validation_folder = r'w:\CNNFormatData\PCaDetection\T2_ADC_DWI1500_3slices-recheck_adding\validation'

    store_folder = r'D:\model\ProstateTumorDetection\Adding-2020-1-3'

    batch_size = 24
    base_shape = [192, 192, 3]
    data_shape = {'base_shape': base_shape, 'output_0': [192, 192, 1]}
    not_roi_info = {'input_0': True, 'input_1': True, 'input_2': True, 'input_3': False, 'output_0': False}

    train_generator = ImageInImageOut2D(data_folder=train_folder, data_shape=data_shape,
                                                 batch_size=batch_size, augment_param=random_2d_augment)
    validation_generator = ImageInImageOut2D(data_folder=validation_folder, data_shape=data_shape,
                                                      batch_size=batch_size, augment_param=random_2d_augment)

    '''Set the path to store model weights'''
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    '''Set the call back parameters'''
    model_checkpoint = ModelCheckpoint(os.path.join(store_folder, 'weights.{epoch:04d}-{val_loss:.4f}.h5'),
                                       monitor='val_FocalLoss', save_best_only=True, save_weights_only=True, period=1)
    model_early_stop = EarlyStopping(monitor='val_FocalLoss', patience=50)
    model_reduce_learning_rate = ReduceLROnPlateau(monitor='val_FocalLoss', factor=0.5, patience=10)

    cbks = [model_checkpoint, model_early_stop, model_reduce_learning_rate]

    model = TrumpetNetWithROI([base_shape, base_shape, base_shape], base_shape)
    # plot_model(model, to_file=os.path.join(store_folder, 'model.png'), show_shapes=True)
    # model.summary()

    SaveModel(model, store_folder)

    model.compile(optimizer=Adam(1e-4), loss=FocalLoss, metrics=[FocalLoss])

    # from CNNModel.Training.Lookahead import Lookahead
    # lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
    # lookahead.inject(model) # add into model

    result = model.fit_generator(train_generator, epochs=10000, validation_data=validation_generator, callbacks=cbks,
                                 workers=1, use_multiprocessing=True, verbose=1)

    SaveHistory(result.history, store_folder)

    '''PLOT result'''

    print(result.history.keys())
    temp = result.history
    print(type(temp))

    plt.plot(result.history['val_FocalLoss'])
    plt.plot(result.history['FocalLoss'])
    plt.legend(['val', 'train'])
    plt.show()
