from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import numpy as np
from keras import backend as K
# import Autoencode
import os
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt


BATCH_SIZE = 128

# load k-mer data features
def load_kmer_feature(filename1):
    filelist = [filename1]
    dataMat = []
    for file in filelist:
        file0 = np.loadtxt(file, encoding='utf_8_sig', delimiter=' ')
        for line in file0:
            curLine = list(line)
            floatLine = map(int, curLine)  # 这里使用的是map函数直接把数据转化成为float类型
            floatLine = list(floatLine)
            length = len(floatLine)
            dataMat.append(floatLine[0:length])

    dataMat = np.array(dataMat)
    return dataMat


def autoencode_train(count_iter, epochs, X_kmer_train):
    # Auto-encoder model.
    input_msg = Input(shape=(1024, 1))
    encoded = Conv1D(filters=16, kernel_size=8, strides=1, padding='same', activation='tanh', bias_initializer='zeros')(
        input_msg)  # (?, 1024, 16)
    encoded = MaxPooling1D(pool_size=2)(encoded)  # (?, 512, 16)
    encoded = Conv1D(filters=1, kernel_size=8, strides=1, padding='same', activation='tanh', bias_initializer='zeros')(
        encoded)  # (?, 512, 1)
    encoder_output = MaxPooling1D(pool_size=4)(encoded)  # (?, 128, 1)
    encoder_model = Model(inputs=input_msg, outputs=encoder_output)

    # 解码器，反过来
    decoded = Conv1D(filters=1, kernel_size=8, strides=1, padding='same', activation='tanh', bias_initializer='zeros')(
        encoder_output)  # (?, 128, 1)
    decoded = UpSampling1D(size=2)(decoded)  # (?, 256, 1)
    decoded = Conv1D(filters=16, kernel_size=8, strides=1, padding='same', activation='tanh', bias_initializer='zeros')(
        decoded)  # (?, 256, 16)
    decoded = UpSampling1D(size=4)(decoded)  ##上采样过程（反卷积）   # (?, 1024, 16)

    # 转成原始数据大小
    decoded_output = Conv1D(filters=1, kernel_size=8, strides=1, padding='same', activation='tanh',
                            bias_initializer='zeros')(decoded)  # (?, 1024, 1)
    autoencoder = Model(input_msg, decoded_output)
    # Train the auto-encoder model.
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    autoencoder.compile(optimizer=adam, loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', verbose=2, patience=2, mode='min')
    # filename = './AE_weight/ratio%d_%s_stride%d_%d.h5' % (ratio, name, stride, ind)
    save_best = ModelCheckpoint(checkpoint_dir + "/ae_model_" + str(count_iter) + 'fold' + '.h5', save_best_only=True,
                                save_weights_only=True)
    autoencoder.fit(X_kmer_train, X_kmer_train, epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks=[early_stopping, save_best])

    # 可视化自编码器训练过程
    # 获取训练过程中的损失值和验证集损失值
    # loss = autoencoder.history.history['loss']
    val_loss = autoencoder.history.history['val_loss']
    # 绘制训练过程图
    # plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Kfold' + str(count_iter) + 'Validation_Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 保存训练过程图
    plt.savefig(checkpoint_dir + '/ae_training_' + str(count_iter) + 'fold' + '.png')

    encoder_model.save(checkpoint_dir + "/encoder_model_" + str(count_iter) + 'fold' + '.hdf5')

    # # Encode the DNase-seq data.
    # kmer_train_X = encoder_model.predict(X_kmer_train).reshape(-1, 128)
    # kmer_test_X = encoder_model.predict(X_kmer_test).reshape(-1, 128)
    # np.savetxt(checkpoint_dir + '/X_kmer_train' + '_' + str(count_iter) + 'fold' + '.txt', kmer_train_X, encoding='utf_8_sig', fmt='%d', delimiter=' ')
    # return kmer_train_X, kmer_test_X
    return 0


if __name__ == "__main__":
    # Hyperparameter settings
    EPCOHS = [50, 60, 70, 80, 90, 110, 120, 130, 140, 150]
    for epoch in EPCOHS:
        checkpoint_dir = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/autoencode'  # Store model files
        if not os.path.exists(checkpoint_dir):  # if the path does not exist, generate it
            os.mkdir(checkpoint_dir)
    
        checkpoint_dir = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/autoencode/' + 'eopchis' + str(
            epoch)  # Store model files
        if not os.path.exists(checkpoint_dir):  # if the path does not exist, generate it
            os.mkdir(checkpoint_dir)
        
        # load kmer feature
        filename1 = "/data/lyli/Silencer/K562/datasets/6924/5fold/K562_kmer_data.txt"
        X_kmer = load_kmer_feature(filename1)
        X_kmer = np.array(X_kmer)
        Y_kmer = [1] * 6923 + [0] * 6923
        Y_kmer = np.array(Y_kmer)
        X_kmer, Y_kmer = shuffle(X_kmer, Y_kmer, random_state=1)
        X_kmer = X_kmer.reshape(-1, 1024, 1)
    
        folds = StratifiedKFold(5)
        count_iter = 1  # Record the number of cycles
        for train_index, test_index in folds.split(X_kmer, Y_kmer):
            # train_index,test_index是数组索引，如[1 2 3]
            K.clear_session()  # Release graph
            X_kmer_train = X_kmer[train_index]  # X_train_first是数组
            X_kmer_test = X_kmer[test_index]
    
            Y_train = Y_kmer[train_index]
            Y_test = Y_kmer[test_index]
    
            autoencode_train(count_iter, epoch, X_kmer_train)
            # kmer_train_X = encoder_model.predict(X_kmer_train).reshape(-1, 128)
            # kmer_test_X = encoder_model.predict(X_kmer_test).reshape(-1, 128)
            # print(kmer_test_X)
            K.clear_session()
            del X_kmer_train
            del X_kmer_test
            del Y_train
            del Y_test
            count_iter += 1


