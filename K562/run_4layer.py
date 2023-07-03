















import os
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import Model_CNN
# import autoencode
from keras import backend as K
from tensorflow.keras.models import load_model
import pandas as pd

# Hyperparameter settings
BATCH_SIZE = 64
EPCOHS = 10
encode_epochs = 100

checkpoint_dir = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/checkpoints_4layer_' + str(encode_epochs)  # Store model files
if not os.path.exists(checkpoint_dir):  # if the path does not exist, generate it
    os.mkdir(checkpoint_dir)

checkpoint_dir = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/checkpoints_4layer_' + str(encode_epochs) + '-/final_model'  # Store model files
if not os.path.exists(checkpoint_dir):  # if the path does not exist, generate it
    os.mkdir(checkpoint_dir)

encode_model = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/autoencode'  # Store model files
if not os.path.exists(encode_model):  # if the path does not exist, generate it
    os.mkdir(encode_model)

encode_model = '/data/lyli/Silencer/K562/datasets/6924/5fold/model/autoencode/' + 'eopchis' + str(encode_epochs)  # Store model files
if not os.path.exists(encode_model):  # if the path does not exist, generate it
    os.mkdir(encode_model)


# load multi_omics data features
def load_omics_feature(filename1, filename2):
    filelist = [filename1, filename2]
    dataMat = []
    for file in filelist:
        file0 = np.loadtxt(file, encoding='utf_8_sig', delimiter=' ')
        for line in file0:
            curLine = list(line)
            floatLine = map(float, curLine)  # 这里使用的是map函数直接把数据转化成为float类型
            floatLine = list(floatLine)
            dataMat.append(floatLine[0:13])

    labelMat = []
    for i in dataMat:
        labelMat.append(int(i[-1]))
        del (i[-1])

    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    return dataMat, labelMat


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


def evaluation(y1, y2):
    y_pred_prob = y1
    y_label = y2
    y_pred_class = np.where(np.array(y_pred_prob) >= 0.5, 1, 0)
    TN, FP, FN, TP = metrics.confusion_matrix(y_label, y_pred_class).ravel()
    # print(metrics.confusion_matrix(y_label, y_pred_class))
    ACC = metrics.accuracy_score(y_label, y_pred_class)
    Sens = metrics.recall_score(y_label, y_pred_class)
    Spec = TN / float(TN + FP)
    MCC = metrics.matthews_corrcoef(y_label, y_pred_class)
    AUC = metrics.roc_auc_score(y_label, y_pred_prob)

    # fpr, tpr, thresholds = roc_curve(test_label, y_pred[:, 0])
    precision, recall, thresholds = precision_recall_curve(y_label, y_pred_prob)
    # auc = roc_auc_score(test_label, y_pred[:, 0])
    # print("AUC(ROC):" + str(auc))
    PRC_AUC = metrics.auc(recall, precision)
    # print("AUC(PRC):" + str(area))

    return ACC, Sens, Spec, MCC, AUC, PRC_AUC, precision, recall


if __name__ == "__main__":
    # load omics feature
    filename1 = "/data/lyli/Silencer/K562/datasets/6924/5fold/0K562_silencer_feature12_6924_lable.txt"
    filename2 = "/data/lyli/Silencer/K562/datasets/6924/5fold/0K562_negative_feature12_6924_lable.txt"
    X_omics, Y_omics = load_omics_feature(filename1, filename2)
    Y_omics = np.array(Y_omics)
    X_omics, Y_omics = shuffle(X_omics, Y_omics, random_state=1)
    X_omics = X_omics.reshape(-1, 12, 1)

    # load kmer feature
    filename3 = "/data/lyli/Silencer/K562/datasets/6924/5fold/K562_kmer_data.txt"
    X_kmer = load_kmer_feature(filename3)
    X_kmer = np.array(X_kmer)
    Y_kmer = [1] * 6923 + [0] * 6923
    Y_kmer = np.array(Y_kmer)
    X_kmer, Y_kmer = shuffle(X_kmer, Y_kmer, random_state=1)
    X_kmer = X_kmer.reshape(-1, 1024, 1)

    ACC_sum = []
    Sens_sum = []
    Spec_sum = []
    MCC_sum = []
    ROC_sum = []
    PRC_sum = []
    y_pred_5fold = []
    model_inds = ["DNA", "multi_omics", "connection"]
    model_ind = "connection"
    folds = StratifiedKFold(5)
    count_iter = 1  # Record the number of cycles
    for train_index, test_index in folds.split(X_kmer, Y_kmer):
        # train_index,test_index是数组索引，如[1 2 3]
        K.clear_session()  # Release graph
        X_omics_train = X_omics[train_index]  # X_train_first是数组
        X_omics_test = X_omics[test_index][:2768]

        X_kmer_train = X_kmer[train_index]  # X_train_first是数组
        X_kmer_test = X_kmer[test_index][:2768]

        Y_train = Y_kmer[train_index]
        Y_test = Y_kmer[test_index][:2768]

        # 加载编码器模型
        encoder_model = load_model(encode_model + "/encoder_model_" + str(count_iter) + 'fold' + '.hdf5')
        # Encode the k-mer data.
        kmer_train_X = encoder_model.predict(X_kmer_train).reshape(-1, 128)
        kmer_test_X = encoder_model.predict(X_kmer_test).reshape(-1, 128)

        if model_ind == "DNA":
            model = Model_CNN.model_cnn(model_inds[0], EPCOHS, 128, 12)  # Get the first layer model
            saveBestModel = ModelCheckpoint(checkpoint_dir + "/model_DNA" + str(count_iter) + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [saveBestModel]
            print('Fitting first model...')
            history = model.fit(kmer_train_X, Y_train, batch_size=BATCH_SIZE, epochs=EPCOHS, callbacks=callbacks_list, validation_split=0.1, verbose=1)  # Start training
            # serialize model to JSON
            model_json = model.to_json()
            with open("model_DNA" + str(count_iter) + ".json", "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            model.save_weights("model_DNA" + str(count_iter) + ".h5")
            print("Saved model to disk")

            # load model
            model_first = Model_CNN.model_cnn(model_inds[0], EPCOHS, 128, 12)
            model_first.load_weights(checkpoint_dir + "/model_DNA" + str(count_iter) + ".hdf5")
            # model_first.load_weights("/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/model_2merl.h5")
            y_pred = model_first.predict(kmer_test_X, batch_size=BATCH_SIZE)
            # print(y_pred)
            test_label = np.array(Y_test)
            auc2 = metrics.roc_auc_score(test_label, y_pred[:, 0])
            print(auc2)

            # print(np.shape(ypred))
            auc2 = metrics.roc_auc_score(test_label, y_pred[:, 0])
            print(auc2)
            fpr, tpr, thresholds = roc_curve(test_label, y_pred[:, 0])
            precision, recall, thresholds = precision_recall_curve(test_label, y_pred[:, 0])
            auc = roc_auc_score(test_label, y_pred[:, 0])
            print("AUC(ROC):" + str(auc))
            area = metrics.auc(recall, precision)
            print("AUC(PRC):" + str(area))
            ROC_sum.append(auc)
            PRC_sum.append(area)

            K.clear_session()
            del X_omics_train
            del X_omics_test
            del X_kmer_train
            del X_kmer_test
            del Y_train
            del Y_test

            count_iter += 1

        if model_ind == "multi_omics":
            model = Model_CNN.model_cnn(model_inds[1], EPCOHS, 128, 12)  # Get the first layer model
            saveBestModel = ModelCheckpoint(checkpoint_dir + "/model_multi_omics" + str(count_iter) + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [saveBestModel]
            print('Fitting first model...')
            history = model.fit(X_omics_train, Y_train, batch_size=BATCH_SIZE, epochs=EPCOHS, callbacks=callbacks_list, validation_split=0.1, verbose=1)  # Start training
            # serialize model to JSON
            model_json = model.to_json()
            with open("model_multi_omics" + str(count_iter) + ".json", "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            model.save_weights("model_multi_omics" + str(count_iter) + ".h5")
            print("Saved model to disk")

            # load model
            model_first = Model_CNN.model_cnn(model_inds[1], EPCOHS, 128, 12)
            model_first.load_weights(checkpoint_dir + "/model_multi_omics" + str(count_iter) + ".hdf5")
            # model_first.load_weights("/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/model_2merl.h5")
            y_pred = model_first.predict(X_omics_test, batch_size=BATCH_SIZE)
            # print(y_pred)
            test_label = np.array(Y_test)
            auc2 = metrics.roc_auc_score(test_label, y_pred[:, 0])
            print(auc2)

            # print(np.shape(ypred))
            auc2 = metrics.roc_auc_score(test_label, y_pred[:, 0])
            print(auc2)
            fpr, tpr, thresholds = roc_curve(test_label, y_pred[:, 0])
            precision, recall, thresholds = precision_recall_curve(test_label, y_pred[:, 0])
            auc = roc_auc_score(test_label, y_pred[:, 0])
            print("AUC(ROC):" + str(auc))
            area = metrics.auc(recall, precision)
            print("AUC(PRC):" + str(area))
            ROC_sum.append(auc)
            PRC_sum.append(area)

            K.clear_session()
            del X_omics_train
            del X_omics_test
            del X_kmer_train
            del X_kmer_test
            del Y_train
            del Y_test

            count_iter += 1

        if model_ind == "connection":
            # model = Model_CNN.model_cnn(model_inds[2], EPCOHS, 128, 12)  # Get the first layer model
            # saveBestModel = ModelCheckpoint(checkpoint_dir + "/model" + str(count_iter) + ".hdf5", monitor='val_acc',
            #                                 verbose=1, save_best_only=True, mode='max')
            # callbacks_list = [saveBestModel]
            # print('Fitting first model...')
            # history = model.fit([kmer_train_X, X_omics_train], Y_train, batch_size=BATCH_SIZE, epochs=EPCOHS,
            #                     callbacks=callbacks_list, validation_split=0.1, verbose=1)  # Start training
            # # serialize model to JSON
            # model_json = model.to_json()
            # with open("model_connection" + str(count_iter) + ".json", "w") as json_file:
            #     json_file.write(model_json)
            #     # serialize weights to HDF5
            # model.save_weights("model_connection" + str(count_iter) + ".h5")
            # print("Saved model to disk")

            # load model
            model_first = Model_CNN.model_cnn(model_inds[2], EPCOHS, 128, 12)
            model_first.load_weights(checkpoint_dir + "/model" + str(count_iter) + ".hdf5")
            # model_first.load_weights("/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/model_2merl.h5")
            y_pred = model_first.predict([kmer_test_X, X_omics_test], batch_size=BATCH_SIZE)
            test_label = np.array(Y_test)
            ACC, Sens, Spec, MCC, ROC_AUC, PRC_ROC, precision, recall = evaluation(y_pred, test_label)
            print('ACC:', ACC)
            print('Sens:', Sens)
            print('Spec:', Spec)
            print('MCC:', MCC)
            print('ROC_AUC:', ROC_AUC)
            print('PRC_ROC:', PRC_ROC)
            ACC_sum.append(ACC)
            Sens_sum.append(Sens)
            Spec_sum.append(Spec)
            MCC_sum.append(MCC)
            ROC_sum.append(ROC_AUC)
            PRC_sum.append(PRC_ROC)
            y_pred_5fold.append(y_pred)

            # 绘制ROC所需的数据
            dataframe = pd.DataFrame({'真实标签': test_label, '预测概率': y_pred[:, 0]})
            dataframe.to_csv(checkpoint_dir + "K562_our_method" + str(count_iter) + ".csv", index=False, sep=',')

            # 绘制PR所需的数据
            dataframe1 = pd.DataFrame({'precision': precision, 'recall': recall})
            dataframe1.to_csv(checkpoint_dir + "K562_our_method_PR" + str(count_iter) + ".csv", index=False, sep=',')

            K.clear_session()
            del X_omics_train
            del X_omics_test
            del X_kmer_train
            del X_kmer_test
            del Y_train
            del Y_test

            count_iter += 1

    print('ACC_average:', sum(ACC_sum) / 5)
    print('Sens_average:', sum(Sens_sum) / 5)
    print('Spec_average:', sum(Spec_sum) / 5)
    print('MCC_average:', sum(MCC_sum) / 5)
    print('ROC_average:', sum(ROC_sum) / 5)
    print('PRC_average:', sum(PRC_sum) / 5)
    # 将这五个数组相加
    y_pred_sum_array = np.sum([y_pred_5fold[0], y_pred_5fold[1], y_pred_5fold[2], y_pred_5fold[3], y_pred_5fold[4]],axis=0)
    dataframe = pd.DataFrame({'真实标签': test_label, '预测概率': y_pred_sum_array[:, 0]})
    # dataframe.to_csv(checkpoint_dir + "K562_our_only-omics_method_average.csv", index=False, sep=',')
    dataframe.to_csv(checkpoint_dir + "K562_our_method_average.csv", index=False, sep=',')