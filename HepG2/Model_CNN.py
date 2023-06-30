from keras.layers import Concatenate
from keras.layers import Dropout, Flatten, BatchNormalization, Conv1D, Lambda, Input, Dense, Softmax, Permute, Dot
from keras.optimizers import Adam
from keras.models import Model


def model_cnn(model_ind, EPCOHS, input_seq_length, input_omics_length):
    '''
    Define the classification model
    '''

    def att(drop2):
        attention = Dense(64, activation='tanh')(drop2)
        attention = Dense(1)(attention)
        attention = Softmax(axis=1)(attention)
        return attention

    if model_ind == "DNA":
        input_seq = Input(shape=(input_seq_length,))  # (?, 128)
        seq_dense1 = Dense(128, name="chip_first_layer")(input_seq)  # (?, 128)
        seq_dense2 = Dense(64, name="chip_second_layer", activation="relu")(seq_dense1)  # (?, 64)
        seq_dense3 = Dense(64, activation='relu')(seq_dense2)  # (?, 64)
        seq_dense4 = Dense(128, activation='relu')(seq_dense3)  # (?, 128)
        seq_dense4 = Dropout(0.4)(seq_dense4)
        merge_seq_dense2_dense3 = Concatenate(axis=-1)([seq_dense2, seq_dense3])  # (?, 128)
        x = Concatenate(axis=1)([seq_dense1, merge_seq_dense2_dense3, seq_dense4])
        dense1_ = Dense(512, activation='relu')
        dense1 = dense1_(x)
        dense2 = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3 = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[input_seq], output=[pred_output])

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001 / EPCOHS)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    if model_ind == "multi_omics":
        inputs_omics = Input(shape=(input_omics_length, 1))  # (?, 12, 1)
        omics_conv1_ = Conv1D(128, 8, activation='relu', border_mode='same')
        omics_conv1 = omics_conv1_(inputs_omics)  # (?, 12, 128)
        omics_conv2_ = Conv1D(64, 1, activation='relu', border_mode='same')
        omics_conv2 = omics_conv2_(omics_conv1)  # (?, 12, 64)
        omics_conv3_ = Conv1D(64, 3, activation='relu', border_mode='same')
        omics_conv3 = omics_conv3_(omics_conv2)  # (?, 12, 64)
        omics_conv4_ = Conv1D(128, 1, activation='relu', border_mode='same')
        omics_conv4 = omics_conv4_(omics_conv3)  # (?, 12, 128)
        omics_drop2 = Dropout(0.5)(omics_conv4)  # (?, 12, 128)

        attention = att(omics_drop2)  # (?, 12, 1)
        omics_drop2 = Permute((2, 1))(omics_drop2)  # (?, 128, 12)
        omics_drop2 = Dot(axes=(2, 1))([omics_drop2, attention])  # (?, 128, 1)
        omics_drop2 = Permute((2, 1))(omics_drop2)  # (?, 1, 128)
        merge_omics_conv2_conv3 = Concatenate(axis=-1)([omics_conv2, omics_conv3])  # (?, 12, 128)
        x = Concatenate(axis=1)([omics_drop2, merge_omics_conv2_conv3, omics_conv1])
        x = Flatten()(x)
        dense1_ = Dense(512, activation='relu')
        dense1 = dense1_(x)
        dense2 = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3 = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[inputs_omics], output=[pred_output])

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001 / EPCOHS)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    if model_ind == "connection":
        input_seq = Input(shape=(input_seq_length,))  # (?, 128)
        seq_dense1 = Dense(128, name="chip_first_layer")(input_seq)  # (?, 128)
        seq_dense1_ = Lambda(lambda x: x[:, :, None])(seq_dense1)  # (?, 128, 1)
        seq_dense__ = Permute((2, 1))(seq_dense1_)  # (?, 1, 128)
        seq_dense2 = Dense(64, name="chip_second_layer", activation="relu")(seq_dense1)  # (?, 64)
        seq_dense3 = Dense(64, activation='relu')(seq_dense2)  # (?, 64)
        seq_dense4 = Dense(128, activation='relu')(seq_dense3)  # (?, 128)
        seq_dense4 = Dropout(0.4)(seq_dense4)
        seq_dense4_ = Lambda(lambda x: x[:, :, None])(seq_dense4)  # (?, 128 1)
        seq_dense4__ = Permute((2, 1))(seq_dense4_)  # (?, 1, 128)

        inputs_omics = Input(shape=(input_omics_length, 1))  # (?, 12, 1)
        omics_conv1_ = Conv1D(128, 8, activation='relu', border_mode='same')
        omics_conv1 = omics_conv1_(inputs_omics)  # (?, 12, 128)
        omics_conv2_ = Conv1D(64, 1, activation='relu', border_mode='same')
        omics_conv2 = omics_conv2_(omics_conv1)  # (?, 12, 64)
        omics_conv3_ = Conv1D(64, 3, activation='relu', border_mode='same')
        omics_conv3 = omics_conv3_(omics_conv2)  # (?, 12, 64)
        omics_conv4_ = Conv1D(128, 1, activation='relu', border_mode='same')
        omics_conv4 = omics_conv4_(omics_conv3)  # (?, 12, 128)
        omics_drop2 = Dropout(0.5)(omics_conv4)  # (?, 12, 128)

        attention = att(omics_drop2)  # (?, 12, 1)
        omics_drop2 = Permute((2, 1))(omics_drop2)  # (?, 128, 12)
        omics_drop2 = Dot(axes=(2, 1))([omics_drop2, attention])  # (?, 128, 1)
        omics_drop2 = Permute((2, 1))(omics_drop2)  # (?, 1, 128)

        # merge_seq_conv2_conv3 = Concatenate(axis=-1)([seq_conv2, seq_conv3])    # (?, 9, 128)
        # merge_seq_omics_first = Concatenate(axis=1)([seq_dense__, omics_conv1])  # (?, 13, 128)
        merge_seq_dense2_dense3 = Concatenate(axis=-1)([seq_dense2, seq_dense3])  # (?, 128)
        merge_seq_dense2_dense3 = Lambda(lambda x: x[:, :, None])(merge_seq_dense2_dense3)  # (?, 128 1)
        merge_seq_dense2_dense3 = Permute((2, 1))(merge_seq_dense2_dense3)  # (?, 1, 128)

        merge_omics_conv2_conv3 = Concatenate(axis=-1)([omics_conv2, omics_conv3])  # (?, 12, 128)
        merge_seq_omics_last = Concatenate(axis=1)([seq_dense4__, omics_drop2])  # (?, 2, 128)
        x = Concatenate(axis=1)([seq_dense__, merge_seq_dense2_dense3, merge_seq_omics_last, merge_omics_conv2_conv3, omics_conv1])
        x = Flatten()(x)
        dense1_ = Dense(512, activation='relu')
        dense1 = dense1_(x)
        dense2 = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3 = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[input_seq, inputs_omics], output=[pred_output])

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001 / EPCOHS)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model
