import math
import os
import random
import sys
from datetime import datetime

import keras
from keras import backend as K
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import sklearn as sk


from dumb_regressor import dumb_regressor_result
from global_parameters import *
from direct_controller import Controller
from model_creator import model_creator, generator
from utils import isdebugging


def train_model(idx, batch_size, epochs, model_name, save_dir, xs_train, xs_validation, xs_test, ys_train, ys_validation, model_type, summary_f, ys_test=None, save_model=True):
    """
         Cnn method runs:
            -train
            -test
            -save model as a h5py file

    Args:
        idx:
        save_model:
        summary_f:
        ys_test:
        ys_validation:
        ys_train:
        xs_validation:
        tr_size: dimension of the train set
        model_type:
        xs_train:
        xs_test:
        batch_size: size of a batch
        epochs: number of epochs
        model_name: name of the model, used for naming saved models
        save_dir: directory of the running test folder

    Returns:
        history: metric history
        y_pred: prediction on test set
    """
    if model_type == "v1":
        x_train = xs_train[idx, :]
        y_train = ys_train[idx, :]
        x_validation = xs_validation
        y_validation = ys_validation
        x_test = xs_test
        batch_per_epoch = math.ceil(x_train.shape[0] / batch_size)
        model = model_creator(show_summary=summary_f, model_type=model_type)
        gen = generator(x_train, y_train, batch_size, model_type=model_type)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.1e-6)
        if save_model:
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = save_dir + model_name
            model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
            history = model.fit_generator(generator=gen,
                                          validation_data=(x_validation, [y_validation[:, 0], y_validation[:, 1], y_validation[:, 2], y_validation[:, 3]]),
                                          epochs=epochs,
                                          steps_per_epoch=batch_per_epoch,
                                          callbacks=[lr_reducer, early_stop, model_checkpoint],
                                          verbose=0)
        else:
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
            history = model.fit_generator(generator=gen,
                                          validation_data=(x_validation, [y_validation[:, 0], y_validation[:, 1], y_validation[:, 2], y_validation[:, 3]]),
                                          epochs=epochs,
                                          steps_per_epoch=batch_per_epoch,
                                          callbacks=[lr_reducer, early_stop],
                                          verbose=2)
        y_pred = model.predict(x_test)
        return history, y_pred

    elif model_type == "v2":
        x1_train = xs_train[0]
        x1_train = x1_train[idx, :]
        x2_train = xs_train[1]
        x2_train = x2_train[idx, :]
        y_train = ys_train[idx, :]
        x1_validation = xs_validation[0]
        x2_validation = xs_validation[1]
        y_validation = ys_validation
        x1_test = xs_test[0]
        x2_test = xs_test[1]

        model = model_creator(show_summary=summary_f, model_type=model_type)
        batch_per_epoch = math.ceil(x1_train.shape[0] / batch_size)
        gen = generator([x1_train, x2_train], y_train, batch_size, model_type=model_type)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.1e-6)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + model_name
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        history = model.fit_generator(generator=gen,
                                      validation_data=([x1_validation, x2_validation], [y_validation[:, 0], y_validation[:, 1], y_validation[:, 2], y_validation[:, 3]]),
                                      epochs=epochs,
                                      steps_per_epoch=batch_per_epoch,
                                      callbacks=[lr_reducer, early_stop, model_checkpoint],
                                      verbose=0)
        y_pred = model.predict([x1_test, x2_test])
        return history, y_pred

    elif model_type == "v3":

        x2_train = xs_train[0]
        x2_train = x2_train[idx, :]
        x3_train = xs_train[1]
        x3_train = x3_train[idx, :]
        x2_test = xs_test[0]
        x3_test = xs_test[1]

        x2_validation = xs_validation[0]
        x3_validation = xs_validation[1]
        y_validation = ys_validation
        y_train = ys_train[idx, :]
        model = model_creator(show_summary=summary_f, model_type=model_type)
        batch_per_epoch = math.ceil(x2_train.shape[0] / batch_size)
        gen = generator([x2_train, x3_train], y_train, batch_size, model_type=model_type)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.1e-6)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + model_name
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        history = model.fit_generator(generator=gen,
                                      validation_data=([x2_validation, x3_validation], [y_validation[:, 0], y_validation[:, 1], y_validation[:, 2], y_validation[:, 3]]),
                                      epochs=epochs,
                                      steps_per_epoch=batch_per_epoch,
                                      callbacks=[lr_reducer, early_stop, model_checkpoint],
                                      verbose=0)
        y_pred = model.predict([x2_test, x3_test])
        return history, y_pred
    else:
        print("ERROR: Model type is not correct: " + model_type + " in CNNMethod")
        sys.exit(1)


# ------------------- Main ----------------------
def main():
    """
        -read pickle file for train and validation
        -calls method for train and predict
        -calls method to run dumb prediction
        -calls method to create a video for qualitative evaluation
        -calls method to plot data for quantitative evaluation
    """

    scelta = raw_input("experiment_version:\n        -v1[1]\n        -v2[2]\n        -v3[3]\n        -1,2,3[4]\n        -only v3[5]\n")
    summary_f = raw_input("summary:[y/n]") == "y"
    if isdebugging():
        print("debugging-settings")
        batch_size = 64
        epochs = 2
    else:
        epochs = int(raw_input("epochs:"))
        batch_size = 64
        epochs = epochs
        # epochs = 2

    start_time = datetime.now()
    if scelta == "1":
        K.clear_session()
        model_type = "v1"
        train_set = pd.read_pickle("../dataset/version1/train.pickle").values
        test_set = pd.read_pickle("../dataset/version1/test.pickle").values
        print('train shape: ' + str(train_set.shape))
        print('test shape: ' + str(test_set.shape))
        save_dir = '../saved_models'
        n_val = 13000
        np.random.seed(100)
        # split between train and test sets:
        x_train = 255 - train_set[:, 0]  # otherwise is inverted
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))
        y_train = train_set[:, 1]
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(x_train.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        x_test = 255 - test_set[:, 0]
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
        y_test = test_set[:, 1]
        y_test = np.vstack(y_test[:]).astype(np.float32)
        visual_odom = test_set[:, 2]
        visual_odom = np.vstack(visual_odom[:]).astype(np.float32)

        model_name = "Model_v1"
        shape_ = x_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        history, y_pred = train_model(sel_idx, batch_size, epochs, model_name, save_dir, x_train, x_validation, x_test, y_train, y_validation, save_model=False, model_type=model_type,
                                      summary_f=summary_f)
        dumb_metrics = dumb_regressor_result(x_test, x_train, y_test, y_train)
        # plot_results_history_v1(history.history, y_pred, y_test, dumb_metrics)

        d_ctrl = Controller()
        for i in range(y_pred[0].shape[0]):
            v_drone = [visual_odom[i][0], visual_odom[i][1], 0]
            p = [y_pred[0][i][0], y_pred[1][i][0], y_pred[2][i][0]]
            yaw_ = y_pred[3][i][0]
            res = d_ctrl.new_controller(p, yaw_, v_drone)
            y_pred[0][i][0] = res[0]
            y_pred[1][i][0] = res[1]
            y_pred[2][i][0] = res[2]
            y_pred[3][i][0] = res[3]
        test_set_2 = pd.read_pickle("../dataset/version2/test.pickle").values
        y_test_2 = test_set_2[:, 2]
        y_test_2 = np.asarray([np.asarray(sublist) for sublist in y_test_2])


    elif scelta == "2":
        model_type = "v2"
        train_set = pd.read_pickle("../dataset/version2/train.pickle").values
        test_set = pd.read_pickle("../dataset/version2/test.pickle").values
        print('train shape: ' + str(train_set.shape))
        print('test shape: ' + str(test_set.shape))
        save_dir = '../saved_models'
        model_name = 'keras_bebop_trained_model_v2.h5'
        n_val = 13000
        np.random.seed(100)
        # split between train and test sets:
        x1_train = 255 - train_set[:, 0]  # otherwise is inverted
        ix_val, ix_tr = np.split(np.random.permutation(x1_train.shape[0]), [n_val])

        x1_train = np.vstack(x1_train[:]).astype(np.float32)
        x1_train = np.reshape(x1_train, (-1, image_height, image_width, 3))
        x2_train = train_set[:, 1]
        x2_train = np.vstack(x2_train[:]).astype(np.float32)
        y_train = train_set[:, 2]
        y_train = np.vstack(y_train[:]).astype(np.float32)
        del train_set
        # y_train = np.asarray([np.asarray(sublist) for sublist in y_train])

        x1_validation = x1_train[ix_val, :]
        x1_train = x1_train[ix_tr, :]
        x2_validation = x2_train[ix_val, :]
        x2_train = x2_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        x1_test = 255 - test_set[:, 0]
        x1_test = np.vstack(x1_test[:]).astype(np.float32)
        x1_test = np.reshape(x1_test, (-1, image_height, image_width, 3))
        x2_test = test_set[:, 1]
        x2_test = np.vstack(x2_test[:]).astype(np.float32)  # needed by keras or TF for correct input shape
        y_test = test_set[:, 2]
        y_test = np.vstack(y_test[:]).astype(np.float32)
        del test_set
        shape_ = x1_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        history, y_pred = train_model(sel_idx, batch_size, epochs, model_name, save_dir, (x1_train, x2_train), (x1_validation, x2_validation), (x1_test, x2_test), y_train, y_validation,
                                      save_model=False,
                                      model_type=model_type, summary_f=summary_f)
        dumb_metrics = dumb_regressor_result(x1_test, x1_train, y_test, y_train)

    elif scelta == "3":
        model_type = "v3"
        train = pd.read_pickle("../dataset/version3/train.pickle").values
        test = pd.read_pickle("../dataset/version3/test.pickle").values
        # img, rel_head, np.asarray([vel_x, vel_y]), target
        # split between train and test sets:
        n_val = 13000
        np.random.seed(100)
        x2_train = train[:, 1]
        x2_train = np.vstack(x2_train[:]).astype(np.float32)
        ix_val, ix_tr = np.split(np.random.permutation(x2_train.shape[0]), [n_val])
        x3_train = train[:, 2]
        x3_train = np.vstack(x3_train[:]).astype(np.float32)
        y_train = train[:, 3]
        y_train = np.asarray([np.asarray(sublist) for sublist in y_train])

        x2_validation = x2_train[ix_val, :]
        x2_train = x2_train[ix_tr, :]
        x3_validation = x3_train[ix_val, :]
        x3_train = x3_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        x1_test = 255 - test[:, 0]
        x1_test = np.vstack(x1_test[:]).astype(np.float32)
        x1_test = np.reshape(x1_test, (-1, image_height, image_width, 3))


        v1_model = keras.models.load_model("PATH TO V1MODEL")
        v1_pred = v1_model.predict(x1_test)
        K.clear_session()
        x2_test = np.swapaxes(np.squeeze(v1_pred), 0, 1)
        x3_test = test[:, 2]
        x3_test = np.vstack(x3_test[:]).astype(np.float32)  # needed by keras or TF for correct input shape
        y_test = test[:, 3]
        y_test = np.asarray([np.asarray(sublist) for sublist in y_test])
        save_dir = '../saved_models'
        model_name = '/keras_bebop_trained_model_v3.h5'
        shape_ = x2_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        history, y_pred = train_model(sel_idx, batch_size, epochs, model_name, save_dir, (x2_train, x3_train), (x2_validation, x3_validation), (x2_test, x3_test), y_train, y_validation,
                                      save_model=True, model_type=model_type,
                                      summary_f=summary_f, ys_test=y_test)
        dumb_metrics = dumb_regressor_result(x2_test, x2_train, y_test, y_train)

    elif scelta == "4":
        save_path = '../runs_save_files/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        try:
            os.makedirs(save_path)
        except OSError:
            if not os.path.isdir(save_path):
                raise
        v1_history_list = []
        v1_y_pred_list = []
        v1_rmse_list = []
        v1_r2_list = []

        v2_history_list = []
        v2_y_pred_list = []
        v2_rmse_list = []
        v2_r2_list = []

        v3_history_list = []
        v3_y_pred_list = []
        v3_rmse_list = []
        v3_r2_list = []
        v3_x2_test_list = []
        idx_list = []
        for k, v in enumerate(train_size):
            save_dir = save_path + '/models'
            v1_data, v2_data, v3_data = data_preparation()

            v1_rmse_rep_list = []
            v1_r2_rep_list = []
            v1_y_pred_rep_list = []
            v1_history_rep_list = []

            v2_rmse_rep_list = []
            v2_r2_rep_list = []
            v2_y_pred_rep_list = []
            v2_history_rep_list = []
            #
            v3_rmse_rep_list = []
            v3_r2_rep_list = []
            v3_y_pred_rep_list = []
            v3_history_rep_list = []

            v3_x2_test_rep_list = []

            idx_rep_list = []
            for rep in range(experiment_repetitions[k]):
                random.seed(None)
                shape_ = v1_data['v1_x_train'].shape[0]
                sel_idx = random.sample(range(0, shape_), k=v)
                idx_rep_list.append(sel_idx)
            idx_list.append(idx_rep_list)

            for rep in range(experiment_repetitions[k]):
                sel_idx = idx_rep_list[rep]
                y_pred, v3_x_test, history = exp_model_v1(sel_idx, batch_size, epochs, rep, save_dir, summary_f, v,
                                                          v1_data['visual_odom'], v1_data['v1_x_test'], v1_data['v1_x_train'], v1_data['v1_x_validation'], v1_data['v1_y_train'], v1_data['v1_y_validation'])
                r2_score, rmse_score = compute_metrics(v1_data['v1_y_test'], y_pred)
                v1_rmse_rep_list.append(rmse_score)
                v1_r2_rep_list.append(r2_score)
                v1_history_rep_list.append(history.history)
                v1_y_pred_rep_list.append(y_pred)
                v3_x2_test_rep_list.append(np.swapaxes(np.squeeze(v3_x_test), 0, 1))

            v1_history_list.append(v1_history_rep_list)
            v1_y_pred_list.append(v1_y_pred_rep_list)
            v1_rmse_list.append([np.std(np.array(v1_rmse_rep_list), axis=0), np.mean(np.array(v1_rmse_rep_list), axis=0)])
            v1_r2_list.append([np.std(np.array(v1_r2_rep_list), axis=0), np.mean(np.array(v1_r2_rep_list), axis=0)])
            v3_x2_test_list.append(v3_x2_test_rep_list)
            K.clear_session()

            for rep in range(experiment_repetitions[k]):
                sel_idx = idx_rep_list[rep]
                history, y_pred = exp_model_v2(sel_idx, batch_size, epochs, rep, save_dir, summary_f, v,
                                               v2_data['v2_x1_test'], v2_data['v2_x2_train'], v2_data['v2_x1_validation'], v2_data['v2_x2_test'], v2_data['v2_x2_train'], v2_data['v2_x2_validation'], v2_data['v2_y_train'], v2_data['v2_y_validation'])
                r2_score, rmse_score = compute_metrics(v2_data['v2_y_test'], y_pred)
                v2_rmse_rep_list.append(rmse_score)
                v2_r2_rep_list.append(r2_score)
                v2_history_rep_list.append(history.history)
                v2_y_pred_rep_list.append(y_pred)
            v2_rmse_list.append([np.std(np.array(v2_rmse_rep_list), axis=0), np.mean(np.array(v2_rmse_rep_list), axis=0)])
            v2_r2_list.append([np.std(np.array(v2_r2_rep_list), axis=0), np.mean(np.array(v2_r2_rep_list), axis=0)])
            v2_history_list.append(v2_history_rep_list)
            v2_y_pred_list.append(v2_y_pred_rep_list)
            K.clear_session()

            for rep in range(experiment_repetitions[k]):
                sel_idx = idx_rep_list[rep]
                v3_x2_test = v3_x2_test_rep_list[rep]
                history, y_pred = exp_model_v3(sel_idx, batch_size, epochs, rep, save_dir, summary_f, v,
                                               v3_x2_test, v3_data['v3_x2_train'], v3_data['v3_x2_validation'], v3_data['v3_x3_test'], v3_data[' v3_x3_train'], v3_data['v3_x3_validation'], v3_data['v3_y_train'], v3_data['v3_y_validation'])
                r2_score, rmse_score = compute_metrics(v3_data['v3_y_test'], y_pred)
                v3_rmse_rep_list.append(rmse_score)
                v3_r2_rep_list.append(r2_score)
                v3_history_rep_list.append(history.history)
                v3_y_pred_rep_list.append(y_pred)

            v3_history_list.append(v3_history_rep_list)
            v3_y_pred_list.append(v3_y_pred_rep_list)
            v3_rmse_list.append([np.std(np.array(v3_rmse_rep_list), axis=0), np.mean(np.array(v3_rmse_rep_list), axis=0)])
            v3_r2_list.append([np.std(np.array(v3_r2_rep_list), axis=0), np.mean(np.array(v1_r2_rep_list), axis=0)])
            K.clear_session()

        dumb_metrics, y_test = general_dumb_metrics()

        save_data_to_pickle(dumb_metrics, idx_list, save_path, v1_history_list, v1_r2_list, v1_rmse_list, v1_y_pred_list, v2_history_list, v2_r2_list, v2_rmse_list, v2_y_pred_list, v3_history_list, v3_r2_list, v3_rmse_list, v3_y_pred_list, y_test,
                            v3_x2_test_list)
        end_time = datetime.now()
        # plot_exp_result((v1_rmse_list, v2_rmse_list, v3_rmse_list), (v1_r2_list, v2_r2_list, v3_r2_list), dumb_metrics, save_path)
    else:
        print("ERROR: Model flag selection is not correct in keras_train main() " + scelta)
        sys.exit(1)


def general_dumb_metrics():
    """
        Create dumb metrics to compare model versions
    Returns:
        dumb_metrics, y_test
    """
    train_set = pd.read_pickle("../dataset/version2/train.pickle").values
    test_set = pd.read_pickle("../dataset/version2/test.pickle").values
    n_val = 13000
    np.random.seed(100)
    # split between train and test sets:
    x1_train = 255 - train_set[:, 0]  # otherwise is inverted
    train_shape_ = x1_train.shape[0]
    ix_val, ix_tr = np.split(np.random.permutation(train_shape_), [n_val])
    x1_train = np.vstack(x1_train[:]).astype(np.float32)
    x1_train = np.reshape(x1_train, (-1, image_height, image_width, 3))
    x1_train = x1_train[ix_tr, :]
    y_train = train_set[:, 2]
    y_train = np.vstack(y_train[:]).astype(np.float32)
    y_train = y_train[ix_tr, :]
    x1_test = 255 - test_set[:, 0]
    x1_test = np.vstack(x1_test[:]).astype(np.float32)
    x1_test = np.reshape(x1_test, (-1, image_height, image_width, 3))
    y_test = test_set[:, 2]
    y_test = np.vstack(y_test[:]).astype(np.float32)
    dumb_metrics = dumb_regressor_result(x1_test, x1_train, y_test, y_train)
    return dumb_metrics, y_test


def save_data_to_pickle(dumb_metrics, idx_list, save_path, v1_history_list, v1_r2_list, v1_rmse_list, v1_y_pred_list, v2_history_list, v2_r2_list, v2_rmse_list, v2_y_pred_list, v3_history_list, v3_r2_list, v3_rmse_list, v3_y_pred_list, y_test,
                        v3_x2_test_list):
    """
        Function to save info about run
    Args:
       each parameter is a list to be saved
    """
    pd.DataFrame(y_test).to_pickle(save_path + "/y_test.pickle")
    pd.DataFrame(dumb_metrics).to_pickle(save_path + "/dumb_metrics.pickle")
    pd.DataFrame(idx_list).to_pickle(save_path + "/idx_list.pickle")

    pd.DataFrame(v1_history_list).to_pickle(save_path + "/v1_history_list.pickle")
    pd.DataFrame(v1_y_pred_list).to_pickle(save_path + "/v1_y_pred_list.pickle")
    pd.DataFrame(v1_rmse_list).to_pickle(save_path + "/v1_rmse_list.pickle")
    pd.DataFrame(v1_r2_list).to_pickle(save_path + "/v1_r2_list.pickle")

    pd.DataFrame(v2_history_list).to_pickle(save_path + "/v2_history_list.pickle")
    pd.DataFrame(v2_y_pred_list).to_pickle(save_path + "/v2_y_pred_list.pickle")
    pd.DataFrame(v2_rmse_list).to_pickle(save_path + "/v2_rmse_list.pickle")
    pd.DataFrame(v2_r2_list).to_pickle(save_path + "/v2_r2_list.pickle")

    pd.DataFrame(v3_history_list).to_pickle(save_path + "/v3_history_list.pickle")
    pd.DataFrame(v3_y_pred_list).to_pickle(save_path + "/v3_y_pred_list.pickle")
    pd.DataFrame(v3_rmse_list).to_pickle(save_path + "/v3_rmse_list.pickle")
    pd.DataFrame(v3_r2_list).to_pickle(save_path + "/v3_r2_list.pickle")
    pd.DataFrame(v3_x2_test_list).to_pickle((save_path + "/v3_x2_test_list.pickle"))


def data_preparation():
    """
        Generate dataset dictionaries one for each model
    Returns:
        v1_data, v2_data, v3_data
    """
    n_val = 13000
    np.random.seed(100)
    v1_train_set = pd.read_pickle("../dataset/version1/train.pickle").values
    v1_test_set = pd.read_pickle("../dataset/version1/test.pickle").values

    train_shape_ = v1_train_set[:, 0].shape[0]
    ix_val, ix_tr = np.split(np.random.permutation(train_shape_), [n_val])

    x_train = 255 - v1_train_set[:, 0]
    x_train = np.vstack(x_train[:]).astype(np.float32)
    x_train = np.reshape(x_train, (-1, image_height, image_width, 3))
    x_validation = x_train[ix_val, :]
    x_train = x_train[ix_tr, :]

    y_train = v1_train_set[:, 1]
    y_train = np.vstack(y_train[:]).astype(np.float32)
    v1_y_validation = y_train[ix_val, :]
    v1_y_train = y_train[ix_tr, :]

    x_test = 255 - v1_test_set[:, 0]
    x_test = np.vstack(x_test[:]).astype(np.float32)
    x_test = np.reshape(x_test, (-1, image_height, image_width, 3))

    v1_y_test = v1_test_set[:, 1]
    v1_y_test = np.vstack(v1_y_test[:]).astype(np.float32)

    visual_odom = v1_test_set[:, 2]
    visual_odom = np.vstack(visual_odom[:]).astype(np.float32)

    v1_data = {'v1_x_train': x_train, 'v1_x_validation': x_validation, 'v1_x_test': x_test,
               'v1_y_train': v1_y_train, 'v1_y_validation': v1_y_validation, 'v1_y_test': v1_y_test,
               'visual_odom': visual_odom}

    v2_train_set = pd.read_pickle("../dataset/version2/train.pickle").values
    v2_test_set = pd.read_pickle("../dataset/version2/test.pickle").values

    v2_x2_train = v2_train_set[:, 1]
    v2_x2_train = np.vstack(v2_x2_train[:]).astype(np.float32)
    v2_x2_validation = v2_x2_train[ix_val, :]
    v2_x2_train = v2_x2_train[ix_tr, :]
    v2_x2_test = v2_test_set[:, 1]
    v2_x2_test = np.vstack(v2_x2_test[:]).astype(np.float32)  # needed by keras or TF for correct input shape
    v2_y_train = v2_train_set[:, 2]
    v2_y_train = np.vstack(v2_y_train[:]).astype(np.float32)
    v2_y_validation = v2_y_train[ix_val, :]
    v2_y_train = v2_y_train[ix_tr, :]
    v2_y_test = v2_test_set[:, 2]
    v2_y_test = np.vstack(v2_y_test[:]).astype(np.float32)

    v2_data = {'v2_x1_train': x_train, 'v2_x2_train': v2_x2_train,
               'v2_x1_validation': x_validation, 'v2_x2_validation': v2_x2_validation,
               'v2_x1_test': x_test, 'v2_x2_test': v2_x2_test,
               'v2_y_train': v2_y_train, 'v2_y_validation': v2_y_validation, 'v2_y_test': v2_y_test}

    v3_train = pd.read_pickle("../dataset/version3/train.pickle").values
    v3_test_set = pd.read_pickle("../dataset/version3/test.pickle").values
    v3_x2_train = v3_train[:, 1]
    v3_x2_train = np.vstack(v3_x2_train[:]).astype(np.float32)
    v3_x3_train = v3_train[:, 2]
    v3_x3_train = np.vstack(v3_x3_train[:]).astype(np.float32)
    v3_y_train = v3_train[:, 3]
    v3_y_train = np.asarray([np.asarray(sublist) for sublist in v3_y_train])

    v3_x2_validation = v3_x2_train[ix_val, :]
    v3_x2_train = v3_x2_train[ix_tr, :]
    v3_x3_validation = v3_x3_train[ix_val, :]
    v3_x3_train = v3_x3_train[ix_tr, :]
    v3_y_validation = v3_y_train[ix_val, :]
    v3_y_train = v3_y_train[ix_tr, :]
    v3_x3_test = v3_test_set[:, 2]
    v3_x3_test = np.vstack(v3_x3_test[:]).astype(np.float32)  # needed by keras or TF for correct input shape

    v3_data = {'v3_x2_train': v3_x2_train, 'v3_x3_train': v3_x3_train,
               'v3_x2_validation': v3_x2_validation, 'v3_x3_validation': v3_x3_validation,
               'v3_x3_test': v3_x3_test,
               'v3_y_train': v3_y_train, 'v3_y_validation': v3_y_validation, 'v3_y_test': v2_y_test}
    return v1_data, v2_data, v3_data


def compute_metrics(y_test, y_pred):
    """
        compute r2 and rmse
    Args:
        y_test: set of labels
        y_pred: set of predictions

    Returns:
        scores
    """
    rmse_score = []
    r2_score = []
    for i in range(len(output_names_v2)):
        rmse_score.append(np.math.sqrt(sk.metrics.mean_squared_error(y_test[:, i], y_pred[i])))
        r2_score.append(sk.metrics.r2_score(y_test[:, i], y_pred[i]))
    return r2_score, rmse_score


def exp_model_v3(sel_idx, batch_size, epochs, rep, save_dir, summary_f, v, v3_x2_test, v3_x2_train, v3_x2_validation, v3_x3_test, v3_x3_train, v3_x3_validation, v3_y_train, v3_y_validation):
    """
        function to call to test v3
    Args:
        sel_idx: indexes of selected samples
        batch_size: batch size
        epochs: number of epochs
        rep: number of repetitions
        save_dir: save directory
        summary_f: flag, True= show keras Summary
        v: model version
        datasets
    Returns:
        keras history for the run
        list of predition of test set

    """
    model_type = "v3"
    print("---------------Model v3 runnning with train size: " + str(v) + " rep: " + str(rep + 1) + "----------------")
    model_name = "/v3_model_train_size_" + str(v) + "_rep_" + str(rep + 1) + ".h5"
    history, y_pred = train_model(sel_idx, batch_size, epochs, model_name, save_dir, (v3_x2_train, v3_x3_train), (v3_x2_validation, v3_x3_validation), (v3_x2_test, v3_x3_test), v3_y_train, v3_y_validation,
                                  save_model=True, model_type=model_type,
                                  summary_f=summary_f)
    return history, y_pred


def exp_model_v2(idx, batch_size, epochs, rep, save_dir, summary_f, v, x1_test, x1_train, x1_validation, x2_test, x2_train, x2_validation, y_train, y_validation):
    """
         function to call to test v2
     Args:
         idx: indexes of selected samples
         batch_size: batch size
         epochs: number of epochs
         rep: number of repetitions
         save_dir: save directory
         summary_f: flag, True= show keras Summary
         v: model version
         datasets
     Returns:
         keras history for the run
         list of predition of test set

     """
    model_type = "v2"
    print("---------------Model v2 runnning with train size: " + str(v) + " rep: " + str(rep + 1) + "----------------")
    model_name = "/v2_model_train_size_" + str(v) + "_rep_" + str(rep + 1) + ".h5"
    history, y_pred = train_model(idx, batch_size, epochs, model_name, save_dir, (x1_train, x2_train), (x1_validation, x2_validation), (x1_test, x2_test), y_train, y_validation,
                                  save_model=True,
                                  model_type=model_type, summary_f=summary_f)
    return history, y_pred


def exp_model_v1(idx, batch_size, epochs, rep, save_dir, summary_f, v, visual_odom, x_test, x_train, x_validation, y_train, y_validation):
    """
         function to call to test v1
     Args:
         idx: indexes of selected samples
         batch_size: batch size
         epochs: number of epochs
         rep: number of repetitions
         save_dir: save directory
         summary_f: flag, True= show keras Summary
         v: model version
         datasets
     Returns:
         keras history for the run
         list of predition of test set

     """
    model_type = "v1"
    print("---------------Model v1 runnning with train size: " + str(v) + " rep: " + str(rep + 1) + "----------------")
    model_name = "/v1_model_train_size_" + str(v) + "_rep_" + str(rep + 1) + ".h5"
    history, y_pred = train_model(idx, batch_size, epochs, model_name, save_dir, x_train, x_validation, x_test, y_train, y_validation, save_model=True, model_type=model_type,
                                  summary_f=summary_f)
    v3_input = np.copy(y_pred)
    d_ctrl = Controller()
    for i in range(y_pred[0].shape[0]):
        v_drone = [visual_odom[i][0], visual_odom[i][1], 0]
        p = [y_pred[0][i][0], y_pred[1][i][0], y_pred[2][i][0]]
        yaw_ = y_pred[3][i][0]
        res = d_ctrl.new_controller(p, yaw_, v_drone)
        y_pred[0][i][0] = res[0]
        y_pred[1][i][0] = res[1]
        y_pred[2][i][0] = res[2]
        y_pred[3][i][0] = res[3]
    return y_pred, v3_input, history


if __name__ == "__main__":
    main()
