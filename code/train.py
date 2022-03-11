from __future__ import absolute_import

import argparse
import time
import json
import os
from datetime import datetime
import random
import numpy as np
import torch as th
from tqdm import tqdm

import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from constants import SEQ_PER_EPISODE_C, LEN_SEQ, RES_DIR
from utils import loadLabels, gen_dict_for_json, write_result, use_pretrainted
from utils import JsonDataset_universal as JsonDataset

from earlyStopping import EarlyStopping
from models.CNN import CNN_PR_FC
from models.CNN_LSTM import CNN_LSTM_encoder_decoder_images_PR
from models.autoencoder import AutoEncoder

from get_hyperparameters_configuration import get_params
from hyperband import Hyperband
import matplotlib.pyplot as plt
plt.switch_backend('agg')


num_epochs = 50
batchsize = 24
learning_rate = 0.0001
optimizer = "adam"
future_window_size = 12
past_window_size = 10
weight_decay = 0.001

seed = 42
encoder_latent_vector = 300
decoder_latent_vector = 300

episodes = 540

start_time = time.time()
print("Start training...")



# We iterate over epochs:
# tqdm prints for-loop progress
for epoch in tqdm(range(num_epochs)):
    # Do a full pass on training data
    # Switch to training mode
    model.train()
    train_loss, val_loss = 0.0, 0.0

    for k, data in enumerate(train_loader):
        # unpacked data
        inputs, p_and_roll, targets = data[0], data[1], data[2]
        # move data to GPU
        if cuda:
            inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
        # Convert to pytorch variables
        inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

        # training process
        optimizer.zero_grad()
        predictions = model(inputs, p_and_roll, use_n_im)
        loss = loss_fn(predictions, targets) / predict_n_pr

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # train MSE for pitch and roll
    train_error = train_loss / n_train

    # if the train_error is to big stop training and save time (usefull for Hyperband)
    if np.isnan(train_error):
        return {"best_train_loss": np.inf, "best_val_loss": np.inf, "final_test_loss": np.inf}

    # Do a full pass on validation data
    with th.no_grad():
        # Switch to evaluation mode
        model.eval()
        for data in val_loader:
            # use right validation process for different models
            if use_LSTM:
                # unpacked data
                inputs, p_and_roll = data[0], data[1]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)
                # validation through the sequence
                loss = eval(cuda, change_fps, inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im, use_2_encoders)
                val_loss += loss

            else:
                # unpacked data
                inputs, p_and_roll, targets = data[0], data[1], data[2]
                # move data to GPU
                if cuda:
                    inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
                # Convert to pytorch variables
                inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

                # validation process
                predictions = model(inputs, p_and_roll, use_n_im)
                loss = loss_fn(predictions, targets)/ predict_n_pr
                val_loss += loss.item()

        # validation MSE for pitch and roll
        val_error = val_loss / n_val

        early_stopping(val_error, model, best_model_weight_path, cuda)

        if train_error < best_train_loss:
            best_train_loss = train_error

    if (epoch + 1) % evaluate_print == 0:
        # update figure value and drawing

        xdata.append(epoch+1)
        train_err_list.append(train_error)
        val_err_list.append(val_error)
        li.set_xdata(xdata)
        li.set_ydata(train_err_list)
        l2.set_xdata(xdata)
        l2.set_ydata(val_err_list)
        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw()

        # Save evolution of train and validation loss functions
        json.dump(train_err_list, open(ress_dir+tmp_str+"_train_loss.json",'w'))
        json.dump(val_err_list, open(ress_dir+tmp_str+"_val_loss.json",'w'))
        print("  training avg loss [normalized -1, 1]   :\t\t{:.6f}".format(train_error))
        print("  validation avg loss [normalized -1, 1] :\t\t{:.6f}".format(val_error))

    if early_stopping.early_stop:
        print("----Early stopping----")
        break

# Save figure
plt.savefig(img_dir+tmp_str +'_log_losses.png')
plt.close()

# Load the best weight configuration
model.load_state_dict(th.load(best_model_weight_path))

print("Start testing...")
test_loss = 0.0

with th.no_grad():

    # Preparation files for saving origin and predicted pitch and roll for visualization
    origins = [{} for i in range(predict_n_pr)]
    origin_names = [lable_dir+ '/origin' + model_type +'_use_' + str(past_window_size) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]
    preds = [{} for i in range(predict_n_pr)]
    pred_names = [lable_dir+'/pred' + model_type +'_use_' + str(past_window_size) + '_s_to_predict_'+str(i+1)+':'+str(predict_n_pr)+'_lr_'+str(learning_rate)+'.json' for i in range(predict_n_pr)]

    for key, data  in enumerate(test_loader):
        # use right testing process for different models
        if use_LSTM:
            # unpacked data
            inputs, p_and_roll = data[0], data[1]
            # move data to GPU
            if cuda:
                inputs, p_and_roll = inputs.cuda(), p_and_roll.cuda()
            # Convert to pytorch variables
            inputs, p_and_roll = Variable(inputs), Variable(p_and_roll)
            # test through the sequence
            loss, origins, preds  = test(cuda, change_fps, key, origins, preds , batchsize, inputs, p_and_roll, model, loss_fn, predict_n_pr, use_n_im, use_2_encoders)
            test_loss += loss

        else:
            # unpacked data
            inputs, p_and_roll, targets = data[0], data[1], data[2]
            # move data to GPU
            if cuda:
                inputs, p_and_roll, targets = inputs.cuda(), p_and_roll.cuda(), targets.cuda()
            # Convert to pytorch variables
            inputs, p_and_roll, targets = Variable(inputs), Variable(p_and_roll),Variable(targets)

            predictions = model(inputs, p_and_roll, use_n_im)

            # save results of prediction for visualization
            key_tmp = np.linspace(key*batchsize , (key+1)*batchsize, batchsize, dtype =int )
            for pred_im in range(predict_n_pr):
                tmp1 = gen_dict_for_json(key_tmp, targets[:,pred_im,:].cpu())
                tmp2 = gen_dict_for_json(key_tmp, predictions[:,pred_im,:].cpu())

                origins[pred_im] = {**origins[pred_im], **tmp1}
                preds[pred_im] = {**preds[pred_im], **tmp2}

            loss = loss_fn(predictions, targets)/ predict_n_pr
            test_loss += loss.item()

    for i in range(predict_n_pr):
        json.dump(preds[i], open(pred_names[i],'w'))
        json.dump(origins[i], open(origin_names[i],'w'))

final_test_loss = test_loss /n_test

print("Final results:")
print("  best avg training loss [normalized (-1 : 1) ]:\t\t{:.6f}".format(best_train_loss))
print("  best avg validation loss [normalized (-1 : 1) ]:\t\t{:.6f}".format(min(val_err_list)))
print("  test avg loss[normalized (-1 : 1) ]:\t\t\t{:.6f}".format(final_test_loss))

# write result into result.txt
final_time = (time.time() - start_time)/60
print("Total train time: {:.2f} mins".format(final_time))

# set lenght of sequence used
tmp_seq_len = use_n_im
if use_LSTM:
    tmp_seq_len = LEN_SEQ

# write configuration in file
write_result(args, [model], [optimizer], result_file_name = ress_dir + "/result.txt",
            best_train_loss = best_train_loss, best_val_loss = early_stopping.val_loss_min,
            final_test_loss = final_test_loss, time = final_time, seq_per_ep = seq_per_ep,
            seq_len = tmp_seq_len, num_epochs = num_epochs
            )
return {"best_train_loss": best_train_loss, "best_val_loss": early_stopping.val_loss_min, "final_test_loss": final_test_loss, "early_stop": early_stopping.early_stop}





