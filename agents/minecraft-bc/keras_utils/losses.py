#!/usr/bin/env python3
#
# losses.py
#
# Losses for training Keras models
#

import numpy as np
# import tensorflow.keras.backend as K
# import tensorflow as tf
import torch
from torch import nn

class create_multicategorical_loss(nn.Module):
    def __init__(self,action_nvec, weights, target_value=0.995):
            super(create_multicategorical_loss, self).__init__()
            self.action_nvec=action_nvec
            self.weights=weights
            self.target_value=target_value

    def forward(self,y_true, y_pred):
        y_true=y_true.type('torch.int32')
        # y_true = tf.cast(y_true, tf.int32)
        losses = []
        current_index = 0
        for i, action_size in enumerate(self.action_nvec):
            preds = y_pred[:, current_index: current_index + action_size]
            trues=torch.nn.functional.one_hot(y_true[:, i], action_size)
            # trues = tf.one_hot(y_true[:, i], action_size)

            # Do not aim for strict {0,1} as this
            # is not reachable by softmax. Instead aim for
            # something reasonable, e.g. 0.995.
            # This is called "label smoothing" in Keras and Tensorflow
            trues = trues * self.target_value + ((1 - self.target_value) / action_size)

            # Clipping
            epsilon=1e-07
            # epsilon = K.epsilon()
            preds=torch.clamp(preds, epsilon, 1.0)
            trues=torch.clamp(preds, epsilon, 1.0)
            # preds = tf.clip_by_value(preds, epsilon, 1.0)
            # trues = tf.clip_by_value(trues, epsilon, 1.0)

            # Include weighting to both positive and negative samples,
            # such that variables with high weighting should happen more
            # often
            option_weights = np.array(self.weights[i])

            # KL loss
            loss = trues * torch.log(trues / preds)
            # loss = trues * tf.compat.v1.log(trues / preds)
            # Apply weighting and sum over support
            loss =torch.sum(loss * option_weights, axis=-1)
            # loss = tf.reduce_sum(loss * option_weights, axis=-1)

            losses.append(loss)
            current_index += action_size
        # Sum over different actions and then mean over batch elements
        loss = torch.mean(torch.sum(losses, dim=0))
        return loss
# def create_multicategorical_loss(action_nvec, weights, target_value=0.995):
    # """ Returns loss appropiate for multicategorical training.

    # y_target is a list of integers (not one-hot), and y_pred is
    # list of one-hot/raveled actions.

    # weights is a list of lists according to action_nvec, each being
    # a weight for that one value.

    # target_value is the target for selected actions. Normally you try
    # to predict {0, 1}, but problem there is that logits need to be
    # {-inf, inf} to reach that. Instead aim for something more relaxed,
    # e.g. {0.005, 0.995}. This is similar to Keras's/Tensorflow's label
    # smoothing.
    # """
    # def forward(y_true, y_pred):
    #     y_true=y_true.type('torch.int32')
    #     # y_true = tf.cast(y_true, tf.int32)
    #     losses = []
    #     current_index = 0
    #     for i, action_size in enumerate(self.action_nvec):
    #         preds = y_pred[:, current_index: current_index + action_size]
    #         trues=torch.nn.functional.one_hot(y_true[:, i], action_size)
    #         # trues = tf.one_hot(y_true[:, i], action_size)

    #         # Do not aim for strict {0,1} as this
    #         # is not reachable by softmax. Instead aim for
    #         # something reasonable, e.g. 0.995.
    #         # This is called "label smoothing" in Keras and Tensorflow
    #         trues = trues * target_value + ((1 - target_value) / action_size)

    #         # Clipping
    #         epsilon=1e-07
    #         # epsilon = K.epsilon()
    #         preds=torch.clamp(preds, epsilon, 1.0)
    #         trues=torch.clamp(preds, epsilon, 1.0)
    #         # preds = tf.clip_by_value(preds, epsilon, 1.0)
    #         # trues = tf.clip_by_value(trues, epsilon, 1.0)

    #         # Include weighting to both positive and negative samples,
    #         # such that variables with high weighting should happen more
    #         # often
    #         option_weights = np.array(weights[i])

    #         # KL loss
    #         loss = trues * torch.log(trues / preds)
    #         # loss = trues * tf.compat.v1.log(trues / preds)
    #         # Apply weighting and sum over support
    #         loss =torch.sum(loss * option_weights, axis=-1)
    #         # loss = tf.reduce_sum(loss * option_weights, axis=-1)

    #         losses.append(loss)
    #         current_index += action_size
    #     # Sum over different actions and then mean over batch elements
    #     loss = torch.mean(torch.sum(losses, dim=0))
    #     return loss

