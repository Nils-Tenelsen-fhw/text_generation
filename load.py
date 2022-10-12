#Script used to load models of the rnn in test.py, generates a handful of test outputs.
import tensorflow as tf

import numpy as np
import os
import time
from tensorflow.keras import backend

one_step_reloaded = tf.saved_model.load('test_with_all_mails_60_model_32_batch_acc_metric_learning_00025_v3')

def get_output(input, length):
    states = None
    next_char = tf.constant([input])
    result = [next_char]

    for n in range(length):
      next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
      result.append(next_char)

    print(tf.strings.join(result)[0].numpy().decode("utf-8"))
    print('--------------------------------END OF MESSAGE--------------------------------')


get_output('This is an automated message.', 600)
get_output('This is an automated message.', 600)
get_output('This is an automated message.', 600)
get_output('This is an automated message.', 600)
get_output('This is an automated message.', 600)
get_output('This is an automated message.', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Notification: We regret to inform you that ', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: Unfortunately, your', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
get_output('Automated Message: You exceeded your monthly email quota and', 600)
