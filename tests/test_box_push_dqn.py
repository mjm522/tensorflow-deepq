from os.path import join, dirname, abspath

import sys

sys.path.insert(1, dirname(dirname(abspath(__file__))) )

import numpy as np
import tempfile
import tensorflow as tf

from tf_rl.controller.discrete_deepq import DiscreteDeepQ
from tf_rl.simulation.mujoco_magic_forces import PushBox
from tf_rl.models import MLP
from tf_rl import simulate

LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)

# create the game simulator
g = PushBox()

# Tensorflow business - it is always good to reset a graph before creating a new controller.
tf.reset_default_graph()
session = tf.InteractiveSession()

# This little guy will let us run tensorboard
#      tensorboard --logdir [LOG_DIR]
journalist = tf.train.SummaryWriter(LOG_DIR)

# Brain maps from observation to Q values for different actions.
# Here it is a done using a multi layer perceptron with 2 hidden
# layers
brain = MLP([g.observation_size,], [200, 200, g.num_actions], 
            [tf.tanh, tf.tanh, tf.identity])

# The optimizer to use. Here we use RMSProp as recommended
# by the publication
optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

# DiscreteDeepQ object
current_controller = DiscreteDeepQ((g.observation_size,), g.num_actions, brain, optimizer, session,
                                   discount_rate=0.99, exploration_period=5000, max_experience=10000, 
                                   store_every_nth=4, train_every_nth=4,
                                   summary_writer=journalist)

session.run(tf.initialize_all_variables())
session.run(current_controller.target_network_update)
# graph was not available when journalist was created  
journalist.add_graph(session.graph)


FPS          = 30
ACTION_EVERY = 3
    
fast_mode = True
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 50
else:
    WAIT, VISUALIZE_EVERY = True, 1

    
try:
    with tf.device("/gpu:0"):
        simulate(simulation=g,
                 controller=current_controller,
                 fps=FPS,
                 visualize_every=VISUALIZE_EVERY,
                 action_every=ACTION_EVERY,
                 wait=WAIT,
                 disable_training=False,
                 simulation_resolution=0.001,
                 save_path=None)
except KeyboardInterrupt:
    print("Interrupted")


session.run(current_controller.target_network_update)

current_controller.q_network.input_layer.Ws[0].eval()

g.plot_reward(smoothing=100)

session.run(current_controller.target_network_update)

current_controller.q_network.input_layer.Ws[0].eval()

current_controller.target_q_network.input_layer.Ws[0].eval()

