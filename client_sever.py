from flask import Flask
from flask import request
import threading
from threading import Lock
import traceback
import numpy as np

# queue data structures

import os
import argparse
#import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Models import Models
from clients import clients, user
import random


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=20, help='local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

myClients = None

# queue for each topic
class TopicQueue:
    def __init__(self, topic_name):
        # each topic queue has its own lock to ensure broker ordering
        self.lock = Lock()
        self.messages = []
        self.topic_name = topic_name

# stores producer information
class Producers:
    def __init__(self):
        self.count = 0       # count for assigning producer_id
        self.lock = Lock()   # lock for getting producer_id
        self.topics = dict() # stores topic: producer_id

class Consumers:
    def __init__(self):
        self.count = 0          # count for assigning consumer_id
        self.lock = Lock()      # lock for getting consumer_id
        self.topics = dict()    # stores topic: consumer_id
        self.offsets = dict()   # stores message offset for each consumer_id

app = Flask(__name__)

# debugging functions
def print_thread_id():
    print('Request handled by worker thread:')

def return_message(status:str, message=None):
    content = dict()
    content['status'] = status
    if message is not None:
        content['message'] = message
    return content

# functions for handelling each endpoint

@app.route('/local_vars', methods=['POST'])
def topic_register_request():
    print_thread_id()
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/json':
        return return_message('failure', 'Content-Type not supported')
    
    try:
        receive = request.json
        client = receive['client_id']
        global_vars = receive['global_vars']
    except:
        return return_message('failure', 'Error While Parsing json')
    
    global myClients
    sess.run(tf.initialize_all_variables())
    global_ndarray = []
    for g in global_vars:
        global_ndarray.append(np.array(g))

    local_vars = myClients.ClientUpdate(client, global_ndarray)
    local_list = []
    for g in local_vars:
        local_list.append(g.tolist())
    return {
        "local_vars": local_list
    }
    
    return return_message('error while getting TF lock')


@app.route('/size',methods=['GET'])
def consumer_size():
    print_thread_id()   
    try:
        topic = request.args.get('topic')
        consumer_id = request.args.get('consumer_id')
        consumer_id = int(consumer_id)
    except:
        return return_message('failure', 'error while parsing request')
        
    return{
        "status": "success",
    }

@app.route('/')
def index():
    return 'Web App with Python Flask!'
            

if __name__ == "__main__":
    args = parser.parse_args()
    
    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_mkdir(args.save_path)

    if args.modelname == 'mnist_2nn' or args.modelname == 'mnist_cnn':
        datasetname = 'mnist'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 784])
            inputsy = tf.placeholder(tf.float32, [None, 10])
    elif args.modelname == 'cifar10_cnn':
        datasetname = 'cifar10'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 24, 24, 3])
            inputsy = tf.placeholder(tf.float32, [None, 10])

    myModel = Models(args.modelname, inputsx)

    predict_label = tf.nn.softmax(myModel.outputs)
    with tf.variable_scope('loss') as scope:
        Cross_entropy = -tf.reduce_mean(inputsy * tf.log(predict_label), axis=1)

    with tf.variable_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        train = optimizer.minimize(Cross_entropy)

    with tf.variable_scope('validation') as scope:
        correct_prediction = tf.equal(tf.argmax(predict_label, axis=1), tf.argmax(inputsy, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.initialize_all_variables())

        myClients = clients(args.num_of_clients, datasetname,
                            args.batchsize, args.epoch, sess, train, inputsx, inputsy, is_IID=args.IID)
    

        app.run(host='0.0.0.0', port=4000, debug=False, threaded=False, processes=1)