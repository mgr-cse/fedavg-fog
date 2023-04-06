import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Models import Models
from clients import clients, user
from copy import deepcopy
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=200, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.3, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=200, help='local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=201, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-ns', '--num_servers', type=int, default=3, help='Number of servers')
parser.add_argument('-of', '--obeserve_file', type=str, default='test_run_hierar', help='file for obeservations')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=='__main__':
    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_mkdir(args.save_path)
    test_mkdir('obeserve/')
    data_to_save = []

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
        y_pred = tf.argmax(predict_label, axis=1)
        y_true = tf.argmax(inputsy, axis=1)
        correct_prediction = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver(max_to_keep=3)

    # ---------------------------------------- server -------------------------------------------- #
    num_of_servers = args.num_servers

    class servers:
        def __init__(self, num_servers, myClients: clients):
            self.num = num_servers
            self.serverSet = {}
            self.serverVars = {}
            
            self.energy = 0.0
            self.energy_rate = 2.0

            # for latency to global aggregation
            self.client_thresh = 10
            self.base_lat = 100
            self.lat_factor = 10
            self.noise_fact = 0.1
            
            
            # create mapping of clients to server in serverSet
            client_index = 0
            clientsPerServer = myClients.num_of_clients // num_servers
            for i in range(num_servers):
                self.serverSet[f'server{i}'] = [ f'client{j}' for j in range(i*clientsPerServer, (i+1)*clientsPerServer)]
                client_index = (i+1)*clientsPerServer
            
            # add the remaining clients to last server
            for i in range(client_index, myClients.num_of_clients):
                self.serverSet[f'server{num_servers-1}'].append(f'client{i}')

            for k in self.serverSet:
                self.serverVars[k] = None
        
        def getlat(self, tot_client):
            lam = self.base_lat + tot_client * (self.lat_factor if tot_client > self.client_thresh else 0)
            lam += random.uniform(0, self.noise_fact)*lam
            return lam


    # ---------------------------------------- train --------------------------------------------- #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.initialize_all_variables())

        myClients = clients(args.num_of_clients, datasetname,
                            args.batchsize, args.epoch, sess, train, inputsx, inputsy, is_IID=args.IID)
        myServers = servers(num_of_servers, myClients)

        # for global aggregation
        global_energy = 0.0
        global_energy_rate = 1000.0
        
        agg_global_vars = sess.run(tf.trainable_variables())
        for i in tqdm(range(args.num_comm)):
            #print("communicate round {}".format(i))
            max_all_client_groups_lat = 0
            for j in range(myServers.num):
                #print(f'server{j} running:')
                
                # select random clients
                client_keys = myServers.serverSet[f'server{j}']
                num_in_comm = int(max(len(client_keys)* args.cfraction, 1))
                clients_in_comm = random.sample(client_keys, num_in_comm)
                
                sum_vars = None
                max_lat_client = 0
                global_vars = myServers.serverVars[f'server{j}']
                
                # init global vars for server
                if global_vars is None:
                    global_vars = sess.run(tf.trainable_variables())
                
                # train clients
                for client in clients_in_comm:
                    # add latency call for clients
                    curr_lat_client = myClients.getlat(client, len(clients_in_comm), True)
                    max_lat_client = max([curr_lat_client, max_lat_client])
                    local_vars = myClients.ClientUpdate(client, global_vars)
                    
                    if sum_vars is None:
                        sum_vars = local_vars
                    else:
                        for sum_var, local_var in zip(sum_vars, local_vars):
                            sum_var += local_var
                
                # max latency for global aggregation
                max_all_client_groups_lat = max([max_all_client_groups_lat, max_lat_client])
                
                # aggregate results (local)
                global_vars = []
                for var in sum_vars:
                    global_vars.append(var / num_in_comm)
                myServers.serverVars[f'server{j}'] = global_vars
                
                # update local energy
                myServers.energy += myServers.energy_rate + random.uniform(0, myServers.energy_rate)

                # print server specific validation data
                if i % args.save_freq == 0:
                    checkpoint_name = os.path.join(args.save_path, '{}_comm'.format(args.modelname) +
                                                   'IID{}_communication{}'.format(args.IID, i+1)+ f'server{j}'+'.ckpt')
                    save_path = saver.save(sess, checkpoint_name)
                
            # global aggregation
            if i % args.val_freq == 0:
                print('*** global aggregation ***')
                max_glob_lat = 0
                serv_sum_vars = None
                
                # global fedavg
                for serv in myServers.serverSet:
                    # latency
                    curr_lat = myServers.getlat(myServers.num)
                    max_glob_lat = max([curr_lat, max_glob_lat])
                    
                    local_vars = myServers.serverVars[serv]
                    if serv_sum_vars is None:
                        serv_sum_vars = local_vars
                    else:
                       for sum_var, local_var in zip(serv_sum_vars, local_vars):
                           sum_var += local_var
                agg_global_vars = []
                for var in serv_sum_vars:
                    agg_global_vars.append(var / myServers.num)
                # give vars back to servers
                for serv in myServers.serverSet:
                    myServers.serverVars[serv] = deepcopy(agg_global_vars)
                
                # at global aggregation evaluate cloud energy
                global_energy += global_energy_rate + random.uniform(0, global_energy_rate/2)
                # eval model
                for variable, value in zip(tf.trainable_variables(), agg_global_vars):
                    variable.load(value, sess)
                test_data = myClients.test_data
                test_label = myClients.test_label
                acc, cross, y_pred_run, y_true_run = sess.run([accuracy, Cross_entropy, y_pred, y_true], feed_dict={inputsx: test_data, inputsy: test_label})
                
                # data to note
                print('communication round:', i// args.val_freq)
                print('Accuracy:', acc, 'Loss:', cross)
                print('Local fog Energy Consumed:', myServers.energy)
                print('Global agg Energy Consumed:', global_energy)
                print('client Energy consumed:', myClients.energy)
                print('Latency (local, client to fog):', max_all_client_groups_lat )
                print('Latency (fog to cloud):', max_glob_lat)
                mig_cost = myClients.getmodcost()
                print('Migration cost:', mig_cost)
                f1_val = f1_score(y_true_run, y_pred_run, average='macro')
                prec_val = precision_score(y_true_run, y_pred_run, average='macro')
                rec_val = recall_score(y_true_run, y_pred_run, average='macro')
                print(f'f1={f1_val} precision={prec_val} recall={rec_val}')

                # save data in dictionary
                data_note = {
                    "round": i,
                    "accuracy": float(acc),
                    "loss": [float(l) for l in cross],
                    "energy_global": global_energy,
                    "energy_client": myClients.energy,
                    "energy_fog": myServers.energy,
                    "latency_local_agg": max_all_client_groups_lat,
                    "latency_global_agg": max_glob_lat,
                    "f1": f1_val,
                    "prec": prec_val,
                    "rec": rec_val,
                    "mig_cost": mig_cost
                }
                data_to_save.append(data_note)

        # save obeserved_data
        
        final_data = {
            "num_clients": myClients.num_of_clients,
            "servers": myServers.num,
            "serv_frac": 1.0,
            "client_frac": args.cfraction,
            "data": data_to_save
        }

        with open(f'./obeserve/{args.obeserve_file}', 'w') as f:
            json.dump(final_data, f)
            print('+++ written to file')