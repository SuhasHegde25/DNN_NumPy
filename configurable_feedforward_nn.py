import numpy as np
import matplotlib.pyplot as plt
import cifar10_dm
import utils
import random
import logging
import logging.handlers
import sys

#np.set_printoptions(threshold=sys.maxsize)


class configurable_feedforward_nn():

    LOG_FILENAME = '/Users/suhashegde/Desktop/ML/Code/LogSM.txt'
    ##
    def __init__(self):
        self.cifar10_dm_obj = cifar10_dm.cifar10_dm()
        self.utils_obj = utils.utils()

        self.m = 0
        self.parameters = {}

        sys.stdout = open(configurable_feedforward_nn.LOG_FILENAME, 'a')

        #self.logger = logging.getLogger('logger')
        #self.logger.setLevel(logging.DEBUG)
        ##logging.handlers.RotatingFileHandler()
        #handler = logging.handlers.RotatingFileHandler(configurable_feedforward_nn.LOG_FILENAME, maxBytes=20000, backupCount=5)
        #self.logger.addHandler(handler)
        ##self.logger.basicConfig(filename='/Users/suhashegde/Desktop/ML/Code/Log.rtf', filemode='w', level=logging.DEBUG)


    ##
    def normalize_input(self, input_features):
        return np.array(input_features) / 255.0


    ##
    def initialize_parameters(self, node_count_list, initialization_method='random'):
        #self.logger.info("initialize_parameters::Parameter initialization method: {}".format(initialization_method))
        print('\nPARAMETER INITIALIZATION\n')
        print("Parameter initialization method: {}".format(initialization_method))

        parameters = {}

        # Initialize weights randomly
        if initialization_method == 'random':
            for i in range(1,len(node_count_list)):
                parameters["W"+str(i)] = np.random.rand(node_count_list[i],node_count_list[i-1]) #* 0.01

        # Xavier initialization
        if initialization_method == 'xavier':
            for i in range(1,len(node_count_list)):
                parameters["W"+str(i)] = np.random.rand(node_count_list[i],node_count_list[i-1]) * np.sqrt(1./node_count_list[i - 1])

        if initialization_method == 'he':
            for i in range(1,len(node_count_list)):
                parameters["W"+str(i)] = np.random.rand(node_count_list[i],node_count_list[i-1]) * 0.01 * np.sqrt(2./node_count_list[i - 1])

        # Initialize bias
        #for i in range(1,len(node_count_list)):
            #parameters["b"+str(i)] = np.zeros([node_count_list[i],1], dtype = int)

        print("Parameters are initialized successfully")
        print('Parameter matrices:')
        #self.logger.info("initialize_parameters::Parameters are initialized successfully")
        self.utils_obj.print_dict(parameters)#, self.logger)

        return parameters


    ##
    def apply_activation(self, nodes, activation_type):
        if activation_type is 'sigmoid':
            nodes = 1./(1 + np.exp(-nodes))
        elif activation_type is 'tanh':
            nodes = np.tanh(nodes)
        elif activation_type is 'relu':
            nodes = np.maximum(nodes, 0)
        elif activation_type is 'leaky_relu':
            nodes = np.maximum(nodes, nodes * 0.01)
        elif activation_type is 'softmax':
            #max_node = np.max(nodes, axis=0).reshape(1, nodes.shape[1])
            max_node = np.max(nodes)
            nodes = np.exp(nodes - max_node)
            nodes = np.divide(nodes,np.sum(nodes,axis=0).reshape(1,nodes.shape[1]))

        return nodes


    ##
    def calculate_loss_and_gradient(self, y_pred, y_label, loss_type='cross_entropy'):
        print('Calculating loss and loss gradients for loss type:{}'.format(loss_type))
        if loss_type is 'cross_entropy':
            loss = np.multiply(y_label,-np.log(y_pred)) + np.multiply(1 - y_label,-np.log(1 - y_pred))
            loss = (1./self.m) * np.sum(loss)
            loss_gradient = self.calculate_loss_gradient(loss_type=loss_type, predictions=y_pred, labels=y_label)
        elif loss_type in 'cross_entropy_softmax':
            #loss = (1./self.m) * np.sum(np.multiply(y_label,-np.log(y_pred)))
            loss = np.sum(np.multiply(-y_label,np.log(y_pred)))
            loss_gradient = self.calculate_loss_gradient(loss_type, y_pred, y_label)          # @TODO: Need to change the code for gradient

        print('Loss:{}'.format(loss))
        print('Loss gradient: \n{}'.format(loss_gradient))
        return loss, loss_gradient   #np.ones((loss_gradient.shape)) #


    ##
    def feed_forward(self, features, parameters, activation_list):
        print('\nFORWARD PROPAGATION\n')
        cache = [(features,features)]
        (n,m) = features.shape
        a_prev = features
        print('Features/Layer 0')
        print(features)
        counter = 0
        for activation in activation_list:
            counter += 1
            print("W{}:{}".format(counter,parameters["W"+str(counter)].shape))
            print(parameters["W"+str(counter)])
            z = np.dot(parameters["W"+str(counter)],a_prev)
            print('Z{}:{}'.format(counter,z.shape))
            print(z)
            a = self.apply_activation(z, activation)
            print('Activation function applied: {}'.format(activation))
            print('A{}:{}'.format(counter,a.shape))
            print(a)
            cache.append((z,a))
            a_prev = a
        #print(cache[1][0])
        return a, cache


    ##
    def calculate_loss_gradient(self, loss_type, predictions, labels):
        if loss_type is 'cross_entropy':
            gradient = -(np.divide(labels,predictions) - np.divide((1.0 - labels),(1.0 - predictions)))
        if loss_type is 'cross_entropy_softmax':
            #gradient = -np.divide(labels,predictions)
            gradient = labels              # Need to do proper implementation

        return gradient


    ##
    def calculate_local_gradient(self, activation_type, cache, incoming_gradient):
        dZ = []
        dW = []
        dA_prev = []

        a_prev = cache['a_prev']
        z_prev = cache['z_prev']
        a = cache['a']
        z = cache['z']
        W = cache['W']

        if activation_type is 'sigmoid':
            dZ = np.multiply(np.multiply(a,(1 - a)),incoming_gradient)
        elif activation_type is 'softmax':
            dZ = a - incoming_gradient                  # @TODO: Need to write the proper equation.
        elif activation_type is 'relu':
            dZ = np.array(a)
            dZ[dZ <= 0] = 0
            dZ[dZ > 0] = 1
            dZ = np.multiply(dZ,incoming_gradient)
        elif activation_type is 'tanh':
            dZ = np.multiply((1 - np.power(a,2)),incoming_gradient)
        elif activation_type is 'leaky_relu':
            pass

        dW = np.multiply((1./self.m),np.dot(dZ, a_prev.T))
        dA_prev = np.dot(W.T, dZ)

        return dW, dZ, dA_prev


    ##
    def create_cache_slice_dict(self, cache, entry_num, parameters):
        cache_slice = {}
        cache_slice['z'] = cache[entry_num][0]
        cache_slice['a'] = cache[entry_num][1]
        cache_slice['z_prev'] = cache[entry_num - 1][0]
        cache_slice['a_prev'] = cache[entry_num - 1][1]
        cache_slice['W'] = parameters['W'+str(entry_num)]

        return cache_slice


    ##
    def backward_propagation(self, a, cache, parameters, loss_gradient, activation_list):
        print('\nBACK PROPAGATION\n')
        gradients = {}
        counter = len(activation_list)
        incoming_gradient = loss_gradient
        print('Loss gradient:{}'.format(loss_gradient.shape))
        print(loss_gradient)

        for activation_type in reversed(activation_list):
            cache_slice = self.create_cache_slice_dict(cache, counter, parameters)
            dW, dZ, dA_prev = self.calculate_local_gradient(activation_type=activation_type, cache=cache_slice, incoming_gradient=incoming_gradient)

            #print('Shape of weight gradient {}: {}'.format('dW'+str(counter),dW.shape))
            gradients['dW'+str(counter)] = dW

            print('dW{}:{}'.format(counter,dW.shape))
            print(dW)
            print('dZ{}:{}'.format(counter,dZ.shape))
            print(dZ)
            print('dA{}:{}'.format(counter-1,dA_prev.shape))
            print(dA_prev)

            incoming_gradient = dA_prev
            counter -= 1

        return gradients


    ##
    def gradient_descent_optimizer(self, parameters, gradients, learning_rate):
        print('Gradient optimization')
        for key in parameters:
            parameters[key] -= learning_rate * gradients["d"+key]

        print('Updated parameters:')
        self.utils_obj.print_dict(dictionary=parameters)
        return parameters


    ##
    def initialize_momentum_parameters(self, parameters):
        velocity = {}
        for key in parameters:
            velocity["d"+key] = np.zeros((parameters[key].shape[0],parameters[key].shape[1]))

        print('Initialized velocity parameters:')
        self.utils_obj.print_dict(dictionary=velocity)
        return velocity


    ##
    def momentum_optimizer(self, parameters, gradients, velocity, learning_rate, beta=0.9):
        print('Momentum optimization')
        for key in parameters:
            velocity["d"+key] = beta * velocity["d"+key] + (1 - beta) * gradients["d"+key]
            parameters[key] -= learning_rate * velocity["d"+key]

        print('Velocity:')
        self.utils_obj.print_dict(dictionary=velocity)
        print('Updated parameters:')
        self.utils_obj.print_dict(dictionary=parameters)

        return velocity, parameters


    ##
    def initialize_adam_parameters(self, parameters):
        velocity = {}
        rms = {}

        for key in parameters:
            velocity["d"+key] = np.zeros((parameters[key].shape[0],parameters[key].shape[1]))
            rms["d"+key] = np.zeros((parameters[key].shape[0],parameters[key].shape[1]))

        print('Initialized velocity parameters:')
        self.utils_obj.print_dict(dictionary=velocity)
        print('Initialized rms parameters:')
        self.utils_obj.print_dict(dictionary=rms)

        return velocity, rms


    ##
    def adam_optimizer(self, parameters, gradients, velocity, rms, epoch_num, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        print('Adam optimization')
        for key in parameters:
            # Velocity update, same as in momentum
            velocity["d"+key] = beta1 * velocity["d"+key] + (1 - beta1) * gradients["d"+key]
            # Bias correction, required to nullify the bias in the initial stages of optimization caused due to initializing velocity to 0
            velocity["d"+key] = velocity["d"+key] / (1 - np.power(beta1,epoch_num+1))

            # Squared velocity update or rms update
            rms["d"+key] = beta2 * rms["d"+key] + (1 - beta2) * np.power(gradients["d"+key],2)
            # Bias correction, required to nullify the bias in the initial stages of optimization caused due to initializing rms to 0
            rms["d"+key] = rms["d"+key] / (1 - np.power(beta2,epoch_num+1))

            # Update parameters
            parameters[key] -= learning_rate * velocity["d"+key] / (np.sqrt(rms["d"+key]) + epsilon)

        print('Velocity:')
        self.utils_obj.print_dict(dictionary=velocity)
        print('rms:')
        self.utils_obj.print_dict(dictionary=rms)
        print('Updated parameters:')
        self.utils_obj.print_dict(dictionary=parameters)

        return velocity, rms, parameters


    ##
    def train(self, features, labels, node_count_list, activation_list, initialization_type, optimization_type, loss_type, learning_rate, epochs):
        # Initialize weights of the network
        print('Initializing parameters...')
        parameters = self.initialize_parameters(node_count_list, initialization_method=initialization_type)

        # Initialize optimization parameters if required
        if optimization_type is 'adam':
            print('Initializing velocity and rms parameters for Adam optimizer...')
            velocity,rms = self.initialize_adam_parameters(parameters)
        elif optimization_type is 'momentum':
            print('Initializing velocity parameters for Momentum optimizer...')
            velocity = self.initialize_momentum_parameters(parameters)

        # Initialize lists to save the data required to plot the graph
        loss_list = []
        epoch_list = []
        accuracy_list = []

        for epoch_num in range(epochs):
            for batch_num in range(len(features)):
                a, cache = self.feed_forward(features[batch_num], parameters, activation_list)
                loss, loss_gradient = self.calculate_loss_and_gradient(a, labels[batch_num], loss_type=loss_type)
                gradients = self.backward_propagation(a, cache, parameters, loss_gradient, activation_list)

                difference = self.gradient_checking(parameters, gradients, features[batch_num], activation_list, loss_type, labels[batch_num])

                # Choose the optimization method
                if optimization_type is 'adam':
                    velocity,rms,parameters = self.adam_optimizer(parameters, gradients, velocity, rms, epoch_num, learning_rate = learning_rate)
                elif optimization_type is 'momentum':
                    velocity,parameters = self.momentum_optimizer(parameters, gradients, velocity, learning_rate = learning_rate)
                else:
                    parameters = self.gradient_descent_optimizer(parameters, gradients, learning_rate)

                self.parameters = parameters

            if epoch_num % 1000 == 0:
                epoch_list.append(epoch_num)
                loss_list.append(loss)
                accuracy = self.predict(features, labels, activation_list)
                accuracy_list.append(accuracy)
                #self.logger.info('Quick Check:: Epoch: {}, Loss: {}'.format(epoch_num,loss))
                print('Quick Check:: Epoch: {}, Loss: {}'.format(epoch_num,loss))

        plt.plot(epoch_list, loss_list)
        plt.show()
        plt.plot(epoch_list, accuracy_list)
        plt.show()


    ##
    def predict(self, features, labels, activation_list):
        print('START OF PREDICTION')
        correct = 0.0
        wrong = 0.0

        for batch_num in range(len(features)):
            a, _ = self.feed_forward(features[batch_num], self.parameters, activation_list)

            for column in range(a.shape[1]):
                max_index = np.where(a[:,column] == np.max(a[:,column]))[0][0]
                if labels[batch_num][max_index][column] == 0:
                    wrong += 1
                else:
                    correct += 1

        return (correct / (correct + wrong)) *100.0


    ##
    def gradient_checking(self, parameters, gradients, features, activation_list, loss_type, labels, epsilon = 1e-7):
        grad_vector = self.utils_obj.dictionary_to_vector(gradients)
        num_parameters = grad_vector.shape[0]
        J_plus = np.zeros((num_parameters,1))
        J_minus = np.zeros((num_parameters,1))
        grad_approx = np.zeros((num_parameters,1))
        counter = 0

        for key in parameters:
            for i in range(parameters[key].shape[0]):
                for j in range(parameters[key].shape[1]):
                    theta_plus = parameters.copy()
                    theta_plus[key][i][j] = theta_plus[key][i][j] + epsilon
                    a,_ = self.feed_forward(features, theta_plus, activation_list)
                    J_plus[counter],_ = self.calculate_loss_and_gradient(a, labels, loss_type)

                    theta_minus = parameters.copy()
                    theta_minus[key][i][j] = theta_minus[key][i][j] - epsilon
                    a,_ = self.feed_forward(features, theta_minus, activation_list)
                    J_minus[counter],_ = self.calculate_loss_and_gradient(a, labels, loss_type)

                    grad_approx[counter] = (J_plus[counter] - J_minus[counter]) / (2*epsilon)

                    counter += 1

        diff = grad_vector - grad_approx
        numerator = np.linalg.norm(diff)
        denominator = np.linalg.norm(grad_vector) + np.linalg.norm(grad_approx)
        difference = numerator/denominator

        if difference > 2e-7:
            print('\n!!!!!!!!!!WRONG DERIVATIVES!!!!!!!!!\n')
        else:
            print('\nGRADIENT CHECK PASSED\n')

        return difference


if __name__ == "__main__":
    print('***************************** START OF PROGRAM **************************')

    cfg_nn_obj = configurable_feedforward_nn()
    cifar_dm_obj = cifar10_dm.cifar10_dm()

    num_categories = 10

    print('Input pixels with each image in a column:')

    features = []
    labels = []
    train_data = cifar_dm_obj.get_train_data()
    for i in range(cifar_dm_obj.num_train_batches):
        print('\nBatch{}:\nFeatures -'.format(i))

        #feat = np.array(train_data['data']['batch'+str(i+1)],dtype=np.float128)
        #feat = feat[:6,:5]
        #print(feat)

        #print(np.array(train_data['data']['batch'+str(i+1)]))
        features.append(np.array(train_data['data']['batch'+str(i+1)],dtype=np.float128))
        #features.append(feat)       #(np.array(train_data['data']['batch'+str(i+1)][0:6][0:3]))
        label_len = len(train_data['labels']['batch'+str(i+1)])   #[0:3][0:5])
        label = np.array(train_data['labels']['batch'+str(i+1)]) #,dtype=np.float128)    #[0:3][0:5]
        label = label.reshape(1,label_len)
        #label = np.array([[0,1,2,0,1]])
        print('\nLabels -')
        print('Before encoding')
        print(label)
        encoded_labels = cifar_dm_obj.one_hot_encoding(label,num_categories,label.shape[1])
        print('After encoding')
        print(encoded_labels)
        labels.append(encoded_labels)

    features = cfg_nn_obj.normalize_input(features)
    print('FEATURES AFTER NORMALIZATION -')
    for i in range(cifar_dm_obj.num_train_batches):
        print('\nBatch{}:'.format(i))
        print(features[i])

    print('Lenth of features list: {}\nLength of labels list: {}'.format(len(features),len(labels)))
    print('Shape of single input batch: {}\nShape of single label batch: {}'.format(features[0].shape,labels[0].shape))

    #cfg_nn_obj.m = labels[0].shape[1]

    #node_count_list = [features[0].shape[0], num_categories]
    #activation_list = ['softmax']

    #print('\nLayers Information:')
    #counter = 1
    #for activation in activation_list:
        #print('Layer{} -'.format(counter))
        #print('Number of nodes-{}\nActivation function-{}'.format(node_count_list[counter],activation))
        #counter += 1

    features = np.array([[[1,10,1,10,1],
                         [2,9,2,9,2],
                         [3,8,3,8,3],
                         [4,7,4,7,4],
                         [5,6,5,6,5],
                         [6,5,6,5,6],
                         [7,4,7,4,7],
                         [8,3,8,3,8],
                         [9,2,9,2,9],
                         [10,1,10,1,10]]])
    labels = np.array([[[1,0,1,0,1],
                        [0,1,0,1,0]]])
    cfg_nn_obj.m = labels[0].shape[1]
    node_count_list = [features[0].shape[0],2]
    activation_list = ['softmax']

    cfg_nn_obj.train(features, labels, node_count_list, activation_list, initialization_type='he', optimization_type='gd', loss_type='cross_entropy_softmax',
                     learning_rate=0.01, epochs=15000)

    print('***************************** END OF PROGRAM **************************')












    #parameters = cfg_nn_obj.initialize_parameters([4,3,1], initialization_method='xavier')
    #features = np.array([[1,2,3],[4,5,6],[7,8,9],[1,1,1]])
    #activation_list = ['sigmoid','sigmoid']
    #a,cache = cfg_nn_obj.feed_forward(features,parameters,activation_list)
    #cfg_nn_obj.calculate_loss_and_gradient(a,np.array([1,1,1]))