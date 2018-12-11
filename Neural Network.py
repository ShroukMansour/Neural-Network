import numpy as np
np.random.seed(100)

# k -> # of samples
# m -> # of features
# L -> # of hidden nodes
# n -> # of outputs

class NeuralNetwork:
    def __init__(self, m, l, n, learning_rate = 0.3):
        self.hidden_weights = np.random.rand(m, l)
        self.output_weights = np.random.rand(l, n)
        self.hidden_layer_values = None
        self.learning_rate = learning_rate

    def feed_forward(self, x_train):
        hidden_layer_values = np.dot(x_train, self.hidden_weights)  # (k*m).(m*L) = k*L
        self.activation_funcv = np.vectorize(self.activation_func)  # to apply on each elem
        hidden_layer_values = self.activation_funcv(hidden_layer_values)
        self.hidden_layer_values = np.array(hidden_layer_values)
        output = np.dot(hidden_layer_values, self.output_weights)  # (k*L).(L*n) = k*n
        output = self.activation_funcv(output)
        return output

    def feed_backward(self, x_train, y_train, iterations_num, error_threshold):
        for i in range(iterations_num):
            o = self.feed_forward(x_train)  # k*n
            delta = y_train - o  # k*n
            error = self.calc_mse(delta)
            print(error)
            if any(error > error_threshold):
                delta_o = o*(1-o)*delta # k*n
                # L*n = L*n + (1*1).(L*k).(k*n)
                new_output_weights = self.output_weights + self.learning_rate*self.hidden_layer_values.T.dot(delta_o)
                # k*L  = ((k*L) * (k*L))* (k*n).(n*L)
                delta_h = self.hidden_layer_values*(1-self.hidden_layer_values)*np.dot(delta, self.output_weights.T)
                # m*L = m*L + (1*1).(m*k).(k*L)
                new_hidden_weights = self.hidden_weights + self.learning_rate* np.dot(x_train.T,delta_h)
                self.hidden_weights = new_hidden_weights
                self.output_weights = new_output_weights
            else:
                break



    def activation_func(self, net):
        return 1 / (1 + np.math.exp(-net))

    def calc_mse(self, delta):
        return np.sum(delta ** 2, axis=0) / delta.shape[0]


info = [int(x) for x in input().split()]
m = info[0]  # features
l = info[1]  # hidden neurons
n = info[2]  # classes
k = int(input())  # num of samples
x_train = []  # k * m
y_train = []  #  k * n
for i in range(k):
    sample = [float(x) for x in input().split()]
    x_train.append(np.array(sample[0:m]))
    y_train.append(np.array(sample[m:m+n]))

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train/np.max(y_train, axis=0)
x_train = x_train/np.max(x_train, axis=0)
model = NeuralNetwork(m, l, n)
yp = model.feed_forward(x_train)
mse = model.calc_mse(y_train - yp)
print("Error before back probagation: " , str(mse))
model.feed_backward(x_train, y_train, 1000, 0.00001)
# print("Error after back probagation: " , str(mse))
