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

    def feed_forward(self, x_train):
        hidden_layer_values = np.dot(x_train, self.hidden_weights)  # (k*m).(m*L) = k*L
        self.activation_funcv = np.vectorize(self.activation_func)  # to apply on each elem
        hidden_layer_values = self.activation_funcv(hidden_layer_values)
        self.hidden_layer_values = np.array(hidden_layer_values)
        output = np.dot(hidden_layer_values, self.output_weights)  # (k*L).(L*n) = k*n
        return output

    def feed_backward(self, x_train, y_train):
        o = self.feed_forward(x_train).T  # k*n
        delta = y_train - o  # k*n
        delta_o = o*(1-o)*delta # k*n
        new_output_weights = self.output_weights + self.learning_rate*self.hidden_output.T.dot(delta_o) # L*n + (1*1).(L*k).(k*n) = L*n



    def activation_func(self, net):
        return 1 / (1 + np.math.exp(-net))

    def calc_mse(self, yt, yp):
        return sum((yt - yp)**2)


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
model = NeuralNetwork(m, l, n)
yp = model.feed_forward(x_train)
mse = model.calc_mse(y_train, yp)
print(mse)