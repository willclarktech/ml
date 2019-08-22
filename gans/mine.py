import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Data parameters

data_mean = 4
data_standard_deviation = 1.25

# Model parameters

g_input_size = 1
g_hidden_size = 5
g_output_size = 1
g_activation_function = torch.tanh
g_learning_rate = 1e-2

d_input_size = 500
d_hidden_size = 10
d_output_size = 1
d_activation_function = torch.sigmoid
d_learning_rate = 1e-3

sgd_momentum = 0.9

minibatch_size = d_input_size
calculate_error = nn.BCELoss()

# Training parameters
num_epochs = 2000
d_steps_per_epoch = 20
g_steps_per_epoch = 20


def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


sample_distribution = get_distribution_sampler(
    data_mean, data_standard_deviation)


def sample_generator_input(m, n):
    return torch.rand(m, n)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, input_layer):
        layer1 = self.f(self.map1(input_layer))
        layer2 = self.f(self.map2(layer1))
        return self.map3(layer2)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, input_layer):
        layer1 = self.f(self.map1(input_layer))
        layer2 = self.f(self.map2(layer1))
        return self.f(self.map3(layer2))


def stats(d):
    return np.mean(d), np.std(d)


def extract(v):
    return v.data.storage().tolist()


def train_d(D, G, d_optimizer):
    D.zero_grad()

    d_real_data = Variable(sample_distribution(d_input_size))
    d_real_decision = D.forward(d_real_data)
    d_real_error = calculate_error(
        d_real_decision,
        Variable(torch.ones([1, 1]))
    )
    d_real_error.backward()

    d_generator_input = Variable(
        sample_generator_input(minibatch_size, g_input_size)
    )
    d_fake_data = G.forward(d_generator_input).detach()
    d_fake_decision = D.forward(d_fake_data.t())
    d_fake_error = calculate_error(
        d_fake_decision,
        Variable(torch.zeros([1, 1]))
    )
    d_fake_error.backward()

    d_optimizer.step()
    return d_real_error, d_fake_error, d_real_data, d_fake_data


def train_g(D, G, g_optimizer):
    G.zero_grad()

    generated_input = Variable(
        sample_generator_input(minibatch_size, g_input_size)
    )
    g_fake_data = G.forward(generated_input)
    d_decision = D.forward(g_fake_data.t())
    g_error = calculate_error(d_decision, Variable(torch.ones([1, 1])))
    g_error.backward()

    g_optimizer.step()
    return g_error


def train(epochs, d_steps, g_steps, print_interval=100):
    G = Generator(
        g_input_size,
        g_hidden_size,
        g_output_size,
        g_activation_function
    )
    D = Discriminator(
        d_input_size,
        d_hidden_size,
        d_output_size,
        d_activation_function
    )
    d_optimizer = optim.SGD(
        D.parameters(),
        lr=d_learning_rate,
        momentum=sgd_momentum
    )
    g_optimizer = optim.SGD(
        G.parameters(),
        lr=g_learning_rate,
        momentum=sgd_momentum
    )

    for epoch in range(epochs):
        for _ in range(d_steps):
            d_real_error, d_fake_error, d_real_data, d_fake_data = train_d(
                D,
                G,
                d_optimizer
            )

        for _ in range(g_steps):
            g_error = train_g(D, G, g_optimizer)

        if epoch % print_interval == 0:
            print("Epoch %d:\nD (%s real error, %s fake error) G (%s error)\nReal dist %s, Fake dist %s\n"
                  % (epoch, extract(d_real_error)[0], extract(d_fake_error)[0], extract(g_error)[0], stats(extract(d_real_data)), stats(extract(d_fake_data)))
                  )


train(num_epochs, d_steps_per_epoch, g_steps_per_epoch)
