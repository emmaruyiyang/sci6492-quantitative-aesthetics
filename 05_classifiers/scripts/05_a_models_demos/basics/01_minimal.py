import torch

import matplotlib.pyplot as plt
import matplotlib.collections as collections

import time
import os

import random


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(THIS_FOLDER, 'data.csv')

x_list = []
y_list = []

with open(CSV_FILE, 'r') as f:
    for line in f:
        x, y = line.split(',')
        x_list.append(float(x))
        y_list.append(float(y))


x = torch.tensor(x_list)
y = torch.tensor(y_list)


params = [0.2, 0.4]
p = torch.tensor(params, requires_grad=True)

learning_rate = 0.02
optimizer = torch.optim.Adam([p], lr=learning_rate)


for epoch in range(4000):
    optimizer.zero_grad()

    y_pred = p[0]*x + p[1] 

    loss = torch.mean((y_pred - y)**2)

    print(f'epoch: {epoch}, loss: {loss.item()}')

    loss.backward()

    optimizer.step()


print(f'Result: y = {p[1].item():.3f}*x + {p[0].item():.3f}')


