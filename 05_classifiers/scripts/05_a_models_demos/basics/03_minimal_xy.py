import torch

import matplotlib.pyplot as plt
import matplotlib.collections as collections

import time
import os
import math


#the folder where this script is located
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#the path to the data.csv file to load
CSV_FILE = os.path.join(THIS_FOLDER, 'dataXY.csv')

#extraction of data from the csv file and storing it in lists
x_list = []
y_list = []

length = 0.0
l_list = []
with open(CSV_FILE, 'r') as f:
    for line in f:
        x, y = line.split(',')
        x = float(x)
        y = float(y)
        
        if len(x_list) > 0:
            dx = x - x_list[-1]
            dy = y - y_list[-1]
            l = math.sqrt(dx**2 + dy**2)
        else :
            l = 0.0
            
        length += l
        l_list.append(length)

        x_list.append(x)
        y_list.append(y)


count = len(x_list)
print(f'count: {count}')

#calculation of min and max values of x and y (used for visualization)
min_x = min(x_list)
max_x = max(x_list)
min_y = min(y_list)
max_y = max(y_list)
print(f'min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}')

#tenors from the lists
x = torch.tensor(x_list)
y = torch.tensor(y_list)
s = torch.tensor(l_list)/length


#initialization of the parameters of the model
param_count = 1000
p = torch.randn(param_count)*5.0

p.requires_grad = True

learning_rate = 0.02
optimizer = torch.optim.Adam([p], lr=learning_rate)
#optimizer = torch.optim.SGD([p], lr=learning_rate)

#enable interactive mode so that the plot is updated in real-time during training
plt.ion() 
#define some RGB colors
background_color = (0.2, 0.4, 0.5)
grid_color = (0.3, 0.5, 0.6)
text_color = (1.0, 1.0, 1.0)
data_points_color = (1.0, 1.0, 1.0)
data_line_color = (0.5, 0.5, 0.5)
predicted_color = (1.0, 0.5, 1.0)
gradient_color =    (0.3, 1.0, 1.0)
#define the size of the window in pixels
win_size = (800, 800)

#create a figure and axis object. An axis object is the object that we use to plot data on. The figure is the window that contains the axis.
#matplot lib calls "axis" an area within which we plot data with defined x and y ranges 
figure, ax = plt.subplots()

#set window size by converting pixels to inches
figure.set_size_inches(win_size[0]/figure.dpi, win_size[1]/figure.dpi)

#set the background color of the figure and axis
figure.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

#set the color of the axis spines (the lines that separate the axis from the plot area)
ax.spines['bottom'].set_color(grid_color)
ax.spines['top'].set_color(grid_color)
ax.spines['right'].set_color(grid_color)
ax.spines['left'].set_color(grid_color)
#set the color of the axis labels
ax.yaxis.label.set_color(text_color)
ax.xaxis.label.set_color(text_color)
#set the color of the ticks on the x and y axis
ax.tick_params(axis='x', colors=text_color)
ax.tick_params(axis='y', colors=text_color)

#create a line graph for the x,y data we loaded from the csv file
xy_data_graph, = ax.plot(x, y, linestyle='dashed', color = data_line_color)
#overlay the x,y data as dots on the line graph
xy_data_points_graph = ax.scatter(x, y, s = 10, color = data_points_color)
#create a line graph for the predicted y values. We will update this line graph during training
#for now we just create an empty line graph
xy_predicted_graph, = ax.plot([], [], color = predicted_color)
#create a line graph for the gradient of the predicted y values. We will update this line graph during training too
#the gradient will be a list of disconnected line segments that show the direction and magnitude of the gradient at each x value
grad_graph = collections.LineCollection([], color=gradient_color)
ax.add_collection(grad_graph)

#set the x and y axis limits taking into account the min and max x and y values
ax.set_xlim(min_x, max_x)
#we add a small margin to the y axis limits to make the plot fit better
ax.set_ylim(min_y-1.0, max_y+1.0)
#show the grid and set the color of the grid lines
ax.grid(color=grid_color)

#draw the figure and flush the event queue so that the figure is displayed
figure.canvas.draw()
figure.canvas.flush_events()

#the show function would normally block the execution of the program until the figure is closed
#but because we set the interactive mode plt.ion() at the start
#will be able to continue with the trainnig loop and update the plot in real-time
plt.show()
#we add a small delay of five seconds to make sure the plot is displayed before we start training
time.sleep(1.0)

#start the training loop
for epoch in range(8000):
    #zero the gradients of the parameters (reset the optimizer)
    optimizer.zero_grad()

    #apply the model to the x values to get the model predictions
    #simple line
    #y_pred = p[0]*x + p[1]

    #line + quadratic term (parabola)
    #y_pred = p[0]*x + p[1] + p[2]*(x - p[3])**2

    #line + quadratic term (parabola) + cubic
    # y_pred = p[0]*x + p[1]
    # y_pred += p[2]*(x - p[3])**2
    # y_pred += p[4]*(x - p[5])**3

    #sine wave
    #y_pred = p[0]*torch.sin(x*p[1]*5.0 + p[2]) + p[3]

    #line and waves
    # y_pred = p[0]*x + p[1]
    # y_pred += p[2]*torch.sin(x*p[3]*5.0 + p[4]) 
    # y_pred += p[5]*torch.cos(x*p[6]*5.0 + p[7])
    # y_pred += p[8]*torch.sin(x*p[9]*5.0 + p[10])

    #t = torch.linspace(0.0, 1.0, count)
    j = 0


    x_pred = p[j]*s + p[j+1]
    j += 2
    y_pred = p[j]*s + p[j+1]
    j += 2
    for i in range(0):
        x_pred += p[j]*torch.sigmoid(10.0*p[j+1] * (s - p[j+2]))
        j += 3
        y_pred += p[j]*torch.sigmoid(10.0*p[j+1] * (s - p[j+2]))
        j += 3

    #gaussians
    for i in range(0):
        x_pred += p[j]*torch.exp(-(s - p[j+1])**2/(p[j+2]**2))
        j += 3
        y_pred += p[j]*torch.exp(-(s - p[j+1])**2/(p[j+2]**2))
        j += 3

    #sines
    for i in range(0):
        x_pred += p[j]*torch.sin(s*p[j+1]*5.0 + p[j+2])
        j += 3
        y_pred += p[j]*torch.sin(s*p[j+1]*5.0 + p[j+2])
        j += 3

    #relus
    for i in range(5):
        x_pred += p[j]*torch.relu(x_pred - p[j+1]) + p[j+2]
        j += 3
        y_pred += p[j]*torch.relu(y_pred - p[j+1]) + p[j+2]
        j += 3

    # for i in range(30):
    #     x_pred += p[j]*torch.sin(s*i + p[j+2])
    #     j += 3
    #     y_pred += p[j]*torch.sin(s*i + p[j+2])
    #     j += 3

    #gaussian bump
    #y_pred = p[0]*x + p[1]
    # j = 2
    # for i in range(5):
    #     y_pred += p[j]*torch.exp(-(x - p[j+1])**2/(p[j+2]**2))
    #     j += 3

    # y_pred += p[2]*torch.exp(-(x - p[4])**2/(p[3]**2))
    # y_pred += p[5]*torch.exp(-(x - p[7])**2/(p[6]**2))
    # y_pred += p[8]*torch.exp(-(x - p[10])**2/(p[9]**2))

    #add smooth step

    #this is needed here for visualization purposes. We will update the line graph of the predicted y values
    #so we need to tell python to not discard the gradient of y_pred after the backward pass
    x_pred.retain_grad()
    y_pred.retain_grad()
    

    #the loss function is the mean squared error between the predicted y values and the actual y values
    loss = torch.mean((y_pred - y)**2) + torch.mean((x_pred - x)**2)

    #print(f'epoch: {epoch}, loss: {loss.item()}')

    #calculate the gradients of the loss with respect to the parameters
    loss.backward()

    #update the parameters using the gradients
    optimizer.step()

    #update the line graph of the predicted y values every 5 epochs
    if epoch % 5 == 0:
        print(epoch, loss.item())

        x_pred_array = x_pred.detach().numpy()
        y_pred_array = y_pred.detach().numpy()      

        x_pred_grad_array = x_pred.grad.detach().numpy() * 10.0
        y_pred_grad_array = y_pred.grad.detach().numpy() * 10.0

        xy_predicted_graph.set_xdata(x_pred_array)
        xy_predicted_graph.set_ydata(y_pred_array)
        grad_graph.set_segments([[(x_pred_array[i], y_pred_array[i]), (x_pred_array[i] - x_pred_grad_array[i], y_pred_array[i] - y_pred_grad_array[i])] for i in range(count)])

        figure.canvas.draw()
        figure.canvas.flush_events()
        #time.sleep(0.05)


print(f'Result: y = {p[1].item()}*x + {p[0].item()}')

plt.ioff()
plt.show()

