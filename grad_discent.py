from MLP.micrograd import Value
import numpy as np
import math
from MLP.nn import MLP

print("Enter the list of inputs e.g., [1, 2, 3] : ")
inputs = eval(input())

print("Enter the target output for the given inputs : ")
ys = eval(input())

nin = len(inputs)
print("Enter the structure of outputs in nn in list format (e.g., [10, 5]) : ")
nouts = eval(input())

print("Enter the number of epochs for training: ")
epochs = int(input())

n = MLP(nin , nouts)

# y_pred = model(inputs)
# print("Predicted output : ", y_pred)

for epoch in range(epochs):
    y_pred = [n(x) for x in inputs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys,y_pred)])

    print(f"Epoch {epoch+1} : Loss = {loss.data}")
    loss.grad = 1.0
    loss.backward()

    for param in n.parameters():
        param.data -= 0.01 * param.grad

    
