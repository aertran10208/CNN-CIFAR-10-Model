import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

train_acc = 'Output1.txt'
test_acc = 'Output2.txt'

train_file = open(train_acc,'r')
lines = train_file.readlines()
xtrain = []
ytrain = []
for line in lines:
    data = line.split(" ")
    xtrain.append(int(data[0]))
    ytrain.append(float(data[1]))

test_file = open(test_acc,'r')
lines = test_file.readlines()
xtest = []
ytest = []
for line in lines:
    data = line.split(" ")
    xtest.append(int(data[0]))
    ytest.append(float(data[1]))

red_patch = mpatches.Patch(color='red', label='Test')
blue_patch = mpatches.Patch(color='blue', label='Train')
plt.legend(handles=[red_patch,blue_patch])
plt.plot(xtest, ytest, 'r-', xtrain, ytrain, 'b-')
plt.ylabel('Accuracy')
plt.xlabel('Steps')
plt.title('0.001 Learn Rate w/ Keep Probability 0.3')
plt.grid(True)
plt.show()
