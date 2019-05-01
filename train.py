import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torchvision
import torchvision.transforms as transforms

from load_data import QAdataset
from model import Net

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Hyper-parameters
input_sizes = [10, 3]
hidden_sizes = [64*2, 128*2]
num_class = 1
num_epochs = 5
batch_size = 32
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(
    QAdataset(
        para_path='./train_para.csv',
        ques_path='./train_ques.csv',
        label_path='./train_label.csv',
        test_mode=False
    ),
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    QAdataset(
        para_path='./val_para.csv',
        ques_path='./val_ques.csv',
        label_path='./val_label.csv',
        test_mode=False
    ),
    batch_size=batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True
)
model = Net(input_sizes, hidden_sizes, num_class).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, h, y) in enumerate(train_loader):
        x = x.to(device)
        h = h.to(device)
        y = y.to(device)
        # Forward pass
        outputs = model(x, h)
        loss = criterion(outputs, y)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for x, h, y in val_loader:
        x, h, y = x.to(device), h.to(device), y.to(device)
        outputs = model(x, h)
        thres = Variable(torch.Tensor([0.5])).to(device)
        predicted = (outputs > thres).float() * 1.
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Accuracy of the network on the 3200 val examples:{}%'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')