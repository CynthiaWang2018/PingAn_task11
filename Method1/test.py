import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torchvision
import torchvision.transforms as transforms

from load_data import QAdataset
from model import Net
import sys
import pandas as pd

MODEL_PATH = sys.argv[1:]
print(MODEL_PATH)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_sizes = [10, 3]
hidden_sizes = [64*2, 128*2]
num_class = 1
batch_size = 1000
learning_rate = 0.001

test_loader = torch.utils.data.DataLoader(
    QAdataset(
        para_path='./test_para.csv',
        ques_path='./test_ques.csv',
        label_path='./test_label.csv',
        test_mode=False
    ),
    batch_size=batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True
)
print('ok')
model = Net(input_sizes, hidden_sizes, num_class).to(device)
print('ok2')
model.load_state_dict(torch.load('./model.ckpt'))
print('ok3')
pred_result = []
with torch.no_grad():
    correct = 0
    total = 0
    for x, h, y in test_loader:
        x, h, y = x.to(device), h.to(device), y.to(device)
        outputs = model(x, h)
        thres = Variable(torch.Tensor([0.5])).to(device)
        predicted = (outputs > thres).float() * 1.
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Accuracy of the network on the 1000 test examples:{}%'.format(100 * correct/total))
pred_result = outputs.cpu().detach().numpy()
df = pd.DataFrame({'pred':list(pred_result.reshape(-1))})
df2 = pd.DataFrame({'predicted':list(predicted.reshape(-1))})

df2.to_csv('./test_pred.csv', header=None, index=None)