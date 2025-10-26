import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fcl = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 == nn.Linear(10, 1)

    def forward(self, x):
        out = self.fcl(x)
        out = self.sigmoid(out)
        out = self.fc2(out)

        return out
    
model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(epochs):
    for x, y in dataloader:


        x = x.to(device)
        y = y.to(device)

        outputs = model(x)   
        loss = criterion(outputs, y)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
