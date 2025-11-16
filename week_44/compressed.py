# compressed_safe.py
import os
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --------------------
# Config (conservative)
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16   # small batch to reduce peak memory
teacher_epochs = 3
student_epochs = 3
simple_epochs = 3
seed = 42
prune_threshold = 0.03
torch.manual_seed(seed)
np.random.seed(seed)

# --------------------
# Ensure directories exist
# --------------------
os.makedirs("models", exist_ok=True)
os.makedirs("models/sparse_layers", exist_ok=True)

# --------------------
# Helpers
# --------------------
def mb(x): return f"{x/(1024**2):.2f} MB"

def print_mem(prefix=""):
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"{prefix} CPU mem: total={mb(vm.total)} used={mb(vm.used)} free={mb(vm.available)}")
    except Exception:
        pass
    if torch.cuda.is_available():
        print(f"{prefix} CUDA alloc: {mb(torch.cuda.memory_allocated())}, cached: {mb(torch.cuda.memory_reserved())}")

def safe_to_cpu_numpy(t: torch.Tensor):
    t_cpu = t.detach().cpu()
    arr = t_cpu.numpy()
    del t_cpu
    gc.collect()
    return arr

# --------------------
# Data
# --------------------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                         pin_memory=(device.type == "cuda"))
testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=0,
                         pin_memory=(device.type == "cuda"))

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# --------------------
# Models
# --------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool  = nn.MaxPool2d(5, 5)
        self.fc1   = nn.Linear(32 * 4 * 4, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# --------------------
# Train teacher
# --------------------
teacher = TeacherNet().to(device)
opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for ep in range(teacher_epochs):
    teacher.train()
    tot_loss = 0.0
    n = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        opt_t.zero_grad()
        out = teacher(inputs)
        loss = crit(out, labels)
        loss.backward()
        opt_t.step()
        tot_loss += loss.item(); n += 1
    print(f"[Teacher] epoch {ep+1}/{teacher_epochs} loss={tot_loss/max(1,n):.4f} acc={evaluate(teacher):.3f}")

torch.save(teacher.state_dict(), "models/teacher_state.pt")

# --------------------
# Knowledge distillation (student)
# --------------------
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, labels)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return alpha * ce + (1.0 - alpha) * (T * T) * kl

student = StudentNet().to(device)
opt_s = optim.Adam(student.parameters(), lr=1e-3)
teacher.eval()

for ep in range(student_epochs):
    student.train()
    tot_loss = 0.0; n = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        opt_s.zero_grad()
        s_logits = student(inputs)
        with torch.no_grad():
            t_logits = teacher(inputs)
        loss = distillation_loss(s_logits, t_logits, labels, T=4.0, alpha=0.5)
        loss.backward()
        opt_s.step()
        tot_loss += loss.item(); n += 1
    print(f"[Student KD] epoch {ep+1}/{student_epochs} loss={tot_loss/max(1,n):.4f} acc={evaluate(student):.3f}")

torch.save(student.state_dict(), "models/student_state.pt")

# Free teacher
del teacher
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# --------------------
# Train SimpleNet
# --------------------
net = SimpleNet().to(device)
opt = optim.Adam(net.parameters(), lr=1e-3)

for ep in range(simple_epochs):
    net.train()
    tot_loss = 0.0; n=0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        out = net(inputs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        tot_loss += loss.item(); n += 1
    print(f"[SimpleNet] epoch {ep+1}/{simple_epochs} loss={tot_loss/max(1,n):.4f} acc={evaluate(net):.3f}")

torch.save(net.state_dict(), "models/simple_state.pt")

# --------------------
# Dense weights size
# --------------------
def dense_weights_bytes(model):
    tot = 0
    for n,p in model.named_parameters():
        if "weight" in n:
            tot += p.element_size() * p.numel()
    return tot

def mb(x): return f"{x/(1024**2):.2f} MB"
print("Dense weights size:", mb(dense_weights_bytes(net)))

# --------------------
# Prune & save sparse layers
# --------------------
with torch.no_grad():
    for name, param in net.named_parameters():
        if "weight" not in name:
            continue
        mask = (param.abs() >= prune_threshold).to(param.dtype)
        param.data.mul_(mask)
        nnz = int((mask != 0).sum().item())
        total = mask.numel()
        print(f"Layer {name}: nnz={nnz}/{total} ({100.0*nnz/total:.2f}%)")

        # Sparse conversion & save
        try:
            cpu_t = param.detach().cpu()
            sp = cpu_t.to_sparse().coalesce()
            nnz = sp._nnz()
            if nnz == 0:
                np.savez_compressed(f"models/sparse_layers/{name.replace('.', '_')}_empty.npz",
                                    shape=np.array(sp.size()))
            else:
                inds = sp.indices().numpy()
                vals = sp.values().numpy()
                np.savez_compressed(f"models/sparse_layers/{name.replace('.', '_')}.npz",
                                    indices=inds, values=vals, shape=np.array(sp.size()))
                del inds, vals
            del cpu_t, sp
            gc.collect()
        except Exception as e:
            print(f"Error converting/saving layer {name}: {type(e).__name__}: {e}")
            print_mem("After error:")
            raise

del net
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Pruning & per-layer sparse save completed. Done.")
