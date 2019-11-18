from torch import nn, optim, from_numpy
import torch.nn.functional as F
import numpy as np

# Load the diabets data
data = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(data[:, 0:-1])
y_data = from_numpy(data[:, -1])

print(f"X's shape: {x_data.shape} | Y's shape: {y_data.shape}")

class Model(nn.Module):
	def __init__(self):
		"""
		Instantiate the model
		"""
		super(Model, self).__init__()
		self.fc1 = nn.Linear(8, 5)
		self.fc2 = nn.Linear(5, 3)
		self.fc3 = nn.Linear(3, 1)

	def forward(self, x):
		out1 = F.relu(self.fc1(x))
		out2 = F.relu(self.fc2(out1))
		
		y_preds = F.sigmoid(self.fc3(out2))

		return y_preds


# our model
model = Model()

# Loss function and optimizer
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
	# Forward pass: pass the full dataset (batch learning)
	y_preds = model(x_data)
	
	# Compute the loss
	loss = criterion(y_preds, y_data)
	
	if epoch % 1000 == 0:
		print(f'Epoch {epoch+1}/10000 | Loss: {loss.item():.4f}')
	
	# Zero grads, backward, update weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
