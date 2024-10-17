import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load the Data
data = pd.read_csv('mobdatakaluga3.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Step 2: Adjust Latitude and Longitude
# Adjust latitude to be between 36.1 and 36.45
data['latitude'] = np.clip(data['latitude'], 36.1, 36.45)

# Adjust longitude to be between 54.2 and 54.6
data['longitude'] = np.clip(data['longitude'], 54.2, 54.6)

# Step 3: Preprocess the Data
# One-hot encode the 'district' variable
data = pd.get_dummies(data, columns=['district'], drop_first=True)

# Encode the 'network_type' variable
data['network_type'] = data['network_type'].astype('category').cat.codes

# Features and target variable
X = data[['network_type', 'longitude', 'latitude', 'population_density'] + [col for col in data.columns if 'district_' in col]]
y = data['user'].values  # Assuming 'user' is the target variable

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Step 4: Build the FNN Model
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = FNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
num_epochs = 4000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the Model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Step 7: Make Predictions
predictions = test_outputs.numpy()

# Step 8: Visualization
# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Define colors for each network type
colors = {0: 'blue', 1: 'green', 2: 'red'}  # Assuming 0=2G, 1=3G, 2=4G
network_labels = {0: '2G Users', 1: '3G Users', 2: '4G Users'}

# Plot each network type in a separate subplot
for network_type in range(3):
    subset = data[data['network_type'] == network_type]
    axs[network_type].scatter(subset['longitude'], subset['latitude'], s=subset['user']*10, alpha=0.5, c=colors[network_type], edgecolors='w', linewidth=0.5)
    axs[network_type].set_title(network_labels[network_type])
    axs[network_type].set_xlabel('Longitude')
    axs[network_type].set_ylabel('Latitude')
    axs[network_type].set_xlim(54.575, 54.625)  # Set x-axis limits
    axs[network_type].set_ylim(36.05, 36.5)  # Set y-axis limits
    axs[network_type].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

