# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load Dataset
data = pd.read_csv("customers.csv")

# Encode categorical features
categorical_columns = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score"
]

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable
data["Segmentation"] = LabelEncoder().fit_transform(
    data["Segmentation"]
)

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# Neural Network Model
class PeopleClassifier(nn.Module):

    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# Initialize Model
model = PeopleClassifier(
    input_size=X_train.shape[1]
)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training Loop
for epoch in range(100):

    model.train()

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/100], "
            f"Loss: {loss.item():.4f}"
        )
