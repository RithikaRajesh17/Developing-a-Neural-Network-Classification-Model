# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="566" height="815" alt="image" src="https://github.com/user-attachments/assets/e28e99c3-7d55-4405-96bf-f2f8d6390fca" />

## DESIGN STEPS
### Step 1: Load and Preprocess Data
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### Step 2: Feature Scaling and Data Split
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### Step 3: Convert Data to PyTorch Tensors
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### Step 4: Define the Neural Network Model
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### Step 5: Train the Model
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### Step 6: Evaluate and Predict
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.




## PROGRAM

### Name:Rithika R

### Register Number:212224240136

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
      super(PeopleClassifier, self).__init__()
      self.fc1=nn.Linear(input_size,32)
      self.fc2=nn.Linear(32,16)
      self.fc3=nn.Linear(16,8)
      self.fc4=nn.Linear(8,4)


    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x


# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
      for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model=PeopleClassifier(X_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
train_model(model,train_loader,criterion,optimizer,epochs=100)


# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name:Rithika R      ")
print("Register No:212224240136       ")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: Rithika R       ")
print("Register No:212224240136 ")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
        
```

### Dataset Information

<img width="1287" height="257" alt="image" src="https://github.com/user-attachments/assets/072bc24e-4002-4cd0-b17e-9696fb1dece4" />


### OUTPUT

## Confusion Matrix
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/23c26fcf-76ac-4383-b684-2534cfff2a13" />


## Classification Report
<img width="789" height="456" alt="image" src="https://github.com/user-attachments/assets/1bf7f328-2146-4f43-a86d-0ecde56d7912" />


### New Sample Data Prediction
<img width="506" height="120" alt="image" src="https://github.com/user-attachments/assets/a1fd346b-8e0c-4846-a127-f6d2caca29a9" />

## RESULT
This program has been executed successfully.
