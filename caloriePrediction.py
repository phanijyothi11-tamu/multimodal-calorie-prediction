import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

cgm_dataframe = pd.read_csv('cgm_train.csv')
cgm_dataframe.head()

cgm_dataframe['Lunch Time'] = pd.to_datetime(cgm_dataframe['Lunch Time'], errors='coerce')
cgm_dataframe['Lunch Time']= cgm_dataframe['Lunch Time'].dt.time
valid_times = cgm_dataframe['Lunch Time'].dropna()
valid_seconds = [t.hour * 3600 + t.minute * 60 + t.second for t in valid_times]
avg_seconds = int(np.mean(valid_seconds))
avg_time = pd.to_datetime(avg_seconds, unit='s').time()
cgm_dataframe['Lunch Time'] = cgm_dataframe['Lunch Time'].fillna(avg_time)

import ast

cgm_dataframe['CGM Data'] = cgm_dataframe['CGM Data'].apply(ast.literal_eval)

cgm_dataframe=cgm_dataframe.drop(['Breakfast Time'],axis=1)

cgm_dataframe = cgm_dataframe[cgm_dataframe['CGM Data'].apply(lambda x: len(x) > 0)]

print(cgm_dataframe.shape)

from datetime import datetime
for index, row in cgm_dataframe.iterrows():
    cgm_data_list = row['CGM Data']
    updated_cgm_data = [(datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S"), item[1]) for item in cgm_data_list]
    cgm_dataframe.at[index, 'CGM Data'] = updated_cgm_data
print(cgm_dataframe)

from datetime import datetime, timedelta
for index, row in cgm_dataframe.iterrows():
    cgm_data_list = row['CGM Data']
    updated_cgm_data = []
    for i in range(len(cgm_data_list)):
        time_str, value = cgm_data_list[i]
        time = datetime.strptime(time_str, "%H:%M:%S").time()
        updated_cgm_data.append((time, value))
    new_cgm_data = []
    if len(updated_cgm_data) > 0 :
      current_time = updated_cgm_data[0][0]
      current_value = updated_cgm_data[0][1]
      new_cgm_data.append((current_time, current_value))
      for i in range(1,len(updated_cgm_data)):
          next_time = updated_cgm_data[i][0]
          next_value = updated_cgm_data[i][1]
          time_diff = timedelta(hours=next_time.hour, minutes=next_time.minute, seconds=next_time.second) - timedelta(hours=current_time.hour, minutes=current_time.minute, seconds=current_time.second)
          if time_diff >= timedelta(minutes=5):
              while time_diff >= timedelta(minutes=5):
                  current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=5)).time()
                  avg_val = (current_value + next_value)/2
                  new_cgm_data.append((current_time, avg_val))
                  time_diff -= timedelta(minutes=5)
          new_cgm_data.append((next_time,next_value))
          current_time = next_time
          current_value = next_value

      cgm_dataframe.at[index, 'CGM Data'] = new_cgm_data
cgm_dataframe

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
for index, row in cgm_dataframe.iterrows():
    lunch_time = row['Lunch Time']
    cgm_data_list = row['CGM Data']
    rounded_lunch_time = (datetime.combine(datetime.today(), lunch_time) +
                          timedelta(minutes=5 - lunch_time.minute % 5)).time()
    post_lunch_cgm = []
    for time_str, value in cgm_data_list:
        if isinstance(time_str, str):
            time = datetime.strptime(time_str, "%H:%M:%S").time()
        else:
            time = time_str

        if time >= rounded_lunch_time:
            post_lunch_cgm.append((time, value))
    if not post_lunch_cgm:
        post_lunch_cgm = cgm_data_list[-13:]
    cgm_values = [value for _, value in post_lunch_cgm]
    cgm_diff = np.diff(cgm_values).tolist()
    while len(cgm_diff) < 12:
        cgm_diff.append(np.nan)
    cgm_features = {}
    for i in range(12):
        cgm_features[f'cgm_diff{i+1}'] = cgm_diff[i]
    if len(cgm_diff) > 0 and not all(np.isnan(cgm_diff)):
        diff_mean = np.nanmean(cgm_diff)
        diff_max = np.nanmax(cgm_diff)
        diff_min = np.nanmin(cgm_diff)
        diff_std = np.nanstd(cgm_diff)
        diff_median = np.nanmedian(cgm_diff)
        diff_range = diff_max - diff_min
        diff_iqr = np.percentile(cgm_diff, 75) - np.percentile(cgm_diff, 25)
        diff_variance = np.nanvar(cgm_diff)
    else:
        diff_mean = diff_max = diff_min = diff_std = diff_median = diff_range = diff_iqr = diff_variance = np.nan
    for feature_name, feature_value in cgm_features.items():
        cgm_dataframe.loc[index, feature_name] = feature_value
    cgm_dataframe.loc[index, 'diff_mean'] = diff_mean
    cgm_dataframe.loc[index, 'diff_max'] = diff_max
    cgm_dataframe.loc[index, 'diff_min'] = diff_min
    cgm_dataframe.loc[index, 'diff_std'] = diff_std
    cgm_dataframe.loc[index, 'diff_median'] = diff_median
    cgm_dataframe.loc[index, 'diff_range'] = diff_range
    cgm_dataframe.loc[index, 'diff_iqr'] = diff_iqr
    cgm_dataframe.loc[index, 'diff_variance'] = diff_variance
from scipy.signal import find_peaks
from scipy.integrate import simps
for index, row in cgm_dataframe.iterrows():
    lunch_time = row['Lunch Time']
    cgm_data_list = row['CGM Data']
    rounded_lunch_time = (datetime.combine(datetime.today(), lunch_time) +
                          timedelta(minutes=5 - lunch_time.minute % 5)).time()
    post_lunch_cgm = []
    for time_str, value in cgm_data_list:
        if isinstance(time_str, str):
            time = datetime.strptime(time_str, "%H:%M:%S").time()
        else:
            time = time_str
        if time >= rounded_lunch_time:
            post_lunch_cgm.append((time, value))
    if not post_lunch_cgm:
        post_lunch_cgm = cgm_data_list[-13:]
    cgm_values = [value for _, value in post_lunch_cgm]
    cgm_diff = np.diff(cgm_values).tolist()
    while len(cgm_diff) < 12:
        cgm_diff.append(np.nan)
    cgm_features = {}
    for i in range(12):
        cgm_features[f'cgm_diff{i+1}'] = cgm_diff[i]
    if len(cgm_diff) > 0 and not all(np.isnan(cgm_diff)):
        diff_mean = np.nanmean(cgm_diff)
        diff_max = np.nanmax(cgm_diff)
        diff_min = np.nanmin(cgm_diff)
        diff_std = np.nanstd(cgm_diff)
        diff_median = np.nanmedian(cgm_diff)
        diff_range = diff_max - diff_min
        diff_iqr = np.percentile(cgm_diff, 75) - np.percentile(cgm_diff, 25)
        diff_variance = np.nanvar(cgm_diff)
    else:
        diff_mean = diff_max = diff_min = diff_std = diff_median = diff_range = diff_iqr = diff_variance = np.nan
    if post_lunch_cgm:
        times = [datetime.strptime(str(t), "%H:%M:%S").time() for t, _ in post_lunch_cgm]
        values = [v for _, v in post_lunch_cgm]
        time_to_peak = (datetime.combine(datetime.today(), max(times)) -
                        datetime.combine(datetime.today(), rounded_lunch_time)).seconds / 60.0
        time_to_trough = (datetime.combine(datetime.today(), min(times)) -
                          datetime.combine(datetime.today(), rounded_lunch_time)).seconds / 60.0
    else:
        time_to_peak = time_to_trough = np.nan
    rate_of_change = np.gradient(cgm_values).tolist() if cgm_values else []
    max_rate_of_change = max(rate_of_change) if rate_of_change else np.nan
    min_rate_of_change = min(rate_of_change) if rate_of_change else np.nan
    peaks, _ = find_peaks(cgm_values)
    troughs, _ = find_peaks([-v for v in cgm_values])
    peak_count = len(peaks)
    trough_count = len(troughs)
    peak_to_peak_diff = max(cgm_values) - min(cgm_values) if cgm_values else np.nan
    auc = simps(cgm_values) if cgm_values else np.nan
    for feature_name, feature_value in cgm_features.items():
        cgm_dataframe.loc[index, feature_name] = feature_value

    cgm_dataframe.loc[index, 'diff_mean'] = diff_mean
    cgm_dataframe.loc[index, 'diff_max'] = diff_max
    cgm_dataframe.loc[index, 'diff_min'] = diff_min
    cgm_dataframe.loc[index, 'diff_std'] = diff_std
    cgm_dataframe.loc[index, 'diff_median'] = diff_median
    cgm_dataframe.loc[index, 'diff_range'] = diff_range
    cgm_dataframe.loc[index, 'diff_iqr'] = diff_iqr
    cgm_dataframe.loc[index, 'diff_variance'] = diff_variance
    cgm_dataframe.loc[index, 'time_to_peak'] = time_to_peak
    cgm_dataframe.loc[index, 'time_to_trough'] = time_to_trough
    cgm_dataframe.loc[index, 'max_rate_of_change'] = max_rate_of_change
    cgm_dataframe.loc[index, 'min_rate_of_change'] = min_rate_of_change
    cgm_dataframe.loc[index, 'peak_count'] = peak_count
    cgm_dataframe.loc[index, 'trough_count'] = trough_count
    cgm_dataframe.loc[index, 'peak_to_peak_diff'] = peak_to_peak_diff
    cgm_dataframe.loc[index, 'auc'] = auc
print(cgm_dataframe)

cgm_dataframe=cgm_dataframe.drop(['CGM Data'],axis=1)

cgm_dataframe['Lunch Time'] = cgm_dataframe['Lunch Time'].apply(lambda x: x.hour * 60 + x.minute)

cgm_dataframe.head()

cgm_dataframe.isnull().sum()

print(cgm_dataframe.shape)

demo_viome_dataframe = pd.read_csv('demo_viome_train.csv')
demo_viome_dataframe.head()

print(demo_viome_dataframe.shape)

img_dataframe = pd.read_csv('img_train.csv')
img_dataframe.head()

print(img_dataframe.isnull().sum())

print(img_dataframe.shape)

label_dataframe = pd.read_csv('label_train.csv')
label_dataframe.head()

print(label_dataframe.shape)

img_label_dataframe = pd.merge(img_dataframe, label_dataframe, on=['Day','Subject ID'])
img_label_dataframe.head()

print(img_label_dataframe.shape)

img_label_cgm_dataframe = pd.merge(img_label_dataframe, cgm_dataframe, on=['Day','Subject ID'])
img_label_cgm_dataframe.head()

print(img_label_cgm_dataframe.shape)

print(img_label_cgm_dataframe.columns)

print(img_label_cgm_dataframe.isnull().sum())

print(img_label_cgm_dataframe.columns)

img_label_cgm_dataframe.head()

img_label_cgm_demo_viome_dataframe= pd.merge(img_label_cgm_dataframe, demo_viome_dataframe, on='Subject ID')
img_label_cgm_demo_viome_dataframe.head()

print(img_label_cgm_demo_viome_dataframe.shape)

print(img_label_cgm_demo_viome_dataframe.columns)

print(img_label_cgm_demo_viome_dataframe.isnull().sum())

img_label_cgm_demo_viome_dataframe= img_label_cgm_demo_viome_dataframe.drop(['Subject ID','Day'],axis=1)

print(img_label_cgm_demo_viome_dataframe.shape)

img_label_cgm_demo_viome_dataframe.head()

img_label_cgm_demo_viome_dataframe['Image Before Lunch'] = img_label_cgm_demo_viome_dataframe['Image Before Lunch'].apply(ast.literal_eval)
#str to tensor

img_label_cgm_demo_viome_dataframe = img_label_cgm_demo_viome_dataframe[img_label_cgm_demo_viome_dataframe['Image Before Lunch'].apply(lambda x: len(x) > 0)]

testing= img_label_cgm_demo_viome_dataframe.drop(['Image Before Breakfast','Lunch Protein','Breakfast Carbs','Lunch Carbs','Breakfast Fat','Lunch Fat','Breakfast Protein'],axis=1)

test = testing['Lunch Calories']
training = testing.drop(['Lunch Calories'],axis=1)
#split output and input features

testing.head(1)
testing.columns

print(training.shape)
print(test.shape)

training.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(training,test,test_size=0.1,random_state=1)

x_train['Viome_split'] = x_train['Viome'].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) else [])
print(x_train[['Viome', 'Viome_split']].head())
x_train['Viome_empty'] = x_train['Viome_split'].apply(lambda x: len(x) == 0)
print(x_train[['Viome', 'Viome_empty']].head())
x_train['Viome_mean'] = x_train['Viome_split'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
x_train['Viome_std'] = x_train['Viome_split'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
x_train['Viome_min'] = x_train['Viome_split'].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
x_train['Viome_max'] = x_train['Viome_split'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
print(x_train[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].head())
for i in range(27):
    x_train[f'Viome_value_{i+1}'] = x_train['Viome_split'].apply(lambda x: x[i] if len(x) > i else np.nan)
print(x_train[[f'Viome_value_{i+1}' for i in range(27)]].head())

x_test['Viome_split'] = x_test['Viome'].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) else [])
print(x_test[['Viome', 'Viome_split']].head())
x_test['Viome_empty'] = x_test['Viome_split'].apply(lambda x: len(x) == 0)
print(x_test[['Viome', 'Viome_empty']].head())
x_test['Viome_mean'] = x_test['Viome_split'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
x_test['Viome_std'] = x_test['Viome_split'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
x_test['Viome_min'] = x_test['Viome_split'].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
x_test['Viome_max'] = x_test['Viome_split'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
print(x_test[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].head())
for i in range(27):
    x_test[f'Viome_value_{i+1}'] = x_test['Viome_split'].apply(lambda x: x[i] if len(x) > i else np.nan)
print(x_test[[f'Viome_value_{i+1}' for i in range(27)]].head())

x_train[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']] = x_train[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].apply(pd.to_numeric, errors='coerce')
x_test[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']] = x_test[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].apply(pd.to_numeric, errors='coerce')
for i in range(27):
    x_train[f'Viome_value_{i+1}'] = pd.to_numeric(x_train[f'Viome_value_{i+1}'], errors='coerce')
    x_test[f'Viome_value_{i+1}'] = pd.to_numeric(x_test[f'Viome_value_{i+1}'], errors='coerce')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
x_train['Race'] = encoder.fit_transform(x_train['Race'])
x_test['Race'] = encoder.transform(x_test['Race'])
#encoding

x_train=x_train.drop(['Viome','Viome_split','Breakfast Calories','Lunch Fiber','Breakfast Fiber','Viome_empty'],axis=1)
x_test=x_test.drop(['Viome','Viome_split','Breakfast Calories','Lunch Fiber','Breakfast Fiber','Viome_empty'],axis=1)

print(x_train.columns)

missing_in_test = set(x_train.columns) - set(x_test.columns)
print("Columns in x_train but missing in x_test:", missing_in_test)
missing_in_train = set(x_test.columns) - set(x_train.columns)
print("Columns in x_test but missing in x_train:", missing_in_train)

image_column = 'Image Before Lunch'
cgm_columns = ['cgm_diff1', 'cgm_diff2',
       'cgm_diff3', 'cgm_diff4', 'cgm_diff5', 'cgm_diff6', 'cgm_diff7',
       'cgm_diff8', 'cgm_diff9', 'cgm_diff10', 'cgm_diff11', 'cgm_diff12',
       'diff_mean', 'diff_max', 'diff_min', 'diff_std', 'diff_median',
       'diff_range', 'diff_iqr', 'diff_variance', 'time_to_peak',
       'time_to_trough', 'max_rate_of_change', 'min_rate_of_change',
       'peak_count', 'trough_count', 'peak_to_peak_diff', 'auc']
x_cgm = x_train[cgm_columns].values.astype(np.float32)
x_image = x_train[image_column]
x_image = np.array(x_image.tolist(), dtype=np.float32) / 255.0
remaining_columns = [col for col in x_train.columns if col not in cgm_columns+ [image_column]]
x_neural = x_train[remaining_columns].values.astype(np.float32)
y_train = y_train.astype(np.float32)

print("Type of x_cgm:", type(x_cgm), "Shape:", np.shape(x_cgm))
print("Type of x_neural:", type(x_neural), "Shape:", np.shape(x_neural))
print("Type of x_image:", type(x_image), "Shape:", np.shape(x_image))
print("Type of y_train:", type(y_train), "Shape:", np.shape(y_train))
x_cgm = torch.tensor(x_cgm, dtype=torch.float32)
x_neural = torch.tensor(x_neural, dtype=torch.float32)
x_image = torch.tensor(x_image, dtype=torch.float32).view(267,3,64,64)
y_train = torch.tensor(y_train, dtype=torch.float32)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset,random_split
# import matplotlib.pyplot as plt

# class RMSRELoss(nn.Module):
#     def __init__(self):
#         super(RMSRELoss, self).__init__()

#     def forward(self, predicted, target):
#         epsilon = 1e-6  # Small constant to avoid division by zero
#         relative_error = (predicted - target) / (target + epsilon)
#         squared_relative_error = relative_error ** 2
#         rmsre = torch.sqrt(torch.mean(squared_relative_error))
#         return rmsre
# # Define the PyTorch Model
# class MultimodalNetwork(nn.Module):
#     def __init__(self):
#         super(MultimodalNetwork, self).__init__()

#         # CGM Input Subnetwork
#         self.cgm_rnn = nn.LSTM(input_size=1, hidden_size=64, bidirectional=True, batch_first=True)
#         self.cgm_dense = nn.Sequential(
#             nn.Linear(64 * 2, 64),  # BiLSTM doubles the hidden size
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         # Neural Input Subnetwork
#         self.neural_dense = nn.Sequential(
#             nn.Linear(50, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         # Image Input Subnetwork
#         self.image_cnn = nn.Sequential(
#             nn.Conv2d(3, 32,kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.image_dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, 64),  # Flattened dimensions depend on the input image size
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         # Fully Connected Layers
#         self.fc = nn.Sequential(
#             nn.Linear(64 + 64 + 64, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 1)
#         )

#     def forward(self, cgm_input, neural_input, image_input):
#         # CGM Subnetwork
#         cgm_input = cgm_input.unsqueeze(-1)  # Reshape for LSTM
#         cgm_output, _ = self.cgm_rnn(cgm_input)
#         cgm_output = cgm_output[:, -1, :]  # Take the last output (many-to-one)
#         cgm_output = self.cgm_dense(cgm_output)

#         # Neural Subnetwork
#         neural_output = self.neural_dense(neural_input)

#         # Image Subnetwork
#         image_output = self.image_cnn(image_input)
#         image_output = self.image_dense(image_output)

#         # Merge and Final Layers
#         merged = torch.cat([cgm_output, neural_output, image_output], dim=1)
#         output = self.fc(merged)

#         return output

# # Instantiate the Model
# model = MultimodalNetwork()

# # Loss and Optimizer
# criterion = RMSRELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Create DataLoader
# dataset = TensorDataset(x_cgm, x_neural, x_image, y_train)
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# # Training Loop
# epochs = 50
# train_losses = []
# val_losses = []

# for epoch in range(epochs):
#     # Training Phase
#     model.train()
#     train_loss = 0.0
#     for cgm_batch, neural_batch, image_batch, target_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(cgm_batch, neural_batch, image_batch)
#         loss = criterion(outputs, target_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Validation Phase
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for cgm_batch, neural_batch, image_batch, target_batch in val_loader:
#             outputs = model(cgm_batch, neural_batch, image_batch)
#             loss = criterion(outputs, target_batch)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)

#     # Print Training and Validation Loss
#     print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# # Plot Training and Validation Loss Graph
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training RMSRE', color='blue', linewidth=2)
# plt.plot(val_losses, label='Validation RMSRE', color='orange', linewidth=2)
# plt.title('Training and Validation RMSRE Loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt

# class RMSRELoss(nn.Module):
#     def __init__(self):
#         super(RMSRELoss, self).__init__()

#     def forward(self, predicted, target):
#         epsilon = 1e-6  # Small constant to avoid division by zero
#         relative_error = (predicted - target) / (target + epsilon)
#         squared_relative_error = relative_error ** 2
#         rmsre = torch.sqrt(torch.mean(squared_relative_error))
#         return rmsre

# # Updated Multimodal Network with additional layers
# class MultimodalNetwork(nn.Module):
#     def __init__(self):
#         super(MultimodalNetwork, self).__init__()

#         # CGM Input Subnetwork
#         self.cgm_rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True)
#         self.cgm_dense = nn.Sequential(
#             nn.Linear(64 * 2, 128),  # BiLSTM doubles the hidden size
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Neural Input Subnetwork
#         self.neural_dense = nn.Sequential(
#             nn.Linear(50, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Image Input Subnetwork
#         self.image_cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.image_dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, 128),  # Adjust based on image input size
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Fully Connected Layers
#         self.fc = nn.Sequential(
#             nn.Linear(64 + 64 + 64, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, cgm_input, neural_input, image_input):
#         # CGM Subnetwork
#         cgm_input = cgm_input.unsqueeze(-1)  # Reshape for LSTM
#         cgm_output, _ = self.cgm_rnn(cgm_input)
#         cgm_output = cgm_output[:, -1, :]  # Take the last output (many-to-one)
#         cgm_output = self.cgm_dense(cgm_output)

#         # Neural Subnetwork
#         neural_output = self.neural_dense(neural_input)

#         # Image Subnetwork
#         image_output = self.image_cnn(image_input)
#         image_output = self.image_dense(image_output)

#         # Merge and Final Layers
#         merged = torch.cat([cgm_output, neural_output, image_output], dim=1)
#         output = self.fc(merged)

#         return output

# # Instantiate Model
# model = MultimodalNetwork()

# # Loss and Optimizer
# criterion = RMSRELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-2)

# # DataLoader
# dataset = TensorDataset(x_cgm, x_neural, x_image, y_train)
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# # Training Loop with Early Stopping
# epochs = 100
# patience = 10
# best_val_loss = float('inf')
# patience_counter = 0

# train_losses = []
# val_losses = []

# for epoch in range(epochs):
#     # Training Phase
#     model.train()
#     train_loss = 0.0
#     for cgm_batch, neural_batch, image_batch, target_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(cgm_batch, neural_batch, image_batch)
#         loss = criterion(outputs, target_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Validation Phase
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for cgm_batch, neural_batch, image_batch, target_batch in val_loader:
#             outputs = model(cgm_batch, neural_batch, image_batch)
#             loss = criterion(outputs, target_batch)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)

#     # Early Stopping Check
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), 'best_model.pth')  # Save best model
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping triggered")
#             break

#     # Print Training and Validation Loss
#     print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# # Load Best Model
# model.load_state_dict(torch.load('best_model.pth'))

# # Plot Training and Validation Loss
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training RMSRE', color='blue', linewidth=2)
# plt.plot(val_losses, label='Validation RMSRE', color='orange', linewidth=2)
# plt.title('Training and Validation RMSRE Loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt

# class RMSRELoss(nn.Module):
#     def __init__(self):
#         super(RMSRELoss, self).__init__()

#     def forward(self, predicted, target):
#         epsilon = 1e-6  # Small constant to avoid division by zero
#         relative_error = (predicted - target) / (target + epsilon)
#         squared_relative_error = relative_error ** 2
#         rmsre = torch.sqrt(torch.mean(squared_relative_error))
#         return rmsre

# class MultimodalNetwork(nn.Module):
#     def __init__(self):
#         super(MultimodalNetwork, self).__init__()

#         # CGM Input Subnetwork
#         self.cgm_lstm1 = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
#         self.cgm_lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
#         self.cgm_lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
#         self.cgm_lstm4 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)

#         self.cgm_dense = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Neural Input Subnetwork
#         self.neural_dense = nn.Sequential(
#             nn.Linear(50, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Image Input Subnetwork
#         self.image_cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2)
#         )
#         self.image_dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, 256),  # Adjust based on image input size
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3)
#         )

#         # Fully Connected Layers
#         self.fc = nn.Sequential(
#             nn.Linear(64 + 64 + 64, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, cgm_input, neural_input, image_input):
#         # CGM Subnetwork
#         cgm_input = cgm_input.unsqueeze(-1)  # Reshape for LSTM
#         cgm_output, _ = self.cgm_lstm1(cgm_input)
#         cgm_output, _ = self.cgm_lstm2(cgm_output)
#         cgm_output, _ = self.cgm_lstm3(cgm_output)
#         cgm_output, _ = self.cgm_lstm4(cgm_output)
#         cgm_output = cgm_output[:, -1, :]  # Take the last output (many-to-one)
#         cgm_output = self.cgm_dense(cgm_output)

#         # Neural Subnetwork
#         neural_output = self.neural_dense(neural_input)

#         # Image Subnetwork
#         image_output = self.image_cnn(image_input)
#         image_output = self.image_dense(image_output)

#         # Merge and Final Layers
#         merged = torch.cat([cgm_output, neural_output, image_output], dim=1)
#         output = self.fc(merged)

#         return output


# # Instantiate Model
# model = MultimodalNetwork()

# # Loss and Optimizer
# criterion = RMSRELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# # DataLoader
# dataset = TensorDataset(x_cgm, x_neural, x_image, y_train)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# # Training Loop with Early Stopping
# epochs = 100
# patience = 10
# best_val_loss = float('inf')
# patience_counter = 0

# train_losses = []
# val_losses = []

# for epoch in range(epochs):
#     # Training Phase
#     model.train()
#     train_loss = 0.0
#     for cgm_batch, neural_batch, image_batch, target_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(cgm_batch, neural_batch, image_batch)
#         loss = criterion(outputs, target_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Validation Phase
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for cgm_batch, neural_batch, image_batch, target_batch in val_loader:
#             outputs = model(cgm_batch, neural_batch, image_batch)
#             loss = criterion(outputs, target_batch)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)

#     # Early Stopping Check
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), 'best_model.pth')  # Save best model
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping triggered")
#             break

#     # Print Training and Validation Loss
#     print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# # Load Best Model
# model.load_state_dict(torch.load('best_model.pth'))

# # Plot Training and Validation Loss
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training RMSRE', color='blue', linewidth=2)
# plt.plot(val_losses, label='Validation RMSRE', color='orange', linewidth=2)
# plt.title('Training and Validation RMSRE Loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt


# # Define RMSRE Loss
# class RMSRELoss(nn.Module):
#     def __init__(self):
#         super(RMSRELoss, self).__init__()

#     def forward(self, predicted, target):
#         epsilon = 1e-6  # Avoid division by zero
#         relative_error = (predicted - target) / (target + epsilon)
#         squared_relative_error = relative_error ** 2
#         rmsre = torch.sqrt(torch.mean(squared_relative_error))
#         return rmsre


# # Attention Layer
# class Attention(nn.Module):
#     def __init__(self, input_dim):
#         super(Attention, self).__init__()
#         self.attention_weights = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         attention_scores = self.attention_weights(x)  # Compute attention scores
#         attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize scores
#         weighted_sum = torch.sum(attention_weights * x, dim=1)  # Weighted sum of features
#         return weighted_sum


# # Define Multimodal Network
# class MultimodalNetwork(nn.Module):
#     def __init__(self):
#         super(MultimodalNetwork, self).__init__()

#         # CGM Input Subnetwork
#         self.cgm_lstm1 = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
#         self.cgm_lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
#         self.cgm_attention = Attention(128)  # Attention layer for CGM

#         self.cgm_dense = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Dropout(0.7),
#             nn.Linear(256, 256)
#         )

#         # Neural Input Subnetwork
#         self.neural_dense = nn.Sequential(
#             nn.Linear(50, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 20)
#         )

#         # Image Input Subnetwork
#         self.image_cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout2d(0.2)
#         )
#         self.image_attention = Attention(128)  # Attention layer for image features
#         self.image_dense = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128)
#         )

#         # Fully Connected Layers
#         self.fc = nn.Sequential(
#             nn.Linear(256 + 20 + 128, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, cgm_input, neural_input, image_input):
#         # CGM Subnetwork
#         cgm_input = cgm_input.unsqueeze(-1)  # Add channel dimension
#         cgm_output, _ = self.cgm_lstm1(cgm_input)
#         cgm_output, _ = self.cgm_lstm2(cgm_output)
#         cgm_output = self.cgm_attention(cgm_output)  # Apply attention
#         cgm_output = self.cgm_dense(cgm_output)

#         # Neural Subnetwork
#         neural_output = self.neural_dense(neural_input)

#         # Image Subnetwork
#         image_features = self.image_cnn(image_input)  # Extract image features
#         image_features = image_features.view(image_features.size(0), -1, 128)  # Reshape for attention
#         image_output = self.image_attention(image_features)  # Apply attention
#         image_output = self.image_dense(image_output)

#         # Merge and Final Layers
#         merged = torch.cat([cgm_output, neural_output, image_output], dim=1)
#         output = self.fc(merged)
#         return output



# # Dataset and DataLoader
# dataset = TensorDataset(x_cgm, x_neural, x_image, y_train)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# # Model Initialization
# model = MultimodalNetwork()
# criterion = RMSRELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# # Training Loop with Early Stopping
# epochs = 100
# patience = 15
# best_val_loss = float('inf')
# patience_counter = 0

# train_losses = []
# val_losses = []

# for epoch in range(epochs):
#     # Training Phase
#     model.train()
#     train_loss = 0.0
#     for cgm_batch, neural_batch, image_batch, target_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(cgm_batch, neural_batch, image_batch)
#         loss = criterion(outputs, target_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Validation Phase
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for cgm_batch, neural_batch, image_batch, target_batch in val_loader:
#             outputs = model(cgm_batch, neural_batch, image_batch)
#             loss = criterion(outputs, target_batch)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)

#     # Early Stopping Check
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), 'best_model.pth')  # Save best model
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping triggered")
#             break

#     # Print Training and Validation Loss
#     print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# # Load Best Model
# model.load_state_dict(torch.load('best_model.pth'))

# # Plot Training and Validation Loss
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training RMSRE', color='blue', linewidth=2)
# plt.plot(val_losses, label='Validation RMSRE', color='orange', linewidth=2)
# plt.title('Training and Validation RMSRE Loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
class RMSRELoss(nn.Module):
    def __init__(self):
        super(RMSRELoss, self).__init__()

    def forward(self, predicted, target):
        epsilon = 1e-6
        relative_error = (predicted - target) / (target + epsilon)
        squared_relative_error = relative_error ** 2
        rmsre = torch.sqrt(torch.mean(squared_relative_error))
        return rmsre
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        attention_scores = self.attention_weights(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)
        return weighted_sum
class MultimodalNetwork(nn.Module):
    def __init__(self):
        super(MultimodalNetwork, self).__init__()
        self.cgm_lstm1 = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.cgm_lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.cgm_attention = Attention(128)

        self.cgm_dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 256)
        )
        self.neural_dense = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 20)
        )
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )
        self.image_attention = Attention(128)
        self.image_dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 + 20 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, cgm_input, neural_input, image_input):
        cgm_input = cgm_input.unsqueeze(-1)
        cgm_output, _ = self.cgm_lstm1(cgm_input)
        cgm_output, _ = self.cgm_lstm2(cgm_output)
        cgm_output = self.cgm_attention(cgm_output)
        cgm_output = self.cgm_dense(cgm_output)
        neural_output = self.neural_dense(neural_input)
        image_features = self.image_cnn(image_input)
        image_features = image_features.view(image_features.size(0), -1, 128)
        image_output = self.image_attention(image_features)
        image_output = self.image_dense(image_output)
        merged = torch.cat([cgm_output, neural_output, image_output], dim=1)
        output = self.fc(merged)
        return output
dataset = TensorDataset(x_cgm, x_neural, x_image, y_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
learning_rates = [0.0001, 0.001, 0.01]
weight_decays = [1e-4, 1e-5, 1e-3]
best_model = None
best_lr = None
best_wd = None
best_val_loss = float('inf')

for lr in learning_rates:
    for wd in weight_decays:
        print(f"Training with lr={lr} and weight_decay={wd}")
        model = MultimodalNetwork()
        criterion = RMSRELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        epochs = 1
        patience = 15
        best_val_loss = float('inf')
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for cgm_batch, neural_batch, image_batch, target_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(cgm_batch, neural_batch, image_batch)
                loss = criterion(outputs, target_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for cgm_batch, neural_batch, image_batch, target_batch in val_loader:
                    outputs = model(cgm_batch, neural_batch, image_batch)
                    loss = criterion(outputs, target_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        model.load_state_dict(torch.load('best_model.pth'))

        if val_loss < best_val_loss:
            best_model = model
            best_lr = lr
            best_wd = wd
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training RMSRE', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation RMSRE', color='orange', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

model.eval()
x_cgm_test = x_test[cgm_columns].values.astype(np.float32)
x_cgm_test_tensor = torch.tensor(x_cgm_test, dtype=torch.float32)
print(f"x_cgm_test_tensor shape: {x_cgm_test_tensor.shape}")
remaining_columns = [col for col in x_test.columns if col not in cgm_columns + [image_column]]
x_neural_test = x_test[remaining_columns].values.astype(np.float32)
x_neural_test_tensor = torch.tensor(x_neural_test, dtype=torch.float32)
x_image_test = np.stack(x_test[image_column].values) / 255.0
x_image_test_tensor = torch.tensor(x_image_test, dtype=torch.float32).view(30,3,64,64)
print(x_image_test_tensor.shape)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

with torch.no_grad():
    y_pred_tensor = model(x_cgm_test_tensor, x_neural_test_tensor, x_image_test_tensor)
y_pred = y_pred_tensor.numpy().flatten()
print("Predictions:", y_pred)
def rmsre(y_true, y_pred):
    relative_errors = (y_true - y_pred) / y_true
    squared_relative_errors = relative_errors ** 2
    mean_squared_relative_error = np.mean(squared_relative_errors)
    rmsre_value = np.sqrt(mean_squared_relative_error)
    return rmsre_value
rmsre_value = rmsre(y_test.values, y_pred)
print(f"RMSRE: {rmsre_value:.6f}")

cgm_test_dataframe = pd.read_csv('cgm_test.csv')
cgm_test_dataframe.head()

cgm_test_dataframe.isnull().sum()

cgm_test_dataframe['Lunch Time'] = pd.to_datetime(cgm_test_dataframe['Lunch Time'], errors='coerce')
cgm_test_dataframe['Lunch Time']= cgm_test_dataframe['Lunch Time'].dt.time
valid_times = cgm_test_dataframe['Lunch Time'].dropna()
valid_seconds = [t.hour * 3600 + t.minute * 60 + t.second for t in valid_times]
avg_seconds = int(np.mean(valid_seconds))
avg_time = pd.to_datetime(avg_seconds, unit='s').time()
cgm_test_dataframe['Lunch Time'] = cgm_test_dataframe['Lunch Time'].fillna(avg_time)

import ast
print(cgm_test_dataframe.shape)
cgm_test_dataframe['CGM Data'] = cgm_test_dataframe['CGM Data'].apply(ast.literal_eval)

cgm_test_dataframe=cgm_test_dataframe.drop(['Breakfast Time'],axis=1)

cgm_test_dataframe = cgm_test_dataframe[cgm_test_dataframe['CGM Data'].apply(lambda x: len(x) > 0)]

cgm_test_dataframe.shape

from datetime import datetime
for index, row in cgm_test_dataframe.iterrows():
    cgm_data_list = row['CGM Data']
    updated_cgm_data = [(datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S"), item[1]) for item in cgm_data_list]
    cgm_test_dataframe.at[index, 'CGM Data'] = updated_cgm_data

print(cgm_test_dataframe)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from datetime import datetime, timedelta
for index, row in cgm_test_dataframe.iterrows():
    cgm_data_list = row['CGM Data']
    updated_cgm_data = []
    for i in range(len(cgm_data_list)):
        time_str, value = cgm_data_list[i]
        time = datetime.strptime(time_str, "%H:%M:%S").time()
        updated_cgm_data.append((time, value))
    new_cgm_data = []
    if len(updated_cgm_data) > 0 :
      current_time = updated_cgm_data[0][0]
      current_value = updated_cgm_data[0][1]
      new_cgm_data.append((current_time, current_value))
      for i in range(1,len(updated_cgm_data)):
          next_time = updated_cgm_data[i][0]
          next_value = updated_cgm_data[i][1]
          time_diff = timedelta(hours=next_time.hour, minutes=next_time.minute, seconds=next_time.second) - timedelta(hours=current_time.hour, minutes=current_time.minute, seconds=current_time.second)
          if time_diff >= timedelta(minutes=5):
              while time_diff >= timedelta(minutes=5):
                  current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=5)).time()
                  avg_val = (current_value + next_value)/2
                  new_cgm_data.append((current_time, avg_val))
                  time_diff -= timedelta(minutes=5)
          new_cgm_data.append((next_time,next_value))
          current_time = next_time
          current_value = next_value

      cgm_test_dataframe.at[index, 'CGM Data'] = new_cgm_data
cgm_test_dataframe

from scipy.signal import find_peaks
from scipy.integrate import simps
for index, row in cgm_test_dataframe.iterrows():
    lunch_time = row['Lunch Time']
    cgm_data_list = row['CGM Data']


    rounded_lunch_time = (datetime.combine(datetime.today(), lunch_time) +
                          timedelta(minutes=5 - lunch_time.minute % 5)).time()

    post_lunch_cgm = []
    for time_str, value in cgm_data_list:
        if isinstance(time_str, str):
            time = datetime.strptime(time_str, "%H:%M:%S").time()
        else:
            time = time_str

        if time >= rounded_lunch_time:
            post_lunch_cgm.append((time, value))


    if not post_lunch_cgm:
        post_lunch_cgm = cgm_data_list[-13:]


    cgm_values = [value for _, value in post_lunch_cgm]


    cgm_diff = np.diff(cgm_values).tolist()


    while len(cgm_diff) < 12:
        cgm_diff.append(np.nan)


    cgm_features = {}
    for i in range(12):
        cgm_features[f'cgm_diff{i+1}'] = cgm_diff[i]


    if len(cgm_diff) > 0 and not all(np.isnan(cgm_diff)):
        diff_mean = np.nanmean(cgm_diff)
        diff_max = np.nanmax(cgm_diff)
        diff_min = np.nanmin(cgm_diff)
        diff_std = np.nanstd(cgm_diff)
        diff_median = np.nanmedian(cgm_diff)
        diff_range = diff_max - diff_min
        diff_iqr = np.percentile(cgm_diff, 75) - np.percentile(cgm_diff, 25)
        diff_variance = np.nanvar(cgm_diff)
    else:
        diff_mean = diff_max = diff_min = diff_std = diff_median = diff_range = diff_iqr = diff_variance = np.nan


    if post_lunch_cgm:
        times = [datetime.strptime(str(t), "%H:%M:%S").time() for t, _ in post_lunch_cgm]
        values = [v for _, v in post_lunch_cgm]
        time_to_peak = (datetime.combine(datetime.today(), max(times)) -
                        datetime.combine(datetime.today(), rounded_lunch_time)).seconds / 60.0
        time_to_trough = (datetime.combine(datetime.today(), min(times)) -
                          datetime.combine(datetime.today(), rounded_lunch_time)).seconds / 60.0
    else:
        time_to_peak = time_to_trough = np.nan


    rate_of_change = np.gradient(cgm_values).tolist() if cgm_values else []
    max_rate_of_change = max(rate_of_change) if rate_of_change else np.nan
    min_rate_of_change = min(rate_of_change) if rate_of_change else np.nan


    peaks, _ = find_peaks(cgm_values)
    troughs, _ = find_peaks([-v for v in cgm_values])
    peak_count = len(peaks)
    trough_count = len(troughs)
    peak_to_peak_diff = max(cgm_values) - min(cgm_values) if cgm_values else np.nan


    auc = simps(cgm_values) if cgm_values else np.nan


    for feature_name, feature_value in cgm_features.items():
        cgm_test_dataframe.loc[index, feature_name] = feature_value


    cgm_test_dataframe.loc[index, 'diff_mean'] = diff_mean
    cgm_test_dataframe.loc[index, 'diff_max'] = diff_max
    cgm_test_dataframe.loc[index, 'diff_min'] = diff_min
    cgm_test_dataframe.loc[index, 'diff_std'] = diff_std
    cgm_test_dataframe.loc[index, 'diff_median'] = diff_median
    cgm_test_dataframe.loc[index, 'diff_range'] = diff_range
    cgm_test_dataframe.loc[index, 'diff_iqr'] = diff_iqr
    cgm_test_dataframe.loc[index, 'diff_variance'] = diff_variance
    cgm_test_dataframe.loc[index, 'time_to_peak'] = time_to_peak
    cgm_test_dataframe.loc[index, 'time_to_trough'] = time_to_trough
    cgm_test_dataframe.loc[index, 'max_rate_of_change'] = max_rate_of_change
    cgm_test_dataframe.loc[index, 'min_rate_of_change'] = min_rate_of_change
    cgm_test_dataframe.loc[index, 'peak_count'] = peak_count
    cgm_test_dataframe.loc[index, 'trough_count'] = trough_count
    cgm_test_dataframe.loc[index, 'peak_to_peak_diff'] = peak_to_peak_diff
    cgm_test_dataframe.loc[index, 'auc'] = auc

print(cgm_test_dataframe)
print(cgm_test_dataframe.shape)

cgm_test_dataframe['Lunch Time'] = cgm_test_dataframe['Lunch Time'].apply(lambda x: x.hour * 60 + x.minute)
cgm_test_dataframe.head()

cgm_test_dataframe.isnull().sum()

cgm_test_dataframe.shape

cgm_test_dataframe=cgm_test_dataframe.drop(['CGM Data'],axis=1)

demo_viome_test_dataframe = pd.read_csv('demo_viome_test.csv')
demo_viome_test_dataframe.head()

img_test_dataframe = pd.read_csv('img_test.csv')
img_test_dataframe.head()
mean_fiber_breakfast = img_test_dataframe['Breakfast Fiber'].mean()
print(mean_fiber_breakfast)
img_test_dataframe.fillna({'Breakfast Fiber': mean_fiber_breakfast}, inplace=True)

label_test_breakfast_only_dataframe = pd.read_csv('label_test_breakfast_only.csv')
label_test_breakfast_only_dataframe.head()

img_test_cgm_test_dataframe = pd.merge(img_test_dataframe, cgm_test_dataframe, on=['Day','Subject ID'])
img_test_cgm_test_dataframe.head()

img_test_cgm_test_dataframe_demo_viome_dataframe= pd.merge(img_test_cgm_test_dataframe, demo_viome_test_dataframe, on='Subject ID')
img_test_cgm_test_dataframe_demo_viome_dataframe.head()

print(img_test_cgm_test_dataframe_demo_viome_dataframe.isnull().sum())

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe= pd.merge(img_test_cgm_test_dataframe_demo_viome_dataframe, label_test_breakfast_only_dataframe, on=['Day','Subject ID'])

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.head()

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.shape

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.columns

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.drop(['Subject ID','Day','Image Before Breakfast','Breakfast Carbs', 'Breakfast Fat', 'Breakfast Protein','Breakfast Calories'],axis=1,inplace=True)

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.shape

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome'].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) else [])
print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome', 'Viome_split']].head())
img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_empty'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: len(x) == 0)
print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome', 'Viome_empty']].head())
img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_mean'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_std'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_min'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_max'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].head())
for i in range(27):  # Assuming you know there are 27 values in the list
    img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[f'Viome_value_{i+1}'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Viome_split'].apply(lambda x: x[i] if len(x) > i else np.nan)
print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[[f'Viome_value_{i+1}' for i in range(27)]].head())

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].apply(pd.to_numeric, errors='coerce')
#img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']] = x_test[['Viome_mean', 'Viome_std', 'Viome_min', 'Viome_max']].apply(pd.to_numeric, errors='coerce')
for i in range(27):
    img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[f'Viome_value_{i+1}'] = pd.to_numeric(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[f'Viome_value_{i+1}'], errors='coerce')

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Race'] = encoder.fit_transform(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Race'])

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.drop(['Viome_empty','Viome','Viome_split','Lunch Fiber','Breakfast Fiber'],axis=1,inplace=True)

print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.dtypes)

print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.shape)

print(img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.columns)

img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Image Before Lunch'] = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe['Image Before Lunch'].apply(ast.literal_eval)

test_x_cgm = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[cgm_columns].values.astype(np.float32)
test_x_image = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[image_column]
test_x_image = np.array(test_x_image.tolist(), dtype=np.float32) / 255.0
remaining_columns = [col for col in img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe.columns if col not in cgm_columns+ [image_column]]
test_x_neural = img_label_cgm_demo_viome_dataframe_label_test_breakfast_only_dataframe[remaining_columns].values.astype(np.float32)

print("Type of x_cgm:", type(x_cgm), "Shape:", np.shape(test_x_cgm))
print("Type of x_neural:", type(x_neural), "Shape:", np.shape(test_x_neural))
print("Type of x_image:", type(x_image), "Shape:", np.shape(test_x_image))
test_x_cgm = torch.tensor(test_x_cgm, dtype=torch.float32)
test_x_neural = torch.tensor(test_x_neural, dtype=torch.float32)
test_x_image = torch.tensor(test_x_image, dtype=torch.float32).view(73,3,64,64)

model.eval()
with torch.no_grad():
    test_outputs = model(test_x_cgm, test_x_neural, test_x_image)

print(test_outputs.shape)

test_outputs  = test_outputs.numpy().flatten()

print(test_outputs)

results_test_df = pd.DataFrame({
    'row_id': range(0, len(test_outputs)),
    'label': test_outputs
})
results_test_df.to_csv('predictions.csv', index=False)
print(results_test_df.head())
