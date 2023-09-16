import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data_frame, window_size):
        self.data = torch.tensor(data_frame[['bt', 'bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse']].values)
        self.masks = torch.tensor(data_frame['mask'].values, dtype=torch.bool)
        self.time = torch.tensor(data_frame[['year_cos', 'year_sin', 'month_cos', 'month_sin', 'day_cos', 'day_sin', 'hour_cos', 'hour_sin', 'minute_cos', 'minute_sin']].values)
        self.window_size = window_size

    def __len__(self):
        return self.data.shape[0]-self.window_size

    def __getitem__(self, i):
        ind = slice(i, i+self.window_size)
        return self.data[ind], self.masks[ind], self.time[ind]

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size) # no clue what it is
    return loader

df = pd.read_csv('data.csv')
train = df[df['month'] < 8]
test = df[df['month'] >= 8]
trainset = TimeSeriesDataset(train, 500)
trainloader = make_loader(trainset, 1)

#Training

config = TimeSeriesTransformerConfig(
    prediction_length=100,
    context_length=393,
    input_size=6,
    num_time_features=10,
    # num_static_categorical_features=1,
    # num_static_real_features=1,
    # lags_sequence=[0], #wtf is this it might be important?
    
    # transformer params: ima just use the default
)

model = TimeSeriesTransformerForPrediction(config)

model.train()

epochs = 30
losses = list()
epoch_losses = list()

for epoch in tqdm(range(epochs), desc='Epochs'):
    total_loss = 0
    for data, mask, time in tqdm(trainloader, desc='Batches', leave=False):
        past_data, future_data = data[:, :400].float(), data[:, 400:].float()
        past_mask, future_mask = mask[:, :400], mask[:, 400:]
        past_time, future_time = time[:, :400].float(), time[:, 400:].float()

        past_mask = past_mask.unsqueeze(-1)
        future_mask = future_mask.unsqueeze(-1)
        # print(past_data.shape, future_data.shape, past_mask.shape, future_mask.shape, past_time.shape, future_time.shape)
        # print(past_data.type(), future_data.type(), past_mask.type(), future_mask.type(), past_time.type(), future_time.type())
        outputs = model(
            past_values=past_data,
            past_time_features=past_time,
            past_observed_mask=past_mask,
            future_values=future_data,
            future_time_features=future_time,
            future_observed_mask=future_mask,
            # static_real_features=torch.zeros((1, 1), dtype=torch.int64),
            # static_categorical_features=id,
            
        )

        loss = outputs.loss
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
    epoch_loss = total_loss / len(trainloader)
    epoch_losses.append(epoch_loss)

torch.save(model.state_dict(), 'model.pth')
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss per Iteration')
plt.savefig('losses.png')
plt.close()

plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.savefig('epoch_losses.png')
plt.close()