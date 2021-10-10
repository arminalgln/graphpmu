import torch
from models.graph_model import graphpmu
#%%
with open('data/positive_graphs.pkl', 'rb') as handle:
  pos_graphs = pickle.load(handle)
#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph = dgl.batch(pos_graphs[1:10])
model = GraphEncoder(25, 32, 64, 8).to(device)
#%%
for i in range(10):
    h1, h2, H = model(graph, graph.ndata['features'])
    print(next(model.parameters())[0][0])
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # predicted = model(pmu_event_data)

    optimizer.zero_grad()
    loss = criterion(H.float(), torch.rand((9,8), device=device).float())
    loss.backward()
    # train_losses.append(loss.item())
    optimizer.step()
    print(next(model.parameters())[0][0])



#%%

# for epoch in range(1, n_epochs + 1):
  model = model.train()
  # for bat_num, batch_input in enumerate(train_dataset):
  #           splitted_batch = np.split(batch_input, pmu_number, axis=-1)

            # for count, pmu_event_data in enumerate(splitted_batch):
                pmu_event_data = pmu_event_data.to(device)
                # train AED with reconstruction loss
                predicted = model(pmu_event_data)
                optimizer.zero_grad()
                loss = criterion(predicted.float(), pmu_event_data.float())
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()




#%%
import matplotlib.pyplot as plt
a = h1.detach().cpu().numpy()
b = h2.detach().cpu().numpy()
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
#%%
for parameter in model.parameters():
    print(parameter.shape)
