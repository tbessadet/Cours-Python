rn18 = resnet18(weights = ResNet18_Weights.DEFAULT).to(device)

for p in rn18.parameters(): 
    p.requires_grad = False
rn18.fc = nn.Linear(in_features=512, out_features=10, bias=True).to(device)

params = {'EPOCH': 60, 'lr': 0.005, 'batch_size': 128, 'patience': 8, 'eps': 1e-4, 'momentum': 0.9, 'lr_decay': 1, 'augment': False, 'freeze': 'rn18'}
perf, t = mlflow_load(rn18, 'ResNet18 gelé', strategie='gelé', **params)
print(f"{t} secondes")