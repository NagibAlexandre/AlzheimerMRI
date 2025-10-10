if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    # --- Config ---
    DATA_PROCESSED = "dataset/Processed"
    IMG_SIZE = 224
    BATCH = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)
    EPOCHS = 10
    CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

    # --- Transforms ---
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # --- Datasets ---
    train_ds = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"train"), transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"val"), transform_test)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"test"), transform_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    # --- Modelo (Transfer Learning) ---
    model = models.resnet18(pretrained=True)

    # --- Congelar todas as camadas exceto a última ---
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))  # última camada treinável
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)  # só a última camada

    # --- Treino ---
    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct/total)

    # --- Avaliação ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Matriz de Confusão ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
