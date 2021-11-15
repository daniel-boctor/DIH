import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_teacher(net, trainloader, lr, wd, epochs, momentum, milestones, gamma, seed):
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"EPOCH {epoch+1}: {running_loss/len(trainloader)}")

    print('Finished Training')

    return net