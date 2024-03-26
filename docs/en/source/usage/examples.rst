Examples
--------
You can build a simple convolutional spiking neural network, trained on the MNIST dataset, as follows::

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from snngrow.base.neuron.LIFNode import LIFNode
    from tqdm import tqdm

    # Define the CSNN model
    class CNN(nn.Module):
        def __init__(self, T):
            super(CNN, self).__init__()
            self.T = T
            self.csnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                LIFNode(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3),
                LIFNode(),
                nn.Flatten(),
                nn.Linear(7744, 128),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x_seq = []
            for _ in range(self.T):
                x_seq.append(self.csnn(x))
            out = torch.stack(x_seq).mean(0)
            return out

    def main():
        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.MNIST(root='./datas', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

        testset = torchvision.datasets.MNIST(root='./datas', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        # Create an instance of the CNN model
        model = CNN(T=4)
        # model to GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(10):
            running_loss = 0.0
            model.train()
            for data in tqdm(trainloader):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.cpu(), labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                for m in model.modules():
                    if hasattr(m, 'reset'):
                        m.reset()

            # Test the model
            correct = 0
            total = 0
            with torch.no_grad():
                model.eval()
                for data in testloader:
                    images, labels = data
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.detach().cpu().data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for m in model.modules():
                        if hasattr(m, 'reset'):
                            m.reset()

            print(f'Epoch: {epoch + 1}, Loss: {running_loss / 100}, Accuracy on the test set: {(correct / total) * 100}%')
            running_loss = 0.0

        print('Training finished.')