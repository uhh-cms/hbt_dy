input_dim = 10
num_classes = 3

model=MultiClassNN(input_dim, num_classes)
model.load_state_dict(torch.load("test.pth"))


