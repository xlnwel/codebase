# to use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# to flatten
x = x.view(x.size()[0], -1)