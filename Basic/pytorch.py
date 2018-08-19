# to use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# to flatten
x = x.view(x.size()[0], -1)

# to save & load parameters in a neural net
torch.save(module.state_dict(), 'checkpoint.pth')
module.load_state_dict(torch.load('checkpoint.pth'))