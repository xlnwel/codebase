# to show progress bar for urlretrieve using tqdm
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download(url, file):
    if not isfile(file):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                url,
                file,
                pbar.hook)


# to save & load data
pickle.dump((features, labels), open(filename, 'wb'))
features, labels = pickle.load(open(filename, mode='rb'))