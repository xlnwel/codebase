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
# filename usually ends with .p, .pkl, .pickle, but any other's are fine since pickle has defines its own serialization format
# pickle is efficient but restricted: other language cannot reuse pickled file and pickle, so do different python versions 
with open(filename, 'wb') as file_desc:
    pickle.dump((features, labels), file_desc)
with open(filename, 'rb') as file_desc:
    features, labels = pickle.load(file_desc)


# to unzip file
def _unzip(save_path, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)