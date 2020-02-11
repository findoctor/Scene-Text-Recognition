# Some cnofiguration and parameter-setting

lmdb_train_path = "Data/lmdb"  # where the lmdb file is stored
imgH = 32
imgW = 100
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz-'
batch_size = 100
n_epoch = 20
n_hidden = 256   # lstm hidden state
n_class = len(alphabet)
lr = 1e-4           # learning rate 
beta1= 0.5

n_channel = 1   # c=1 in this dataset IIITK5

def load_data(v, img):
    v.data.resize_(img.size()).copy_(img)