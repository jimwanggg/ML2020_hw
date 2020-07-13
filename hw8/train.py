from config import configurations
from utils import train_process
import sys

if __name__ == '__main__':
    config = configurations()
    config.data_path = sys.argv[1]
    config.load_model = False
    print ('config:\n', vars(config))
    train_losses, val_losses, bleu_scores = train_process(config)