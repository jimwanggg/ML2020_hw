from config import configurations
from utils import test_process
import sys


# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置
if __name__ == '__main__':
    config = configurations()
    config.load_model = True
    config.data_path = sys.argv[1]
    print ('config:\n', vars(config))
    test_loss, bleu_score = test_process(config)
    print (f'test loss: {test_loss}, bleu_score: {bleu_score}')

