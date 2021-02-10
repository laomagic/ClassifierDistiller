import argparse
import config
import os
import torch
from torch.utils.data import DataLoader
from data_processor import DataPrecessForSentence
# from create_dataset import load_data, DataPrecessForSentence
from utils import train, validate, test
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from utils import create_logger, compute_loss
from match_model import BaseModel, DgcnnModel
from swa import SWA


root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger = create_logger("logs/dgcnn.log")
# 随机数固定
seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(args,
         train_data,
         dev_data,
         test_data,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading dataloader")
    bert_tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_dataset = DataPrecessForSentence(bert_tokenizer, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = DataPrecessForSentence(bert_tokenizer, dev_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float('inf')
    improve = 0
    logger.info("loading model")
    bert = BertModel.from_pretrained(args.model_path)
    model = BaseModel(bert, config)
    # model = DgcnnModel(bert, config)
    if checkpoint:
        model.load_state_dict(torch.load(args.save_model_path))
    logger.info("loading optimizer")
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.swa:
        optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=args.lr)
    for epoch in range(args.epochs):
        train_epoch_time, train_epoch_loss, train_epoch_accuracy = \
            train(args, model, train_dataloader, optimizer, epoch, max_grad_norm, logger, device)
        logger.info('Training epoch:{}  Time:{:.2f} loss:{:.4f} accuracy:{:.4f}'.format
                    (epoch, train_epoch_time, train_epoch_loss, train_epoch_accuracy))

        valid_epoch_time, valid_epoch_loss, valid_accuracy_score, valid_auc_score = validate(model, dev_dataloader,
                                                                                             device)
        logger.info('Validating epoch:{} Time:{:.2f} loss:{:.4f}  accuracy:{:.4f}  auc:{:.4f}'.format
                    (epoch, valid_epoch_time, valid_epoch_loss, valid_accuracy_score, valid_auc_score))

        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            logger.info('saving model epoch:{}'.format(epoch))
            if not os.path.exists(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.state_dict(), args.save_model_path + "/roberta_model.pt")
            improve = 0
        else:
            improve += 1
            if improve > patience:
                logger.info("early stopping")
                break
    logger.info('=' * 70)
    if args.test_data:
        logger.info('testing dataset')
        test_dataset = DataPrecessForSentence(bert_tokenizer, test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        _, test_total_time, test_acc, test_auc_score = test(model, test_dataloader, device)
        logger.info('Testing: Time:{:.2f} accuracy:{:.4f}  auc:{:.4f}'.format
                    (test_total_time, test_acc, test_auc_score))


if __name__ == "__main__":
    # train_data = load_data("train")
    # train_data, dev_data = train_test_split(train_data, test_size=0.2, shuffle=True, random_state=2020)
    # test_data = load_data("test")
    parser = argparse.ArgumentParser(description="semantic match model training")
    parser.add_argument('--train_data', dest="train_data", action="store", default=True, help="")
    parser.add_argument('--eval_data', dest="eval_data", action="store", default=True, help="")
    parser.add_argument('--test_data', dest="test_data", action="store", default=False, help="")
    parser.add_argument('--run_mode', dest="run_mode", action="store", default="train",
                        help="Running mode: train or eval")
    parser.add_argument("--swa", dest="swa", action="store", type=bool, default=False, help="")
    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=64, help="")
    parser.add_argument("--epochs", dest="epochs", action="store", type=int, default=20, help="")
    parser.add_argument("--lr", dest="lr", action="store", type=float, default=0.00002, help="")
    parser.add_argument("--save_model_path", dest="save_model_path", action="store", default="save_dir",
                        help="")
    parser.add_argument("--model_path", dest="model_path", action="store", default="model/roberta/",
                        help="pretrained model")
    parser.add_argument("--log_path", dest="log_path", action="store", default="logs", help="")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", action="store",
                        type=int, default=4, help="")
    args = parser.parse_args()
    main(args, config.train_path, config.dev_path, None)
