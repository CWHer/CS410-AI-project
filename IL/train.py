import copy
import json
import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch
from icecream import ic
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from agent.network import Imitator
from config import TRAIN_CONFIG, saveSettings
from train_utils.data_set import DataSet
from utils import printInfo, printWarning, timeLog


class Trainer():
    def __init__(self, data_set) -> None:
        self.net = Imitator()
        self.data_set = data_set

    @timeLog
    def evaluate(self):
        ic("evaluate model")
        # self.net.setDevice(torch.device("cpu"))
        # self.best_net.setDevice(torch.device("cpu"))

        results = {0: 0, 1: 0, -1: 0}
        with tqdm(total=TRAIN_CONFIG.num_contest) as pbar:
            # parallel
            # games = [
            #     (self.net, self.best_net, np.random.randint(2 ** 30))
            #     for _ in range(TRAIN_CONFIG.num_contest // 2)]
            # with Pool(TRAIN_CONFIG.process_num) as pool:
            #     winners = pool.starmap(contest, games)
            #     for winner in winners:
            #         results[winner] += 1
            #         pbar.update()

            # games = [
            #     (self.best_net, self.net, np.random.randint(2 ** 30))
            #     for _ in range(TRAIN_CONFIG.num_contest // 2)]
            # with Pool(TRAIN_CONFIG.process_num) as pool:
            #     winners = pool.starmap(contest, games)
            #     for winner in winners:
            #         winner = winner ^ 1 if winner != -1 else winner
            #         results[winner] += 1
            #         pbar.update()

            # serial
            for _ in range(TRAIN_CONFIG.num_contest // 2):
                winner = contest(
                    self.net, self.best_net,
                    np.random.randint(2 ** 30))
                results[winner] += 1
                pbar.update()

            for _ in range(TRAIN_CONFIG.num_contest // 2):
                winner = contest(
                    self.best_net, self.net,
                    np.random.randint(2 ** 30))
                winner = winner ^ 1 if winner != -1 else winner
                results[winner] += 1
                pbar.update()

        message = "result: {} win, {} lose, {} draw".format(
            results[0], results[1], results[-1])
        ic(message)
        return (results[0] + 0.5 * results[-1]) / TRAIN_CONFIG.num_contest

    def train(self, data_iter, is_train=True):
        """[summary]
        train one epoch
        """
        total_num = 0
        mean_loss, mean_acc = 0, 0
        with tqdm(total=len(data_iter)) as pbar:
            for data_batch in data_iter:
                loss, acc = self.net.trainStep(data_batch, is_train)
                batch_num = data_batch[-1].shape[0]
                total_num += batch_num
                mean_loss += loss * batch_num
                mean_acc += acc * batch_num
                pbar.update()

        return mean_loss / total_num, mean_acc / total_num

    def run(self):
        """[summary]
        pipeline: collect data, train, evaluate, update and repeat
        """
        best_acc, epoch_cnt = 0, 0
        train_iter, valid_iter = self.data_set.getIter()
        lr_scheduler = ExponentialLR(self.net.optimizer, 0.2)

        for i in range(TRAIN_CONFIG.train_epochs):
            # >>>>> train
            loss, acc = self.train(train_iter)
            printInfo(
                "train loss: {:>4f}, train acc: {:>4f}".format(loss, acc))

            # >>>>> validate
            loss, acc = self.train(
                valid_iter, is_train=False)
            printInfo(
                "valid loss: {:>4f}, valid acc: {:>4f}".format(loss, acc))
            printInfo(f"epoch {i} finish.")
            
            # >>>>> check
            if acc > best_acc:
                epoch_cnt = 0
                best_acc = acc
                self.net.save(version="best")
            else:
                epoch_cnt += 1
                if epoch_cnt % TRAIN_CONFIG.patience == 0:
                    lr_scheduler.step()
                    ic(lr_scheduler.get_last_lr())
                if epoch_cnt % TRAIN_CONFIG.early_stop == 0:
                    printWarning(True, "early stop")
                    break

            # >>>>> evaluate
            # if i % TRAIN_CONFIG.check_freq == 0:
            #     win_rate = self.evaluate()
            #     if win_rate >= TRAIN_CONFIG.update_threshold:
            #         self.cnt += 1
            #         self.best_net = copy.deepcopy(self.net)
            #         self.best_net.save(version=self.cnt)
            #         message = "new best model {}!".format(self.cnt)
            #         ic(message)
            #     else:
            #         ic("reject new model.")


if __name__ == "__main__":
    # NOTE: multiprocessing with CUDA is available on Linux
    # multiprocessing.set_start_method("spawn")

    saveSettings()

    data_set = DataSet()
    data_set.load("../dataset/data_Top1.pkl")
    trainer = Trainer(data_set)
    trainer.run()
