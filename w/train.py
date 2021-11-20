import copy
import json
import multiprocessing
from itertools import chain
from multiprocessing import Pool

import numpy as np
import torch
from icecream import ic
from tqdm import tqdm

from agent.network import PolicyValueNet
from config import TRAIN_CONFIG, saveSettings
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer
from utils import timeLog


class Trainer():
    def __init__(self) -> None:
        self.net = PolicyValueNet()
        self.cnt = 0  # version of best net
        self.best_net = copy.deepcopy(self.net)
        self.replay_buffer = ReplayBuffer()

    @timeLog
    def collectData(self):
        ic("collect data")
        # self.net.setDevice(torch.device("cpu"))

        games = [
            (self.net, np.random.randint(2 ** 30))
            for _ in range(TRAIN_CONFIG.game_num)]
        with tqdm(total=TRAIN_CONFIG.game_num) as pbar:
            with Pool(TRAIN_CONFIG.process_num) as pool:
                results = pool.starmap(selfPlay, games)
                for result in results:
                    self.replay_buffer.add(*result)
                    pbar.update()

    @timeLog
    def evaluate(self):
        ic("evaluate model")
        # self.net.setDevice(torch.device("cpu"))
        # self.best_net.setDevice(torch.device("cpu"))

        results = {0: 0, 1: 0, -1: 0}
        with tqdm(total=TRAIN_CONFIG.num_contest) as pbar:
            games = [
                (self.net, self.best_net, np.random.randint(2 ** 30))
                for _ in range(TRAIN_CONFIG.num_contest // 2)]
            with Pool(TRAIN_CONFIG.process_num) as pool:
                winners = pool.starmap(contest, games)
                for winner in winners:
                    results[winner] += 1
                    pbar.update()

            games = [
                (self.best_net, self.net, np.random.randint(2 ** 30))
                for _ in range(TRAIN_CONFIG.num_contest // 2)]
            with Pool(TRAIN_CONFIG.process_num) as pool:
                winners = pool.starmap(contest, games)
                for winner in winners:
                    winner = winner ^ 1 if winner != -1 else winner
                    results[winner] += 1
                    pbar.update()

        message = "result: {} win, {} lose, {} draw".format(
            results[0], results[1], results[-1])
        ic(message)
        return (results[0] + 0.5 * results[-1]) / TRAIN_CONFIG.num_contest

    def train(self):
        ic("train model")
        # self.net.setDevice(torch.device("cuda:0"))

        train_iter = self.replay_buffer.trainIter()
        iter_len = len(train_iter[0]) + len(train_iter[1])
        for i in range(1, TRAIN_CONFIG.train_epochs + 1):
            losses, mean_loss, mean_acc = [], 0, 0
            with tqdm(total=iter_len) as pbar:
                for data_batch in chain(*train_iter):
                    loss, acc = self.net.trainStep(data_batch)
                    losses.append(loss)
                    mean_loss += loss * data_batch[-1].shape[0]
                    mean_acc += acc * data_batch[-1].shape[0]
                    pbar.update()
            print("epoch {} finish".format(i))
            mean_loss /= self.replay_buffer.size()
            mean_acc /= self.replay_buffer.size()
            ic(mean_loss, mean_acc)

    def run(self):
        """[summary]
        pipeline: collect data, train, evaluate, update and repeat
        """
        for i in range(1, TRAIN_CONFIG.train_num + 1):
            # >>>>> collect data
            self.collectData()
            print("Round {} finish, buffer size {}".format(
                i, self.replay_buffer.size()))
            # save data
            if i % TRAIN_CONFIG.data_save_freq == 0:
                self.replay_buffer.save(version=str(i))

            # >>>>> train
            if self.replay_buffer.enough():
                self.train()

            # >>>>> evaluate
            if i % TRAIN_CONFIG.check_freq == 0:
                win_rate = self.evaluate()
                if win_rate >= TRAIN_CONFIG.update_threshold:
                    self.cnt += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=str(self.cnt))
                    message = "new best model {}!".format(self.cnt)
                    ic(message)
                else:
                    ic("reject new model.")


if __name__ == "__main__":
    # NOTE: multiprocessing with CUDA is available on Linux
    multiprocessing.set_start_method("spawn")

    saveSettings()

    trainer = Trainer()
    trainer.run()
