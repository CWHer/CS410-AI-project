import copy
import json
import multiprocessing
from itertools import chain
from multiprocessing import Pool

import numpy as np
import torch
from icecream import ic
from tqdm import tqdm

from agent.network import D3QN
from config import TRAIN_CONFIG, saveSettings
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer
from utils import timeLog


class Trainer():
    def __init__(self) -> None:
        self.net = D3QN()
        self.cnt = 0  # version of best net
        self.best_net = D3QN()
        self.replay_buffer = ReplayBuffer()
        self.epsilon = TRAIN_CONFIG.init_epsilon

    @timeLog
    def collectData(self):
        ic("collect data")
        # self.net.setDevice(torch.device("cpu"))

        # parallel
        # games = [
        #     (self.net, self.epsilon, np.random.randint(2 ** 30))
        #     for _ in range(TRAIN_CONFIG.game_num)]
        # with tqdm(total=TRAIN_CONFIG.game_num) as pbar:
        #     with Pool(TRAIN_CONFIG.process_num) as pool:
        #         episode_data = pool.starmap(selfPlay, games)
        #         for data in episode_data:
        #             self.replay_buffer.add(data)
        #             pbar.update()

        # serial
        for _ in tqdm(range(TRAIN_CONFIG.game_num)):
            episode_data = selfPlay(
                self.net, self.epsilon,
                np.random.randint(2 ** 30))
            self.replay_buffer.add(episode_data)

        self.epsilon -= TRAIN_CONFIG.delta_epsilon
        self.epsilon = max(self.epsilon, TRAIN_CONFIG.min_epsilon)

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

    def train(self):
        ic("train model")
        # self.net.setDevice(torch.device("cuda:0"))

        total_num, mean_loss = 0, 0
        for _ in tqdm(range(TRAIN_CONFIG.train_epochs)):
            data_batch = self.replay_buffer.sample()
            loss = self.net.trainStep(data_batch)
            total_num += data_batch[-1].shape[0]
            mean_loss += loss * data_batch[-1].shape[0]
        mean_loss /= total_num
        ic(mean_loss)

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
                self.replay_buffer.save(version=i)

            # >>>>> train
            if self.replay_buffer.enough():
                self.train()

            # >>>>> evaluate
            if i % TRAIN_CONFIG.check_freq == 0:
                win_rate = self.evaluate()
                if win_rate >= TRAIN_CONFIG.update_threshold:
                    self.cnt += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=self.cnt)
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
