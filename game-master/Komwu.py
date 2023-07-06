import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, generateOutcome, exploitability, RegretSolverPlus, generateB
import time
from scipy.special import logsumexp


class KOMWU:
    def __init__(self, game, Type="regretmatching", thres=0.0):
        print("initializing solver")
        self.thres = thres
        self.time = 0

        self.game = game

        self.Type = Type
        Solver = None
        if Type == "regretmatching":
            Solver = RegretSolver
        else:
            solver = RegretSolverPlus

        self.cfvCache = []
        self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))
        self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))

        # print("HEREE")
        # print(game.numIsets[0], game.numIsets[1])
        # for x in range(game.numHists):
        # 	print(game.nactsOnHist[x], "  player: ", game.playerOfHist[x])

        self.probNotUpdated = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.probNotPassed = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.histflag = [-1 * np.ones(game.numHists), -1 * np.ones(game.numHists)]
        self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]

        self.reachp = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]

        self.solvers = []
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[0][x]), range(game.numIsets[0]))))
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[1][x]), range(game.numIsets[1]))))

        """
        """
        self.stgy = [[], []]
        self.total_seqs = [-1,-1]
        self.new_seqs = [[], []]

        for i, iset in enumerate(range(game.numIsets[0])):
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                self.stgy[0].append(np.ones(nact) / nact)

                seqs0 = []
                for ii in range(1, nact + 1):
                    seqs0.append(self.total_seqs[0] + ii)
                self.total_seqs[0] += nact
                self.new_seqs[0].append(seqs0)

            else:
                self.stgy[0].append(np.ones(0))
                self.new_seqs[0].append([])
                # self.total_seqs[0] += 1

        for i, iset in enumerate(range(game.numIsets[1])):
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                self.stgy[1].append(np.ones(nact) / nact)

                seqs1 = []
                for ii in range(1, nact + 1):
                    seqs1.append(self.total_seqs[1] + ii)
                self.total_seqs[1] += nact
                self.new_seqs[1].append(seqs1)

            else:
                self.stgy[1].append(np.ones(0))
                self.new_seqs[1].append([])
                # self.total_seqs[1] += 1

        self.round = -1
        self.nodestouched = 0
        self.AMMO = 50
        # self.outcome, self.reward = generateOutcome(game, self.stgy)
        self.b = [np.zeros(self.AMMO), np.zeros(self.AMMO)] #[0 for _ in range(12)]
        # self.b1 = np.zeros(12) #[0 for _ in range(12)]

        self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_gradient = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        print("STGY")
        # THESE ARE 31, 29, USE AS ABOVE?
        print(self.stgy[0])
        print(self.new_seqs[0])
        print(self.stgy[0], self.stgy[1])

        print(len(self.stgy[0]), len(self.stgy[1]))
        print("INFOSETS")
        print(self.game.infoSets)
        print(self.new_seqs[1])

        self.sumstgy = [[], []]
        for i, iset in enumerate(range(game.numIsets[0])):
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                self.sumstgy[0].append(np.ones(nact) / nact)
            else:
                self.sumstgy[0].append(np.ones(0))
        for i, iset in enumerate(range(game.numIsets[1])):
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                self.sumstgy[1].append(np.ones(nact) / nact)
            else:
                self.sumstgy[1].append(np.ones(0))

    def updateAll(self):
        t1 = time.time()
        # game = self.game
        self.round += 1

        self.grad[0] = np.array(generateB(self.game, self.stgy, 0))
        # print("B0000")
        # print(self.grad[0])

        self.grad[1] = np.array(generateB(self.game, self.stgy, 1))
        # print("B1111")
        # print(self.grad[1])

        self.updateKomwu(0)
        self.updateKomwu(1)


    def updateKomwu(self, player):
        eta = 1.0
        optimistic_gradient = 2.0 * self.grad[player] - 1.0 * self.last_gradient[player]
        self.last_gradient[player] = self.grad[player].copy()
        self.b[player] += eta * optimistic_gradient

        K_j = [None] * 50 # This should be num infosets (the actual total not the collected ones) ? Not sure, look into this
        # This starts at the bottom and works upwards
        print("FIRST PASS: ", self.game.infoSets[player][::-1])
        for infoset_id in self.game.infoSets[player][::-1]:
            seq_values = []
            for seq in self.game.seqs[player][infoset_id]:
                child_values = []
                if seq in self.game.childrenInfosets[player]:
                    for child_infoset in self.game.childrenInfosets[player][seq]:
                        print("CHILD: ", child_infoset)
                        child_values.append(K_j[child_infoset])
                print(player, infoset_id, seq, child_values)
                seq_value = self.b[player][seq] + sum(child_values)
                seq_values.append(seq_value)
            K_j[infoset_id] = logsumexp(seq_values)

        # print(K_j)


        # print("SECOND PASS: ", self.game.infoSets[player])
        # This starts at the top and works downwards
        y = np.zeros(self.total_seqs[player] + 1)
        for infoset_id in self.game.infoSets[player]:
            for sequence_id in self.new_seqs[player][infoset_id]:
                if sequence_id in self.game.parSeq[player]:
                    y_par = 0.0 #y[self.game.parSeq[player][sequence_id]] THIS GIVES BEHAVE POLICY BY COMMENTING OUT
                else:
                    y_par = 0.0
                y[sequence_id] = y_par + self.b[player][sequence_id]
                if sequence_id in self.game.childrenInfosets[player]:
                    child_sum = 0.0
                    for child in self.game.childrenInfosets[player][sequence_id]:
                        child_sum += K_j[child]
                    y[sequence_id] += child_sum
                y[sequence_id] -= K_j[infoset_id]

        # print(np.exp(y))
        self.exp_y = np.exp(y)

        for s in range(len(self.stgy[player])):
            x = self.new_seqs[player][s]
            if len(x) > 0:
                denom = 0.0
                for i in x:
                    denom += self.exp_y[i]
                for i, ii in enumerate(x):
                    self.stgy[player][s][i] = self.exp_y[ii] / denom

        # print("FINAL STGY")
        # print(self.stgy[player])

                # y[sequence_id] = y_par + self.b[player][sequence_id] + sum(
                #                 [K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
                #                              - K_j[infoset.infoset_id]

        # y = np.zeros(self.tpx.n_sequences)
        # for infoset in self.tpx.infosets[::-1]:
        #     for sequence_id in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
        #         # Proposition 5.3 in logarithmic form
        #         if first_step:
        #             y[sequence_id] = y[infoset.parent_sequence_id] \
        #                              + self.b_xi[sequence_id] + sum(
        #                 [K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
        #                              - K_j[infoset.infoset_id]
        #         else:
        #             y[sequence_id] = y[infoset.parent_sequence_id] \
        #                              + self.b[sequence_id] + sum(
        #                 [K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
        #                              - K_j[infoset.infoset_id]

    def update(self, game):

        K_j = [None] * self.tpx.n_infosets

        # for infoset_id in range(len(self.tpx.infosets)):
        #     infoset = self.tpx.infosets[infoset_id]
        #     seq_values = []
        #     for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
        #         child_values = []
        #         for child_infoset in self.tpx.children[seq]:
        #             child_values.append(K_j[child_infoset.infoset_id])
        #         if first_step:
        #             seq_value = self.b_xi[seq] + sum(child_values)
        #         else:
        #             seq_value = self.b[seq] + sum(child_values)
        #         seq_values.append(seq_value)
        #     K_j[infoset_id] = logsumexp(seq_values)

        # y = np.zeros(self.tpx.n_sequences)
        # for infoset in self.tpx.infosets[::-1]:
        #     for sequence_id in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
        #         # Proposition 5.3 in logarithmic form
        #         if first_step:
        #             y[sequence_id] = y[infoset.parent_sequence_id] \
        #                              + self.b_xi[sequence_id] + sum(
        #                 [K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
        #                              - K_j[infoset.infoset_id]
        #         else:
        #             y[sequence_id] = y[infoset.parent_sequence_id] \
        #                              + self.b[sequence_id] + sum(
        #                 [K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
        #                              - K_j[infoset.infoset_id]
        #
        # if first_step:
        #     self.x_xi = np.exp(y)
        # else:
        #     self.x = np.exp(y)

    def getAvgStgy(self, owner, iset):
        game = self.game
        player = game.playerOfIset[owner][iset]
        if player == owner:
            # THIS SHOULD BE = not +=
            self.sumstgy[owner][iset] += self.stgy[player][iset]         #self.reachp[owner][iset] * self.solvers[owner][iset].curstgy
        # print(enumerate(game.isetSucc[owner][iset]))
        for _i, niset in enumerate(game.isetSucc[owner][iset]):
            # if player == owner:
            #     self.reachp[owner][niset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy[_i]
            # else:
            #     self.reachp[owner][niset] += self.reachp[owner][iset]
            self.getAvgStgy(owner, niset)
        # self.reachp[owner][iset] = 0

    def getExploitability(self):
        stgy_prof = []

        def avg(_x):
            s = np.sum(_x)
            l = _x.shape[0]
            if s < 1e-5:
                return np.ones(l) / l
            return _x / s

        self.getAvgStgy(0, 0)
        self.getAvgStgy(1, 0)
        stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[0])))
        stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[1])))
        self.mike = stgy_prof
        return exploitability(self.game, stgy_prof)
