import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, generateOutcome, exploitability, RegretSolverPlus, generateB, updateB, generateChildren
import time
from scipy.special import logsumexp
import numexpr as ne



class LazyCFR:
    def __init__(self, game, Type="regretmatching", thres=0.0, params=None):
        print("initializing solver")

        self.thres = params["thres"]
        self.entropy = params["entropy"]
        self.KL = params["KL"]
        self.opt = [params["optimism"], params["optimism"]]
        self.last_opt = [params["optimism"] - 1.0, params["optimism"] - 1.0]
        self.eta = params["eta"]
        # self.entropy_twice = params["entropy_twice"]
        # self.mod_value = params["mod_value"]
        self.AMMO = params["AMMO"]
        self.KL_mod = params["KL_mod"]
        self.new_b = params["new_b"]
        self.b_count_check = params["b_count"]
        self.b_count_count_at = params["b_count_count_at"]


        self.RESTART = False

        self.time = 0

        self.game = game

        self.Type = Type
        Solver = None
        if Type == "regretmatching":
            Solver = RegretSolver
        else:
            solver = RegretSolverPlus


        # WONT NEED THIS
        self.cfvCache = []
        self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))
        self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))


        self.probNotUpdated = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.probNotPassed = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.histflag = [-1 * np.ones(game.numHists), -1 * np.ones(game.numHists)]
        self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]

        # self.tmp_opt = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]
        self.tmp_opt = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]

        self.reachp = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]

        self.b = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.b_store = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.b_tally = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.b_ten = [[],[]]
        self.b_count = 1
        self.last_entropy = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        # self.b = [np.ones(self.AMMO) * 5.0, np.ones(self.AMMO) * 5.0]

        self.seqCount = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.seqCountFlag = [np.zeros(self.AMMO), np.zeros(self.AMMO)]


        self.solvers = []
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[0][x], None), range(game.numIsets[0]))))
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[1][x], None), range(game.numIsets[1]))))

        self.stgy = [[], []]

        self.stgy = [[], []]
        self.last_stgy = [[], []]
        self.total_seqs = [-1, -1]
        self.new_seqs = [[], []]

        for i, iset in enumerate(range(game.numIsets[0])):
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                self.stgy[0].append(np.ones(nact) / nact)
                self.last_stgy[0].append(np.ones(nact) / nact)

                seqs0 = []
                for ii in range(1, nact + 1):
                    seqs0.append(self.total_seqs[0] + ii)
                self.total_seqs[0] += nact
                self.new_seqs[0].append(seqs0)

            else:
                self.stgy[0].append(np.ones(0))
                self.last_stgy[0].append(np.ones(0))
                self.new_seqs[0].append([])

        for i, iset in enumerate(range(game.numIsets[1])):
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                self.stgy[1].append(np.ones(nact) / nact)
                self.last_stgy[1].append(np.ones(nact) / nact)

                seqs1 = []
                for ii in range(1, nact + 1):
                    seqs1.append(self.total_seqs[1] + ii)
                self.total_seqs[1] += nact
                self.new_seqs[1].append(seqs1)

            else:
                self.stgy[1].append(np.ones(0))
                self.last_stgy[1].append(np.ones(0))
                self.new_seqs[1].append([])


        self.total_entropy = [[], []]

        self.round = -1
        self.nodestouched = 0
        self.outcome, self.reward = generateOutcome(game, self.stgy)

        # WAS 5.9m with regular


        self.total = 2
        self.current = 0

        self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.grad_flag = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_gradient = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_last_gradient = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.seqs_updated = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.K_j = [[0.0] * self.AMMO, [0.0] * self.AMMO]
        self.visited = [[],[]]
        self.last_visited = [[],[]]
        self.exp_y = [np.zeros(self.total_seqs[0] + 1),np.zeros(self.total_seqs[1] + 1)]

        self.regret = [np.zeros(self.total_seqs[0] + 1), np.zeros(self.total_seqs[1] + 1)]

        self.opt_levels = []

        self.p = 1

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


        b0 = generateChildren(game, self.stgy, 0)
        # print("CHILDREN0:", b0)
        b1 = generateChildren(game, self.stgy, 1)
        # print("CHILDREN1:", b1)
        self.game.childrenInfosets[0] = b0
        self.game.childrenInfosets[1] = b1

        self.last_stgy = [None, None]

    def receiveProb(self, owner, histind, prob, amount=1.0):
        self.probNotPassed[owner][histind] += prob * amount
        self.probNotUpdated[owner][histind] += prob * amount
        self.nodestouched += 1

    def passProbOnHist(self, owner, histind, last_seqs, true_owner):
        game = self.game
        if game.isTerminal[histind]:
            return

        parhind, pactind = game.histPar[histind]
        # THIS DOESN"T SEEM TO GET CALLED EVER
        if parhind != -1 and self.histflag[owner][parhind] < self.round:
            print("Recursive call with owner=", owner, " for parhind: ", parhind)
            self.passProbOnHist(owner, parhind)

        player = game.playerOfHist[histind]
        stgy = None
        if player == 2:
            stgy = game.chanceprob[histind]
            isetind = None
        else:
            isetind = game.Hist2Iset[player][histind]
            stgy = self.stgy[player][isetind]

        for aind, nxthind in enumerate(game.histSucc[histind]):
            tmp = self.probNotPassed[owner][histind].copy()
            if player == owner:
                tmp[owner] *= stgy[aind]
            else:
                tmp[1 - owner] *= stgy[aind]
            # outcome looks like this: [ 0.125 -0.875 -0.875] and tmp[1-owner] like this: 0.3333
            self.cfvCache[owner][nxthind] += np.array(self.outcome[nxthind]) * tmp[1 - owner]

            if game.isTerminal[nxthind]:  # and true_owner == owner:
                i1 = game.Hist2Iset[owner][histind]  # original
                i2 = game.Hist2Iset[owner][nxthind]
                sign = 1.0
                if owner == 1:
                    sign = -1.0
                if i2 in self.game.isetSuccSeq[owner][i1]:
                    # game.reward[hist]
                    self.grad[owner][self.game.isetSuccSeq[owner][i1][i2]] += tmp[1 - owner] * self.reward[nxthind] * sign
                    # self.grad[owner][self.game.isetSuccSeq[owner][i1][i2]] += tmp[1 - owner] * self.game.reward[nxthind][owner] #* sign

            self.receiveProb(owner, nxthind, tmp, 1.0)
        self.probNotPassed[owner][histind] = np.zeros(2)
        self.histflag[owner][histind] = self.round


    def updateIset(self, owner, isetind, last_seqs):

        self.isetflag[owner][isetind] = self.round
        game = self.game
        if game.playerOfIset[owner][isetind] == owner:
            # TODO Won't need any of this here
            sumcfv = np.zeros(game.nactsOnIset[owner][isetind])
            weight = 0
            for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
                sumcfv += self.cfvCache[owner][hind]
                weight = self.probNotUpdated[owner][hind][owner]
            #     self.cfvCache[owner][hind] *= 0.0
            if owner == 1:
                sumcfv *= -1.0

            # TODO Won't need this here
            self.sumstgy[owner][isetind] += self.reachp[owner][isetind] * self.stgy[owner][isetind]# THIS IS FOR LAZY CFRself.solvers[owner][isetind].curstgy
            # TODO Won't need this here
            self.solvers[owner][isetind].receive(sumcfv, weight=weight)
            self.visited[owner].append(isetind)


        for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
            self.probNotUpdated[owner][hind] = np.zeros(2)
            # These will pass cfv down to next successor and give them probabilities
            self.passProbOnHist(0, hind, last_seqs, true_owner=owner)
            self.passProbOnHist(1, hind, last_seqs, true_owner=owner)

        nacts = game.nactsOnIset[owner][isetind]

        for innerIind, nxtisetind in enumerate(game.isetSucc[owner][isetind]):
            sumprob = 0.0
            nxthists = game.Iset2Hists[owner][nxtisetind]
            if game.isTerminal[nxthists[0]] == False:
                for nxth in game.Iset2Hists[owner][nxtisetind]:
                    sumprob += self.probNotUpdated[owner][nxth][1 - owner]
                if game.playerOfIset[owner][isetind] == owner:
                    self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * self.stgy[owner][isetind][innerIind] #self.solvers[owner][isetind].curstgy[innerIind]
                else:
                    self.reachp[owner][nxtisetind] += self.reachp[owner][isetind]
                if sumprob > self.thres:
                    owner_of_iset = game.playerOfIset[owner][nxtisetind]
                    if owner_of_iset != 2 and owner == owner_of_iset:
                        cur_seqs = game.seqs[owner_of_iset][nxtisetind].copy()
                        next_seqs = last_seqs.copy()
                        # Might be more actions - check this again to make a decent comment
                        if len(cur_seqs) < len(next_seqs):
                            for _ in range(len(next_seqs) - len(cur_seqs) + 1):
                                cur_seqs.append(0)
                        next_seqs[owner_of_iset] = cur_seqs[innerIind]
                    else:
                        next_seqs = last_seqs.copy()
                    self.updateIset(owner, nxtisetind, next_seqs)
                else:
                    dummy = 0

        self.reachp[owner][isetind] = 0

    def updateAll(self, curexpl, t):

        if t == 2:
            print("Setting initial curexpl:", curexpl)
            self.curexpl = curexpl

        # if curexpl < 0.9 * self.curexpl: # and len(self.b_ten[0]) >= 10:
        #     print("RESTART: ", "CURRENT: ", curexpl, " PREV: ", self.curexpl)
        #     self.curexpl = curexpl
        #     self.RESTART = True

        t1 = time.time()
        game = self.game
        self.round += 1
        ff = [2, 12, 24, 36, 48, 60, 72]
        ff = [3]
        for i in range(2, 100):
            ff.append(i * 3)
        # print(ff)
        if True: #t in ff:
            if self.Type == "regretmatching":
                self.reachp[0][0] += 1  # These are set to zero in line just above so only ever have values 0 or 1
                self.reachp[1][0] += 1
            else:
                # print("HERE")
                self.reachp[0][0] += self.round
                self.reachp[1][0] += self.round
            self.receiveProb(0, 0, np.ones(2))  # These set probNotPass and probNotUpdate for history
            self.receiveProb(1, 0, np.ones(2))  # to all 1 as in: probNotPassed before:  [0. 0.] prob not passed after:  [1. 1.]

            self.updateIset(0, 0, [[], [], [], [], [], [], [], []])
            self.updateIset(1, 0, [[], [], [], [], [], [], [], []])

        self.getAveragePolicy() 

        # grad0 = generateB(game,self.stgy_prof, 0)
        # grad1 = generateB(game,self.stgy_prof, 1)
        # self.grad = [np.array(grad0), np.array(grad1)]

        # mod = self.round // self.mod_value
        # ent = self.entropy / (mod + 1.0) #np.log(self.round + 2.0) # (mod + 1.0)
        # ent_twice = 2.0 if self.entropy_twice else 1.0
        # total_entropy0 = 0.0
        # total_entropy1 = 0.0
        #
        # entropy = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        # for infoset_id in self.visited[0][::-1]:
        #     vals = []
        #     sum_vals = 0.0
        #     for i, seq in enumerate(self.game.seqs[0][infoset_id]):
        #         pol = self.stgy[0][infoset_id][i]
        #         if pol <= 0:
        #             pol = 1e-32
        #         sum_vals += pol * np.log(pol)
        #         total_entropy0 += pol * np.log(pol)
        #         vals.append(0)#ent_twice * np.log(pol))
        #     for i, seq in enumerate(self.game.seqs[0][infoset_id]):
        #         self.b[0][seq] += ent*(vals[i] - sum_vals)
        #         # entropy[0][seq] += ent*(vals[i] - sum_vals)
        #         entropy[0][seq] += ent*-sum_vals
        #         # if t > 20:
        #         #     print("Iteration:", t, "seq:", seq, " Value:", ent*(vals[i] - sum_vals))
        #
        #
        # for infoset_id in self.visited[1][::-1]:
        #     vals = []
        #     sum_vals = 0.0
        #     for i, seq in enumerate(self.game.seqs[1][infoset_id]):
        #         pol = self.stgy[1][infoset_id][i]
        #         if pol <= 0:
        #             pol = 0.00000000000000000000000000000001
        #         sum_vals += pol * np.log(pol)
        #         total_entropy1 += pol * np.log(pol)
        #         vals.append(0)#ent_twice * np.log(pol))
        #     for i, seq in enumerate(self.game.seqs[1][infoset_id]):
        #         self.b[1][seq] += ent*(vals[i] - sum_vals) # Always negative
        #         entropy[1][seq] += ent * (vals[i] - sum_vals)

        # self.b[0] -= self.last_entropy[0]
        # self.b[1] -= self.last_entropy[1]
        # self.last_entropy[0] = entropy[0].copy()
        # self.last_entropy[1] = entropy[1].copy()

        # For plotting entropy of stgy
        # if len(self.total_entropy[0]) > 1:
        #     self.total_entropy[0].append(self.total_entropy[0][-1] + -total_entropy0)
        #     self.total_entropy[1].append(self.total_entropy[1][-1] + -total_entropy1)
        # else:
        #     self.total_entropy[0].append(-total_entropy0)
        #     self.total_entropy[1].append(-total_entropy1)


        # Plot strategy
        # self.total_entropy[0].append(self.stgy[0][4][0])
        # self.total_entropy[1].append(self.stgy[1][5][1])

        self.updateKomwu(0, t)
        self.updateKomwu(1, t)

        self.b_count += 1
        if self.b_count == self.b_count_check[0] and self.new_b:
            self.b = self.b_store.copy()
            # self.b[0] *= 0.95
            # self.b[1] *= 0.95

            self.b_count = 1
            if len(self.b_count_check) > 1:
                self.b_count_check.pop(0)
                self.b_count_count_at.pop(0)


        if t > 1 and t % 8 == 0:
            self.last_stgy[0] = copy.deepcopy(self.stgy[0])
            self.last_stgy[1] = copy.deepcopy(self.stgy[1])

        # for infoset_id in self.visited[0][::-1]:
        #     reg = self.solvers[0][infoset_id].cfrreg()
        #     # print(reg)
        #     for i, sequence_id in enumerate(self.new_seqs[0][infoset_id]):
        #         if reg[i] >= 0.0:
        #             # print("HERE")
        #             self.b[0][sequence_id] *= 0.95
        #
        # for infoset_id in self.visited[1][::-1]:
        #     reg = self.solvers[1][infoset_id].cfrreg()
        #     for i, sequence_id in enumerate(self.new_seqs[1][infoset_id]):
        #         if reg[i] >= 0.0:
        #             self.b[1][sequence_id] *= 0.95

        # self.last_gradient[0] = self.grad[0].copy()
        # self.last_gradient[1] = self.grad[1].copy()
        if True: #t + 1 in ff:
            self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
            self.visited = [[], []]

        # print(self.stgy[0])
        # ff = [10, 40, 70, 110, 210, 410]
        # if t in ff: #$ == 0:
        #     # self.b = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        #     print("t:", t)
        #     for i in range(len(self.sumstgy[0])):
        #         self.sumstgy[0][i] *= 0
        #     for i in range(len(self.sumstgy[1])):
        #          self.sumstgy[1][i] *= 0

        # def updateoutcome(hist):
        #     if game.isTerminal[hist]:
        #         return
        #     if self.histflag[0][hist] < self.round:
        #         return
        #     self.reward[hist] = 0.0
        #     nacts = game.nactsOnHist[hist]
        #     _stgy = None
        #     player = game.playerOfHist[hist]
        #     if player == 2:
        #         _stgy = game.chanceprob[hist]
        #     else:
        #         piset = game.Hist2Iset[player][hist]
        #         _stgy = self.solvers[player][piset].curstgy  # self.stgy[player][piset]
        #     for a in range(nacts):
        #         nh = game.histSucc[hist][a]
        #         updateoutcome(nh)
        #         self.outcome[hist][a] = self.reward[nh]
        #         self.reward[hist] += _stgy[a] * self.reward[nh]
        #
        # updateoutcome(0)

        self.time += time.time() - t1
        return self.stgy


    def getAvgStgy(self, owner, iset):
        game = self.game
        player = game.playerOfIset[owner][iset]
        if player == owner:
            self.sumstgy[owner][iset] += 1# USED ABOVE NOW self.stgy[player][iset] * self.reachp[owner][iset]
        for _i, niset in enumerate(game.isetSucc[owner][iset]):
            if player == owner:
                self.reachp[owner][niset] += self.reachp[owner][iset] * self.stgy[owner][iset][_i] # self.solvers[owner][iset].curstgy[_i]
            else:
                self.reachp[owner][niset] += self.reachp[owner][iset]

            self.getAvgStgy(owner, niset)
        self.reachp[owner][iset] = 0

    def getExploitability(self):
        stgy_prof = []

        def avg(_x):
            s = np.sum(_x)
            l = _x.shape[0]
            if s < 1e-5:
                return np.ones(l) / l
            return _x / s

        # self.reachp[0][0] = 1
        # self.getAvgStgy(0, 0)
        # self.reachp[1][0] = 1
        # self.getAvgStgy(1, 0)
        # stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[0])))
        # stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[1])))
        stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[0])))
        stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[1])))
        self.mike = stgy_prof
        return exploitability(self.game, stgy_prof)

    def getAveragePolicy(self):
        self.stgy_prof = []
        def avg(_x):
            s = np.sum(_x)
            l = _x.shape[0]
            if s < 1e-5:
                return np.ones(l) / l
            return _x / s

        self.stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[0])))
        self.stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[1])))
        return

    def updateKomwu(self, player, t):

        # entropy = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        mod = self.round // self.KL_mod
        KL = self.KL / (mod + 1.0) #np.log(self.round + 2.0) #(mod + 1.0)
        # THIS IS THE KL CODE THAT WAS BEING USED
        # if self.last_stgy[player] != None:
        #     vals = []
        #     for infoset_id in self.visited[player][::-1]:
        #         sum_vals = 0.0
        #         max_val = 1e-100
        #         for i, seq in enumerate(self.game.seqs[player][infoset_id]):
        #             pol = self.stgy[player][infoset_id][i]
        #             old_pol = self.last_stgy[player][infoset_id][i]
        #             pol = np.clip(pol, a_min=np.finfo(float).eps, a_max=None)
        #             old_pol = np.clip(old_pol, a_min=np.finfo(float).eps, a_max=None)
        #             sum_vals += -pol * np.log(pol / old_pol)
        #             # vals.append(-pol * np.log(pol / old_pol))
        #             # max_val = max(max_val, self.b[player][seq])
        #         for i, seq in enumerate(self.game.seqs[player][infoset_id]):
        #             self.b[player][seq] += -KL * sum_vals
        #             if self.b_count >= self.b_count_count_at[0]:
        #                 self.b_store[player] += -KL * sum_vals
                    # print(-KL * sum_vals)

        if self.last_stgy[player] != None:
            vals = []
            u = 0.0
            for infoset_id in self.visited[player][::-1]:
                sum_vals = 0.0
                max_val = 1e-100
                for i, seq in enumerate(self.game.seqs[player][infoset_id]):
                    pol = self.stgy[player][infoset_id][i]
                    ref_pol = self.last_stgy[player][infoset_id][i]
                    pol = np.clip(pol, a_min=np.finfo(float).eps, a_max=None)
                    ref_pol = np.clip(ref_pol, a_min=np.finfo(float).eps, a_max=None)
                    rew = u*(ref_pol - pol) # (KL / pol) *
                    self.b[player][seq] += self.eta * rew



        # self.b[player] -= self.last_entropy[player]
        # self.last_entropy[player] = entropy[player].copy()

        optimistic_gradient = self.opt[player] * (self.grad[player] / 1.0) - self.last_opt[player] * (self.last_gradient[player] / 1.0)
        # optimistic_gradient = 1.0 * self.grad[player] #- 0.5 * self.last_gradient[player] - 0.5*self.last_last_gradient[player]
        # print(optimistic_gradient)
        # if player == 0:
        #     print(self.grad[player])
        #     print(self.last_gradient[player])
        #     print(self.last_last_gradient[player])
        #     print("")
        self.last_last_gradient[player] = self.last_gradient[player].copy()
        self.last_gradient[player] = self.grad[player].copy()

        self.b[player] += self.eta * optimistic_gradient

        # for infoset_id in self.visited[player][::-1]:
        #     for i, seq in enumerate(self.game.seqs[player][infoset_id]):
        #         self.b_tally[player][seq] += 1
        #         if self.b_tally[player][seq] >= 8:
        #             self.b_store[player][seq] += self.eta * optimistic_gradient[seq]

        if self.b_count > self.b_count_count_at[0]:
            self.b_store[player] += self.eta * optimistic_gradient

        self.last_opt[player] = self.opt[player] - 1.0
        # self.last_stgy[player] = copy.deepcopy(self.stgy[player])

        y = np.zeros(self.total_seqs[player] + 1)
        for infoset_id in self.visited[player][::-1]:
            seq_values = []
            for i, seq in enumerate(self.game.seqs[player][infoset_id]):
                child_values = []
                if seq in self.game.childrenInfosets[player]:
                    for child_infoset in self.game.childrenInfosets[player][seq]:
                        if self.K_j[player][child_infoset] == None:
                            aff = 0.0
                        else:
                            aff = self.K_j[player][child_infoset]
                        child_values.append(aff)
                seq_value = self.b[player][seq] + sum(child_values)
                seq_values.append(seq_value)
                # total_seq_values[seq] = seq_value ###
            self.K_j[player][infoset_id] = logsumexp(seq_values)
            # NEW BELOW
            # ys = np.zeros(len(self.new_seqs[player][infoset_id]))
            # for i, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            #     ys[i] = self.b[player][sequence_id]
            #     if sequence_id in self.game.childrenInfosets[player]:
            #         child_sum = 0.0
            #         for child in self.game.childrenInfosets[player][sequence_id]:
            #             if self.K_j[player][child] == None:
            #                 eff = 0
            #             else:
            #                 eff = self.K_j[player][child]
            #             child_sum += eff #self.K_j[player][child]
            #         ys[i] += child_sum
            #     ys[i] -= self.K_j[player][infoset_id]
            # bb = ne.evaluate('exp(ys)')
            # summ = np.sum(bb)
            # for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            #     self.stgy[player][infoset_id][ii] = bb[ii] / summ

        # print("SECOND PASS: ", self.game.infoSets[player])
        # This starts at the top and works downwards
        y = np.zeros(self.total_seqs[player] + 1)
        for infoset_id in self.visited[player]: #self.game.infoSets[player]:
            for sequence_id in self.new_seqs[player][infoset_id]:
                y_par = 0.0
                y[sequence_id] = y_par + self.b[player][sequence_id] # + y_par
                if sequence_id in self.game.childrenInfosets[player]:
                    child_sum = 0.0
                    for child in self.game.childrenInfosets[player][sequence_id]:
                        if self.K_j[player][child] == None:
                            eff = 0
                        else:
                            eff = self.K_j[player][child]
                        child_sum += eff #self.K_j[player][child] #self.K_j[player][child]
                    y[sequence_id] += child_sum
                y[sequence_id] -= self.K_j[player][infoset_id]

            denom = 0.0
            for sequence_id in self.new_seqs[player][infoset_id]:
                self.exp_y[player][sequence_id] = np.exp(y[sequence_id])
                denom += self.exp_y[player][sequence_id]
            for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
                self.stgy[player][infoset_id][ii] = self.exp_y[player][sequence_id] / denom

        # y_exp = np.e ** y # ne.evaluate('exp(y)')
        # for infoset_id in self.visited[player]:  # self.game.infoSets[player]:
        #     summ = 0.0
        #     for sequence_id in self.new_seqs[player][infoset_id]:
        #         summ += y_exp[sequence_id]
        #     for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
        #         self.stgy[player][infoset_id][ii] = y_exp[sequence_id] / summ

            # NORMALIZE POLICY HERE
            # This is if doing it right away
            # denom = 0.0
            # ys = np.zeros(len(self.new_seqs[player][infoset_id]))
            # for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            #     ys[ii] = y[sequence_id]
            # # NEW
            # # max_ys = max(ys)
            # # if max_ys > 10:
            # #     dif = max_ys - 10.0
            # #     for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            # #         if ys[ii] > 10:
            # #             ys[ii] -= dif
            # # for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            # #     if ys[ii] < 0:
            # #         amount = np.abs(ys[ii])
            # #         ys[ii] += amount
            # #         self.b[player][sequence_id] += amount
            # ###
            # # print(ys)
            # b = ne.evaluate('exp(ys)')
            # b_sum = np.sum(b)
            # for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            #     self.stgy[player][infoset_id][ii] = b[ii] / b_sum #self.exp_y[player][sequence_id] / denom #+ (1.0-self.alpha)*(1.0 / len(self.new_seqs[player][infoset_id]))



            # ORIGINAL - WORKING
            # for sequence_id in self.new_seqs[player][infoset_id]:
            #     # b = ne.evaluate('exp(a)')
            #     self.exp_y[player][sequence_id] = np.exp(y[sequence_id])
            #     denom += self.exp_y[player][sequence_id]
            # for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
            #     # self.alpha = 1.0 - (1.0 / (self.round + 1.0) ** 2.0)
            #     self.stgy[player][infoset_id][ii] = self.exp_y[player][sequence_id] / denom #+ (1.0-self.alpha)*(1.0 / len(self.new_seqs[player][infoset_id]))


        # if player == 0:
        #     print(self.stgy[player][4][0], self.last_stgy[player][4][0])

        # Might be able to get this down to times 1
        # But this doesn't bring it down that much - most nodes are in histories not visited infosets
        self.nodestouched += len(self.visited[player][::-1]) * 1

        # for infoset_id in self.visited[player][::-1]:        #self.game.infoSets[player][::-1]:
        #     for seq in self.game.seqs[player][infoset_id]:
        #         self.regret[player][seq] += total_seq_values[seq] - exp_value[infoset_id] ###
        #         if self.regret[player][seq] < 0:
        #             self.regret[player][seq] *= 0.5
        #             self.b[player][seq] *= 0.5
        #         else:
        #             alpha = 3.0 / 2.0
        #             we = (self.round ** alpha) / ((self.round + 1.0) ** alpha)
        #             self.regret[player][seq] *= we
        #             self.b[player][seq] *= we

                # total_value += self.stgy[player][infoset_id][seq] * self.b[player][seq]
                # seq_values.append(self.b)



        # for infoset_id in self.visited[player][::-1]:        #self.game.infoSets[player][::-1]:
        #     seq_values = []
        #     for seq in self.game.seqs[player][infoset_id]:
        #         child_values = []
        #         if seq in self.game.childrenInfosets[player]:
        #             for child_infoset in self.game.childrenInfosets[player][seq]:
        #                 if self.K_j[player][child_infoset] == None:
        #                     aff = 0.0
        #                 else:
        #                     aff = self.K_j[player][child_infoset]
        #                 child_values.append(aff)
        #         seq_value = self.b[player][seq] + sum(child_values)
        #         seq_values.append(seq_value)
        #     self.K_j[player][infoset_id] = logsumexp(seq_values)
        #
        #
        # # print("SECOND PASS: ", self.game.infoSets[player])
        # # This starts at the top and works downwards
        # y = np.zeros(self.total_seqs[player] + 1)
        # # print("NEXTVISITED: ", self.visited[player], "  player:", player)
        # for infoset_id in self.visited[player]: #self.game.infoSets[player]:
        #     for sequence_id in self.new_seqs[player][infoset_id]:
        #         if sequence_id in self.game.parSeq[player]:
        #             y_par = 0.0  # y[self.game.parSeq[player][sequence_id]] THIS GIVES BEHAVE POLICY BY COMMENTING OUT
        #         else:
        #             y_par = 0.0
        #         y[sequence_id] = y_par + self.b[player][sequence_id]
        #         if sequence_id in self.game.childrenInfosets[player]:
        #             child_sum = 0.0
        #             for child in self.game.childrenInfosets[player][sequence_id]:
        #                 if self.K_j[player][child] == None:
        #                     eff = 0
        #                 else:
        #                     eff = self.K_j[player][child]
        #                 child_sum += eff #self.K_j[player][child]
        #             y[sequence_id] += child_sum
        #         y[sequence_id] -= self.K_j[player][infoset_id]
        #
        #     # NORMALIZE POLICY HERE
        #     denom = 0.0
        #     for sequence_id in self.new_seqs[player][infoset_id]:
        #         self.exp_y[player][sequence_id] = np.exp(y[sequence_id])
        #         denom += self.exp_y[player][sequence_id]
        #     for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
        #         self.stgy[player][infoset_id][ii] = self.exp_y[player][sequence_id] / denom

        # PREVIOUS
        # for s in range(len(self.stgy[player])):
        #     x = self.new_seqs[player][s]
        #     if len(x) > 0:
        #         denom = 0.0
        #         for i in x:
        #             denom += self.exp_y[player][i]
        #         for i, ii in enumerate(x):
        #             # print(self.exp_y[player][ii] / denom)
        #             self.stgy[player][s][i] = self.exp_y[player][ii] / denom




    # def init(self):
    #     copy_thres = self.thres
    #     self.thres = -100000.0
    #     game = self.game
    #     self.round += 1
    #     if self.Type == "regretmatching":
    #         self.reachp[0][0] += 1  # These are set to zero in line just above so only ever have values 0 or 1
    #         self.reachp[1][0] += 1
    #     else:
    #         self.reachp[0][0] += self.round
    #         self.reachp[1][0] += self.round
    #     self.receiveProb(0, 0, np.ones(2))  # These set probNotPass and probNotUpdate for history
    #     self.receiveProb(1, 0,
    #                      np.ones(2))  # to all 1 as in: probNotPassed before:  [0. 0.] prob not passed after:  [1. 1.]
    #     self.updateIset(0, 0, [[], []])
    #     self.updateIset(1, 0, [[], []])
    #
    #
    #     def updateoutcome(hist):
    #         if game.isTerminal[hist]:
    #             return
    #         if self.histflag[0][hist] < self.round:
    #             # print("HISTFLAG LOWER: ", hist)
    #             return
    #         self.reward[hist] = 0.0
    #         nacts = game.nactsOnHist[hist]
    #         _stgy = None
    #         player = game.playerOfHist[hist]
    #         if player == 2:
    #             _stgy = game.chanceprob[hist]
    #         else:
    #             piset = game.Hist2Iset[player][hist]
    #             _stgy = self.solvers[player][piset].curstgy  # self.stgy[player][piset]
    #         for a in range(nacts):
    #             nh = game.histSucc[hist][a]
    #             updateoutcome(nh)
    #             self.outcome[hist][a] = self.reward[nh]
    #             self.reward[hist] += _stgy[a] * self.reward[nh]
    #
    #     updateoutcome(0)
    #
    #     self.updateKomwu(0)
    #     self.updateKomwu(1)
    #     self.grad = [np.zeros(12), np.zeros(12)]
    #     self.last_last_gradient = [np.zeros(12), np.zeros(12)]
    #
    #     self.visited = [[], []]
    #     self.thres = copy_thres




"""
game = Game(bidmaximum=5)#maximumhists=maximumhists)
print(game.numHists, game.numIsets)

lazycfr = LazyCFR(game, Type="regretmatchingplus")

T = 10000
for r in range(T):
	lazycfr.updateAll()

	if r % 300 == 0:
		print("time 0", time.time())
		print("exploitability:", lazycfr.getExploitability(), lazycfr.time)
		print("time 1", time.time())
"""
