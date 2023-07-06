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
        # self.outcome, self.reward = generateOutcome(game, self.stgy)
        self.b = [np.zeros(12), np.zeros(12)] #[0 for _ in range(12)]
        # self.b1 = np.zeros(12) #[0 for _ in range(12)]

        self.grad = [np.zeros(12), np.zeros(12)]
        self.last_gradient = [np.zeros(12), np.zeros(12)]
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

    def receiveProb(self, owner, histind, prob):
        self.probNotPassed[owner][histind] += prob
        self.probNotUpdated[owner][histind] += prob
        self.nodestouched += 1

    def passProbOnHist(self, owner, histind):
        game = self.game
        if game.isTerminal[histind]:
            return

        parhind, pactind = game.histPar[histind]
        # THIS DOESN"T SEEM TO GET CALLED EVER
        if parhind != -1 and self.histflag[owner][parhind] < self.round:
            # other = game.playerOfHist[parhind]
            # print("OWNER: ", owner, " OTHER: ", other)
            self.passProbOnHist(owner, parhind)

        player = game.playerOfHist[histind]
        stgy = None
        if player == 2:
            stgy = game.chanceprob[histind]
        else:
            isetind = game.Hist2Iset[player][histind]
            # stgy = self.solvers[player][isetind].curstgy
            stgy = self.stgy[player][isetind]

        for aind, nxthind in enumerate(game.histSucc[histind]):
            tmp = self.probNotPassed[owner][histind].copy()
            if player == owner:
                tmp[owner] *= stgy[aind]
            else:
                tmp[1 - owner] *= stgy[aind]
            self.cfvCache[owner][nxthind] += np.array(self.outcome[nxthind]) * tmp[1 - owner]
            self.receiveProb(owner, nxthind, tmp)
        self.probNotPassed[owner][histind] = np.zeros(2)
        self.histflag[owner][histind] = self.round

    def updateIset(self, owner, isetind):
        self.isetflag[owner][isetind] = self.round
        # Add infoset to list of infosets that KOMWU will update

        # This is applying the probNotUpdated that could be accumulated to the cfvCache
        # This is where probNotUpdated needs to be used by KOMWU

        # if self.game.playerOfIset[owner][isetind] == owner:
        #     sumcfv = np.zeros(game.nactsOnIset[owner][isetind])
        #     weight = 0
        #     print("About to loop through hists in this iset. Hists: ", game.Iset2Hists[owner][isetind])
        #     for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
        #         sumcfv += self.cfvCache[owner][hind]
        #         weight = self.probNotUpdated[owner][hind][owner]
        #         self.cfvCache[owner][hind] *= 0.0
        #     if owner == 1:
        #         sumcfv *= -1.0
        #     self.sumstgy[owner][isetind] += self.reachp[owner][isetind] * self.solvers[owner][isetind].curstgy
        #     self.solvers[owner][isetind].receive(sumcfv, weight=weight)
        #     print("Done looping through the hists and updating cfvs")

        for innerhind, hind in enumerate(self.game.Iset2Hists[owner][isetind]):
            # This will happen when an infoset "opens up"
            # The if statement above is where the probNotUpdated is used if it is stored
            # Need to somehow "apply" it to komwu before it gets zeroed out
            # See what probNotPassed has during the run
            self.probNotUpdated[owner][hind] = np.zeros(2)
            self.passProbOnHist(0, hind)
            self.passProbOnHist(1, hind)

        nacts = self.game.nactsOnIset[owner][isetind]

        for innerIind, nxtisetind in enumerate(self.game.isetSucc[owner][isetind]):
            sumprob = 0.0
            nxthists = self.game.Iset2Hists[owner][nxtisetind]
            if self.game.isTerminal[nxthists[0]] == False:
                for nxth in self.game.Iset2Hists[owner][nxtisetind]:
                    sumprob += self.probNotUpdated[owner][nxth][1 - owner]
                if self.game.playerOfIset[owner][isetind] == owner:
                    self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * self.stgy[owner][isetind][innerIind] #self.solvers[owner][isetind].curstgy[innerIind]
                else:
                    self.reachp[owner][nxtisetind] += self.reachp[owner][isetind]
                if sumprob > self.thres:
                    self.updateIset(owner, nxtisetind)
        self.reachp[owner][isetind] = 0

    def updateAll(self):
        t1 = time.time()
        self.round += 1

        self.reachp[0][0] += 1  # These are set to zero in line just above so only ever have values 0 or 1
        self.reachp[1][0] += 1

        # These should be moved to init and then put updateB calls here
        self.grad[0] = np.array(generateB(self.game, self.stgy, 0))
        self.grad[1] = np.array(generateB(self.game, self.stgy, 1))

        self.updateIset(0, 0)
        self.updateIset(1, 0)

        self.updateKomwu(0)
        self.updateKomwu(1)


    def updateKomwu(self, player):
        eta = 1.0
        optimistic_gradient = 2.0 * self.grad[player] - 1.0 * self.last_gradient[player]
        self.last_gradient[player] = self.grad[player].copy()
        self.b[player] += eta * optimistic_gradient

        K_j = [None] * 22 # This should be num infosets (the actual total not the collected ones) ? Not sure, look into this
        # This starts at the bottom and works upwards
        # print("FIRST PASS: ", self.game.infoSets[player][::-1])
        for infoset_id in self.game.infoSets[player][::-1]:
            seq_values = []
            for seq in self.game.seqs[player][infoset_id]:
                child_values = []
                if seq in self.game.childrenInfosets[player]:
                    for child_infoset in self.game.childrenInfosets[player][seq]:
                        child_values.append(K_j[child_infoset])
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

    # Use probNotPassed here
    # If it has accumulation, it will get passed down
    #   If it doesn't make it to a terminal, it stops going down and isnt used
    #   If it gets used
    def updateoutcome(self, hist):
        if self.game.isTerminal[hist]:
            return
        if self.histflag[0][hist] < self.round:
            return
        self.grad[hist] = 0.0
        nacts = self.game.nactsOnHist[hist]
        _stgy = None
        player = self.game.playerOfHist[hist]
        if player == 2:
            _stgy = self.game.chanceprob[hist]
        else:
            piset = self.game.Hist2Iset[player][hist]
            _stgy = self.solvers[player][piset].curstgy  # self.stgy[player][piset]
        for a in range(nacts):
            nh = self.game.histSucc[hist][a]
            self.updateoutcome(nh)
            # self.outcome[hist][a] = self.reward[nh]
            # self.reward[hist] += _stgy[a] * self.reward[nh]

    updateoutcome(0)

    def updateB(game, stgy_prof, p):
        outcome = np.zeros(game.numHists).tolist()
        rew = np.zeros(game.numHists)
        b = [0] * 12

        def solve(hist, p, last_seq, cfrp):
            if game.isTerminal[hist]:
                rew[hist] = game.reward[hist][p]
                outcome[hist] = []
                if last_seq >= 0:
                    b[last_seq] += cfrp * game.reward[hist][p]
                return rew[hist]
            outcome[hist] = []
            player = game.playerOfHist[hist]
            stgy = None
            if player < 2:
                iset = game.Hist2Iset[player][hist]
                stgy = stgy_prof[player][iset]
            else:
                stgy = game.chanceprob[hist]
            for a in range(game.nactsOnHist[hist]):
                if player == p:
                    cur_seqs = game.seqs[p][iset]
                    next_seq = cur_seqs[a]
                else:
                    cur_seqs = None
                    next_seq = last_seq
                if player == 2 or player != p:
                    next_cfrp = cfrp * stgy[a]
                else:
                    next_cfrp = cfrp
                srew = solve(game.histSucc[hist][a], p, next_seq, next_cfrp)  # get the reward returned here
                outcome[hist].append(srew)  # set the [infoset][history][action] = srew
                rew[hist] += stgy[a] * srew  # sew rew[infoset][history] += stgy[a] * srew

            return rew[hist]  # return that value

        solve(0, p, -1, 1)

    def getAvgStgy(self, owner, iset):
        game = self.game
        player = game.playerOfIset[owner][iset]
        if player == owner:
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
