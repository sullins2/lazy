import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, generateOutcome, exploitability, RegretSolverPlus, generateB, generateChildren
import time
from scipy.special import logsumexp


class LazyCFR:
    def __init__(self, game, Type="regretmatching", thres=0.0, params=None):
        print("initializing solver")
        self.thres = thres
        self.time = 0
        self.eta = params["eta"]
        self.b_count_check = params["b_count"]
        self.b_count_count_at = params["b_count_count_at"]
        self.b_count = 1

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

        self.probNotUpdatedFirst = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.probNotPassedFirst = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
        self.probNotUpdatedFirstCopy = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]


        self.histflag = [-1 * np.ones(game.numHists), -1 * np.ones(game.numHists)]
        self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]

        self.tmp_opt = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]

        self.reachp = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]
        self.reachp_first_step = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]

        self.AMMO = 12000
        self.b = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.b_store = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.seqCount = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.seqCountFlag = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.solvers = []
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[0][x], None), range(game.numIsets[0]))))
        self.solvers.append(list(map(lambda x: RegretSolver(game.nactsOnIset[1][x], None), range(game.numIsets[1]))))



        """
        """
        self.stgy = [[], []]
        first = 0
        second = 0

        self.stgy = [[], []]
        self.stgy_xi = [[], []]
        self.stgy_br = [[], []]

        self.total_seqs = [-1, -1]
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



        for i, iset in enumerate(range(game.numIsets[0])):
            nact = game.nactsOnIset[0][iset]
            if game.playerOfIset[0][iset] == 0:
                self.stgy_xi[0].append(np.ones(nact) / nact)
                self.stgy_br[0].append(np.ones(nact) / nact)
            else:
                self.stgy_xi[0].append(np.ones(0))
                self.stgy_br[0].append(np.ones(0))

        for i, iset in enumerate(range(game.numIsets[1])):
            nact = game.nactsOnIset[1][iset]
            if game.playerOfIset[1][iset] == 1:
                self.stgy_xi[1].append(np.ones(nact) / nact)
                self.stgy_br[1].append(np.ones(nact) / nact)
            else:
                self.stgy_xi[1].append(np.ones(0))
                self.stgy_br[1].append(np.ones(0))

        # hist_seqs0 = []
        # for h in range(game.numHists):
        #     pla = game.playerOfHist[h]
        #     if pla == 0:
        #         i1 = game.Hist2Iset[0][h]
        #         seqqs = self.game.seqs[0][i1]
        #         hist_seqs0.append(np.zeros(self.total_seqs[0] + 1))
        #     else:
        #         hist_seqs0.append(np.zeros(self.total_seqs[0] + 1))
        #
        # hist_seqs1 = []
        # for h in range(game.numHists):
        #     pla = game.playerOfHist[h]
        #     if pla == 1:
        #         i1 = game.Hist2Iset[1][h]
        #         seqqs = self.game.seqs[1][i1]
        #         hist_seqs1.append(np.zeros(self.total_seqs[1] + 1))
        #     else:
        #         hist_seqs1.append(np.zeros(self.total_seqs[1] + 1))

        # self.hist_seqs = [hist_seqs0, hist_seqs1]

        # hist_seqs0_first_step = []
        # for h in range(game.numHists):
        #     pla = game.playerOfHist[h]
        #     if pla == 0:
        #         i1 = game.Hist2Iset[0][h]
        #         seqqs = self.game.seqs[0][i1]
        #         hist_seqs0_first_step.append(np.zeros(self.total_seqs[0] + 1))
        #     else:
        #         hist_seqs0_first_step.append(np.zeros(self.total_seqs[0] + 1))
        #
        # hist_seqs1_first_step = []
        # for h in range(game.numHists):
        #     pla = game.playerOfHist[h]
        #     if pla == 1:
        #         i1 = game.Hist2Iset[1][h]
        #         seqqs = self.game.seqs[1][i1]
        #         hist_seqs1_first_step.append(np.zeros(self.total_seqs[1] + 1))
        #     else:
        #         hist_seqs1_first_step.append(np.zeros(self.total_seqs[1] + 1))
        #
        # self.hist_seqs_first_step = [hist_seqs0_first_step, hist_seqs1_first_step]

        # print("HIST SEQS")
        # print(self.hist_seqs)
        # print(self.hist_seqs_first_step)

        self.round = 1
        self.nodestouched = 0
        self.outcome, self.reward = generateOutcome(game, self.stgy)

        b0 = generateChildren(game,self.stgy, 0)
        # print("CHILDREN0:", b0)
        b1 = generateChildren(game, self.stgy, 1)
        # print("CHILDREN1:", b1)
        self.game.childrenInfosets[0] = b0
        self.game.childrenInfosets[1] = b1

        self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.grad_flag = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_gradient = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.last_last_gradient = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.seqs_updated = [np.zeros(self.AMMO), np.zeros(self.AMMO)]

        self.K_j = [[None] * self.AMMO, [None] * self.AMMO]
        self.K_j_first_step = [[None] * self.AMMO, [None] * self.AMMO]

        self.visited = [[],[]]
        self.last_visited = [[],[]]
        self.exp_y = [np.zeros(self.total_seqs[0] + 1),np.zeros(self.total_seqs[1] + 1)]
        self.exp_y_xi = [np.zeros(self.total_seqs[0] + 1), np.zeros(self.total_seqs[1] + 1)]


        self.last_stgy = [None, None]
        self.seq_flag = [np.zeros(self.total_seqs[0] + 1), np.zeros(self.total_seqs[1] + 1)]

        self.b_xi = [[],[]]

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

        # self.init()
        # print("INIT DONE")
        # self.updateKomwu(0, True)
        # self.updateKomwu(1, True)
        # self.updateKomwu(0, False)
        # self.updateKomwu(1, False)

    def receiveProb(self, owner, histind, prob):
        self.probNotPassed[owner][histind] += prob
        self.probNotUpdated[owner][histind] += prob
        self.nodestouched += 1

    def receiveProbFirst(self, owner, histind, prob):
        self.probNotPassedFirst[owner][histind] += prob
        self.probNotUpdatedFirst[owner][histind] += prob
        self.nodestouched += 1

    def passProbOnHist(self, owner, histind, last_seqs, true_owner, first_step):
        game = self.game
        if game.isTerminal[histind]:
            return

        parhind, pactind = game.histPar[histind]
        # THIS DOESN"T SEEM TO GET CALLED EVER
        if parhind != -1 and self.histflag[owner][parhind] < self.round:
            for _ in range(100000):
                print("Go look in passProbOnHist(), this condition was met!")
                print("Recursive call with owner=", owner, " for parhind: ", parhind)
            self.passProbOnHist(owner, parhind)

        player = game.playerOfHist[histind]
        stgy = None
        # Get the strategy at this infoset
        if player == 2:
            stgy = game.chanceprob[histind]
            isetind = None
        else:
            isetind = game.Hist2Iset[player][histind]
            if first_step:
                # self.stgy is used on first step
                stgy = self.stgy[player][isetind]
            else:
                # self.stgy_xi is used on second step
                stgy = self.stgy_xi[player][isetind]

        for aind, nxthind in enumerate(game.histSucc[histind]):
            if first_step:
                tmp = self.probNotPassedFirst[owner][histind].copy()
            else:
                tmp = self.probNotPassed[owner][histind].copy()

            if player == owner:
                tmp[owner] *= stgy[aind]
            else:
                tmp[1 - owner] *= stgy[aind]
            # outcome looks like this: [ 0.125 -0.875 -0.875] and tmp[1-owner] like this: 0.3333
            self.cfvCache[owner][nxthind] += np.array(self.outcome[nxthind]) * tmp[1 - owner]


            if game.isTerminal[nxthind]: # and true_owner == owner:
                i1 = game.Hist2Iset[owner][histind]  # original
                i2 = game.Hist2Iset[owner][nxthind]

                sign = 1.0
                if owner == 1:
                    sign = -1.0
                if i2 in self.game.isetSuccSeq[owner][i1]:
                    # seqq = self.game.isetSuccSeq[owner][i1][i2]
                    # if first_step:
                    #     amount = self.hist_seqs_first_step[owner][histind][seqq]
                    # else:
                    #     amount = self.hist_seqs[owner][histind][seqq]
                    # if first_step:
                    #     if self.round > amount:
                    #         f = self.round - amount #+ 1.0
                    #     else:
                    #         f = 1.0
                    f = 1.0
                    self.grad[owner][self.game.isetSuccSeq[owner][i1][i2]] += 1.0 * f * tmp[1 - owner] * self.reward[nxthind] * sign

                    # if first_step:
                    #     self.hist_seqs_first_step[owner][histind][seqq] = self.round
                    # else:
                    #     self.hist_seqs[owner][histind][seqq] = self.round

            if first_step:
                self.receiveProbFirst(owner, nxthind, tmp)
            else:
                self.receiveProb(owner, nxthind, tmp)
        if first_step:
            self.probNotPassedFirst[owner][histind] = np.zeros(2)
        else:
            self.probNotPassed[owner][histind] = np.zeros(2)
        self.histflag[owner][histind] = self.round


    def updateIset(self, owner, isetind, last_seqs, first_step):

        self.isetflag[owner][isetind] = self.round
        game = self.game
        if game.playerOfIset[owner][isetind] == owner:
            strat = self.stgy_xi[owner][isetind]
            # sumstgy is the policy that gets checked for exploitability
            # On the second step, this overwrites it to be the policy we care about
            self.sumstgy[owner][isetind] = self.reachp[owner][isetind] * strat
            self.visited[owner].append(isetind)


        for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
            if first_step:
                # On the first step, we copy probNotUpdatedFirst for backup since
                #   on the second pass we only want to update the same infosets
                #   so we will use it below as the check.
                self.probNotUpdatedFirstCopy[owner][hind] = self.probNotUpdatedFirst[owner][hind]
                self.probNotUpdatedFirst[owner][hind] = np.zeros(2)
                self.passProbOnHist(0, hind, last_seqs, true_owner=owner, first_step=first_step)
                self.passProbOnHist(1, hind, last_seqs, true_owner=owner, first_step=first_step)
            else:
                # On second step,
                self.probNotUpdated[owner][hind] = np.zeros(2)
                self.passProbOnHist(0, hind, last_seqs, true_owner=owner, first_step=first_step)
                self.passProbOnHist(1, hind, last_seqs, true_owner=owner, first_step=first_step)

        nacts = game.nactsOnIset[owner][isetind]

        for innerIind, nxtisetind in enumerate(game.isetSucc[owner][isetind]):
            sumprob = 0.0
            nxthists = game.Iset2Hists[owner][nxtisetind]
            if game.isTerminal[nxthists[0]] == False:
                # Sum up probNotUpdated for all the histories in the next infoset
                for nxth in game.Iset2Hists[owner][nxtisetind]:
                    if first_step:
                        # On the first step, probNotUpdatedFirst is keeping track of if we continue
                        sumprob += self.probNotUpdatedFirst[owner][nxth][1 - owner]
                    else:
                        # On second step, if still only check based on what first step did
                        # self.probNotUpdatedFirstCopy[owner][nxth][1 - owner]
                        # sumprob += self.probNotUpdatedFirstCopy[owner][nxth][1 - owner]
                        # Use current probNotUpdated
                        sumprob += self.probNotUpdated[owner][nxth][1 - owner]

                if game.playerOfIset[owner][isetind] == owner:
                    # The strategy used for each step
                    if first_step:
                        strat = self.stgy[owner][isetind][innerIind]
                    else:
                        strat = self.stgy_xi[owner][isetind][innerIind]
                    # Calculate reachp based on which step we are in
                    if first_step:
                        self.reachp_first_step[owner][nxtisetind] += self.reachp_first_step[owner][isetind] * strat
                    else:
                        self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * strat #self.stgy[owner][isetind][innerIind] #self.solvers[owner][isetind].curstgy[innerIind]
                else:
                    # Calculate reachp based on which step we are in
                    if first_step:
                        self.reachp_first_step[owner][nxtisetind] += self.reachp_first_step[owner][isetind]
                    else:
                        self.reachp[owner][nxtisetind] += self.reachp[owner][isetind]
                if sumprob > self.thres:
                    # Track the last seq seen so grad can update it when it sees a terminal above
                    owner_of_iset = game.playerOfIset[owner][nxtisetind]
                    if owner_of_iset != 2 and owner == owner_of_iset:
                        cur_seqs = game.seqs[owner_of_iset][nxtisetind].copy()
                        next_seqs = last_seqs.copy()
                        # There might be more than two actions
                        if len(cur_seqs) < len(next_seqs):
                            for _ in range(len(next_seqs) - len(cur_seqs) + 1):
                                cur_seqs.append(0)
                        next_seqs[owner_of_iset] = cur_seqs[innerIind]
                    else:
                        next_seqs = last_seqs.copy()
                    self.updateIset(owner, nxtisetind, next_seqs, first_step)
                else:
                    dummy = 0

        # Set reachp to 0 depending on step
        if first_step:
            self.reachp_first_step[owner][isetind] = 0
        else:
            self.reachp[owner][isetind] = 0

    def updateAll(self, curexpl, t):
        t1 = time.time()
        game = self.game
        self.round += 1

        # Reset for BR calculations
        self.reachp_first_step[0][0] += 1  # These are set to zero in line just above so only ever have values 0 or 1
        self.reachp_first_step[1][0] += 1
        self.receiveProbFirst(0, 0, np.ones(2))  # These set probNotPass and probNotUpdate for history
        self.receiveProbFirst(1, 0, np.ones(2))  # to all 1 as in: probNotPassed before:  [0. 0.] prob not passed after:  [1. 1.]

        # Reset the utility gradients and infosets visited
        self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.visited = [[], []]

        # Get the initial grads with strategy stgy
        self.updateIset(0, 0, [[], [], [], []], first_step=True)
        self.updateIset(1, 0, [[], [], [], []], first_step=True)


        # DOING ENT ON BOTH NOW


        # print("LEN: ", len(self.visited[0]))

        # Update to get stgy_xi
        self.updateKomwu(0, first_step=True, t=t)
        self.updateKomwu(1, first_step=True, t=t)

        # if t % 10 == 0:
        #     self.stgy_br[0] = copy.deepcopy(self.stgy_xi[0])
        #     self.stgy_br[1] = copy.deepcopy(self.stgy_xi[1])

        # Reset for second update rule
        self.reachp[0][0] += 1  # These are set to zero in line just above so only ever have values 0 or 1
        self.reachp[1][0] += 1
        self.receiveProb(0, 0, np.ones(2))  # These set probNotPass and probNotUpdate for history
        self.receiveProb(1, 0, np.ones(2))  # to all 1 as in: probNotPassed before:  [0. 0.] prob not passed after:  [1. 1.]

        # Reset utility gradients
        self.grad = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        self.visited = [[], []]

        # Get the grads with stgy_xi
        self.updateIset(0, 0, [[], [], [], []], first_step=False)
        self.updateIset(1, 0, [[], [], [], []], first_step=False)

        # ####### ENTROPY ##################
        # mod = self.round // 50
        # ent = 0 #-0.005 / (mod + 1)
        # for infoset_id in self.visited[0][::-1]:
        #     vals = []
        #     sum_vals = 0.0
        #     # reg = self.solvers[0][infoset_id].cfrreg()
        #     for i, seq in enumerate(self.game.seqs[0][infoset_id]):
        #         pol = self.stgy[0][infoset_id][i]
        #         if pol == 0:
        #             pol = 0.00000000000000000000000000000001
        #         sum_vals += pol * np.log(pol)
        #         vals.append(np.log(pol))
        #     for i, seq in enumerate(self.game.seqs[0][infoset_id]):
        #         self.b[0][seq] += ent * (vals[i] - sum_vals)

        # CHECK MAGNITUDE AND SIGNS
        # for infoset_id in self.visited[1][::-1]:
        #     vals = []
        #     sum_vals = 0.0
        #     # reg = self.solvers[1][infoset_id].cfrreg()
        #     for i, seq in enumerate(self.game.seqs[1][infoset_id]):
        #         pol = self.stgy[1][infoset_id][i]
        #         if pol == 0:
        #             pol = 0.00000000000000000000000000000001
        #         sum_vals += pol * np.log(pol)
        #         vals.append(np.log(pol))
        #     for i, seq in enumerate(self.game.seqs[1][infoset_id]):
        #         self.b[1][seq] += ent * (vals[i] - sum_vals)
        #
        # ###########################################

        # Update to get stgy
        self.updateKomwu(0, first_step=False, t=t)
        self.updateKomwu(1, first_step=False, t=t)

        self.probNotUpdatedFirstCopy = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]

        # self.b_count += 1
        # if self.b_count == self.b_count_check:
        #     self.b = self.b_store.copy()
        #     # self.b_store = [np.zeros(self.AMMO), np.zeros(self.AMMO)]
        #     self.b_count = 1

        self.time += time.time() - t1



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
        stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[0])))
        stgy_prof.append(list(map(lambda _x: avg(_x), self.stgy[1])))
        # stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[0])))
        # stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[1])))
        self.mike = stgy_prof
        return exploitability(self.game, stgy_prof)

    def updateKomwu(self, player, first_step,t):


        # The first step is applying a large learning rate to the utilities updating the logits
        # A different set of K_js are used here, and b_xi is used instead of b
        # self.K_j_first_step = [[None] * self.AMMO, [None] * self.AMMO]

        if first_step:
            xi = 100.0 # Try smaller with laziness
            self.b_xi[player] = self.b[player].copy() # Copy the current b

            ####### ENTROPY ##################
            # mod = self.round // 100
            # ent = 0 #-3.0 / np.log(self.round + 2.0) #(mod + 1)
            # for infoset_id in self.visited[player][::-1]:
            #     vals = []
            #     sum_vals = 0.0
            #     # reg = self.solvers[0][infoset_id].cfrreg()
            #     for i, seq in enumerate(self.game.seqs[player][infoset_id]):
            #         pol = self.stgy[player][infoset_id][i]
            #         if pol == 0:
            #             pol = 0.00000000000000000000000000000001
            #         sum_vals += -pol * np.log(pol)
            #         vals.append(-pol * np.log(pol))
            #     for i, seq in enumerate(self.game.seqs[player][infoset_id]):
            #         self.b_xi[player][seq] += ent * sum_vals #(vals[i] - sum_vals)


            self.b_xi[player] += xi * self.grad[player] # Add the current utility times xi
            # for ii in range(len(self.b_xi[player])):
            #     if ii in self.game.seqs_depth[player]:
            #         xi = 10.0 * self.game.seqs_depth[player][ii]
            #         # print("HERE", xi)
            #     else:
            #         # print("NOW HERE", ii)
            #         # print(self.game.seqs_depth[player])
            #         xi = 1.0
            #     self.b_xi[player][ii] += xi * self.grad[player][ii]


            # mod = self.round // 100
            # KL = 0.0 / (mod + 1.0)
            # if self.last_stgy[player] != None:
            #     vals = []
            #     for infoset_id in self.visited[player][::-1]:
            #         sum_vals = 0.0
            #         for i, seq in enumerate(self.game.seqs[player][infoset_id]):
            #             strat = self.stgy[player][infoset_id][i]
            #             ref_strat = self.stgy_br[player][infoset_id][i]
            #             new_rew = min((KL / strat) * (ref_strat - strat), 10)
            #             new_rew = max(new_rew, 1e-10)
            #             # new_rew = 0.0
            #             self.b_xi[player][seq] += new_rew



                    #     old_pol = self.stgy[player][infoset_id][i]
                    #     pol = self.last_stgy[player][infoset_id][i]
                    #     pol = np.clip(pol, a_min=np.finfo(float).eps, a_max=None)
                    #     old_pol = np.clip(old_pol, a_min=np.finfo(float).eps, a_max=None)
                    #     # p = np.clip(pol / old_pol, a_min=np.finfo(float).eps, a_max=None)
                    #     sum_vals += pol * np.log(pol / old_pol)
                    #     vals.append(pol * np.log(pol / old_pol))
                    # for i, seq in enumerate(self.game.seqs[player][infoset_id]):
                    #     self.b_xi[player][seq] += KL * (vals[i] - sum_vals)


            self.last_stgy[player] = copy.deepcopy(self.stgy[player])

            # self.K_j_first_step = [[None] * self.AMMO, [None] * self.AMMO]
            # First step of KOMWU with K_J_first_step
            for infoset_id in self.visited[player][::-1]:
                seq_values = []
                for seq in self.game.seqs[player][infoset_id]:
                    child_values = []
                    if seq in self.game.childrenInfosets[player]:
                        for child_infoset in self.game.childrenInfosets[player][seq]:
                            if self.K_j_first_step[player][child_infoset] == None:
                                aff = 0.0
                            else:
                                aff = self.K_j_first_step[player][child_infoset]
                            child_values.append(aff)
                    seq_value = self.b_xi[player][seq] + sum(child_values)
                    seq_values.append(seq_value)
                self.K_j_first_step[player][infoset_id] = logsumexp(seq_values)

            # Second step of KOMWU with self.b_xi
            y = np.zeros(self.total_seqs[player] + 1)
            for infoset_id in self.visited[player]:  # self.game.infoSets[player]:
                for sequence_id in self.new_seqs[player][infoset_id]:
                    y_par = 0.0
                    y[sequence_id] = y_par + self.b_xi[player][sequence_id]
                    if sequence_id in self.game.childrenInfosets[player]:
                        child_sum = 0.0
                        for child in self.game.childrenInfosets[player][sequence_id]:
                            if self.K_j_first_step[player][child] == None:
                                eff = 0
                            else:
                                eff = self.K_j_first_step[player][child]
                            child_sum += eff  # self.K_j[player][child]
                        y[sequence_id] += child_sum
                    y[sequence_id] -= self.K_j_first_step[player][infoset_id]

                # NORMALIZE POLICY HERE
                denom = 0.0
                for sequence_id in self.new_seqs[player][infoset_id]:
                    self.exp_y_xi[player][sequence_id] = np.exp(y[sequence_id])
                    denom += self.exp_y_xi[player][sequence_id]
                for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
                    self.stgy_xi[player][infoset_id][ii] = self.exp_y_xi[player][sequence_id] / denom

            self.nodestouched += len(self.visited[player][::-1]) * 1

        else:
            # Second update rule of FLBR
            # KOMWU uses K_j's and b's here
            #eta = 1.0 # / 9.0  #1 flbr not working 44 mil
            # eta = np.sqrt(np.log(2) / (self.round +  1.0))
            self.b[player] += self.eta * self.grad[player]
            # if self.b_count >= self.b_count_count_at:
            #     self.b_store[player] += self.eta * self.grad[player]

            # KL = 1.0
            # if self.last_stgy[player] != None:
            #     vals = []
            #     for infoset_id in self.visited[player][::-1]:
            #         sum_vals = 0.0
            #         for i, seq in enumerate(self.game.seqs[player][infoset_id]):
            #             old_pol = self.stgy[player][infoset_id][i]
            #             pol = self.last_stgy[player][infoset_id][i]
            #             pol = np.clip(pol, a_min=np.finfo(float).eps, a_max=None)
            #             old_pol = np.clip(old_pol, a_min=np.finfo(float).eps, a_max=None)
            #             # p = np.clip(pol / old_pol, a_min=np.finfo(float).eps, a_max=None)
            #             sum_vals += pol * np.log(pol / old_pol)
            #             vals.append(pol * np.log(pol / old_pol))
            #         for i, seq in enumerate(self.game.seqs[player][infoset_id]):
            #             self.b[player][seq] += KL * (vals[i] - sum_vals)
            #
            # self.last_stgy[player] = copy.deepcopy(self.stgy[player])

            # First step of KOMWU - bottom to top
            for infoset_id in self.visited[player][::-1]:
                seq_values = []
                for seq in self.game.seqs[player][infoset_id]:
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
                self.K_j[player][infoset_id] = logsumexp(seq_values)


            # This starts at the top and works downwards
            y = np.zeros(self.total_seqs[player] + 1)
            for infoset_id in self.visited[player]: #self.game.infoSets[player]:
                for sequence_id in self.new_seqs[player][infoset_id]:
                    y_par = 0.0
                    y[sequence_id] = y_par + self.b[player][sequence_id]
                    if sequence_id in self.game.childrenInfosets[player]:
                        child_sum = 0.0
                        for child in self.game.childrenInfosets[player][sequence_id]:
                            if self.K_j[player][child] == None:
                                eff = 0
                            else:
                                eff = self.K_j[player][child]
                            child_sum += eff #self.K_j[player][child]
                        y[sequence_id] += child_sum
                    y[sequence_id] -= self.K_j[player][infoset_id]


                # NORMALIZE POLICY HERE
                denom = 0.0
                for sequence_id in self.new_seqs[player][infoset_id]:
                    self.exp_y[player][sequence_id] = np.exp(y[sequence_id])
                    denom += self.exp_y[player][sequence_id]
                for ii, sequence_id in enumerate(self.new_seqs[player][infoset_id]):
                    self.stgy[player][infoset_id][ii] = self.exp_y[player][sequence_id] / denom

            self.nodestouched += len(self.visited[player][::-1]) * 1



    def getAvgStgy(self, owner, iset):
        game = self.game
        player = game.playerOfIset[owner][iset]
        # print(self.reachp[owner][iset])
        if player == owner:
            # THIS SHOULD BE = not +=
            # print(self.reachp[owner][iset])
            self.sumstgy[owner][iset] += 1# USED ABOVE NOW self.stgy[player][iset] * self.reachp[owner][iset]  # self.reachp[owner][iset] * self.solvers[owner][iset].curstgy
        # print(enumerate(game.isetSucc[owner][iset]))
        for _i, niset in enumerate(game.isetSucc[owner][iset]):
            if player == owner:
                self.reachp[owner][niset] += self.reachp[owner][iset] * self.stgy[owner][iset][_i] # self.solvers[owner][iset].curstgy[_i]
            else:
                self.reachp[owner][niset] += self.reachp[owner][iset]

            self.getAvgStgy(owner, niset)
        self.reachp[owner][iset] = 0

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