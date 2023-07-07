import copy
import numpy as np
import time

"""
		self.numHists = 0
		self.numIsets = [0, 0]
		self.isTerminal = []
		self.reward = []
		self.histSucc = []
		self.histPar = []
		self.playerOfHist = []
		self.nactsOnHist = []
		self.Iset2Hists = [[], []]
		self.Hist2Iset = [[], []]
		self.nactsOnIset = [[], []]
		self.playerOfIset = [[], []]
		self.isetPar = [[],[]]
		self.isetSucc = [[], []]
		self.chanceprob	 = []
"""


class KuhnGame:
    def __init__(self, path=-1, cards=3, bidmaximum=6, tosave=False):
        self.bidmaximum = bidmaximum
        self.cards = cards
        self.numHists = 0
        self.numIsets = [0, 0]
        self.isTerminal = []
        self.reward = []
        self.histSucc = []
        self.histPar = []
        self.playerOfHist = []
        self.nactsOnHist = []
        self.Iset2Hists = [[], []]  #
        self.Hist2Iset = [[], []]  #
        self.nactsOnIset = [[], []]
        self.playerOfIset = [[], []]
        self.isetPar = [[], []]
        self.isetSucc = [[], []]
        self.isetSuccSeq = [{}, {}]


        self.chanceprob = []

        self.infoSets = [[],[]]
        self.numInfoSets = [0, 0]
        self.totalSeqs = [-1, -1]
        self.seqs = [{}, {}]
        self.childrenInfosets = [{},{}]  # seq -> [infosetid, ...]
        self.parSeq = [{}, {}]

        if path == -1:
            self.genGame(np.array([[-1]]), np.array([[-1]]), [np.array([-1]), np.array([-1])],
                         [np.array([-1]), np.array([-1])], 0, np.array([[[-1]], [[-1]]]), -1, np.array([-1, -1]),
                         np.array([0, 0]), np.array([-1, -1]), np.array([-1, -1]), None, None, True, None) # KOMWU - second to last is lastpar, last will be last seq

            for p in range(2):
                for iset in range(self.numIsets[p]):
                    self.Iset2Hists[p][iset] = self.Iset2Hists[p][iset].astype(int)

            # Set children here
            # SEQ -> Children list
            # self.childrenInfosets[0][1] = [19]
            # self.childrenInfosets[0][3] = [20]
            # self.childrenInfosets[0][5] = [21]

        else:
            file = np.load(path)

            self.numHists = file['numHists']
            self.numIsets = file['numIsets']
            self.isTerminal = file['isTerminal']
            self.reward = file['reward']
            self.histSucc = file['histSucc']
            self.histPar = file['histPar']
            self.playerOfHist = file['playerOfHist']
            self.nactsOnHist = file['nactsOnHist']
            self.Iset2Hists = [file['Iset2Hists0'], file['Iset2Hists1']]

            self.Hist2Iset = file['Hist2Iset']
            self.nactsOnIset = file['nactsOnIset']
            self.playerOfIset = file['playerOfIset']
            self.isetPar = file['isetPar']
            self.isetSucc = file['isetSucc']
            self.chanceprob = file['chanceprob']
            for h in range(self.numHists):
                self.Hist2Iset[0][h] = int(self.Hist2Iset[0][h])
                self.Hist2Iset[1][h] = int(self.Hist2Iset[1][h])
            for p in range(2):
                for iset in range(self.numIsets[p]):
                    self.Iset2Hists[p][iset] = self.Iset2Hists[p][iset].astype(int)
        if tosave:
            np.savez("leduc_" + str(cards) + "_" + str(bidmaximum), numHists=self.numHists, isTerminal=self.isTerminal,
                     reward=self.reward,
                     histSucc=self.histSucc, histPar=self.histPar, playerOfHist=self.playerOfHist,
                     nactsOnHist=self.nactsOnHist, Iset2Hists0=self.Iset2Hists[0], Iset2Hists1=self.Iset2Hists[1],
                     Hist2Iset=self.Hist2Iset, nactsOnIset=self.nactsOnIset, playerOfIset=self.playerOfIset,
                     isetPar=self.isetPar, isetSucc=self.isetSucc, numIsets=self.numIsets,
                     chanceprob=self.chanceprob)

    def printGame(self):
        print("numHists", self.numHists, "numIsets", self.numIsets)
        print("histPar", list(enumerate(self.histPar)))
        print("histSucc", list(enumerate(self.histSucc)))
        print("isetPar 0", list(enumerate(self.isetPar[0])))
        print("isetSucc 0", list(enumerate(self.isetSucc[0])))
        print("isTerminal", self.isTerminal)
        print("Hist2Iset", self.Hist2Iset)
        print("Iset2Hists", self.Iset2Hists[0])
        print("reward", list(enumerate(self.reward)))
        print("chanceprob", list(enumerate(self.chanceprob)))
        print("playerOfHist", self.playerOfHist)
        print("playerOfIset 0", self.playerOfIset[0])
        print("nactsOnIset 0", self.nactsOnIset[0])
        print("nactsOnIset 1", self.nactsOnIset[1])

    # parhists = np.array([[-1]]),
    # hacts =  np.array([[-1]]),
    # parisets =  [np.array([-1]), np.array([-1])]
    # iacts [np.array([-1]), np.array([-1])],
    # depth = 0,
    # privatecard - np.array([[[-1]], [[-1]]]),
    # publiccard = -1,
    # bids = np.array([-1, -1]),
    # quitnp.array([0, 0]))
    def genGame(self, parhists, hacts, parisets, iacts, depth, privatecard, publiccard, bids, quit, lastpar, lastseq, lastparseq0, lastparseq1, lastchance, selfparids, isTerminal=False,
                player=2):
        x, y = parhists.shape

        # print("PLAYER: ", player, " DEPTH: ", depth, " x: ", x, " y: ", y)
        # print("CARDS: ", self.cards)

        histids = np.ones((x, y))
        isetids = [np.ones(x), np.ones(y)]

        obs = [True, True]

        # This happens on the first iteration
        if privatecard[0][0][0] == -1:
            obs[1] = False
        elif privatecard[1][0][0] == -1:  # second iteration
            obs[0] = False

        # print("OBS: ", obs)

        def genNactsOnHist():
            if isTerminal:
                return 0
            if player == 2 and obs[1] == False:
                # # print("Numacts on hist (player 2):", int(self.cards))
                return int(self.cards)
            if player == 2 and obs[0] == False:
                return int(self.cards)
            if bids[1 - player] == -1:
                # # print("Numacts on hist: (bids -1)", self.bidmaximum + 1)
                return int(self.bidmaximum + 1)  # pretty sure this means bet 1 or check
            # # print("Numacts on hist: (OTHER)", 2 + int(self.bidmaximum - bids[1 - player]))
            return 2 + int(self.bidmaximum - bids[1 - player])  # call, fold or raise amount left?

        def genNactsOnIset(owner):
            if isTerminal:
                return 0
            if obs[owner] == False:
                return 1
            return genNactsOnHist()

        # # print("isTerminal", isTerminal, "privatecard", privatecard, "publiccard", publiccard, "bids", bids, "quit", quit, "obs", obs)

        if player == 0:
            totseqs0 = [[] for _ in range(genNactsOnIset(0))]
        else:
            totseqs0 = None
        if player == 1:
            totseqs1 = [[] for _ in range(genNactsOnIset(1))]
        else:
            totseqs1 = None

        # print("RER")
        # print(totseqs0)
        for i in range(x):
            # print("FOR X")
            # print("selfparids[0]:", selfparids)
            self.isetSucc[0].append([])
            self.isetPar[0].append(0)
            self.playerOfIset[0].append(0)
            self.nactsOnIset[0].append(0)
            self.Iset2Hists[0].append(0)
            isetids[0][i] = int(self.numIsets[0])
            # print("isetid: ", isetids[0][i])
            self.isetPar[0][self.numIsets[0]] = (int(parisets[0][i]), int(iacts[0][i]))
            # print("isetPar: ", self.isetPar[0][self.numIsets[0]])
            self.playerOfIset[0][self.numIsets[0]] = player
            # print("playerOfIset: ", player)

            self.nactsOnIset[0][self.numIsets[0]] = genNactsOnIset(0)
            # print("Num acts on iset: ", self.nactsOnIset[0][self.numIsets[0]])
            ################# KOMWU
            seqs0 = []
            # nxtpars0 = []
            if player == 0:
                # print("Here would have lastparseq0: ", lastparseq0)
                if lastparseq0 is not None:
                    # print("Setting parSeq to be: ", lastparseq0[i])
                    self.parSeq[0][self.numIsets[0]] = lastparseq0[i]
                self.infoSets[0].append(self.numIsets[0]) # self.numInfoSets[0]) # create the infoset
                for ii in range(1, genNactsOnIset(0) + 1):
                    seqs0.append(self.totalSeqs[0] + ii)
                    totseqs0[ii-1].append(self.totalSeqs[0] + ii)
                self.seqs[0][self.numIsets[0]] = seqs0
                self.numInfoSets[0] += 1
                self.totalSeqs[0] += genNactsOnIset(0)

            #################
            # print("TOTAL SEQS0")
            # print(totseqs0)
            # # print(" x acts: ", self.nactsOnIset[0][self.numIsets[0]])

            if parisets[0][i] >= 0:
                # if True: #player == 0:
                #     # print("****************************")
                #     # print("For parisets:", parisets[0][i], " to get to isetids:", isetids[0][i])
                #     if lastparseq0 is not None:
                #         # print("lastparseq0[i]: i:", i, " value:", lastparseq0[i])
                #     # print("****************************")
                # print("PARISET ABOVE 0. lastchance:", lastchance, "  iset:", isetids[0][i])
                self.isetSucc[0][int(parisets[0][i])].append(int(isetids[0][i]))
                if int(parisets[0][i]) not in self.isetSuccSeq[0] and int(parisets[0][i]) is not None:
                    # print("  setting:",parisets[0][i])
                    self.isetSuccSeq[0][int(parisets[0][i])] = {}
                if int(parisets[0][i]) is not None and int(isetids[0][i]) is not None:
                    if lastparseq0 is not None:
                        # print("  lastparseq0 not None. Setting", int(parisets[0][i]), " to ", int(isetids[0][i]), " via seq:", lastparseq0[i])
                        self.isetSuccSeq[0][int(parisets[0][i])][int(isetids[0][i])] = lastparseq0[i]

            self.numIsets[0] += 1

        for i in range(y):
            # print("FOR Y")
            # print("selfparids[1]:", selfparids)
            self.isetSucc[1].append([])
            self.isetPar[1].append(0)
            self.playerOfIset[1].append(0)
            self.nactsOnIset[1].append(0)
            self.Iset2Hists[1].append(0)
            isetids[1][i] = int(self.numIsets[1])
            # print("isetids: ", isetids[1][i])
            self.isetPar[1][self.numIsets[1]] = (int(parisets[1][i]), int(iacts[1][i]))
            # print("isetPar: ", self.isetPar[1][self.numIsets[1]])
            self.playerOfIset[1][self.numIsets[1]] = player
            # print("playerOfIset: ", player)

            self.nactsOnIset[1][self.numIsets[1]] = genNactsOnIset(1)
            # print("Num acts on iset: ", self.nactsOnIset[1][self.numIsets[1]])


            ################# KOMWU
            seqs1 = []
            # nxtpars1 = []
            if player == 1:
                if lastparseq1 is not None:
                    # print(lastparseq1)
                    # print("Setting parSeq to be: ", lastparseq1[i])
                    self.parSeq[1][self.numIsets[1]] = lastparseq1[i]
                self.infoSets[1].append(self.numIsets[1]) #self.numInfoSets[1])  # create the infoset

                for ii in range(1, genNactsOnIset(1) + 1):
                    seqs1.append(self.totalSeqs[1] + ii)
                    totseqs1[ii - 1].append(self.totalSeqs[1] + ii)
            #     # self.seqs[1].append(seqs1)
                self.seqs[1][self.numIsets[1]] = seqs1
                self.numInfoSets[1] += 1
                self.totalSeqs[1] += genNactsOnIset(1)
            #################

            # print("TOTAL SEQS1")
            # print(totseqs1)

            if parisets[1][i] >= 0:
                # if True: #player == 0:
                #     # print("****************************")
                #     # print("For parisets:", parisets[1][i], " to get to isetids:", isetids[1][i])
                #     if lastparseq1 is not None:
                #         # print("lastparseq0[i]: i:", i, " value:", lastparseq1[i])
                #     # print("****************************")
                self.isetSucc[1][int(parisets[1][i])].append(int(isetids[1][i]))
                if int(parisets[1][i]) not in self.isetSuccSeq[1] and int(parisets[1][i]) is not None:
                    self.isetSuccSeq[1][int(parisets[1][i])] = {}
                if int(parisets[1][i]) is not None and int(isetids[1][i]) is not None:
                    if lastparseq1 is not None:
                        self.isetSuccSeq[1][int(parisets[1][i])][int(isetids[1][i])] = lastparseq1[i]
            self.numIsets[1] += 1

        # # print("HISTORIES")
        for i in range(x):
            for j in range(y):
                self.isTerminal.append(False)
                self.reward.append((np.inf, -np.inf))

                self.histSucc.append([])
                self.histPar.append(0)
                self.playerOfHist.append(0)
                self.nactsOnHist.append(0)
                self.Hist2Iset[0].append(0)
                self.Hist2Iset[1].append(0)
                histids[i][j] = int(self.numHists)
                # print("histids: ", histids[i][j])
                self.histPar[self.numHists] = (int(parhists[i][j]), int(hacts[i][j]))
                # print("histPar: ", self.histPar[self.numHists])
                self.playerOfHist[self.numHists] = player
                # print("playerOfHis: ", player)
                self.isTerminal[self.numHists] = isTerminal
                self.Hist2Iset[0][self.numHists] = int(isetids[0][i])
                # print("Hist2Iset[0]: ", int(isetids[0][i]))
                self.Hist2Iset[1][self.numHists] = int(isetids[1][j])
                # print("Hist2Iset[1]: ", int(isetids[1][j]))

                if player == 2:
                    if privatecard[0][i][j] == -1:
                        # print("Setting chance prob to: ", np.ones(self.cards) / self.cards)
                        h = np.ones(self.cards) / self.cards
                        # h[0] -= 0.33 # 0.13333333333 * 0.4 < 0.1
                        # h[1] += 0.33
                        self.chanceprob.append(h)
                        # self.chanceprob.append(np.ones(self.cards) / self.cards)

                    elif privatecard[1][i][j] == -1:
                        self.chanceprob.append(np.ones(self.cards))
                        for _a in range(self.cards):
                            if _a == int(privatecard[0][i][j]):
                                self.chanceprob[int(self.numHists)][_a] = 0.0
                                # print("Setting chance prob to: ", 0.0)
                            else:
                                # if _a == 0:
                                #     f = 1.0 / 10000.0
                                # else:
                                #     f = 1.0 / 2.0
                                self.chanceprob[int(self.numHists)][_a] = 1.0  / 2.0
                                # print("Setting chance prob to: ", 1.0 / 2.0)
                                # print("FFFFF: ", _a)
                    # else:
                    #     self.chanceprob.append(np.ones(self.cards))
                    #     for _a in range(self.cards):
                    #         _cnt = 2
                    #         if _a == int(privatecard[0][i][j]):
                    #             _cnt -= 1
                    #         if _a == int(privatecard[1][i][j]):
                    #             _cnt -= 1
                    #         self.chanceprob[int(self.numHists)][_a] = _cnt / (2 * self.cards - 2)
                else:
                    self.chanceprob.append(0)

                """
                if isTerminal:
                    self.nactsOnHist[self.numHists] = 0
                else:
                    self.nactsOnHist[self.numHists] = 1 - (self.bidmaximum - bids[1 - player])
                """
                self.nactsOnHist[self.numHists] = genNactsOnHist()
                # # print("nactsOnHist: ", self.nactsOnHist[self.numHists])
                if parhists[i][j] >= 0:
                    self.histSucc[int(parhists[i][j])].append(int(histids[i][j]))
                # print("Setting parhist: ", parhists[i][j], " to have histSucc: ", int(histids[i][j]))
                self.numHists += 1

        for i in range(x):
            self.Iset2Hists[0][int(isetids[0][i])] = histids[i, :].astype(int)
        # print("Iset2Hists[0] for isetids::", int(isetids[0][i]), " is ", histids[i, :].astype(int))
        for i in range(y):
            self.Iset2Hists[1][int(isetids[1][i])] = histids[:, i].astype(int)
        # print("Iset2Hists[1] for isetids::", int(isetids[1][i]), " is ", histids[:, i].astype(int))

        if isTerminal == True:
            # print("Is terminal. Bids: ", bids)
            for i in range(x):
                for j in range(y):
                    win = 1
                    if quit[0]:
                        # # print("Player 0 Folded")
                        win = -1
                        self.reward[int(histids[i][j])] = (-1.0, 1.0)
                        # print("Reward -1")
                    elif quit[1]:
                        # # print("Player 1 Folded")
                        win = 1
                        self.reward[int(histids[i][j])] = (1.0, -1.0)
                        # print("Reward 1")
                    elif privatecard[0][i][j] == privatecard[1][i][j]:
                        # # print("TIE TIE TIE")
                        win = 0
                        self.reward[int(histids[i][j])] = (0, 0)
                        # print("Reward 0")
                    # elif privatecard[0][i][j] == publiccard:
                    #     win = 1
                    # elif privatecard[1][i][j] == publiccard:
                    #     win = -1
                    # Check the bids to see what to set the reward to
                    #   different for check check and bet call
                    elif privatecard[0][i][j] > privatecard[1][i][j]:
                        # # print("Player 0 Wins. Bids: ", bids, " depth: ", depth)
                        win = 1
                        if depth == 4 or depth == 5:
                            if int(bids[0]) == 1:
                                self.reward[int(histids[i][j])] = (2.0, -2.0)
                                # print("Reward 2")
                            if int(bids[0]) == -1:
                                self.reward[int(histids[i][j])] = (1.0, -1.0)
                                # print("Reward 1")
                    elif privatecard[0][i][j] < privatecard[1][i][j]:
                        win = -1
                        # # print("Player 1 Wins. Bids: ", bids, " depth: ", depth)
                        if depth == 4 or depth == 5:
                            if int(bids[0]) == 1:
                                self.reward[int(histids[i][j])] = (-2.0, 2.0)
                                # print("Reward -2")
                            if int(bids[0]) == -1:
                                self.reward[int(histids[i][j])] = (-1.0, 1.0)
                                # print("Reward -1")
                    # if win == 0:
                    #     self.reward[int(histids[i][j])] = (0, 0)
                    # elif win == 1:
                    #     # # print("Player 0 wins. Bids: ", bids)
                    #     self.reward[int(histids[i][j])] = (bids[1] + 0.5, -bids[1] - 0.5)
                    # elif win == -1:
                    #     # # print("Player 1 wins. Bids: ", bids)
                    #     self.reward[int(histids[i][j])] = (-bids[0] - 1, bids[0] + 1)
                # # print("Just set a terminal")
            return  # histids, isetids

        if obs[0] == False:
            # print("obs[0] == False")
            nxtparhist = np.zeros((x, 0))
            # print("nxtparhist: ", nxtparhist)
            nxthacts = np.zeros((x, 0))
            # print("nxthacts: ", nxthacts)
            nxtpariset = [isetids[0].copy(), np.zeros(0)]
            # print("nxtpariset: ", nxtpariset)
            nxtiacts = [np.zeros(x), np.zeros(0)]
            # print("nxtiacts: ", nxtiacts)

            nxtprivatecard = np.zeros((2, x, 0))
            nxtpubliccard = -1
            nxtbid = -1 * np.ones(2)
            nxtquit = np.zeros(2)

            # print("Looping over self.cards: ", self.cards)
            for i in range(self.cards):
                nxtparhist = np.concatenate((nxtparhist, histids), axis=1)
                # print("nxtparhist: ", nxtparhist)
                nxthacts = np.concatenate((nxthacts, i * np.ones((x, y))), axis=1)
                # print("nxthacts: ", nxthacts)
                nxtpariset[1] = np.concatenate((nxtpariset[1], isetids[1]))
                # print("nxtpariset[1]: ", nxtpariset[1])
                nxtiacts[1] = np.concatenate((nxtiacts[1], i * np.ones(y)))
                # print("nxtiacts[1]: ", nxtiacts[1])

                tmpprivate0 = privatecard[0, :, :]
                tmpprivate1 = i * np.ones((x, y))
                tmpprivate = np.array((tmpprivate0, tmpprivate1))
                nxtprivatecard = np.concatenate((nxtprivatecard, tmpprivate), axis=2)
                # print("nxtprivatecard: ", nxtprivatecard)

            # print("CAL WITH: ", depth)
            # self.genGame(nxtparhist, nxthacts, nxtpariset, nxtiacts, depth + 1, nxtprivatecard, nxtpubliccard, nxtbid,
                         # nxtquit, lastpar, lastseq, lastparseq0, lastparseq1, lastchance=True, isTerminal=False, player=0)

            self.genGame(nxtparhist, nxthacts, nxtpariset, nxtiacts, depth + 1, nxtprivatecard, nxtpubliccard, nxtbid,
                         nxtquit, lastpar, lastseq, None, None, lastchance=True, selfparids=None, isTerminal=False,
                         player=0)


        elif obs[1] == False:

            # print("obs[1] == False")
            nxtparhist = np.zeros((0, y))
            # print("nxtparhist: ", nxtparhist)
            nxthacts = np.zeros((0, y))
            # print("nxthacts: ", nxthacts)
            nxtpariset = [np.zeros(0), isetids[1].copy()]
            # print("nxtpariset: ", nxtpariset)
            nxtiacts = [np.zeros(0), np.zeros(y)]
            # print("nxtiacts: ", nxtiacts)

            nxtprivatecard = np.zeros((2, 0, y))
            # print("nxtprivatecard: ", nxtprivatecard)
            nxtpubliccard = -1
            nxtbid = -1 * np.ones(2)
            nxtquit = np.zeros(2)

            # print("Looping over self.cards: ", self.cards)
            for i in range(self.cards):
                nxtparhist = np.concatenate((nxtparhist, histids), axis=0)
                # print("nxtparhist: ", nxtparhist)
                nxthacts = np.concatenate((nxthacts, i * np.ones((x, y))), axis=0)
                # print("nxthacts: ", nxthacts)
                nxtpariset[0] = np.concatenate((nxtpariset[0], isetids[0]))
                # print("nxtpariset[0]: ", nxtpariset[0])
                nxtiacts[0] = np.concatenate((nxtiacts[0], i * np.ones(x)))
                # print("nxtiacts[0]: ", nxtiacts[0])

                tmpprivate1 = privatecard[1, :, :]
                tmpprivate0 = i * np.ones((x, y))
                tmpprivate = np.array((tmpprivate0, tmpprivate1))
                nxtprivatecard = np.concatenate((nxtprivatecard, tmpprivate), axis=1)
                # print("nxtprivatecard: ", nxtprivatecard)

            # self.genGame(nxtparhist, nxthacts, nxtpariset, nxtiacts, depth + 1, nxtprivatecard, nxtpubliccard, nxtbid,
            #              nxtquit, lastpar, lastseq, lastparseq0, lastparseq1, lastchance=True, isTerminal=False, player=2)

            self.genGame(nxtparhist, nxthacts, nxtpariset, nxtiacts, depth + 1, nxtprivatecard, nxtpubliccard, nxtbid,
                         nxtquit, lastpar, lastseq, None, None, lastchance=True, selfparids=None, isTerminal=False,
                         player=2)



        else:
            nacts = genNactsOnHist()
            # print("Player: ", player, " nacts: ", nacts, " depth: ", depth)

            for i in range(nacts):  # This will always be 2
                # print("Player: ", player, " nacts: ", nacts, " depth: ", depth)
                nxtparhist = histids.copy()
                nxthacts = i * np.ones((x, y))
                nxtpariset = [isetids[0].copy(), isetids[1].copy()]
                nxtiacts = [i * np.ones(x), i * np.ones(y)]

                if player == 0:
                    nxtparseq0 = totseqs0[i].copy()
                else:
                    nxtparseq0 = lastparseq0

                if player == 1:
                    nxtparseq1 = totseqs1[i].copy()
                else:
                    nxtparseq1 = lastparseq1

                # want the seq to
                # nxtparseq0 = seqs0[i]
                # nxtparseq0 = [seqs0[i] * np.ones(x), seqs1[i] * np.ones(y)]

                nxtprivatecard = privatecard.copy()
                nxtpubliccard = -1 # There is no public card in Kuhn

                nxtlastseq = None #[seqs0.copy(), seqs1.copy()]
                nxtlastpar = None #lastpar.copy()  #  parisets.copy()  #[nxtpars0.copy(), nxtpars1.copy()]
                # if player == 0:
                #     nxtlastseq[0] = seqs0[i]
                #     nxtlastpar[0] = nxtpariset[0].copy()
                # elif player == 1:
                #     nxtlastseq[1] = seqs1[i]
                #     nxtlastpar[1] = nxtpariset[1].copy()

                # if player == 2:
                #     nxtpubliccard = i
                # else:
                #     nxtpubliccard = publiccard

                nxtbid = bids.copy()
                # print("bids: ", bids)
                nxtquit = quit.copy()
                nxtisTerminal = False
                # Possible moves are
                #   Bet, check
                #   Call, fold
                # If not chance and is the first action (BET or CALL)

                if player != 2 and i != nacts - 1:
                    # print("First action")
                    if player == 0:
                        if depth == 2:
                            nxtbid[player] = 1
                            # print("Player 0 BET")
                        if depth == 4:
                            nxtbid[player] = nxtbid[1 - player]
                            # nxtquit[player] = True
                            nxtisTerminal = True
                            # print("Player 0 CALL")
                    if player == 1:
                        if depth == 3:
                            if nxtbid[1-player] == 1:
                                nxtbid[player] = nxtbid[1-player]
                                # nxtquit[player] = True
                                nxtisTerminal = True
                                # print("Player 1 CALL")
                            if nxtbid[1-player] == -1:
                                nxtbid[player] = 1
                                # print("Player 1 BET")
                if player != 2 and i == nacts-1:
                    # print("Second action")
                    if player == 0:
                        if depth == 2:
                            nxtbid[player] = -1
                            # print("Player 0 CHECK")
                        if depth == 4:
                            nxtbid[player] = -1
                            nxtquit[player] = True
                            # print("Player 0 FOLD")
                    if player == 1:
                        if depth == 3:
                            if nxtbid[1-player] == 1:
                                nxtbid[player] = -1
                                nxtquit[player] = True
                                # print("Player 1 FOLD")
                            if nxtbid[1-player] == -1:
                                nxtbid[player] = -1
                                nxtisTerminal = True
                                # nxtquit[player] = True
                                # print("Player 1 CHECK")


                    # print("Player not 2 and i = 0")
                    # if nxtbid[1 - player] == -1:
                    #     print("other player not bet, betting 1 BET")
                    #     nxtbid[player] = 1 #
                    # else:
                    #     nxtbid[player] = nxtbid[1 - player]
                    #     print("other player bet, calling CALL")


                # nxtquit = quit.copy()
                # If not chance and is last action
                # if player != 2 and i == nacts - 1:
                #
                #     print("Player not 2 and i = 1")
                #     if nxtbid[1 - player] == -1:
                #         print("other player not bet, nxtbid=-1 CHECK")
                #         nxtbid[player] = -1
                #     else:
                #         print("other player bet, nxtbid=-1 FOLD")
                #         nxtbid[player] = -1
                #         nxtquit[player] = 1
                #     if bids[player] == -1:  # does this make it a check?
                #         bids[player] = 0

                # nxtisTerminal = None
                nxtplayer = None
                if player == 2:
                    nxtisTerminal = False
                    nxtplayer = 0
                elif nxtquit[0] or nxtquit[1]:
                    nxtisTerminal = True
                    # print("nxtisTerminal == True")
                    nxtplayer = 2
                # elif bids[0] == nxtbid[0] and bids[1] == nxtbid[1]:
                # elif nxtbid[0] == -1 and nxtbid[1] == -1 and depth == 3:
                # elif nxtbid[0] == nxtbid[1] and not (nxtbid[0] == -1 or nxtbid[1] == -1):
                #     nxtplayer = 2
                #     nxtisTerminal = True
                    # if publiccard == -1:
                    #     nxtisTerminal = False
                    # else:
                    #     nxtisTerminal = True
                    # print("one of bids and nxtbids are equal:")
                    # print("  0: ", bids[0], nxtbid[0])
                    # print("  1: ", bids[1], nxtbid[1])
                else:

                    # nxtisTerminal = False
                    if nxtisTerminal == True:
                        nxtplayer = 2
                        # print("nxtisTerminal = True, going on with player = 2")
                    else:
                        nxtplayer = 1 - player
                        # print("nxtisTerminal = False, going on with next player")
                # print("CAL WITH: ", depth)

                if player == 2:
                    nxtlastchance = True
                else:
                    nxtlastchance = False

                if selfparids is not None:
                    nxtselfparids = selfparids.copy()
                else:
                    nxtselfparids = [[],[]]
                if player == 0:
                    if nxtselfparids is not None:
                        nxtselfparids[0] = isetids[0]
                if player == 1:
                    if nxtselfparids is not None:
                        nxtselfparids[1] = isetids[1]

                self.genGame(nxtparhist, nxthacts, nxtpariset, nxtiacts, depth + 1, nxtprivatecard, nxtpubliccard,
                             nxtbid, nxtquit, nxtlastpar, nxtlastseq, nxtparseq0, nxtparseq1, lastchance=nxtlastchance, selfparids=nxtselfparids, isTerminal=nxtisTerminal, player=nxtplayer)

# for i in range(5, 11):
#	game = Game(bidmaximum=i)
#	print(i, int(time.time()) % 100000, game.numHists)
# game.printGame()