import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, generateOutcome, exploitability, RegretSolverPlus, RegretSolverDCFR
import time


class LazyCFR:
	def __init__(self, game, Type="regretmatching", thres=0.0, params=None):
		print("initializing solver")
		self.thres = thres
		self.time = 0

		self.game = game

		self.Type = Type
		Solver = None
		if Type == "regretmatching":
			Solver = RegretSolver
			self.DCFR = False
		elif Type == "regretmatchingplus":
			Solver = RegretSolverPlus
			self.DCFR = False
		else:
			Solver = RegretSolverDCFR
			self.DCFR = True

		self.cfvCache = []
		self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))
		self.cfvCache.append(list(map(lambda x: np.zeros(game.nactsOnHist[x]), range(game.numHists))))

		self.probNotUpdated = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
		self.probNotPassed = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
		self.histflag = [-1 * np.ones(game.numHists), -1 * np.ones(game.numHists)]
		self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]

		self.reachp = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]

		self.solvers = []
		self.solvers.append(list(map(lambda x: Solver(game.nactsOnIset[0][x], params), range(game.numIsets[0]))))
		self.solvers.append(list(map(lambda x: Solver(game.nactsOnIset[1][x], params), range(game.numIsets[1]))))

		"""
		"""
		self.stgy = [[], []]
		for i, iset in enumerate(range(game.numIsets[0])):
			nact = game.nactsOnIset[0][iset]
			if game.playerOfIset[0][iset] == 0:
				self.stgy[0].append(np.ones(nact) / nact)
			else:
				self.stgy[0].append(np.ones(0))

		for i, iset in enumerate(range(game.numIsets[1])):
			nact = game.nactsOnIset[1][iset]
			if game.playerOfIset[1][iset] == 1:
				self.stgy[1].append(np.ones(nact) / nact)
			else:
				self.stgy[1].append(np.ones(0))
		self.round = -1
		self.nodestouched = 0
		self.outcome, self.reward = generateOutcome(game, self.stgy)

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
		if parhind != -1 and self.histflag[owner][parhind] < self.round:
			self.passProbOnHist(owner, parhind)

		player = game.playerOfHist[histind]
		stgy = None
		if player == 2:
			stgy = game.chanceprob[histind]
		else:
			isetind = game.Hist2Iset[player][histind]
			stgy = self.solvers[player][isetind].curstgy
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
		game = self.game
		if game.playerOfIset[owner][isetind] == owner:
			sumcfv = np.zeros(game.nactsOnIset[owner][isetind])
			weight = 0
			for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
				sumcfv += self.cfvCache[owner][hind]
				weight = self.probNotUpdated[owner][hind][owner]

				self.cfvCache[owner][hind] *= 0.0
			if owner == 1:
				sumcfv *= -1.0
			if self.DCFR:
				gamma = self.solvers[owner][isetind].gamma
				gamma_weight = (((self.round + 1.0) - 1.0) / (self.round + 1.0)) ** gamma
			else:
				gamma_weight = 1.0
			self.sumstgy[owner][isetind] += self.reachp[owner][isetind] * self.solvers[owner][isetind].curstgy * gamma_weight
			self.solvers[owner][isetind].receive(sumcfv, weight=weight)

		for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
			self.probNotUpdated[owner][hind] = np.zeros(2)
			self.passProbOnHist(0, hind)
			self.passProbOnHist(1, hind)
		nacts = game.nactsOnIset[owner][isetind]
		for innerIind, nxtisetind in enumerate(game.isetSucc[owner][isetind]):
			sumprob = 0.0
			nxthists = game.Iset2Hists[owner][nxtisetind]
			if game.isTerminal[nxthists[0]] == False:
				for nxth in game.Iset2Hists[owner][nxtisetind]:
					sumprob += self.probNotUpdated[owner][nxth][1 - owner]
				if game.playerOfIset[owner][isetind] == owner:
					self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * \
													  self.solvers[owner][isetind].curstgy[innerIind]
				else:
					self.reachp[owner][nxtisetind] += self.reachp[owner][isetind]
				if sumprob > self.thres:
					self.updateIset(owner, nxtisetind)
		self.reachp[owner][isetind] = 0

	def updateAll(self, curexpl, t):
		t1 = time.time()
		game = self.game
		self.round += 1
		if self.Type == "regretmatching" or self.Type == "DCFR":
			self.reachp[0][0] += 1
			self.reachp[1][0] += 1
		else: # "regretmatchingplus"
			self.reachp[0][0] += self.round
			self.reachp[1][0] += self.round
		self.receiveProb(0, 0, np.ones(2))
		self.receiveProb(1, 0, np.ones(2))
		self.updateIset(0, 0)
		self.updateIset(1, 0)

		"""
		def updstgy(owner, iset):
			hists = game.Iset2Hists[owner][iset]
			if game.isTerminal[hists[0]]:
				return
			if self.isetflag[owner][iset] < self.round:
				return
			if game.playerOfIset[owner][iset] == owner:
				self.stgy[owner][iset] = self.solvers[owner][iset].curstgy
			nacts = game.nactsOnIset[owner][iset]
			for a in range(nacts):
				updstgy(owner, game.isetSucc[owner][iset][a])
		updstgy(0, 0)
		updstgy(1, 0)
		"""

		def updateoutcome(hist):
			if game.isTerminal[hist]:
				return
			if self.histflag[0][hist] < self.round:
				return
			self.reward[hist] = 0.0
			nacts = game.nactsOnHist[hist]
			_stgy = None
			player = game.playerOfHist[hist]
			if player == 2:
				_stgy = game.chanceprob[hist]
			else:
				piset = game.Hist2Iset[player][hist]
				_stgy = self.solvers[player][piset].curstgy  # self.stgy[player][piset]
			for a in range(nacts):
				nh = game.histSucc[hist][a]
				updateoutcome(nh)
				self.outcome[hist][a] = self.reward[nh]
				self.reward[hist] += _stgy[a] * self.reward[nh]

		updateoutcome(0)

		self.time += time.time() - t1
		self.getExploitability()
		return self.mike

	def getAvgStgy(self, owner, iset):
		game = self.game
		player = game.playerOfIset[owner][iset]
		if player == owner:
			self.sumstgy[owner][iset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy
		for _i, niset in enumerate(game.isetSucc[owner][iset]):
			if player == owner:
				self.reachp[owner][niset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy[_i]
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

		self.getAvgStgy(0, 0)
		self.getAvgStgy(1, 0)
		stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[0])))
		stgy_prof.append(list(map(lambda _x: avg(_x), self.sumstgy[1])))
		self.mike = stgy_prof
		return exploitability(self.game, stgy_prof)


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















#################################### BELOW WAS MODIFIED #####################################



# import numpy as np
# from LeducHoldem import Game
# import copy
# import queue
# import utils
# from utils import RegretSolver, generateOutcome, exploitability, RegretSolverPlus, generateB
# import time
# from scipy.special import logsumexp
#
#
# class LazyCFR:
# 	def __init__(self, game, Type="regretmatching", thres = 0.0):
# 		print("initializing solver")
# 		self.thres = thres
# 		self.time = 0
#
# 		self.game = game
#
# 		self.Type = Type
# 		Solver = None
# 		if Type == "regretmatching":
# 			Solver = RegretSolver
# 		else:
# 			solver = RegretSolverPlus
#
#
# 		self.cfvCache = []
# 		self.cfvCache.append(list(map(lambda x:  np.zeros(game.nactsOnHist[x]), range(game.numHists))))
# 		self.cfvCache.append(list(map(lambda x:  np.zeros(game.nactsOnHist[x]), range(game.numHists))))
#
# 		# print("HEREE")
# 		# print(game.numIsets[0], game.numIsets[1])
# 		# for x in range(game.numHists):
# 		# 	print(game.nactsOnHist[x], "  player: ", game.playerOfHist[x])
#
# 		self.probNotUpdated = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
# 		self.probNotPassed = [np.zeros((game.numHists, 2)), np.zeros((game.numHists, 2))]
# 		self.histflag = [-1 * np.ones(game.numHists), -1 * np.ones(game.numHists)]
# 		self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]
#
# 		self.reachp = [np.zeros(game.numIsets[0]), np.zeros(game.numIsets[1])]
#
# 		# self.b = [np.zeros(12), np.zeros(12)]
#
# 		# self.b_temp = [[0] * 12 for _ in range(2)]
#
# 		self.solvers = []
# 		self.solvers.append(list(map(lambda x:  RegretSolver(game.nactsOnIset[0][x]), range(game.numIsets[0]))))
# 		self.solvers.append(list(map(lambda x:  RegretSolver(game.nactsOnIset[1][x]), range(game.numIsets[1]))))
#
# 		"""
# 		"""
# 		self.stgy = [[], []]
# 		first = 0
# 		second = 0
# 		for i, iset in enumerate(range(game.numIsets[0])):
# 			nact = game.nactsOnIset[0][iset]
# 			if game.playerOfIset[0][iset] == 0:
# 				first += 1
# 				self.stgy[0].append(np.ones(nact) / nact)
# 			else:
# 				second += 1
# 				# print("HAPPENING HERE: ", game.playerOfIset[0][iset])
# 				self.stgy[0].append(np.ones(0))
# 		# print(first, second)
#
# 		first = 0
# 		second = 0
# 		for i, iset in enumerate(range(game.numIsets[1])):
# 			nact = game.nactsOnIset[1][iset]
# 			if game.playerOfIset[1][iset] == 1:
# 				first += 1
# 				self.stgy[1].append(np.ones(nact) / nact)
# 			else:
# 				second += 1
# 				self.stgy[1].append(np.ones(0))
# 		# print(first, second)
# 		self.round = -1
# 		self.nodestouched = 0
# 		self.outcome, self.reward = generateOutcome(game, self.stgy)
#
# 		# b = generateB(game,self.stgy, 0)
# 		# print("B0")
# 		# print(b)
#
# 		# b = generateB(game, self.stgy, 1)
# 		# print("B1")
# 		# print(b)
#
# 		self.grad = [np.zeros(12), np.zeros(12)]
# 		# self.last_gradient = [np.zeros(12), np.zeros(12)]
#
# 		self.sumstgy = [[], []]
# 		for i, iset in enumerate(range(game.numIsets[0])):
# 			nact = game.nactsOnIset[0][iset]
# 			if game.playerOfIset[0][iset] == 0:
# 				self.sumstgy[0].append(np.ones(nact) / nact)
# 			else:
# 				self.sumstgy[0].append(np.ones(0))
# 		for i, iset in enumerate(range(game.numIsets[1])):
# 			nact = game.nactsOnIset[1][iset]
# 			if game.playerOfIset[1][iset] == 1:
# 				self.sumstgy[1].append(np.ones(nact) / nact)
# 			else:
# 				self.sumstgy[1].append(np.ones(0))
#
# 	def receiveProb(self, owner, histind, prob):
# 		# print("IN RECEIEVEPROB() for owner: ", owner, " histind:", histind)
# 		# print("  Here is before:")
# 		# print("      probNotPassed[", owner, "][", histind, "]:", self.probNotPassed[owner][histind])
# 		# print("      probNotUpdated[", owner, "][", histind, "]:", self.probNotUpdated[owner][histind])
# 		# print("  probNotPassed and probNotUpdated both adding prob:", prob)
# 		self.probNotPassed[owner][histind] += prob
# 		self.probNotUpdated[owner][histind] += prob
# 		# print("  Here is after:")
# 		# print("      probNotPassed[", owner, "][", histind, "]:", self.probNotPassed[owner][histind])
# 		# print("      probNotUpdated[", owner, "][", histind, "]:", self.probNotUpdated[owner][histind])
# 		self.nodestouched += 1
#
# 	def passProbOnHist(self, owner, histind):
# 		game = self.game
# 		if game.isTerminal[histind]:
# 			return
#
# 		# print("Inside passProbOnHist for owner=", owner, " for hist: ", histind)
# 		parhind, pactind = game.histPar[histind]
# 		# THIS DOESN"T SEEM TO GET CALLED EVER
# 		if parhind != -1 and self.histflag[owner][parhind] < self.round:
# 			# other = game.playerOfHist[parhind]
# 			# print("OWNER: ", owner, " OTHER: ", other)
# 			# print("Recursive call with owner=", owner, " for parhind: ", parhind)
# 			self.passProbOnHist(owner, parhind)
#
# 		player = game.playerOfHist[histind]
# 		stgy = None
# 		if player == 2:
# 			stgy = game.chanceprob[histind]
# 		else:
# 			isetind = game.Hist2Iset[player][histind]
# 			stgy = self.solvers[player][isetind].curstgy
#
# 		# print("Looping thru histSucc: ", game.histSucc[histind])
# 		for aind, nxthind in enumerate(game.histSucc[histind]):
# 			tmp = self.probNotPassed[owner][histind].copy()
# 			# print("nxthind: ", nxthind, " probNotPassed: ", tmp, " which is from owner:", owner, " and histind:", histind)
# 			# print("Multiplying probNotPassed by stgy:")
# 			if player == owner:
# 				tmp[owner] *= stgy[aind]
# 				# print("  player: ", player, "= owner:", owner, " tmp is now: ", tmp, " due to stgy[aind]:", stgy[aind])
# 			else:
# 				tmp[1 - owner] *= stgy[aind]
# 				# print("  player: ", player, "!= owner:", owner, " tmp is now: ", tmp, " due to stgy[aind]:", stgy[aind])
# 			# print("Update cfvCache for owner:", owner, " nxthind:", nxthind)
# 			# print("  Also: owner:", owner, " nxthind:", nxthind, " is getting to receiveProb: ", tmp)
# 			# outcome looks like this: [ 0.125 -0.875 -0.875] and tmp[1-owner] like this: 0.3333
# 			self.cfvCache[owner][nxthind] += np.array(self.outcome[nxthind]) * tmp[1 - owner]
#
# 			# If nxthind has any seqs that hit a reward
# 			# apply tmp[1-owner] to the reward and add to b
# 			# Update the b's for just owner using tmp[1-owner]
#
# 			# THIS ONLY HAS LAST_SEQS FOR THE CURRENT OWNER IN UPDATEISET()
# 			# IF YOU PASS THAT IN, THEN YOU CAN UPDATE B'S POTENTIALLY
# 			# if game.isTerminal[nxthind]:# and true_owner == owner:
# 			# 	# if true_owner == 1:
# 			# 	# 	for _ in range(200):
# 			# 	# 		print("LALALALALALALA")
# 			# 	print("TERM2:", nxthind, " OWNER: ", owner)
# 			# 	print("  Last seqs:", last_seqs)
# 			# 	print("  tmp:", tmp)
# 			# 	print("  Reward: ", self.reward[nxthind])
# 			# 	# iset = game.Hist2Iset[owner][histind]
# 			# 	i1 = game.Hist2Iset[owner][histind] #original
# 			# 	i2 = game.Hist2Iset[owner][nxthind]
# 			# 	print("  i1:", i1, " i2:", i2)
# 			# 	print("  This would be seq to update:", self.game.isetSuccSeq[owner][i1][i2])
# 			# 	sign = 1.0
# 			# 	if owner == 1:
# 			# 		sign = -1.0
# 			# 	self.grad[owner][self.game.isetSuccSeq[owner][i1][i2]] += tmp[1-owner] * self.reward[nxthind] * sign
# 				# self.b[player][game.seqs[player][aind]] += tmp[1-owner]
# 			# self.seqs[0][self.numIsets[0]]
#
#
# 			self.receiveProb(owner, nxthind, tmp)
# 		# print("Setting probNotPassed to 0 for owner=",owner, " histind: ", histind)
# 		self.probNotPassed[owner][histind] = np.zeros(2)
# 		self.histflag[owner][histind] = self.round
#
#
# 	def updateIset(self, owner, isetind):
# 		# print("Inside at start of updateIset for owner:", owner, " isetind:", isetind)
# 		# Here, check if the flag is less than self.round - 1
# 		# If it is, then that means it has accumulated prob
#
# 		self.isetflag[owner][isetind] = self.round
# 		# print("Set flag for iset: ", isetind, " to round:", self.round)
# 		game = self.game
# 		# print("Checking to see if playerOfIset: ", game.playerOfIset[owner][isetind], " = Owner: ", owner)
# 		if game.playerOfIset[owner][isetind] == owner:
# 			# print("   player of iset = owner")
# 			sumcfv = np.zeros(game.nactsOnIset[owner][isetind])
# 			weight = 0
# 			# print("About to loop through hists in this iset. Hists: ", game.Iset2Hists[owner][isetind])
# 			for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
# 				sumcfv += self.cfvCache[owner][hind]
# 				weight = self.probNotUpdated[owner][hind][owner]
# 				self.cfvCache[owner][hind] *= 0.0
# 			# print("Using probNotUpdated. Here is what it is: ", weight)
# 			if owner == 1:
# 				sumcfv *= -1.0
# 			self.sumstgy[owner][isetind] += self.reachp[owner][isetind] * self.solvers[owner][isetind].curstgy
# 			self.solvers[owner][isetind].receive(sumcfv, weight=weight)
# 			# print("Done looping through the hists and updating cfvs")
# 		else:
# 			dummy = 0
# 			# print(" player of iset:", game.playerOfIset[owner][isetind], " != owner:", owner, " probNotUpdated NOT used")
#
# 		# print("Starting middle part where looping thru hists again: ", game.Iset2Hists[owner][isetind])
# 		for innerhind, hind in enumerate(game.Iset2Hists[owner][isetind]):
# 			self.probNotUpdated[owner][hind] = np.zeros(2)
# 			# print("Setting probNotUpdate for owner=", owner, " to zero for hist: ", hind)
# 			# These will pass cfv down to next successor and give them probabilities
# 			# print("Calling passProbOnHist for player 0 on hind: ", hind)
# 			self.passProbOnHist(0, hind)
# 			# print("Calling passProbOnHist for player 1 on hind: ", hind)
# 			self.passProbOnHist(1, hind)
#
# 		# print("Finished the middle part for iset: ", isetind)
# 		nacts = game.nactsOnIset[owner][isetind]
#
# 		# print("Start the final part where looping thru isetSuccs: ", game.isetSucc[owner][isetind])
# 		for innerIind, nxtisetind in enumerate(game.isetSucc[owner][isetind]):
# 			sumprob = 0.0
# 			# print("  Successor: ", nxtisetind)
# 			nxthists = game.Iset2Hists[owner][nxtisetind]
# 			if game.isTerminal[nxthists[0]] == False:
# 				# print("  Looping through to sum up other player's probNotUpdated for all of the hists of the infoset ", nxtisetind)
# 				# print("     The hists are:", game.Iset2Hists[owner][nxtisetind])
# 				# print("       This is probNotUpdated[",owner, "][", nxth, "][", 1-owner, "]")
# 				for nxth in game.Iset2Hists[owner][nxtisetind]:
# 					# print("   For hist:", nxth, "probNotUpdated[",owner, "][", nxth, "][", 1-owner, "]: ", self.probNotUpdated[owner][nxth][1- owner])
# 					sumprob += self.probNotUpdated[owner][nxth][1- owner]
# 				# print("    Sum: ", sumprob)
# 				if game.playerOfIset[owner][isetind] == owner:
# 					# THIS IS FOR LAZY CFR
# 					# self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * self.solvers[owner][isetind].curstgy[innerIind]
# 					self.reachp[owner][nxtisetind] += self.reachp[owner][isetind] * self.stgy[owner][isetind][innerIind]  #self.solvers[owner][isetind].curstgy[innerIind]
#
# 				else:
# 					self.reachp[owner][nxtisetind] += self.reachp[owner][isetind]
# 				# print("  Checking if sumprob:", sumprob, " is above thres: ", self.thres)
# 				if sumprob > self.thres:
# 					# print("  It is. Calling updateIset on infoset: ", nxtisetind)
# 					# # Pass the current seqs in this infoset along
# 					# owner_of_iset = game.playerOfIset[owner][nxtisetind]
# 					# print("   Owner of infoset:", owner_of_iset, " owner:", owner, " nxtisetind:", isetind)
# 					# if owner_of_iset != 2 and owner == owner_of_iset:
# 					# 	cur_seqs = game.seqs[owner_of_iset][nxtisetind].copy()
# 					# 	# cur_seq = game.isetSuccSeq
# 					# 	next_seqs = last_seqs.copy()
# 					# 	next_seqs[owner_of_iset] = cur_seqs[innerIind]
# 					#
# 					# else:
# 					# 	next_seqs = last_seqs.copy()
# 					# next_seqs = None
# 					self.updateIset(owner, nxtisetind)
# 				else:
# 					dummy = 0
# 					# print("It isn't.")
#
# 		self.reachp[owner][isetind] = 0
#
#
# 	def updateAll(self):
# 		# print("SELF.B_TEMP BEFORE:", self.b_temp)
# 		t1 = time.time()
# 		game = self.game
# 		self.round += 1
# 		if self.Type == "regretmatching":
# 			self.reachp[0][0] += 1 # These are set to zero in line just above so only ever have values 0 or 1
# 			self.reachp[1][0] += 1
# 		else:
# 			self.reachp[0][0] += self.round
# 			self.reachp[1][0] += self.round
# 		self.receiveProb(0, 0, np.ones(2)) # These set probNotPass and probNotUpdate for history
# 		self.receiveProb(1, 0, np.ones(2)) # to all 1 as in: probNotPassed before:  [0. 0.] prob not passed after:  [1. 1.]
# 		# print("*****************************************")
# 		# print("Calling updateIset for player 0 at iset 0")
# 		# print("*****************************************")
# 		self.updateIset(0, 0)
# 		# print("*****************************************")
# 		# print("Calling updateIset for player 1 at iset 0")
# 		# print("*****************************************")
# 		self.updateIset(1, 0)
#
# 		# print("FINISHED")
# 		# print("NOTPASSED 0")
# 		# for i, x in enumerate(self.probNotPassed[0]):
# 		# 	print(i, x)
# 		# print("NOPASSED 1")
# 		# for i, x in enumerate(self.probNotPassed[1]):
# 		# 	print(i, x)
# 		# print("NOTUPDATED 0")
# 		# for i, x in enumerate(self.probNotUpdated[0]):
# 		# 	print(i, x)
# 		# print("NOTUPDATED 1")
# 		# for i, x in enumerate(self.probNotUpdated[1]):
# 		# 	print(i, x)
# 		# print("cfvCache 0")
# 		# for i, x in enumerate(self.cfvCache[0]):
# 		# 	print(i, x)
# 		# print("cfvCache 1")
# 		# for i, x in enumerate(self.cfvCache[1]):
# 		# 	print(i, x)
#
# 		# print("SELF.B_TEMP AFTER:", self.b_temp)
#
# 		"""
# 		def updstgy(owner, iset):
# 			hists = game.Iset2Hists[owner][iset]
# 			if game.isTerminal[hists[0]]:
# 				return
# 			if self.isetflag[owner][iset] < self.round:
# 				return
# 			if game.playerOfIset[owner][iset] == owner:
# 				self.stgy[owner][iset] = self.solvers[owner][iset].curstgy
# 			nacts = game.nactsOnIset[owner][iset]
# 			for a in range(nacts):
# 				updstgy(owner, game.isetSucc[owner][iset][a])
# 		updstgy(0, 0)
# 		updstgy(1, 0)
# 		"""
#
#
#
# 		def updateoutcome(hist):
# 			if game.isTerminal[hist]:
# 				return
# 			if self.histflag[0][hist] < self.round:
# 				# print("HISTFLAG LOWER: ", hist)
# 				return
# 			self.reward[hist] = 0.0
# 			nacts = game.nactsOnHist[hist]
# 			_stgy = None
# 			player = game.playerOfHist[hist]
# 			if player == 2:
# 				_stgy = game.chanceprob[hist]
# 			else:
# 				piset = game.Hist2Iset[player][hist]
# 				_stgy = self.solvers[player][piset].curstgy #self.stgy[player][piset]
# 			for a in range(nacts):
# 				nh = game.histSucc[hist][a]
# 				updateoutcome(nh)
# 				self.outcome[hist][a] = self.reward[nh]
# 				self.reward[hist] += _stgy[a] * self.reward[nh]
# 		updateoutcome(0)
#
# 		# self.updateKomwu(0)
# 		# self.updateKomwu(1)
# 		# reset grad
# 		# self.grad = [np.zeros(12), np.zeros(12)]
#
# 		self.time += time.time() - t1
#
#
# 	# THIS IS FOR LAZY CFR
# 	def getAvgStgy(self, owner, iset):
# 		game = self.game
# 		player = game.playerOfIset[owner][iset]
# 		if player == owner:
# 			for _ in range(100):
# 				print(self.reachp[owner][iset])
# 			self.sumstgy[owner][iset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy
# 		# print(enumerate(game.isetSucc[owner][iset]))
# 		for _i, niset in enumerate(game.isetSucc[owner][iset]):
# 			if player == owner:
# 				self.reachp[owner][niset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy[_i]
# 			else:
# 				self.reachp[owner][niset] += self.reachp[owner][iset]
# 			self.getAvgStgy(owner, niset)
# 		self.reachp[owner][iset] = 0
#
# 	def getAvgStgy(self, owner, iset):
# 		game = self.game
# 		player = game.playerOfIset[owner][iset]
# 		if player == owner:
# 			# THIS SHOULD BE = not +=
# 			self.sumstgy[owner][iset] += self.stgy[player][iset] * self.reachp[owner][iset] #* self.solvers[owner][iset].curstgy
# 		# print(enumerate(game.isetSucc[owner][iset]))
# 		for _i, niset in enumerate(game.isetSucc[owner][iset]):
# 			if player == owner:
# 				self.reachp[owner][niset] += self.reachp[owner][iset] * self.solvers[owner][iset].curstgy[_i]
# 			else:
# 				self.reachp[owner][niset] += self.reachp[owner][iset]
# 			self.getAvgStgy(owner, niset)
#
# 	# self.reachp[owner][iset] = 0
#
# 	def getExploitability(self):
# 		stgy_prof = []
# 		def avg(_x):
# 			s = np.sum(_x)
# 			l = _x.shape[0]
# 			if s < 1e-5:
# 				return np.ones(l) / l
# 			return _x / s
#
# 		self.getAvgStgy(0, 0)
# 		self.getAvgStgy(1, 0)
# 		stgy_prof.append(list(map( lambda _x: avg(_x), self.sumstgy[0] )))
# 		stgy_prof.append(list(map( lambda _x: avg(_x), self.sumstgy[1] )))
# 		self.mike = stgy_prof
# 		return exploitability(self.game, stgy_prof)
#
# 	# def updateKomwu(self, player):
# 	# 	eta = 1.0
# 	# 	optimistic_gradient = 2.0 * self.grad[player] - 1.0 * self.last_gradient[player]
# 	# 	self.last_gradient[player] = self.grad[player].copy()
# 	# 	self.b[player] += eta * optimistic_gradient
# 	#
# 	# 	K_j = [None] * 22  # This should be num infosets (the actual total not the collected ones) ? Not sure, look into this
# 	# 	# This starts at the bottom and works upwards
# 	# 	# print("FIRST PASS: ", self.game.infoSets[player][::-1])
# 	# 	for infoset_id in self.game.infoSets[player][::-1]:
# 	# 		seq_values = []
# 	# 		for seq in self.game.seqs[player][infoset_id]:
# 	# 			child_values = []
# 	# 			if seq in self.game.childrenInfosets[player]:
# 	# 				for child_infoset in self.game.childrenInfosets[player][seq]:
# 	# 					child_values.append(K_j[child_infoset])
# 	# 			seq_value = self.b[player][seq] + sum(child_values)
# 	# 			seq_values.append(seq_value)
# 	# 		K_j[infoset_id] = logsumexp(seq_values)
# 	#
# 	# 	# print(K_j)
# 	#
# 	# 	# print("SECOND PASS: ", self.game.infoSets[player])
# 	# 	# This starts at the top and works downwards
# 	# 	y = np.zeros(self.total_seqs[player] + 1)
# 	# 	for infoset_id in self.game.infoSets[player]:
# 	# 		for sequence_id in self.new_seqs[player][infoset_id]:
# 	# 			if sequence_id in self.game.parSeq[player]:
# 	# 				y_par = 0.0  # y[self.game.parSeq[player][sequence_id]] THIS GIVES BEHAVE POLICY BY COMMENTING OUT
# 	# 			else:
# 	# 				y_par = 0.0
# 	# 			y[sequence_id] = y_par + self.b[player][sequence_id]
# 	# 			if sequence_id in self.game.childrenInfosets[player]:
# 	# 				child_sum = 0.0
# 	# 				for child in self.game.childrenInfosets[player][sequence_id]:
# 	# 					child_sum += K_j[child]
# 	# 				y[sequence_id] += child_sum
# 	# 			y[sequence_id] -= K_j[infoset_id]
# 	#
# 	# 	# print(np.exp(y))
# 	# 	self.exp_y = np.exp(y)
# 	#
# 	# 	for s in range(len(self.stgy[player])):
# 	# 		x = self.new_seqs[player][s]
# 	# 		if len(x) > 0:
# 	# 			denom = 0.0
# 	# 			for i in x:
# 	# 				denom += self.exp_y[i]
# 	# 			for i, ii in enumerate(x):
# 	# 				self.stgy[player][s][i] = self.exp_y[ii] / denom
#
#
# """
# game = Game(bidmaximum=5)#maximumhists=maximumhists)
# print(game.numHists, game.numIsets)
#
# lazycfr = LazyCFR(game, Type="regretmatchingplus")
#
# T = 10000
# for r in range(T):
# 	lazycfr.updateAll()
#
# 	if r % 300 == 0:
# 		print("time 0", time.time())
# 		print("exploitability:", lazycfr.getExploitability(), lazycfr.time)
# 		print("time 1", time.time())
# """