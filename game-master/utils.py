import copy
import numpy as np


class RegretSolver:
	def __init__(self, dim, params):
		self.round = 0
		self.sumRewVector = np.zeros(dim)
		self.sumStgyVector = np.zeros(dim)
		self.gained = 0
		self.sumWeight = 0.0
		self.dim = dim
		self.curstgy = np.ones(dim) / self.dim
		self.regret = np.zeros(dim)

	# REGRET MATCHING
	# dim IS INITIALIZED TO THE NUMBER OF ACTIONS IN THIS INFOSET
	# BECAUSE THIS EACH INSTANCE OF REGRETMATCHING IS JUST IN ONE INFOSET
	def take(self):
		ret = np.zeros(self.dim)
		for d in range(self.dim):
			if self.sumRewVector[d] > self.gained:  # THIS IS BECAUSE POLICY IS BASED ON POSITIVE REGRET SUMS
				ret[d] = self.sumRewVector[d] - self.gained
		s = sum(ret)
		if s < 1e-8:
			return np.ones(self.dim) / self.dim
		return ret / s

	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:
			stgy = np.array(stgy)
		elif type(stgy) == int:
			stgy = self.take()
		# STGY IS NORMALIZED REGRET MATCHED POLICY AT THIS POINT
		curgain = np.inner(rew, stgy)
		self.gained += curgain
		self.round += 1
		self.sumRewVector += rew
		self.sumStgyVector += self.curstgy * weight # avg policy is cur_pol * rp
		# print("WEIGHTLL", weight)
		self.curstgy = stgy.copy()
		self.sumWeight += weight

	def avg(self):
		if self.sumWeight < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumStgyVector / self.sumWeight

	def regret(self):
		m = -np.inf 
		for i in range(self.dim):
			m = max(m, self.sumRewVector[i])
		return m - self.gained

	def cfrreg(self):
		ret = np.zeros(self.dim)
		whi = np.zeros(self.dim)
		for d in range(self.dim):
			#if self.sumRewVector[d] > self.gained:  # THIS IS BECAUSE POLICY IS BASED ON POSITIVE REGRET SUMS
			ret[d] = self.sumRewVector[d] - self.gained
			if ret[d] < 0:
				# self.sumRewVector[d] = 0
				whi[d] = 1.0
		# s = sum(ret)
		# if s < 1e-8:
		# 	self.sumRewVector = np.zeros(self.dim)
		# 	return False
		return whi
		# return True
		# s = sum(ret)
		# if s < 1e-8:
		# 	return np.ones(self.dim) / self.dim
		# return ret / s

def exploitability(game, stgy_prof):
	def best(owner, iset, probs):
		hists = game.Iset2Hists[owner][iset]
		if game.isTerminal[hists[0]]:
			ret = np.zeros(2)
			for i, h in enumerate(hists):
				tmp = np.array(game.reward[h]) * probs[i]
				ret += tmp

			return ret
		player = game.playerOfIset[owner][iset]
		if player != owner:
			obsnacts = game.nactsOnIset[owner][iset]
			if obsnacts == 1:
				realnacts = game.nactsOnHist[hists[0]]
				nxtprobs = np.zeros(0)
				for i in range(realnacts):
					tmp = np.zeros(len(hists))
					for j, p in enumerate(probs):
						h = hists[j]
						_stgy = None
						if player == 2:
							_stgy = game.chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							_stgy = stgy_prof[player][piset]
						tmp[j] = probs[j] * _stgy[i]
					nxtprobs = np.concatenate((nxtprobs, tmp))
				return best(owner, game.isetSucc[owner][iset][0], nxtprobs)
			else:
				ret = np.zeros(2)
				for i in range(obsnacts):
					nxtprobs = np.zeros(0)
					tmp = np.zeros(len(hists))
					for j, p in enumerate(probs):
						h = hists[j]
						_stgy = None
						if player == 2:
							_stgy = game.chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							_stgy = stgy_prof[player][piset]
						tmp[j] = probs[j] * _stgy[i]
					nxtprobs = np.concatenate((nxtprobs, tmp))
					ret += best(owner, game.isetSucc[owner][iset][i], nxtprobs)
				return ret
		else:
			nacts = game.nactsOnIset[owner][iset]
			ret = -np.inf * np.ones(2)
			for i in range(nacts):
				tmp = best(owner, game.isetSucc[owner][iset][i], probs.copy())
				if tmp[owner] > ret[owner]:
					ret = tmp
			return ret



	b0 = best(0, 0, np.ones(1))
	b1 = best(1, 0, np.ones(1))
	return b0[0] + b1[1]


def generateB(game, stgy_prof, p):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	b = [0] * 1000
	def solve(hist, p, last_seq, cfrp):
		if game.isTerminal[hist]:
			rew[hist] = game.reward[hist][p]
			outcome[hist] = []
			if last_seq >= 0:
				# print(last_seq, hist, p)
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
			srew = solve(game.histSucc[hist][a], p, next_seq, next_cfrp) # get the reward returned here
			outcome[hist].append(srew)           # set the [infoset][history][action] = srew
			rew[hist] += stgy[a] * srew          # sew rew[infoset][history] += stgy[a] * srew

		return rew[hist]						 # return that value

	solve(0, p, -1, 1)
	# print(ff)
	# print("Here")
	# print(outcome[0], rew[0])
	return b #outcome, rew


def updateB(game, stgy_prof, p):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	b = [0] * 50
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
			srew = solve(game.histSucc[hist][a], p, next_seq, next_cfrp) # get the reward returned here
			outcome[hist].append(srew)           # set the [infoset][history][action] = srew
			rew[hist] += stgy[a] * srew          # sew rew[infoset][history] += stgy[a] * srew

		return rew[hist]						 # return that value

	solve(0, p, -1, 1)
	# print(ff)
	# print("Here")
	# print(outcome[0], rew[0])
	return np.array(b) #outcome, rew





def generateOutcome(game, stgy_prof):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	print("GENERATE OUTCOME() called")
	def solve(hist):
		# all this does is returns the reward
		if game.isTerminal[hist]:
			rew[hist] = game.reward[hist][0]
			# print(game.reward[hist])
			outcome[hist] = []
			# print("Terminal: ", hist, " reward: ", game.reward[hist][0])
			return rew[hist]
		outcome[hist] = []
		player = game.playerOfHist[hist]
		stgy = None
		if player < 2:
			iset = game.Hist2Iset[player][hist]
			stgy = stgy_prof[player][iset]
		else:
			stgy = game.chanceprob[hist]
		# print("Player:", player, " hist:", hist, " looping through actions:")
		for a in range(game.nactsOnHist[hist]):
			# print("Taking action ", a, " in hist:", hist)
			srew = solve(game.histSucc[hist][a]) # get the reward returned here
			# print("srew return=", srew, " from taking action ", a, " in hist: ", hist)
			# print("Setting outcome for action ", a, " to be srew=", srew)
			# Note: regret[a] = outcome[a] - rew
			outcome[hist].append(srew)  # children utility of action
			# print("Multiplying srew=", srew, " by stgy[", a, "]:", stgy[a], " to get: ", stgy[a] * srew)
			rew[hist] += stgy[a] * srew  # state (expected) value
		# print("Expected reward rew[", hist, "]: ", rew[hist])

		return rew[hist]						 # return that value

	solve(0)
	# print("Here")
	# print(outcome[0], rew[0])
	# print("OUTCOME")
	# print(outcome)
	return outcome, rew

def generateOutcomeFinal(game, stgy_prof):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	print("GENERATE OUTCOME FINAL")
	def solve(hist):
		# all this does is returns the reward
		if game.isTerminal[hist]:
			rew[hist] = game.reward[hist][0]
			# print(game.reward[hist][0])
			outcome[hist] = []
			# print("Terminal: ", hist, " reward: ", game.reward[hist][0])
			return rew[hist]
		outcome[hist] = []
		player = game.playerOfHist[hist]
		stgy = None
		if player < 2:
			iset = game.Hist2Iset[player][hist]
			stgy = stgy_prof[player][iset]
		else:
			stgy = game.chanceprob[hist]
		# print("Player:", player, " hist:", hist, " looping through actions:")
		for a in range(game.nactsOnHist[hist]):
			# print("Taking action ", a, " in hist:", hist)
			srew = solve(game.histSucc[hist][a]) # get the reward returned here
			# print("srew return=", srew, " from taking action ", a, " in hist: ", hist)
			# print("Setting outcome for action ", a, " to be srew=", srew)
			# Note: regret[a] = outcome[a] - rew
			outcome[hist].append(srew)  # children utility of action
			# print("Multiplying srew=", srew, " by stgy[", a, "]:", stgy[a], " to get: ", stgy[a] * srew)
			rew[hist] += stgy[a] * srew  # state (expected) value
		# print("Expected reward rew[", hist, "]: ", rew[hist])

		return rew[hist]						 # return that value

	solve(0)
	print("Here: ", rew[0])
	# print(outcome[0], rew[0])
	# print("OUTCOME")
	# print(outcome)
	return outcome, rew


def generateChildren(game, stgy_prof, p):
	outcome = np.zeros(game.numHists).tolist()
	rew = np.zeros(game.numHists)
	b = {}
	def solve(hist, p, last_par, cfrp, game):
		if game.isTerminal[hist]:
			rew[hist] = game.reward[hist][p]
			outcome[hist] = []
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
			if player == 2 or player != p:
				next_cfrp = cfrp * stgy[a]
				next_last_par = last_par
			else:
				next_cfrp = cfrp
				if last_par != None:
					if last_par not in b:
						b[last_par] = []
					if iset not in b[last_par]:
						b[last_par].append(iset)
				next_last_par = game.seqs[p][iset][a]

			srew = solve(game.histSucc[hist][a], p, next_last_par, next_cfrp, game) # get the reward returned here
			outcome[hist].append(srew)           # set the [infoset][history][action] = srew
			rew[hist] += stgy[a] * srew          # sew rew[infoset][history] += stgy[a] * srew

		return rew[hist]						 # return that value

	solve(0, p, None, 1, game)
	# print(ff)
	# print("Here")
	# print(outcome[0], rew[0])
	return b #outcome, rew




class RegretSolverPlus:
	def __init__(self, dim, params):
		self.round = 0
		self.sumRewVector = np.zeros(dim)
		self.sumStgyVector = np.zeros(dim)
		self.sumQ = np.zeros(dim)
		self.gained = 0
		self.sumWeight = 0.0
		self.dim = dim
		self.curstgy = np.ones(dim) / self.dim
		
	def take(self):
		s = sum(self.sumQ)
		if s < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumQ / s

	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:
			stgy = np.array(stgy)
		elif type(stgy) == int:
			stgy = self.take()
		curgain = np.inner(rew, stgy)
		for i in range(self.dim):
			self.sumQ[i] += rew[i] - curgain
			self.sumQ[i] = max(self.sumQ[i], 0)
		self.gained += curgain
		self.round += 1
		self.sumRewVector += rew
		self.sumStgyVector += self.curstgy *  weight
		self.curstgy = stgy.copy()
		self.sumWeight += weight

	def avg(self):
		if self.sumWeight < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumStgyVector / self.sumWeight

	def regret(self):
		m = -np.inf 
		for i in range(self.dim):
			m = max(m, self.sumRewVector[i])
		return m - self.gained


class RegretSolverDCFR:
	def __init__(self, dim, params):
		self.round = 0
		self.sumRewVector = np.zeros(dim)
		self.sumStgyVector = np.zeros(dim)
		self.sumQ = np.zeros(dim)
		self.gained = 0
		self.sumWeight = 0.0
		self.dim = dim
		self.curstgy = np.ones(dim) / self.dim
		self.alpha = params[0]
		self.beta = params[1]
		self.gamma = params[2]

	def take(self):
		s = sum(self.sumQ)
		if s < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumQ / s

	def receive(self, rew, stgy=0, weight=1.0):
		if type(stgy) == list:
			stgy = np.array(stgy)
		elif type(stgy) == int:
			stgy = self.take()
		curgain = np.inner(rew, stgy)
		self.round += 1
		for i in range(self.dim):
		# 	if self.sumQ[i] >= 0:
		# 		t = self.round + 1.0
		# 		self.alpha = 1.5
		# 		# self.sumQ[i] *= (t ** self.alpha) / ((t**self.alpha) + 1.0) # Openspiel version
		# 		# info_state.cumulative_regret[action] *= (
		# 		# 		self._iteration ** self.alpha /
		# 		# 		(self._iteration ** self.alpha + 1))
		# 	else:
		# 		t = self.round + 1.0
		# 		self.beta = 0
		# 		# self.sumQ[i] *= (t ** self.beta) / ((t ** self.beta) + 1.0)  # Openspiel version


			# TODO THINK IT CURRENTLY IS:
			#   CONTRIB TO AVG STRAT IS T^2 (THAT'S THERE)
			#   BELOW CLIPS REGRETS
			#  SO ABOVE IS ONLY PART THAT IS WRONG?

			self.sumQ[i] += rew[i] - curgain
			self.sumQ[i] = max(self.sumQ[i], 0)


		self.gained += curgain
		self.sumRewVector += rew
		self.sumStgyVector += self.curstgy * weight
		self.curstgy = stgy.copy()
		self.sumWeight += weight

	def avg(self):
		if self.sumWeight < 1e-8:
			return np.ones(self.dim) / self.dim
		return self.sumStgyVector / self.sumWeight

	def regret(self):
		m = -np.inf
		for i in range(self.dim):
			m = max(m, self.sumRewVector[i])
		return m - self.gained
