import numpy as np
from LeducHoldem_with_KOMWU import Game
from Kuhn_with_KOMWU import KuhnGame
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, generateB, generateOutcomeFinal
import time
import matplotlib.pyplot as plt

import mccfr
import Lazycfr
import Lazycfr_FLBR # FLBR # _nocomments
import Lazycfr_nocomments
import cfr
import cfrnoprune
import Komwu
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--betm", type=int)
parser.add_argument("--Type", type=str)
parser.add_argument("--algo", type=str)
parser.add_argument("--thres", type=float)
parser.add_argument("--noprune", type=int)
args = parser.parse_args()

betm=1
savepath = "leduc_3_"+str(betm)

# algo="cfr"
# algo="lazycfr"
# algo="komwu"
# algo="lazyflbr"
algo="lazycfr_nocomments"  # This is KOMWU

# Type = "regretmatching"
# Type = "regretmatchingplus"
Type = "dcfr"
dcfr_params = [1.5, 0.0, 2.0]


# TODO check exploit calc code


# This is for all Leduc
if betm>7:
	game = Game(path=savepath+".npz")
else:
	betm = 8
	game = Game( bidmaximum =betm) #path=savepath+".npz")#bidmaximum=betmpath=

test = 1
# This is for Kuhn - doesn't have betm..
# if betm>7:
# 	game = KuhnGame(path=savepath+".npz")
# else:
# 	game = KuhnGame( bidmaximum =betm)#path=savepath+".npz")#bidmaximum=betmpath=

# print("-----------------------------")
# print(game.infoSets[0])
# print(game.totalSeqs[0])
# print(game.seqs[0])
# print(game.numInfoSets[0])
# print(game.totalSeqs[0])
# print(game.childrenInfosets[0])
# print(game.parSeq)
# print(game.isetSuccSeq[0])
# print("-----------------------------")
# print("-----------------------------")
# print(game.infoSets[1])
# print(game.totalSeqs[1])
# print(game.seqs[1])
# print(game.numInfoSets[1])
# print(game.totalSeqs[1])
# print(game.childrenInfosets[1])
# print(game.parSeq)
# print(game.isetSuccSeq[1])
# print("-----------------------------")

print("initializing game")
print("game info:", game.numHists, game.numIsets, algo, Type)

reporttimes=[10, 10, 10, 10, 10, 20, 60, 100, 200, 300, 600, 500]
printround=[10000, 8000, 6000, 4000, 2000, 100, 50, 200, 100, 50, 1, 1]

# CHECK EXPLOITABILITY CALC CODE

def run(game, path="result", Type="regretmatching", solvername = "cfr"):
	thres = 0.001 #5  # Lazy-CFR uses 0.1 for larger games
	# TODO COMPARE WITH 0.04
	#  0.05 works ok, but it doesn't get lower exploit very fast
	#  0.04 is fast and exploit -> 0
	#thres = 0.01 #0.008 #0.004
	def solve(gamesolver, reporttime=60, timelim = 30000, minimum=0):
		expl_plot = []
		expl_iters = []
		ITERS = 1  # Total plot iterations
		PLOT_ITERS = 1  # How often to record exploit for plotting
		curexpl = 100
		cumutime = 0
		timestamp = time.time()
		result = []
		expls = []
		times = []
		nodes = []
		rounds = 0
		z = 1
		lastexpl = 0.0
		plot_its = []
		stgy = None
		Z = 170
		while z <= Z: #: 0000000: #cumutime + time.time() - timestamp < timelim or gamesolver.nodestouched < minimum:
			z += 1
			plot_its.append(z)
			rounds += 1
			if z % PLOT_ITERS == 0:
				curexpl = gamesolver.getExploitability()
				expl_plot.append(curexpl)
				expl_iters.append(solver.nodestouched)
				# ITERS += 1
			if z % 300 == 0: #rounds % printround[betm]== 0:
				curexpl = gamesolver.getExploitability()
				# print(solver.mike[0]) #
				generateOutcomeFinal(game, solver.mike)# self.stgy) #
				print("round ", rounds, "time", time.time() - timestamp, cumutime,  betm, solvername, Type, "betm", betm , "thres", thres, "expl", curexpl)
				# if curexpl < 0.1: #1e-14: #0.001:
				# 	print("curexpl: ", curexpl, " nodestouched:", gamesolver.nodestouched)
				# 	z = np.inf
				# 	if solver.round > 6000:
				# 		break
			stgy = gamesolver.updateAll()
			if time.time() - timestamp > reporttimes[betm]:
				cumutime += time.time() - timestamp
				expl = gamesolver.getExploitability()
				tmpresult = (expl, cumutime, gamesolver.nodestouched)
				print("solvername", solvername, Type, "game", savepath, "expl", expl, "time", cumutime, "nodestouched", gamesolver.nodestouched, "thres", thres, "rounds", rounds)
				result.append(tmpresult)
				expls.append(expl)
				times.append(cumutime)
				nodes.append(gamesolver.nodestouched)
				timestamp = time.time()
				res_path = savepath+"_"+solvername+"_"+Type
				if solvername == "lazycfr":
					res_path+="_"+str(thres)
				if solvername == "cfr":
					res_path+="_"+str(args.noprune)
				# np.savez(res_path, expl = expls, times = times, nodes = nodes)

		# Plot exploitability
		plt.plot(expl_iters, expl_plot, 'b-', label='Lazy-KFLBR')
		plt.xlabel('Nodes Touched')
		plt.ylabel('Exploitability')
		plt.ylim(0.0, 1.0)  # Set y-axis range
		plt.legend()
		plt.show()

		# Plot optimism levels throughout run
		# plt.plot(expl_iters, gamesolver.opt_levels, 'b-', label='Lazy-KFLBR')
		# plt.xlabel('Nodes Touched')
		# plt.ylabel('Exploitability')
		# plt.ylim(0.0, 40.0)  # Set y-axis range
		# plt.legend()
		# plt.show()

		# Plot entropy
		# plt.plot(plot_its, solver.total_entropy[0], 'b-', label='player0')
		# plt.plot(plot_its, solver.total_entropy[1], 'r-', label='player1')
		# plt.xlabel('Iteration')
		# plt.ylabel('Total Entropy')
		# plt.ylim(0.0, 40000.0)  # Set y-axis range
		# plt.legend()
		# # This adds text to display the value of the last iteration on the plot
		# last_value_idx = -1
		# last_value_player0 = solver.total_entropy[0][last_value_idx]
		# last_value_player1 = solver.total_entropy[1][last_value_idx]
		# plt.annotate(f'{last_value_player0:.2f}', xy=(plot_its[last_value_idx], last_value_player0), xytext=(10, 10),
		# 			 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
		# plt.annotate(f'{last_value_player1:.2f}', xy=(plot_its[last_value_idx], last_value_player1), xytext=(10, -20),
		# 			 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
		# plt.show()

		print("shape", len(expls), len(times), len(nodes))
		# print("HERE: ", stgy)
		expl = gamesolver.getExploitability()
		return (expls, times, nodes, stgy, expl)
	print("initializing solver")
	solver = None
	if solvername == "cfr":
		if args.noprune:
			solver = cfrnoprune.CFR(game, Type=Type)
		else:
			solver = cfr.CFR(game, Type=Type, params=dcfr_params)
	if solvername == "mccfr":
		solver = mccfr.MCCFR(game, Type=Type)
	if solvername == "lazycfr_nocomments": # THIS IS KOMWU
		solver = Lazycfr_nocomments.LazyCFR(game, Type=Type, thres=thres)
	if solvername == "lazycfr":
		solver = Lazycfr.LazyCFR(game, Type=Type, thres=thres, params=dcfr_params)
	if solvername == "lazyflbr":
		solver = Lazycfr_FLBR.LazyCFR(game, Type=Type, thres=thres)
	if solvername == "komwu":
		solver = Komwu.KOMWU(game, Type=Type, thres=thres)

	res = solve(solver)
	return res

res = run(game, Type=Type, solvername=algo)


#-----------------------------------
# This is to calculate game value between two algs
# TURN OFF THE LINE ABOVE OR IT WILL RUN 3 TIMES
# algo = "lazycfr"
# Type = "regretmatchingplus"
# # Type = "dcfr"
# game = Game(bidmaximum=5)
# # game = KuhnGame( bidmaximum =betm)
# res = run(game, Type=Type, solvername=algo)
# stgy0 = res[3]
# expl0 = res[4]
# r0, v0 = generateOutcome(game, stgy0)
#
# algo = "lazycfr_nocomments"
# game = Game(bidmaximum=5)
# # game = KuhnGame( bidmaximum =betm)
# res = run(game, Type=Type, solvername=algo)
# stgy1 = res[3]
# expl1 = res[4]
# r1, v1 = generateOutcome(game, stgy1)
#
# stgy_prof = [stgy0[0], stgy1[1]]
#
# r, v = generateOutcome(game, stgy_prof)
# print("Exploit0:", expl0, "Exploit1:", expl1)
# print("First gv:, ", v0[0])
# print("Second gv:, ", v1[0])
# print("Against gv:",  v[0])
#------------------------------------

# expl_iters = [t for t in range(150)]
# plt.plot(expl_iters, stgy0, 'b-', label='CFR')
# plt.plot(expl_iters, stgy0, 'r-', label='KOMWU')
# plt.xlabel('Iteration')
# plt.ylabel('Policy')
# plt.ylim(0.0, 1.0)  # Set y-axis range
# plt.legend()
# plt.show()


# ------------- CHECK LATER, PUT BACK IN
# if args.betm:
# 	betm = args.betm
# if args.algo:
# 	algo = args.algo
# if args.Type:
# 	Type=args.Type
