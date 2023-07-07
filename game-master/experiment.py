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



betm=1  #MM10
# if args.betm:
# 	betm = args.betm
savepath = "leduc_3_"+str(betm)

# algo="cfr"
# algo="lazycfr"
# algo="komwu"
# algo="lazyflbr"
algo="lazycfr_nocomments"  # This is KOMWU
# if args.algo:
# 	algo = args.algo#"cfr"
# algo="cfr"
# Type = "regretmatchingplus"
Type = "regretmatching"
if args.Type:
	Type=args.Type

# CHECK EXPLOIT CALC CODE
# LAZY FLBR -0.001 around 21-22.5m
# LAZY FLBR with ANY thres doesn't work yet for betm=4

# print(algo, Type, savepath)

# This is for all Leduc
# if betm>7:
# 	game = Game(path=savepath+".npz")
# else:
# 	betm = 3
# 	game = Game( bidmaximum =betm) #path=savepath+".npz")#bidmaximum=betmpath=

# This is for Kuhn
if betm>7:
	game = KuhnGame(path=savepath+".npz")
else:
	game = KuhnGame( bidmaximum =betm)#path=savepath+".npz")#bidmaximum=betmpath=

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
	# thres = -0.0045 #5  # Lazy-CFR uses 0.1 for larger games
	thres = -0.01
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
		z = 2
		lastexpl = 0.0
		plot_its = []
		stgy = None
		while z <= 40: #: 0000000: #cumutime + time.time() - timestamp < timelim or gamesolver.nodestouched < minimum:
			z += 1
			plot_its.append(z)
			rounds += 1
			if z % PLOT_ITERS == 0:
				curexpl = solver.getExploitability()
				expl_plot.append(curexpl)
				expl_iters.append(solver.nodestouched)
				# ITERS += 1
			if z % 300 == 0: #rounds % printround[betm]== 0:
				curexpl = solver.getExploitability()
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
		# plt.plot(expl_iters, expl_plot, 'b-', label='Lazy-KFLBR')
		# plt.xlabel('Nodes Touched')
		# plt.ylabel('Exploitability')
		# plt.ylim(0.0, 1.0)  # Set y-axis range
		# plt.legend()
		# plt.show()

		# Plot entropy
		# plt.plot(plot_its, solver.total_entropy[0], 'b-', label='player0')
		# plt.plot(plot_its, solver.total_entropy[1], 'r-', label='player1')
		# plt.xlabel('Iteration')
		# plt.ylabel('Total Entropy')
		# plt.ylim(0.0, 1.0)  # Set y-axis range
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
			solver = cfr.CFR(game, Type=Type)
	if solvername == "mccfr":
		solver = mccfr.MCCFR(game, Type=Type)
	if solvername == "lazycfr_nocomments":
		solver = Lazycfr_nocomments.LazyCFR(game, Type=Type, thres=thres)
	if solvername == "lazycfr":
		# if args.thres:
		# 	thres = args.thres
		# THIS IS FOR LAZY CFR solver = Lazycfr.LazyCFR(game, Type=Type, thres =thres)
		solver = Lazycfr.LazyCFR(game, Type=Type, thres=thres)
	if solvername == "lazyflbr":
		solver = Lazycfr_FLBR.LazyCFR(game, Type=Type, thres=thres)
	if solvername == "komwu":
		solver = Komwu.KOMWU(game, Type=Type, thres=thres)

	res = solve(solver)
	return res

# res = run(game, Type=Type, solvername=algo)

algo = "cfr"
game = Game( bidmaximum=3)
# game = KuhnGame( bidmaximum =betm)
res = run(game, Type=Type, solvername=algo)
stgy0 = res[3]
expl0 = res[4]
r0, v0 = generateOutcome(game, stgy0)
# print(stgy0)

algo = "lazycfr_nocomments"
game = Game( bidmaximum=3)
# game = KuhnGame( bidmaximum =betm)
res = run(game, Type=Type, solvername=algo)
stgy1 = res[3]
expl1 = res[4]
r1, v1 = generateOutcome(game, stgy1)

stgy_prof = [stgy0[0], stgy1[1]]

r, v = generateOutcome(game, stgy_prof)

print("Exploit0:", expl0, "Exploit1:", expl1)
print("First:, ", r0[0], v0[0])
print("Second:, ", r1[0], v1[0])
print("Against:", r[0], v[0])


# expl_iters = [t for t in range(150)]
# plt.plot(expl_iters, stgy0, 'b-', label='CFR')
# plt.plot(expl_iters, stgy0, 'r-', label='KOMWU')
# plt.xlabel('Iteration')
# plt.ylabel('Policy')
# plt.ylim(0.0, 1.0)  # Set y-axis range
# plt.legend()
# plt.show()

