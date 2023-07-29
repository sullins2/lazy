import numpy as np
# from LeducHoldem_with_KOMWU import Game
from RandomGame import Game
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

Type = "regretmatching"
# Type = "regretmatchingplus"
# Type = "dcfr"
dcfr_params = [1.5, 0.0, 2.0]

# This is for all Leduc
if betm>7:
	game = Game(path=savepath+".npz")
else:
	betm = 6
	game = Game( bidmaximum =betm) #path=savepath+".npz")#bidmaximum=betmpath=

# This is for Kuhn - doesn't have betm..
# if betm>7:
# 	game = KuhnGame(path=savepath+".npz")
# else:
# 	game = KuhnGame( bidmaximum =betm)

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




# LEDUC-3 works well
# params = {}
# params["thres"] = -0.008 #0.004#0.01 #0.06
# params["entropy"] = -6.0
# params["mod_value"] = 1000
# params["KL"] = -3.0
# params["KL_mod"] = 1000
# params["optimism"] = 2.0
# params["eta"] = 1.0
# params["entropy_twice"] = False
# params["final_exploit"] = 1e-12
# params["AMMO"] = 500000


# figure out why rewards change results
# Story: If you can get last iterate convergence, this helps. Otherwise, doesn't help average policy
# BEER GAME - for early fall
params = {}
params["thres"] = -0.008
params["entropy"] = 0#-1
params["mod_value"] = 10000
params["KL"] = 0 #-400.0
params["KL_mod"] = 10000
params["optimism"] = 2.0
params["eta"] = 20.0
params["entropy_twice"] = False
params["final_exploit"] = 1e-12
params["AMMO"] = 500000
params["b_count"] = 30 # Set to negative to disable
params["b_count_count_at"] = 15

# 10 2 4.26
# 20 10 2.34
# 20 5 3.15
# 25 10 2.50
# 30 15 2.29
# 20 1

# DOES GOOD ON LEDUC-4
# params = {}
# params["thres"] = -0.01
# params["entropy"] = -3.1
# params["mod_value"] = 1000
# params["KL"] = -3.0 #-1.0
# params["KL_mod"] = 1000
# params["optimism"] = 2.0
# params["eta"] = 20.0
# params["entropy_twice"] = False
# params["final_exploit"] = 1e-12
# params["AMMO"] = 500000


#  FORGET KUHN
# params = {}
# params["thres"] = -10.008 #0.004#0.01 #0.06
# params["entropy"] = 0#-0.2
# params["mod_value"] = 100
# params["KL"] = 0#-3.0
# params["KL_mod"] = 1000
# params["optimism"] = 2.0
# params["eta"] = 5.0
# params["entropy_twice"] = False
# params["final_exploit"] = 1e-12
# params["AMMO"] = 500000

# WORKS WELL FOR LEDUC-5
# params = {}
# params["thres"] = 0.008
# params["entropy"] = -5.0
# params["mod_value"] = 1000
# params["KL"] = -3.0
# params["KL_mod"] = 1000
# params["optimism"] = 2.0
# params["eta"] = 20.0
# params["entropy_twice"] = False
# params["final_exploit"] = 1e-12
# params["AMMO"] = 500000


# params = {}
# params["thres"] = 0.008
# params["entropy"] = -5.0
# params["mod_value"] = 1000
# params["KL"] = -3.0
# params["KL_mod"] = 1000
# params["optimism"] = 2.0
# params["eta"] = 20.0
# params["entropy_twice"] = False
# params["final_exploit"] = 1e-12
# params["AMMO"] = 500000



print("params: ", params)


def run(game, path="result", Type="regretmatching", solvername = "cfr"):
	thres = params["thres"]
	# TODO THIS IS WITHOUT ANY ENT OR KL
	#  BETM=3 max=0.04
	#  BETM=4 max=0.01
	#  BETM=5 max=0.01-0.008
	#  BETM=6
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
		quit = False
		Z = 2000
		while z <= Z: #: 0000000: #cumutime + time.time() - timestamp < timelim or gamesolver.nodestouched < minimum:
			z += 1
			plot_its.append(z)
			rounds += 1
			if z % PLOT_ITERS == 0:
				curexpl = gamesolver.getExploitability()
				expl_plot.append(curexpl)
				# expl_plot.append(gamesolver.b[0][0])
				expl_iters.append(gamesolver.nodestouched)
				if curexpl < params["final_exploit"] and quit == False:
					cumutime += time.time() - timestamp
					print("TIME: ", cumutime, "EXPLOIT: ", curexpl, "nodestouched:", gamesolver.nodestouched)
					print("z = ", z)
					z = Z - 10
					quit = True
				ITERS += 1
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
			curexpl = gamesolver.getExploitability()
			stgy = gamesolver.updateAll(curexpl, t=z)
			if z % 15 == 0: #time.time() - timestamp > reporttimes[betm]:
				cumutime += time.time() - timestamp
				expl = gamesolver.getExploitability()
				tmpresult = (expl, cumutime, gamesolver.nodestouched)
				print("solvername", solvername, Type, "game", savepath, "betm", betm, "expl", expl, "time", cumutime, "nodestouched", gamesolver.nodestouched, "thres", thres, "rounds", rounds)
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
		# for i in range(10, 0, -1):
		# 	print("EXL:", expl_plot[-i])
		# print("EXPLOTS:", expl_plot[-1])

		# LOG PLOT (NOT WORKING)
		# plt.xlabel('Nodes Touched')
		# plt.ylabel('Exploitability')
		# plt.xscale('log')  # Set x-axis to log scale
		# plt.yscale('log')  # Set y-axis to log scale
		# This breaks the plot
		# plt.xlim(1e0, 1e5)  # Set x-axis range from 1 to 10^5
		# plt.ylim(1e-12, 1e1)  # Set y-axis range from 10^-12 to 1
		# plt.plot(expl_iters, expl_plot, 'b-', label='Lazy-KFLBR')
		# plt.legend()
		# plt.show()

		plt.plot(expl_iters, expl_plot, 'b-', label='None')
		plt.xlabel('Nodes Touched')
		plt.ylabel('Exploitability')
		# Set the axis scale to logarithmic
		plt.xscale('log')
		plt.yscale('log')
		# Set the axis limits
		plt.xlim(1, 1e7)  # x-axis limits from 1 to 100000 (10^5)
		plt.ylim(1e-8, 1)  # y-axis limits from 10^-12 to 1

		# plt.ylim(0.0, 1.0)  # Set y-axis range
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

		expl = gamesolver.getExploitability()
		print("Last expl:", expl)

		print("shape", len(expls), len(times), len(nodes))
		# print("HERE: ", gamesolver.stgy[0])
		# print("THIS: ", gamesolver.stgy[0][17])
		print(gamesolver.stgy[0][21:61])
		print("JACK:", gamesolver.stgy[0][4][0], "OTHER:", gamesolver.stgy[0][20][0], "JACK+0.3333", gamesolver.stgy[0][4][0] + 0.33333333 )
		print("DIF: ", np.abs(gamesolver.stgy[0][4][0] + 0.333333333333333333 - gamesolver.stgy[0][20][0]))


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
		solver = Lazycfr_nocomments.LazyCFR(game, Type=Type, thres=thres, params=params)
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
