import math
memo = {0:0}
set_coins ={1,4,5}
memo = {i:1 for i in set_coins}
def min_coin_2_sum(set_coins, target, memo):
    if target in memo:
        return memo[target]
    print(set_coins,target,memo)
    sol = 1+min([min_coin_2_sum(set_coins,target-decision,memo) if target >= decision else math.inf for decision in set_coins])
    #print(sol)
    memo[target] = sol
    return sol
min_coin_2_sum(set_coins,1,memo)
min_coin_2_sum(set_coins,3,memo)
min_coin_2_sum(set_coins,13,memo)
min_coin_2_sum(set_coins,43,memo)


#01 knapsack
memo={}
import copy
def best_val_w(vis,wis,maxw,memo):
    if maxw in memo:
        return memo[maxw]

    all_decisions = []

    for i,vw in enumerate(zip(vis,wis)):
        vi,wi = vw
        if maxw - wi < 0:
            result = -1*math.inf
            all_decisions.append(result)
        else:
            new_v = copy.deepcopy(vis)
            new_w = copy.deepcopy(wis)
            new_v.pop(i)
            new_w.pop(i)

            result = vi+best_val_w(new_v,new_w,maxw-wi,memo)
            all_decisions.append(result)
    best = max(all_decisions)
    index = all_decisions.index(best)
    if best != -1*math.inf:
        vis.pop(index)
        wis.pop(index)
        memo[maxw] = best
        return best
    else:
        return 0
vis=[1,6,18,22,28]
wis=[1,2,5,6,7]

best_val_w(vis,wis,11,memo)