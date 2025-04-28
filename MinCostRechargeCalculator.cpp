#include "MinCostRechargeCalculator.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm> // for std::max

// Nom(s) étudiant(s) / Name(s) of student(s):

// ce fichier contient les definitions des methodes de la classe MinCostRechargeCalculator
// this file contains the definitions of the methods of the MinCostRechargeCalculator class

using namespace std;

MinCostRechargeCalculator::MinCostRechargeCalculator()
{
}

int MinCostRechargeCalculator::CalculateMinCostRecharge(const vector<int>& RechargeCost) {
   int n = RechargeCost.size();
   const int MAX_BATTERY = 3;
   const int INF = 1e9; // big number acting as "infinity"
   
   vector<vector<int>> dp(n, vector<int>(MAX_BATTERY + 1, INF));
   
   // After moving from start to station 0 (costs 1 battery), 2 battery left
   dp[0][2] = 0; // No recharge paid yet

   for (int i = 0; i < n; ++i) {
       for (int b = 0; b <= MAX_BATTERY; ++b) {
           if (dp[i][b] == INF) continue; // not reachable
           
           // Option 1: Recharge now at station i
           if (b < MAX_BATTERY) {
               dp[i][MAX_BATTERY] = min(dp[i][MAX_BATTERY], dp[i][b] + RechargeCost[i]);
           }
           
           // Option 2: Move to next station without recharging
           if (i + 1 < n && b >= 1) {
               dp[i+1][b-1] = min(dp[i+1][b-1], dp[i][b]);
           }
       }
   }
   
   // At last station (n-1), we need battery >= 1 to reach destination
   int result = INF;
   for (int b = 1; b <= MAX_BATTERY; ++b) {
       result = min(result, dp[n-1][b]);
   }
   
   return result;
}
