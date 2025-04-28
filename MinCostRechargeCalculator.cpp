#include "MinCostRechargeCalculator.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm> // for std::max

// Nom(s) étudiant(s) / Name(s) of student(s):
// Yorguin José Mantilla Ramos
// Matricule: 20253616

// ce fichier contient les definitions des methodes de la classe MinCostRechargeCalculator
// this file contains the definitions of the methods of the MinCostRechargeCalculator class

using namespace std;

MinCostRechargeCalculator::MinCostRechargeCalculator()
{
}

int MinCostRechargeCalculator::CalculateMinCostRecharge(const vector<int>& RechargeCost) {
   int n = RechargeCost.size(); // Number of stations
   const int MAX_BATTERY = 3; // Maximum battery capacity (battery levels: 0, 1, 2, 3)
   const int INF = 1e9; // A big number representing "infinity" (unreachable state)
   
   vector<vector<int>> dp(n, vector<int>(MAX_BATTERY + 1, INF));
   // dp[i][b] represents the minimum cost to reach station i with exactly b battery units left.
   // MAX_BATTERY + 1 columns to represent battery levels 0, 1, 2, and 3.
   // Initialized to INF to represent unreachable states.
   
   dp[0][2] = 0; // After moving from start to station 0 (costs 1 battery), we have 2 battery units left. No recharge paid yet.
   
   // Example Matrix (initial state):
   //                 battery
   //            0     1     2     3
   // station 0 [INF][INF][  0 ][INF]
   // station 1 [INF][INF][INF][INF]
   // station 2 [INF][INF][INF][INF]
   // station 3 [INF][INF][INF][INF]
   // station 4 [INF][INF][INF][INF]
   
   // Idea: For each station and each battery level, decide whether to recharge or to move forward.
   for (int i = 0; i < n; ++i) {
       for (int b = 0; b <= MAX_BATTERY; ++b) {
           if (dp[i][b] == INF) continue; // Skip if the current state is not reachable.
   
           // Option 1: Recharge now at station i
           if (b < MAX_BATTERY) { // We can recharge if battery is not already full.
               // Update the cost to be at station i with a full battery (after paying RechargeCost[i]).
               dp[i][MAX_BATTERY] = min(dp[i][MAX_BATTERY], dp[i][b] + RechargeCost[i]);
           }
   
           // Option 2: Move to the next station without recharging (costs 1 battery unit)
           if (i + 1 < n && b >= 1) { // Can move if we have at least 1 battery unit and there is a next station.
               // Moving consumes 1 battery unit but adds no cost.
               // Update the cost to reach station i+1 with b-1 batteries left.
               dp[i+1][b-1] = min(dp[i+1][b-1], dp[i][b]);
               // Here dp[i][b] is the cost to be at the current station i with b battery,
               // and moving forward decreases battery by 1 without adding cost.
           }
       }
   }
   
   // At the last station (n-1), we must have at least 1 battery unit left to reach the destination.
   int result = INF;
   for (int b = 1; b <= MAX_BATTERY; ++b) {
       result = min(result, dp[n-1][b]); // Among all possibilities where we have at least 1 battery left, take the minimum cost.
   }
   
   return result;
   }
