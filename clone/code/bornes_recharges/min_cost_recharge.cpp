// CE FICHIER NE SERT QU'A APPELER ET TESTER VOTRE CODE. 
// VOUS NE DEVRIEZ PAS AVOIR BESOIN DE LE MODIFIER, SAUF POUR 
// AJOUTER VOUS-MÊME D'AUTRES TESTS SI VOUS LE VOULEZ.
// NE PAS REMETTRE SUR STUDIUM.

// THIS FILE IS ONLY USED TO CALL AND TEST YOUR CODE.
// YOU SHOULD NOT HAVE TO MODIFY IT, EXCEPT FOR ADDING 
// NEW CUSTOM TESTS IF YOU WISH TO DO SO.
// DO NOT SUBMIT ON STUDIUM.

#include <iostream> // pour l'affichage dans la console // for display in console
#include "MinCostRechargeCalculator.h" // pour la classe principale de l'exercice // for the main class of the exercise
#include <vector> // pour utiliser les vecteurs de la librairie standard // to use vectors from the standard library
#include <cstdlib> // pour convertir le input en int // to convert input to int
#include <cassert>

// commandes / command (PowerShell) :
// g++ -o .\min_cost_recharge.exe .\min_cost_recharge.cpp .\MinCostRechargeCalculator.cpp
// .\min_cost_recharge.exe

// for VS Code, make sure to compile all the files of the project
// you might want to change "${file}", by "${fileDirname}\\**.cpp" in the tasks.json of .vscode -> taks -> args
// pour VS Code, veillez à compiler tous les fichiers du projet
// vous souhaiterez peut-être remplacer "${file}", par "${fileDirname}\\**.cpp" dans le fichier task.json de .vscode -> taks -> args

using namespace std;

bool TestMinCostRechargeCalculator();

int main(int argc, char *argv[])
{

    vector<int> RechargeCost = {1,9,9,1,1,9,1,1,9,9,1};

    MinCostRechargeCalculator Calculator = MinCostRechargeCalculator();
    int MinCost = Calculator.CalculateMinCostRecharge(RechargeCost);

    // tests
    if (TestMinCostRechargeCalculator()){
        std::cout << "Tests reussis / Tests passed !" << std::endl;
    } else {
        std::cout << "Tests echoues / Failed tests :(" << std::endl;
    }

}

bool TestMinCostRechargeCalculator(){
    vector<vector<int>> testCases = {
        {1,9,9,1,1,9,1,1,9,9,1},
        {1,1},
        {1,1,1},
        {9,9,1,9,9},
        {9,1,9,9},
        {9,1,9}
    };

    vector<int> expectedResults = {5, 0, 1, 1, 1, 1};

    MinCostRechargeCalculator Calculator = MinCostRechargeCalculator();

    for (size_t i = 0; i < testCases.size(); i++) {
        const auto& RechargeCost = testCases[i];
        int result = Calculator.CalculateMinCostRecharge(RechargeCost);
        cout << "Test " << i + 1 << ": Expected = " << expectedResults[i] << ", Got = " << result << " --> ";
        if (result == expectedResults[i]) {
            cout << "PASSED" << endl;
        }else{
            assert(result == expectedResults[i]);  // Assertion to verify the correctness
            return false;
        }
    }
    return true;
}
