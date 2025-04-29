//knapsack basic recursion

#include <bits/stdc++.h>
using namespace std;

// Returns the maximum value that
// can be put in a knapsack of capacity W
int knapsackRec(int W, vector<int> &val, vector<int> &wt, int n) {

    // Base Case
    if (n == 0 || W == 0)
        return 0;

    int pick = 0;

    // Pick nth item if it does not exceed the capacity of knapsack
    if (wt[n - 1] <= W)
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1);
    
    // Don't pick the nth item
    int notPick = knapsackRec(W, val, wt, n - 1);
     
    return max(pick, notPick);
}

int knapsack(int W, vector<int> &val, vector<int> &wt) {
    int n = val.size();
    return knapsackRec(W, val, wt, n);
}

int main() {
    vector<int> val = {1, 2, 3};
    vector<int> wt = {4, 5, 1};
    int W = 4;

    cout << knapsack(W, val, wt) << endl;
    return 0;
}











 ##knapsack 0/1 using memorisation dynamic programming

#include <bits/stdc++.h>
using namespace std;

// Returns the maximum value that
// can be put in a knapsack of capacity W
int knapsackRec(int W, vector<int> &val, vector<int> &wt, int n, 
                                        vector<vector<int>> &memo) {

    // Base Case
    if (n == 0 || W == 0)
        return 0;

    // Check if we have previously calculated the same subproblem
    if(memo[n][W] != -1)
        return memo[n][W];

    int pick = 0;

    // Pick nth item if it does not exceed the capacity of knapsack
    if (wt[n - 1] <= W)
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1, memo);
    
    // Don't pick the nth item
    int notPick = knapsackRec(W, val, wt, n - 1, memo);
    
    // Store the result in memo[n][W] and return it
    return memo[n][W] = max(pick, notPick);
}

int knapsack(int W, vector<int> &val, vector<int> &wt) {
    int n = val.size();
    
    // Memoization table to store the results
    vector<vector<int>> memo(n + 1, vector<int>(W + 1, -1));
    
    return knapsackRec(W, val, wt, n, memo);
}

int main() {
    vector<int> val = {1, 2, 3};
    vector<int> wt = {4, 5, 1};
    int W = 4;

    cout << knapsack(W, val, wt) << endl;
    return 0;
}








2 ##coinage

    //basic recursive code

   // using recursion
#include <bits/stdc++.h>
using namespace std;

// Returns the count of ways we can
// sum coins[0...n-1] coins to get sum "sum"
int countRecur(vector<int>& coins, int n, int sum) {
  
    // If sum is 0 then there is 1 solution
    // (do not include any coin)
    if (sum == 0) return 1;

    // 0 ways in the following two cases
    if (sum < 0 || n == 0) return 0;

    // count is sum of solutions (i)
    // including coins[n-1] (ii) excluding coins[n-1]
    return countRecur(coins, n, sum - coins[n - 1]) + 
            countRecur(coins, n - 1, sum);
}

int count(vector<int> &coins, int sum) {
    return countRecur(coins, coins.size(), sum);
}

int main() {
    vector<int> coins = {1, 2, 3};
    int sum = 5;
    cout << count(coins, sum);
    return 0;
}




  //using memorisation dynamic programming
// C++ program for coin change problem
// using memoization
#include <bits/stdc++.h>
using namespace std;

// Returns the count of ways we can
// sum coins[0...n-1] coins to get sum "sum"
int countRecur(vector<int>& coins, int n, int sum, 
               vector<vector<int>> &memo) {
  
    // If sum is 0 then there is 1 solution
    // (do not include any coin)
    if (sum == 0) return 1;

    // 0 ways in the following two cases
    if (sum < 0 || n == 0) return 0;
    
    // If the subproblem is previously calculated then
    // simply return the result
    if (memo[n-1][sum]!=-1) return memo[n-1][sum];

    // count is sum of solutions (i)
    // including coins[n-1] (ii) excluding coins[n-1]
    return memo[n-1][sum] = 
        countRecur(coins, n, sum - coins[n-1], memo) + 
        countRecur(coins, n - 1, sum, memo);
}

int count(vector<int> &coins, int sum) {
    
    vector<vector<int>> memo(coins.size(), vector<int>(sum+1, -1));
    return countRecur(coins, coins.size(), sum, memo);
}

int main() {
    vector<int> coins = {1, 2, 3};
    int sum = 5;
    cout << count(coins, sum);
    return 0;
}




//using tabulation
// C++ program for coin change problem using tabulation
#include <bits/stdc++.h>
using namespace std;

// Returns total distinct ways to make sum using n coins of
// different denominations
int count(vector<int>& coins, int sum) {
    int n = coins.size();
    
    // 2d dp array where n is the number of coin
    // denominations and sum is the target sum
    vector<vector<int> > dp(n + 1, vector<int>(sum + 1, 0));

    // Represents the base case where the target sum is 0,
    // and there is only one way to make change: by not
    // selecting any coin
    dp[0][0] = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= sum; j++) {

            // Add the number of ways to make change without
            // using the current coin,
            dp[i][j] += dp[i - 1][j];

            if ((j - coins[i - 1]) >= 0) {

                // Add the number of ways to make change
                // using the current coin
                dp[i][j] += dp[i][j - coins[i - 1]];
            }
        }
    }
    return dp[n][sum];
}

int main() {
    vector<int> coins = {1, 2, 3};
    int sum = 5;
    cout << count(coins, sum);
    return 0;
}










// 3 Floyd warshall

// C++ program to implement Floyd-Warshall algorithm

#include <iostream>
#include <limits.h>
#include <vector>

#define INF INT_MAX

using namespace std;

// Function to print the solution matrix
void printSolution(const vector<vector<int>> &dist)
{
    int V = dist.size();
    cout << "The following matrix shows the shortest distances"
            " between every pair of vertices:\n";
    for (int i = 0; i < V; ++i)
    {
        for (int j = 0; j < V; ++j)
        {
            if (dist[i][j] == INF)
                cout << "INF"
                     << "\t";
            else
                cout << dist[i][j] << "\t";
        }
        cout << endl;
    }
}

// Function to implement the Floyd-Warshall algorithm
void floydWarshall(vector<vector<int>> &graph)
{
    int V = graph.size();
    vector<vector<int>> dist = graph;

    // Update the solution matrix by considering all vertices
    for (int k = 0; k < V; ++k)
    {
        for (int i = 0; i < V; ++i)
        {
            for (int j = 0; j < V; ++j)
            {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }

    // Print the shortest distance matrix
    printSolution(dist);
}

int main()
{
    /* Let us create the following weighted graph
           10
      (0)------->(3)
       |         /|\
     5 |          |
       |          | 1
      \|/         |
      (1)------->(2)
           3           */
    vector<vector<int>> graph = {{0, 5, INF, 10}, {INF, 0, 3, INF}, {INF, INF, 0, 1}, {INF, INF, INF, 0}};

    // Function call to implement Floyd-Warshall algorithm
    floydWarshall(graph);

    return 0;
}










//matrix multiplication 
 //basic recursive code 
// C++ code to implement the
// matrix chain multiplication using recursion
#include <bits/stdc++.h>
using namespace std;

// Matrix Ai has dimension arr[i-1] x arr[i]
int minMultRec(vector<int> &arr, int i, int j)
{

    // If there is only one matrix
    if (i + 1 == j)
        return 0;

    int res = INT_MAX;

    // Place the first bracket at different
    // positions or k and for every placed
    // first bracket, recursively compute
    // minimum cost for remaining brackets
    // (or subproblems)
    for (int k = i + 1; k < j; k++)
    {
        int curr = minMultRec(arr, i, k) + minMultRec(arr, k, j) + arr[i] * arr[k] * arr[j];

        res = min(curr, res);
    }

    // Return minimum count
    return res;
}

int matrixMultiplication(vector<int> &arr)
{

    int n = arr.size();
    return minMultRec(arr, 0, n - 1);
}

int main()
{

    vector<int> arr = {2, 1, 3, 4};
    cout << matrixMultiplication(arr);
    return 0;
}



//using memorisation dynamic programming
// C++ code to implement the
// matrix chain multiplication using memoization
#include <bits/stdc++.h>
using namespace std;

int minMultRec(vector<int> &arr, int i, int j, vector<vector<int>> &memo)
{

    // If there is only one matrix
    if (i + 1 == j)
        return 0;

    // Check if the result is already
    // computed
    if (memo[i][j] != -1)
        return memo[i][j];

    int res = INT_MAX;

    // Place the first bracket at different positions or k and
    // for every placed first bracket, recursively compute
    // minimum cost for remaining brackets (or subproblems)
    for (int k = i + 1; k < j; k++)
    {
        int curr = minMultRec(arr, i, k, memo) + minMultRec(arr, k, j, memo) + arr[i] * arr[k] * arr[j];

        res = min(curr, res);
    }

    // Store the result in memo table
    memo[i][j] = res;
    return res;
}

int matrixMultiplication(vector<int> &arr)
{

    int n = arr.size();
    vector<vector<int>> memo(n, vector<int>(n, -1));
    return minMultRec(arr, 0, n - 1, memo);
}

int main()
{
    vector<int> arr = {2, 1, 3, 4};
    int res = matrixMultiplication(arr);
    cout << res << endl;
    return 0;
}




//LCS BASIC RECURSIVE CODE
// A Naive recursive implementation of LCS problem
#include <bits/stdc++.h>
using namespace std;

// Returns length of LCS for s1[0..m-1], s2[0..n-1]
int lcsRec(string &s1, string &s2,int m,int n) {
    
    // Base case: If either string is empty, the length of LCS is 0
    if (m == 0 || n == 0)
        return 0;

    // If the last characters of both substrings match
    if (s1[m - 1] == s2[n - 1])
      
        // Include this character in LCS and recur for remaining substrings
        return 1 + lcsRec(s1, s2, m - 1, n - 1);

    else
        // If the last characters do not match
        // Recur for two cases:
        // 1. Exclude the last character of s1 
        // 2. Exclude the last character of s2 
        // Take the maximum of these two recursive calls
        return max(lcsRec(s1, s2, m, n - 1), lcsRec(s1, s2, m - 1, n));
}
int lcs(string &s1,string &s2){
    
    int m = s1.size(), n = s2.size();
    return lcsRec(s1,s2,m,n);
}

int main() {
    string s1 = "AGGTAB";
    string s2 = "GXTXAYB";
    int m = s1.size();
    int n = s2.size();

    cout << lcs(s1, s2) << endl;

    return 0;
}



//USING MEMORISATION DYNAMIC PROGAMMING
// C++ implementation of Top-Down DP
// of LCS problem
#include <bits/stdc++.h>
using namespace std;
// Returns length of LCS for s1[0..m-1], s2[0..n-1]
int lcsRec(string &s1, string &s2, int m, int n, vector<vector<int>> &memo) {

    // Base Case
    if (m == 0 || n == 0)
        return 0;

    // Already exists in the memo table
    if (memo[m][n] != -1)
        return memo[m][n];

    // Match
    if (s1[m - 1] == s2[n - 1])
        return memo[m][n] = 1 + lcsRec(s1, s2, m - 1, n - 1, memo);

    // Do not match
    return memo[m][n] = max(lcsRec(s1, s2, m, n - 1, memo), lcsRec(s1, s2, m - 1, n, memo));
}
int lcs(string &s1,string &s2){
    int m = s1.length();
    int n = s2.length();
    vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
    return lcsRec(s1, s2, m, n, memo);
}

int main() {
    string s1 = "AGGTAB";
    string s2 = "GXTXAYB";
    cout << lcs(s1, s2) << endl;
    return 0;
}













//LONGEST INCREASING SEQUENCE USING RECURSION 
// Cpp program to find lis using recursion
// in Exponential Time and Linear Space
#include <bits/stdc++.h>
using namespace std;

// Returns LIS of subarray ending with index i.
int lisEndingAtIdx(vector<int>& arr, int idx) {
  
    // Base case
    if (idx == 0)
        return 1;

    // Consider all elements on the left of i,
    // recursively compute LISs ending with 
    // them and consider the largest
    int mx = 1;
    for (int prev = 0; prev < idx; prev++)
        if (arr[prev] < arr[idx])
            mx = max(mx, lisEndingAtIdx(arr, prev) + 1);
    return mx;
}

int lis(vector<int>& arr) {
    int n = arr.size();
    int res = 1;
    for (int i = 1; i < n; i++)
        res = max(res, lisEndingAtIdx(arr, i));
    return res;
}

int main() {
    vector<int> arr = { 10, 22, 9, 33, 21, 50, 41, 60 };
    cout << lis(arr);
    return 0;
}




//USING MEMORISATION DYNAMIC PROGRAMMING
#include <bits/stdc++.h>
using namespace std;

int lisEndingAtIdx(vector<int>& arr, int idx, vector<int>& memo) {
  
    // Base case
    if (idx == 0)
        return 1;

    // Check if the result is already computed
    if (memo[idx] != -1)
        return memo[idx];

    // Consider all elements on left of i,
    // recursively compute LISs ending with 
    // them and consider the largest
    int mx = 1;
    for (int prev = 0; prev < idx; prev++)
        if (arr[prev] < arr[idx])
            mx = max(mx, lisEndingAtIdx(arr, prev, memo) + 1);

    // Store the result in the memo array
    memo[idx] = mx;
    return memo[idx];
}

int lis(vector<int>& arr) {
    
    int n = arr.size();
  
    vector<int> memo(n, -1);
  
    int res = 1;
    for (int i = 1; i < n; i++)
        res = max(res, lisEndingAtIdx(arr, i, memo));
    return res;
}

int main() {
    vector<int> arr = { 10, 22, 9, 33, 21, 50, 41, 60 };
    cout << lis(arr);
    return 0;
}



//string editing recursive 
// C++ program to calculate the minimum edit distance
// between two strings using recursion

#include <cstring>
#include <iostream>
using namespace std;

int min(int a, int b, int c) { return min(min(a, b), c); }

// Recursive function to find the edit distance
int editDistanceRecursive(string str1, string str2, int m,
                          int n)
{
    // If str1 is empty, insert all characters of str2
    if (m == 0)
        return n;
    // If str2 is empty, remove all characters of str1
    if (n == 0)
        return m;

    // If the last characters match, move to the next pair
    if (str1[m - 1] == str2[n - 1])
        return editDistanceRecursive(str1, str2, m - 1,
                                     n - 1);

    // If the last characters don't match, consider all
    // three operations
    return 1
           + min(editDistanceRecursive(str1, str2, m,
                                       n - 1), // Insert
                 editDistanceRecursive(str1, str2, m - 1,
                                       n), // Remove
                 editDistanceRecursive(str1, str2, m - 1,
                                       n - 1) // Replace
           );
}

int main()
{
    // Initialize two strings
    string str1 = "GEEXSFRGEEKKS";
    string str2 = "GEEKSFORGEEKS";
    // print the minimum edit distance
    cout << "Minimum edit distance is "
         << editDistanceRecursive(str1, str2, str1.length(),
                                  str2.length())
         << endl;
    return 0;
}




//string editing using memorisation dyanmmic programming
// C++ program to calculate minimum edit distance using
// dynamic programming
#include <iostream>
#include <vector>
using namespace std;

// Helper function to find the minimum of three values
int min(int a, int b, int c)
{
    // Find the smallest value among a, b, and c
    return min(min(a, b), c);
}

// Function to find the edit distance using dynamic
// programming (memoization)
int editDistanceDP(string str1, string str2, int m, int n)
{
    // Create a 2D vector to store the edit distance values
    vector<vector<int> > dp(m + 1, vector<int>(n + 1));

    // Initialize base cases
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            // If str1 is empty, insert all characters of
            // str2
            if (i == 0)
                dp[i][j] = j;
            // If str2 is empty, remove all characters of
            // str1
            else if (j == 0)
                dp[i][j] = i;
            // No new operation needed if last characters
            // match
            else if (str1[i - 1] == str2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];

            else
                dp[i][j]
                    = 1
                      + min(
                          dp[i][j - 1], // Insert operation
                          dp[i - 1][j], // Remove operation
                          dp[i - 1]
                            [j - 1] // Replace operation
                      );
        }
    }

    // Return the final edit distance
    return dp[m][n];
}

int main()
{
    // Define input strings
    string str1 = "GEEXSFRGEEKKS";
    string str2 = "GEEKSFORGEEKS";

    // Calculate and output the minimum edit distance
    cout << "Minimum edit distance is "
         << editDistanceDP(str1, str2, str1.length(),
                           str2.length())
         << endl;

    return 0;
}




//lab test questions 
//Grid Minimum Path with One Diagonal Move
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

int minPathWithOneDiagonal(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    const int INF = INT_MAX / 2;

    vector<vector<int>> dp_no_diag(m, vector<int>(n, INF));
    vector<vector<int>> dp_with_diag(m, vector<int>(n, INF));

    dp_no_diag[0][0] = grid[0][0];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // Without diagonal
            if (i > 0)
                dp_no_diag[i][j] = min(dp_no_diag[i][j], dp_no_diag[i-1][j] + grid[i][j]);
            if (j > 0)
                dp_no_diag[i][j] = min(dp_no_diag[i][j], dp_no_diag[i][j-1] + grid[i][j]);

            // With diagonal already used
            if (i > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_with_diag[i-1][j] + grid[i][j]);
            if (j > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_with_diag[i][j-1] + grid[i][j]);

            // Diagonal move (use it exactly once)
            if (i > 0 && j > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_no_diag[i-1][j-1] + grid[i][j]);
        }
    }

    return min(dp_no_diag[m-1][n-1], dp_with_diag[m-1][n-1]);
}

int main() {
    vector<vector<int>> grid1 = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
    vector<vector<int>> grid2 = {{1, 2}, {1, 1}};

    cout << minPathWithOneDiagonal(grid1) << endl; // Output: 6
    cout << minPathWithOneDiagonal(grid2) << endl; // Output: 2

    return 0;
}





//lab test
//minimum sum excluding at max one element from array
#include <iostream>
#include <vector>
#include <climits>
#include <cmath>
using namespace std;

int minDiffWithOneExclusion(vector<int>& arr) {
    int n = arr.size();
    int total = 0;
    for (int val : arr) total += val;

    int result = INT_MAX;

    // Try excluding each element (or none)
    for (int skip = -1; skip < n; ++skip) {
        int curr_total = total;
        vector<int> temp;

        // Build the array after skipping one element
        for (int i = 0; i < n; ++i) {
            if (i != skip) {
                temp.push_back(arr[i]);
            } else {
                curr_total -= arr[i];
            }
        }

        // Classic subset sum DP
        vector<bool> dp(curr_total + 1, false);
        dp[0] = true;

        for (int num : temp) {
            for (int j = curr_total; j >= num; --j) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        // Try all possible subset sums
        for (int s = 0; s <= curr_total; ++s) {
            if (dp[s]) {
                int diff = abs((curr_total - s) - s);
                result = min(result, diff);
            }
        }
    }

    return result;
}

int main() {
    vector<int> arr = {1, 6, 11, 5};
    cout << "Minimum difference after excluding at most one element: " 
         << minDiffWithOneExclusion(arr) << endl;
    return 0;
}




// lab test Generalized Problem Statement: Two-Agent Grid Meeting Optimization
Problem:
You are given a 2D grid of size M × N, where each cell grid[i][j] contains an integer value (which could represent XP, goods, or cost). Two agents start from different corners of the grid and are allowed to move only in constrained directions (e.g., only right/down, or up/right, etc.). Each agent must traverse the grid according to their allowed directions, and they are required to meet at exactly one common cell.

#include <iostream>
#include <vector>
#include <algorithm>
#include<climits>
using namespace std;


// Function to find the maximum value both agents can collect
int maxGridValueWithMeeting(vector<vector<int>>& grid) {
    int m = grid.size();       // number of rows
    int n = grid[0].size();    // number of columns

    // dp[x1][y1][x2] stores the maximum value when:
    // Agent A is at (x1, y1)
    // Agent B is at (x2, y2) where y2 = x1 + y1 - x2 (because steps taken by both are the same)
    vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(m, INT_MIN)));

    // Initialize starting position (0,0) for both agents
    dp[0][0][0] = grid[0][0];

    // Traverse all possible positions for Agent A and Agent B
    for (int x1 = 0; x1 < m; ++x1) {
        for (int y1 = 0; y1 < n; ++y1) {
            for (int x2 = 0; x2 < m; ++x2) {
                // Derive y2 using steps constraint
                int y2 = x1 + y1 - x2;

                // If y2 is out of bounds, skip
                if (y2 < 0 || y2 >= n) continue;

                // Skip blocked cells
                if (grid[x1][y1] == -1 || grid[x2][y2] == -1) continue;

                int maxPrev = INT_MIN;

                // Consider all 4 possible previous states:
                // 1. Both moved down
                if (x1 > 0 && x2 > 0)
                    maxPrev = max(maxPrev, dp[x1 - 1][y1][x2 - 1]);
                // 2. A moved down, B moved right
                if (x1 > 0 && y2 > 0)
                    maxPrev = max(maxPrev, dp[x1 - 1][y1][x2]);
                // 3. A moved right, B moved down
                if (y1 > 0 && x2 > 0)
                    maxPrev = max(maxPrev, dp[x1][y1 - 1][x2 - 1]);
                // 4. Both moved right
                if (y1 > 0 && y2 > 0)
                    maxPrev = max(maxPrev, dp[x1][y1 - 1][x2]);

                // If no valid path to current state, skip
                if (maxPrev == INT_MIN) continue;

                // Add current cell values:
                int value = grid[x1][y1];
                if (x1 != x2 || y1 != y2)
                    value += grid[x2][y2];  // Don't double count if they meet at same cell

                dp[x1][y1][x2] = max(dp[x1][y1][x2], maxPrev + value);
            }
        }
    }

    // Final answer is at the bottom-right corner (m-1, n-1) for both agents
    return max(0, dp[m - 1][n - 1][m - 1]);
}

int main() {
    // Sample grid with a valid path
    vector<vector<int>> grid1 = {
        {1, 1, -1},
        {1, -1, 1},
        {1, 1, 1}
    };

    // Sample grid with positive values
    vector<vector<int>> grid2 = {
        {1, 3, 1},
        {1, 5, 1},
        {4, 2, 1}
    };

    cout << "Max value (Grid 1): " << maxGridValueWithMeeting(grid1) << endl;  // Output: 5
    cout << "Max value (Grid 2): " << maxGridValueWithMeeting(grid2) << endl;  // Output: 15

    return 0;
}



















//without comments
 knapsack recursion

#include <bits/stdc++.h>
using namespace std;

int knapsackRec(int W, vector<int> &val, vector<int> &wt, int n) {
    if (n == 0 || W == 0)
        return 0;

    int pick = 0;

    if (wt[n - 1] <= W)
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1);
    
    int notPick = knapsackRec(W, val, wt, n - 1);
     
    return max(pick, notPick);
}

int knapsack(int W, vector<int> &val, vector<int> &wt) {
    int n = val.size();
    return knapsackRec(W, val, wt, n);
}

int main() {
    vector<int> val = {1, 2, 3};
    vector<int> wt = {4, 5, 1};
    int W = 4;

    cout << knapsack(W, val, wt) << endl;
    return 0;
}








knapsack memorisation
#include <bits/stdc++.h>
using namespace std;

int knapsackRec(int W, vector<int> &val, vector<int> &wt, int n, 
                                        vector<vector<int>> &memo) {
    if (n == 0 || W == 0)
        return 0;

    if(memo[n][W] != -1)
        return memo[n][W];

    int pick = 0;

    if (wt[n - 1] <= W)
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1, memo);
    
    int notPick = knapsackRec(W, val, wt, n - 1, memo);
    
    return memo[n][W] = max(pick, notPick);
}

int knapsack(int W, vector<int> &val, vector<int> &wt) {
    int n = val.size();
    
    vector<vector<int>> memo(n + 1, vector<int>(W + 1, -1));
    
    return knapsackRec(W, val, wt, n, memo);
}

int main() {
    vector<int> val = {1, 2, 3};
    vector<int> wt = {4, 5, 1};
    int W = 4;

    cout << knapsack(W, val, wt) << endl;
    return 0;
}
























//coinage recursion
#include <bits/stdc++.h>
using namespace std;

int countRecur(vector<int>& coins, int n, int sum) {
    if (sum == 0) return 1;

    if (sum < 0 || n == 0) return 0;

    return countRecur(coins, n, sum - coins[n - 1]) + 
            countRecur(coins, n - 1, sum);
}

int count(vector<int> &coins, int sum) {
    return countRecur(coins, coins.size(), sum);
}

int main() {
    vector<int> coins = {1, 2, 3};
    int sum = 5;
    cout << count(coins, sum);
    return 0;
}
















//coinage DP
#include <bits/stdc++.h>
using namespace std;

int countRecur(vector<int>& coins, int n, int sum, 
               vector<vector<int>> &memo) {
    if (sum == 0) return 1;

    if (sum < 0 || n == 0) return 0;
    
    if (memo[n-1][sum]!=-1) return memo[n-1][sum];

    return memo[n-1][sum] = 
        countRecur(coins, n, sum - coins[n-1], memo) + 
        countRecur(coins, n - 1, sum, memo);
}

int count(vector<int> &coins, int sum) {
    vector<vector<int>> memo(coins.size(), vector<int>(sum+1, -1));
    return countRecur(coins, coins.size(), sum, memo);
}

int main() {
    vector<int> coins = {1, 2, 3};
    int sum = 5;
    cout << count(coins, sum);
    return 0;
}


























//Floyd warshall
#include <iostream>
#include <limits.h>
#include <vector>

#define INF INT_MAX

using namespace std;

void printSolution(const vector<vector<int>> &dist)
{
    int V = dist.size();
    cout << "The following matrix shows the shortest distances"
            " between every pair of vertices:\n";
    for (int i = 0; i < V; ++i)
    {
        for (int j = 0; j < V; ++j)
        {
            if (dist[i][j] == INF)
                cout << "INF"
                     << "\t";
            else
                cout << dist[i][j] << "\t";
        }
        cout << endl;
    }
}

void floydWarshall(vector<vector<int>> &graph)
{
    int V = graph.size();
    vector<vector<int>> dist = graph;

    for (int k = 0; k < V; ++k)
    {
        for (int i = 0; i < V; ++i)
        {
            for (int j = 0; j < V; ++j)
            {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }

    printSolution(dist);
}

int main()
{
    vector<vector<int>> graph = {{0, 5, INF, 10}, {INF, 0, 3, INF}, {INF, INF, 0, 1}, {INF, INF, INF, 0}};

    floydWarshall(graph);

    return 0;
}
























//matrix chain recursion
#include <bits/stdc++.h>
using namespace std;

int minMultRec(vector<int> &arr, int i, int j)
{
    if (i + 1 == j)
        return 0;

    int res = INT_MAX;

    for (int k = i + 1; k < j; k++)
    {
        int curr = minMultRec(arr, i, k) + minMultRec(arr, k, j) + arr[i] * arr[k] * arr[j];

        res = min(curr, res);
    }

    return res;
}

int matrixMultiplication(vector<int> &arr)
{
    int n = arr.size();
    return minMultRec(arr, 0, n - 1);
}

int main()
{
    vector<int> arr = {2, 1, 3, 4};
    cout << matrixMultiplication(arr);
    return 0;
}
















//matrix chain DP
#include <bits/stdc++.h>
using namespace std;

int minMultRec(vector<int> &arr, int i, int j, vector<vector<int>> &memo)
{
    if (i + 1 == j)
        return 0;

    if (memo[i][j] != -1)
        return memo[i][j];

    int res = INT_MAX;

    for (int k = i + 1; k < j; k++)
    {
        int curr = minMultRec(arr, i, k, memo) + minMultRec(arr, k, j, memo) + arr[i] * arr[k] * arr[j];

        res = min(curr, res);
    }

    memo[i][j] = res;
    return res;
}

int matrixMultiplication(vector<int> &arr)
{
    int n = arr.size();
    vector<vector<int>> memo(n, vector<int>(n, -1));
    return minMultRec(arr, 0, n - 1, memo);
}

int main()
{
    vector<int> arr = {2, 1, 3, 4};
    int res = matrixMultiplication(arr);
    cout << res << endl;
    return 0;
}























//lcs recursion
#include <bits/stdc++.h>
using namespace std;

int lcsRec(string &s1, string &s2,int m,int n) {
    if (m == 0 || n == 0)
        return 0;

    if (s1[m - 1] == s2[n - 1])
        return 1 + lcsRec(s1, s2, m - 1, n - 1);

    else
        return max(lcsRec(s1, s2, m, n - 1), lcsRec(s1, s2, m - 1, n));
}

int lcs(string &s1,string &s2){
    int m = s1.size(), n = s2.size();
    return lcsRec(s1,s2,m,n);
}

int main() {
    string s1 = "AGGTAB";
    string s2 = "GXTXAYB";
    int m = s1.size();
    int n = s2.size();

    cout << lcs(s1, s2) << endl;

    return 0;
}













//lcs DP
#include <bits/stdc++.h>
using namespace std;

int lcsRec(string &s1, string &s2, int m, int n, vector<vector<int>> &memo) {
    if (m == 0 || n == 0)
        return 0;

    if (memo[m][n] != -1)
        return memo[m][n];

    if (s1[m - 1] == s2[n - 1])
        return memo[m][n] = 1 + lcsRec(s1, s2, m - 1, n - 1, memo);

    return memo[m][n] = max(lcsRec(s1, s2, m, n - 1, memo), lcsRec(s1, s2, m - 1, n, memo));
}

int lcs(string &s1,string &s2){
    int m = s1.length();
    int n = s2.length();
    vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
    return lcsRec(s1, s2, m, n, memo);
}

int main() {
    string s1 = "AGGTAB";
    string s2 = "GXTXAYB";
    cout << lcs(s1, s2) << endl;
    return 0;
}















\\Longest increasing susequence recursion
#include <bits/stdc++.h>
using namespace std;

int lisEndingAtIdx(vector<int>& arr, int idx) {
    if (idx == 0)
        return 1;

    int mx = 1;
    for (int prev = 0; prev < idx; prev++)
        if (arr[prev] < arr[idx])
            mx = max(mx, lisEndingAtIdx(arr, prev) + 1);
    return mx;
}

int lis(vector<int>& arr) {
    int n = arr.size();
    int res = 1;
    for (int i = 1; i < n; i++)
        res = max(res, lisEndingAtIdx(arr, i));
    return res;
}

int main() {
    vector<int> arr = { 10, 22, 9, 33, 21, 50, 41, 60 };
    cout << lis(arr);
    return 0;
}













//longest increasing sub DP

#include <bits/stdc++.h>
using namespace std;

int lisEndingAtIdx(vector<int>& arr, int idx, vector<int>& memo) {
    if (idx == 0)
        return 1;

    if (memo[idx] != -1)
        return memo[idx];

    int mx = 1;
    for (int prev = 0; prev < idx; prev++)
        if (arr[prev] < arr[idx])
            mx = max(mx, lisEndingAtIdx(arr, prev, memo) + 1);

    memo[idx] = mx;
    return memo[idx];
}

int lis(vector<int>& arr) {
    int n = arr.size();
    vector<int> memo(n, -1);
    int res = 1;
    for (int i = 1; i < n; i++)
        res = max(res, lisEndingAtIdx(arr, i, memo));
    return res;
}

int main() {
    vector<int> arr = { 10, 22, 9, 33, 21, 50, 41, 60 };
    cout << lis(arr);
    return 0;
}

















//string editing dp
#include <iostream>
#include <vector>
using namespace std;

int min(int a, int b, int c)
{
    return min(min(a, b), c);
}

int editDistanceDP(string str1, string str2, int m, int n)
{
    vector<vector<int> > dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0)
                dp[i][j] = j;
            else if (j == 0)
                dp[i][j] = i;
            else if (str1[i - 1] == str2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];

            else
                dp[i][j]
                    = 1
                      + min(
                          dp[i][j - 1],
                          dp[i - 1][j],
                          dp[i - 1]
                            [j - 1]
                      );
        }
    }

    return dp[m][n];
}

int main()
{
    string str1 = "GEEXSFRGEEKKS";
    string str2 = "GEEKSFORGEEKS";

    cout << "Minimum edit distance is "
         << editDistanceDP(str1, str2, str1.length(),
                           str2.length())
         << endl;

    return 0;
}





















//lab test problems min path one diag move
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

int minPathWithOneDiagonal(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    const int INF = INT_MAX / 2;

    vector<vector<int>> dp_no_diag(m, vector<int>(n, INF));
    vector<vector<int>> dp_with_diag(m, vector<int>(n, INF));

    dp_no_diag[0][0] = grid[0][0];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > 0)
                dp_no_diag[i][j] = min(dp_no_diag[i][j], dp_no_diag[i-1][j] + grid[i][j]);
            if (j > 0)
                dp_no_diag[i][j] = min(dp_no_diag[i][j], dp_no_diag[i][j-1] + grid[i][j]);

            if (i > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_with_diag[i-1][j] + grid[i][j]);
            if (j > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_with_diag[i][j-1] + grid[i][j]);

            if (i > 0 && j > 0)
                dp_with_diag[i][j] = min(dp_with_diag[i][j], dp_no_diag[i-1][j-1] + grid[i][j]);
        }
    }

    return min(dp_no_diag[m-1][n-1], dp_with_diag[m-1][n-1]);
}

int main() {
    vector<vector<int>> grid1 = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
    vector<vector<int>> grid2 = {{1, 2}, {1, 1}};

    cout << minPathWithOneDiagonal(grid1) << endl;
    cout << minPathWithOneDiagonal(grid2) << endl;

    return 0;
}






















//lab test minimum diff exclude one element knapsack 
#include <iostream>
#include <vector>
#include <climits>
#include <cmath>
using namespace std;

int minDiffWithOneExclusion(vector<int>& arr) {
    int n = arr.size();
    int total = 0;
    for (int val : arr) total += val;

    int result = INT_MAX;

    for (int skip = -1; skip < n; ++skip) {
        int curr_total = total;
        vector<int> temp;

        for (int i = 0; i < n; ++i) {
            if (i != skip) {
                temp.push_back(arr[i]);
            } else {
                curr_total -= arr[i];
            }
        }

        vector<bool> dp(curr_total + 1, false);
        dp[0] = true;

        for (int num : temp) {
            for (int j = curr_total; j >= num; --j) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        for (int s = 0; s <= curr_total; ++s) {
            if (dp[s]) {
                int diff = abs((curr_total - s) - s);
                result = min(result, diff);
            }
        }
    }

    return result;
}

int main() {
    vector<int> arr = {1, 6, 11, 5};
    cout << "Minimum difference after excluding at most one element: " 
         << minDiffWithOneExclusion(arr) << endl;
    return 0;
}















//grid problem meeting at one point
#include <iostream>
#include <vector>
#include <algorithm>
#include<climits>
using namespace std;

int maxGridValueWithMeeting(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();

    vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(m, INT_MIN)));

    dp[0][0][0] = grid[0][0];

    for (int x1 = 0; x1 < m; ++x1) {
        for (int y1 = 0; y1 < n; ++y1) {
            for (int x2 = 0; x2 < m; ++x2) {
                int y2 = x1 + y1 - x2;

                if (y2 < 0 || y2 >= n) continue;

                if (grid[x1][y1] == -1 || grid[x2][y2] == -1) continue;

                int maxPrev = INT_MIN;

                if (x1 > 0 && x2 > 0)
                    maxPrev = max(maxPrev, dp[x1 - 1][y1][x2 - 1]);
                if (x1 > 0 && y2 > 0)
                    maxPrev = max(maxPrev, dp[x1 - 1][y1][x2]);
                if (y1 > 0 && x2 > 0)
                    maxPrev = max(maxPrev, dp[x1][y1 - 1][x2 - 1]);
                if (y1 > 0 && y2 > 0)
                    maxPrev = max(maxPrev, dp[x1][y1 - 1][x2]);

                if (maxPrev == INT_MIN) continue;

                int value = grid[x1][y1];
                if (x1 != x2 || y1 != y2)
                    value += grid[x2][y2];

                dp[x1][y1][x2] = max(dp[x1][y1][x2], maxPrev + value);
            }
        }
    }

    return max(0, dp[m - 1][n - 1][m - 1]);
}

int main() {
    vector<vector<int>> grid1 = {
        {1, 1, -1},
        {1, -1, 1},
        {1, 1, 1}
    };

    vector<vector<int>> grid2 = {
        {1, 3, 1},
        {1, 5, 1},
        {4, 2, 1}
    };

    cout << "Max value (Grid 1): " << maxGridValueWithMeeting(grid1) << endl;
    cout << "Max value (Grid 2): " << maxGridValueWithMeeting(grid2) << endl;

    return 0;
}








