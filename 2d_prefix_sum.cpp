#include<bits/stdc++.h>
using namespace std;

int main(){
  int r = 4, c = 3;
  int matrix[r][c] = {{1, 2, 3},{4, 5, 6},{8, 10, 48},{5, 3, 20}};//eg matrix
  int pref[r+1][c+1];
  for(int i=0; i<r+1; ++i){
    for(int j=0; j<c+1; ++j){
      if(i==0 || j==0){
	pref[i][j]=0;//padding with 0
      }
      else{
	pref[i][j] = matrix[i-1][j-1] + pref[i-1][j] + pref[i][j-1] - pref[i-1][j-1];//pref sum formula
      }
    }
  }
  pair<int, int>top_left;//coordinates of the top-left element of your chunk. use indices starting from 1 not 0.
  pair<int, int>bottom_right;//coordinates of the bottom-right element of your chunk, use indices starting from 1 not 0.
  cin >> top_left.first;//the row coordinate of top left.
  cin >> top_left.second;//the column coordinate of top left.
  cin >> bottom_right.first;//the row coordinate of bottom right;
  cin >> bottom_right.second;//the column coorindate of bottom right;
  int ans = pref[bottom_right.first][bottom_right.second]-pref[top_left.first-1][bottom_right.second]-pref[bottom_right.first][top_left.second-1]+pref[top_left.first-1][top_left.second-1];//using pref sum to find chunk sum
  cout << ans;
}
