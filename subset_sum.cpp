#include<bits/stdc++.h>
using namespace std;

bool sum_dp(vector<int> v, int V){
  bool K[v.size()+1][V+1];
  for(int i=0; i<=v.size(); ++i){
    for(int w=0; w<=V; ++w){
      if(i==0 || w==0){
	K[i][w] = false;
      }
      else if (v[i-1] <= w){
	if(K[i-1][w-v[i-1]] || v[i-1] == w){
	  K[i][w] = true;
	}
	else{
	  K[i][w] = K[i-1][w];
	}
      }
      else{
	K[i][w] = K[i-1][w];
      }
    }
  }
  return K[v.size()][V];
}


int main(){
  cout << "Enter the size of your array." << endl;
  int n;
  cin >> n;
  vector<int> v(n);
  cout << "Provide the elements of the array." << endl;
  for(int i=0; i<n; ++i){
    cin >> v[i];
  }
  cout << "Provide the value you want to sum to." << endl;
  int V;
  cin >> V;
  if(sum_dp(v, V)){
    cout << "You can!" << endl;
  }
  else{
    cout << "You can't!" << endl;
  }
}
