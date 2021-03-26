#include <bits/stdc++.h>
using namespace std;

int flag = 0;
map<int, int>mp;

void check(int edge, vector<pair<int, int>>G_1, vector<pair<int, int>>G_2){
  int count = 0;
  for(int i=0; i<edge; ++i){
    int first = mp[G_1[i].first];
    int second = mp[G_1[i].second];
    for(int j=0; j<edge; ++j){
      if((first == G_2[j].first && second == G_2[j].second) || (first == G_2[j].second && second == G_2[j].first)){
	count++;
      }
    }
  }
  if(count == edge){
    cout << endl;
    cout << "The graphs are isomorphic." << endl << endl;
    cout << "The one-one onto function is: " << endl << endl;
    cout << "Graph 1  |  " << "Graph 2" << endl << endl;
    for(auto itr=mp.begin(); itr!=mp.end(); ++itr){
      cout << "   " << itr->first << " --------> " << itr->second << endl;
    }
    cout << endl;
    exit(1);
  }
}

void combinations(int a[], int n, vector<pair<int, int>>G_1, vector<pair<int, int>>G_2, int edge){
  for(int i=0; i<n; ++i){
    if(a[i]==0){
      if(flag == n-1){
	mp[flag] = i;
	check(edge, G_1, G_2);
	break;
      }
      a[i] = 1;
      mp[flag] = i;
      flag++;
      combinations(a, n, G_1, G_2, edge);
      a[i] = 0;
      if(flag != 0){
	flag--;
      }
    }
  }
}

int main(){
  int n_1, n_2;
  cout << "Number of vertices for the first graph: ";
  cin >> n_1;
  cout << "Number of vertices for the second graph: ";
  cin >> n_2;
  if(n_1 != n_2){
    cout << "The graphs aren't isomorphic." << endl;
    return 0;
  }
  int e_1, e_2;
  cout << "Number of edges for the first graph: ";
  cin >> e_1;
  cout << "Number of edges for the second graph: ";
  cin >> e_2;
  if(e_1 != e_2){
    cout << "The graphs aren't isomorphic." << endl;;
    return 0;
  }
  vector<pair<int, int>>G_1(e_1);
  vector<pair<int, int>>G_2(e_2);
  cout << "Please provide all the edges for the first graph." << endl;
  for(int i=0; i<e_1; ++i){
    pair<int, int>input;
    cout << "Edge " << i+1 << ":" << endl;
    cout << "    " << "First Vertex: ";
    cin >> input.first;
    cout << "    " << "Second Vertex: ";
    cin >> input.second;
    G_1[i]=input;
  }
  cout << endl << "Please provide all the edges for the second graph." << endl;
  for(int i=0; i<e_2; ++i){
    pair<int, int>input;
    cout << "Edge " << i+1 << ":" << endl;
    cout << "    " << "First Vertex: ";
    cin >> input.first;
    cout << "    " << "Second Vertex: ";
    cin >> input.second;
    G_2[i]=input;
  }
  int freq_arr[n_1] = {0};
  combinations(freq_arr, n_1, G_1, G_2, e_1);
  cout << "The graphs aren't isomorphic." << endl;
}
