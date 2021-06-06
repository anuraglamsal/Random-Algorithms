  <b><i>A naive algorithm to check if two graphs are isomorphic:</i></b><br>

  From rough calculations, I think the number of iterations in worst case of this algorithm is n<sup>2</sup>•n!•E<sup>2</sup>+n where 'n' is the number of vertices
  and 'E' is the number of edges. The way to use the algorithm is by first counting the number of vertices and edges in your graph. Then give those as inputs. 
  If they are accepted, then you need to do the following:<br>
  
  * Change the labels of the vertices of your first graph using numbers from 0 to n-1 where 'n' is the number of vertices. Each vertex should have a unique label. It doesn't 
    matter what number you attach to a vertex; just make sure that the label of each vertex is unique. 
    
  * Do the same as above for the second graph. 
  
  * Now, provide all the edges of both of those graphs as per asked using the new lables for each vertex. 
  
  Then that is it. At the end, the algorithm will print a one-one function between the vertices of the graphs if they are isomorphic, and if they aren't isomorphic
  then it will say so. Now, you have to manually convert the labels of vertices in the one-one function given at the end to their original labels if you care
  about that. Therefore, it is good to note down the relationship of the new label of each vertex with its original label when you're labeling them. <br>

  <b><i>Subset sum problem using knapsack dp:</i></b><br>
  
  If "S" is the number of elements in your array and "V" is the value that you want to sum to, the number of iterations is (S+1)•(V+1).
