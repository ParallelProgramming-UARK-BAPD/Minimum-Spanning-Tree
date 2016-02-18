/*
 * mst.cpp
 *
 *  Created on: Feb 5, 2016
 *      Authors: Brice Alexander, Patrick Dillingham
 Parallel Boruvka pseudocode
 size(V)=6,400
 max degree of vertex = 1000
 P0 reads “graph.txt” and constructs a matrix of adjacencies
 this matrix is scattered
 P0 constructs the head node array
 each Pi constructs local adjacency list for each local V
 P0 allocates space for the MST which is an array of edges of size V-1
 while size of MST < V-1
        P0 broadcasts the head node array
        each Pi updates the use of each edge from each of its vertices,
        finds light edge, and sends that edge to P0
        P0 receives the edges to be added to the MST,
        updates the head node array (pointer jumping),
        and broadcasts the updated array
        note that some processors will become idle
 P0 writes the edges in he MST to the file “mst.txt”
 */

#include "mpi.h"
#include <string>
#include <iostream>
#include <fstream>
#include <istream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <queue>
#include <climits> //int max

using namespace std;

#define V 32
#define MD 64

//where V is number vertices and MD is max degree of any vertex

//flags for quick debugging
#define PRINTING true
#define MPI true

//flags for quick shortcuts
#define MINT MPI_INTEGER
#define MCOMM MPI_COMM_WORLD
#define MIGN MPI_STATUS_IGNORE

MPI_Datatype mpi_edge;

struct edge {
    int v1, v2;
    int weight;
    int use;
};

class compareEdgeWeights
{
	public:
		compareEdgeWeights(){}
		bool operator()(const edge& a, const edge& b) {
		  return a.weight > b.weight;
		}
};

typedef priority_queue<edge, vector<edge>, compareEdgeWeights> edgepq;

//true if usable
bool check_edge(edge e, int headNodeArray[]) {
    int v1Head = e.v1;
    int v2Head = e.v2;
    int index;

    if(e.v1 == e.v2) return false;

    //find v1's head for set
    while(headNodeArray[v1Head] != v1Head){
        v1Head = headNodeArray[v1Head];
    }

    //find v2's head for set
    while(headNodeArray[v2Head] != v2Head){
        v2Head = headNodeArray[v2Head];
    }

    if(v1Head == v2Head){
        return false; //same head so unusable
    }else {
        headNodeArray[v2Head] = v1Head;
        return true; //not same set (but it is now)
    }
}

//print HEad node array
void printHNA(int headNodeArray[]){
  cout << "HNA: ";
  for(int x = 0; x < V; x++){
    cout << x<<"="<<headNodeArray[x] << " ,";
  }
  cout << endl;
}

//print final MST
void printMST(vector<edge>& MST){
  for (int x = 0; x < MST.size(); x++)
    cout << "MST Edge from v" << MST[x].v1 << " to v" << MST[x].v2
          << ", weight "<<MST[x].weight << endl;
}

/*send an edge to process 0*/
void pass_change(edge e) {

	int edgedata[] = {e.v1, e.v2, e.weight};
  MPI_Send(edgedata, 3, MINT, 0, 0, MCOMM); //send data to process 0
}

/*returns true if still valid edges, false if all done , receive edges from processors
 and update them all with the edges from other processors*/
bool dist_change(int size, int headNodeArray[], edge localEdge, vector<edge>& MST) {
	  int recbuff[3];
    edge tempEdge;
    int validEdges = 0;  //keeps track of how many procs are sending real edges

    priority_queue<edge, vector<edge>, compareEdgeWeights> edgepq;

    //p0 gets each edge
    if(localEdge.v1 != localEdge.v2) {
        edgepq.push(localEdge);
        validEdges++;
    }
    for (int x = 1; x < size; x++) {
        //receive data from each processor, x
        MPI_Recv(recbuff, 3, MINT, x, 0, MCOMM, MIGN);

        //if valid edge
        if(recbuff[0] != recbuff[1]){
            tempEdge.v1 = recbuff[0];
            tempEdge.v2 = recbuff[1];
            tempEdge.weight = recbuff[2];
            edgepq.push(tempEdge);
            validEdges++;
        }
        //if (PRINTING) cout << "P0 received v"<<tempEdge.v1<<" v"<<tempEdge.v2<<" w"<<tempEdge.weight<<" from p"<<x<<" Qsize "<<edgepq.size()<<endl;

    }

    //p0 grabs least weight edge, checks sets and updates headNodeArray
    while( !edgepq.empty()){
        tempEdge = edgepq.top();
        //if not already in the same set
        if (check_edge(tempEdge, headNodeArray)){
              MST.push_back(tempEdge);  //add the edge to the MST!
              //cout << "combined v"<<tempEdge.v1<<" v"<<tempEdge.v2<<", "<<headNodeArray[tempEdge.v1]<<"="<<headNodeArray[tempEdge.v2]<<endl;
        }
        edgepq.pop();
    }

    //return valid edges found or not
    if (validEdges > 0){
      return true;
    } else{
      return false;
    }

}//end dist_change

void MSTSort(int* list, int rank, int size, int localnvert) {

    //head node array for each process
    int headNodeArray[V+1]; //the last element of head node array tells the processes to continue or are they all done (0 == done)
    int localEdgeCount = 0;
    edge fakeEdge;
    //construct MST = array of edges of size V - 1
    vector<edge> MST;

    //process 0
    if (rank == 0) {

        //construct head node array for p0
        for (int x = 0; x < V; x++) {
            //each vertex points to itself initially
            headNodeArray[x] = x;
        } headNodeArray[V] = 1; //flag for continuing

    }//end if rank 0

    //local matrix for each proc's vertices it gets
    int localArraySize = localnvert * (MD * 2);
    int localList[localArraySize];

    //scatter the data into the local array(s)
    MPI_Scatter(list, localArraySize, MPI_INT, localList, localArraySize, MPI_INT, 0, MPI_COMM_WORLD);

    //broadcast the head node array
    MPI_Bcast(headNodeArray, V, MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);

/*    //PRINT LOCAL list (only non zero edges for each process)
    if( PRINTING ){
    cout << "\nPROCESS " << rank <<
            ":: localArraySize::" << localArraySize <<
            ":: localnvert::" << localnvert;
    for (int x = 0; x < localArraySize; x+=2) {
        if (x % (MD*2) == 0 ) cout << "\n\n";
        if (localList[x+1] != 0){
            cout << localnvert * rank + (x / (MD*2))
                 << "->" << localList[x] << ":" << localList[x+1] << "   "; }
    } cout << "\n\n";
  }//end print
*/
    //priority q for each process' edges
    priority_queue<edge, vector<edge>, compareEdgeWeights> edgepq;

    int x = 0; //array offset
    //for each vertex locally owned
    for (int i = 0; i < localArraySize; i += MD*2){
        x = i; //jump to vertex i data in local array
        //while non zero weight from array
        while (localList[x+1] != 0){

          edge temp;
          temp.v1 = (localnvert * rank + (x / (MD*2))); //vertex 1
          temp.v2 = localList[x];
          temp.weight = localList[x+1];
          temp.use = 2;
          edgepq.push(temp);

          x+=2;
          localEdgeCount++;
        }
    }
    //if (PRINTING) cout << localEdgeCount << " localEdgeCount " << " edgepqsize = "<< edgepq.size()<<endl;

    //while processor has more edges
    //while(! edgepq.empty() ){
    while(headNodeArray[V] != 0){
      edge least;
      fakeEdge.v1 = 0; fakeEdge.v2 = 0; fakeEdge.weight = INT_MAX;  //processor is done with valid edges

      if (edgepq.empty()){
        least = fakeEdge;
      }else{
        //find least weight, usable edge
        least = edgepq.top();
        while( !check_edge(least, headNodeArray) && !edgepq.empty()){
            edgepq.pop();
            if( edgepq.empty() ) {
              //cout << "PROCESS DONE = " << rank<<endl;
              least = fakeEdge;
            }else {
              least = edgepq.top();
            }
        }
      }
        //temp printing of found least edges
        //if(PRINTING) cout << "p" << rank<< " found weight " << least.weight <<" "<< least.v1 << "->"<< least.v2 << endl;

        //send least edge to p0
        if(rank != 0){
            //SEND EDGE
            pass_change(least);
            //cout << "\nleft pass_change:"<<rank<<endl;

        } else if(rank == 0){
            //receive edges, and passes it's own leasst weight edge as well
            if( !dist_change(size, headNodeArray, least, MST)){
                headNodeArray[V] = 0; //all done flag
            }
            //cout << "\nleft distchange"<<endl;
        }

        //remove edge
        if (!edgepq.empty())
          edgepq.pop();
        localEdgeCount--;

        //update new Head node array to procs
        MPI_Bcast(headNodeArray, V+1, MPI_INT, 0, MPI_COMM_WORLD);
        //if (PRINTING) {cout << rank; printHNA(headNodeArray);}
    }

    //print the final MST
    if (PRINTING && rank == 0) printMST(MST);

}//end MSTSort

int main(int argc, char** argv) {
    int nvert;
    int nedge;
    int *elist;
    int v1, v2, weight;

    int rank = 0, size = 0;

    MPI_Init(&argc, &argv);

    MPI_Type_contiguous(3, MPI_INT, &mpi_edge);
    MPI_Type_commit(&mpi_edge);

    double starttime, endtime;
    starttime = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localnvert = V / size;

    if (rank == 0) {

        ifstream graphfile("graph.txt");
        string line;
        if (graphfile.is_open()) {
            graphfile >> nvert;
            graphfile >> nedge;

            elist = new int[nvert * (MD * 2)];

            int x = 0;
            for (int i = 0; i < nedge; i++) {
                int v1, v2, w;
                graphfile >> v1 >> v2 >> w;

                int offset = 0;
                while (elist[(v1 * MD * 2) + offset] != 0) {
                    offset += 2;
                }

                elist[(v1 * MD * 2) + offset] = v2;
                elist[(v1 * MD * 2) + offset + 1] = w;

                offset = 0;
                while (elist[(v2 * MD * 2) + offset] != 0) {
                    offset += 2;
                }

                elist[(v2 * MD * 2) + offset] = v1;
                elist[(v2 * MD * 2) + offset + 1] = w;

                x++;
            }//end for
            graphfile.close();
        }//end if open
    }//end if rank 0

    MSTSort(elist, rank, size, localnvert);

    MPI_Finalize();
    return 0;
}//EOF
