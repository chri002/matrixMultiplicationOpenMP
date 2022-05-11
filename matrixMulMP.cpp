/*
	##################################################
	##												##
	##			    PRODOTTO MATRICIALE				##
	##				      OPENMP					##
	##												##
	##################################################

cl -openmp .\matrixMulMP.cpp /D C /D NS /D FILEOUT /D NTHR=10 /D VISC /link /HEAP:0x4096


C       : execute the multiplication
DEBUG   : show all
TEST    : execute multiplication and save the output into txt file
FILEOUT : export into txt file the output matrix
NTHR  	: thread count for moltiplication
N		: number of rows
M  		: number of coloums
MIFN    : memory info 
NS      : show debug thread info
VISAB   : show the original matrix
VISC    : show the output matrix
STEP    : show the step


/link /HEAP:0x4096 : increase the memory

*/




#include <omp.h>
#include <stdio.h>
#include <Math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>



using namespace std;

#ifndef NTHR
const int NTHR = 8;
#endif
#ifndef N
#define N 100
#endif
#ifndef M
#define M 100
#endif
#ifdef DEBUG
#define STEP
#define NS
#define C
#define VISAB
#define VISC
#define FILEOUT
#endif
#ifdef TEST
#define NS
#define C
#define FILEOUT
#endif


chrono::time_point<chrono::high_resolution_clock> st;


int toIntTime(){
	auto timer = chrono::high_resolution_clock::now();
	
	
	
	return (chrono::duration_cast<chrono::nanoseconds>(timer-st)).count();
}

vector<vector<int>> multiply(vector<vector<int>> a, vector<vector<int>> b, vector<vector<int>> c, int dimX, int dimY){
	
	int i=0,j=0,k=0;
	#ifdef STEP
		printf("OP::Init Multiplication (%d ns)\n",toIntTime());
	#endif
	#pragma omp parallel shared(a,b,c)  private(i,j,k) num_threads(NTHR)
	{
				
		int m=0,s=0;
		int maxYX = fmax(dimY,dimX);
		
		if(dimX<=omp_get_num_threads()){
			m=omp_get_thread_num();
			s=(m)+1;
		}else{
			m = (ceil)((float)(dimX)/(float)(omp_get_num_threads()))*omp_get_thread_num();
			s=m+(ceil)((float)(dimX)/(float)(omp_get_num_threads()));	
			
		}
		
		s=(s>=dimX? dimX:s);
		#if defined(NS)
		printf("Thread: (%d, %d, %d, %d, %d)\n",omp_get_thread_num(), m, s, dimX, dimY);
		if(m>=s) printf("Errore %d\n", omp_get_thread_num());
		#endif
		
		#ifdef STEP
			printf("OP::Start Multiplication thread %d\n",omp_get_thread_num());
		#endif
		
		if(m<s)
			for(i = m; i<s; i++){
				//printf("%d\n",i);
				for(j = 0; j<maxYX; j++){
					c[i][j] = 0;
					//#pragma omp critical
					for(k=0; k<M;k++){
						c[i][j] += (a[i][k]*b[k][j]); 
					}
					if(c[i][j]!=M) printf("%d %d %d\n",i,j,c[i][j]);
				}
			}
		#ifdef STEP
			printf("OP::End Multiplication thread %d\n",omp_get_thread_num());
		#endif
	}
	
	#ifdef STEP
		printf("OP::End Multiplication (%d ns)\n",toIntTime());
	#endif
	
	return c;
}

int main(){
	
	
	#ifdef STEP
		st = chrono::high_resolution_clock::now();
		printf("OP::Start (%d ns)\n",toIntTime());
	#endif
	
	vector<vector<int>> a(N);
	vector<vector<int>> b(M);
	vector<vector<int>> c(N);
	int dimX = N;
	int dimY = M;
	int i,j,k;
	#if defined(FILEOUT)
	ofstream myfile;
	string strFile = "matrice"+to_string(N)+"x"+to_string(N)+".txt";
	myfile.open(strFile, ios::out | ios::trunc);
	#endif	
	
	#ifdef STEP
		printf("OP::Init Matrix\n");
	#endif
	
	#ifdef MIFN
	int tm = (N*N+N*M*2)*(sizeof(long));
	printf("Memory ~: %d %cb\n", tm/ (tm>1024? (tm>1024*1024? 1024*1024:1024):1),(tm>1024? (tm>1024*1024? 'M':'k'):' '));
	#endif
	
	
	for(i=0; i<N; i++){
		a[i].resize(M);
		for(j =0; j<M; j++){ a[i][j]=1;}
	}
	for(i=0;i<M; i++){
		b[i].resize(N);
		for(j =0; j<N; j++) b[i][j]=1;
	}
	for(i=0; i<N; i++){
		c[i].resize(N);
		for(j =0; j<N; j++) c[i][j]=0;
	}
		
	
	#ifdef STEP
		printf("OP::End Init Matrix\n");
	#endif
	
	#if defined(C)
	
	c=multiply(a,b,c, dimX, dimX);
	
	
	#endif
	
	int maxVal = 1, t=c[N-1][N-1];
	while(t!=0){
		maxVal++;
		t=t/10;
	}
	
	
	#ifdef STEP
		printf("OP::Init Output (%d ns)\n",toIntTime());
	#endif
	for(i = 0; i<fmax(dimX,dimY); i++){
		#if defined(VISAB)
		for(j = 0; j<dimY; j++){
			if(i<dimX && j<dimY)
				printf("% *d",maxVal,a[i][j]);
			
			else 
				printf("% *c",maxVal,' ');
		}
		printf("\t");
		for(j = 0; j<dimX; j++){
			if(i<dimY)
				printf("% *d",maxVal,(b[i][j]));
			else 
				printf("% *c",maxVal,' ');
			
		}
		printf("\t");
		#endif
		#if defined(VISC) || defined(FILEOUT)
		for(j = 0; j<dimX; j++){
			if(i<dimX){
				#if defined(VISC)
					printf("% *d",maxVal,(c[i][j]));
				#endif
				#if defined(FILEOUT)
					myfile << c[i][j];
					myfile << " ";
				#endif		
			}			
		}
		#endif
		#if defined(VISAB)
		if(i>dimX)
			printf("\n");
		#endif
		#if defined(VISC) || defined(FILEOUT)
		if(i<=dimX){
			#if defined(VISC)
				printf("\n");
			#endif
			#if defined(FILEOUT)
				myfile<<"\n";
			#endif	
			}
		#endif
	}
	#if defined(FILEOUT)
		myfile.close();
	#endif	
	
	#ifdef STEP
		printf("OP::End Output (%d ns)\n",toIntTime());
	#endif
	
	
	
}