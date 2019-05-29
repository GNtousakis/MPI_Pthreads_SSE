#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <pthread.h>
#include "xmmintrin.h"

#define MINSNPS_B 5
#define MAXSNPS_E 20

#define BUSYWAIT 0
#define LOOP 1
#define EXIT 127


double gettime(void);
float randpval (void);    

//Thead stuff
#define THREADS 2
static pthread_t * workerThread; 
static pthread_barrier_t barrier;


double gettime(void)
{
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval (void)
{
	int vr = rand();
	int vm = rand()%vr;
	float r = ((float)vm)/(float)vr;
	assert(r>=0.0f && r<=1.00001f);
	return r;
}

float max(float a, float b) 
{
	if (a>b)
		return a;
	return b;
}

float min(float a, float b) 
{
	if (a<b)
		return a;
	return b;
}

float sum(float a,float b)
{
		return(a+b);
}

typedef struct 
{
	//Thread Data
	int threadID;
	int threadTOTAL;
	int threadBARRIER;
	int threadOPERATION;

	//Where the loop begins and ends
	int begin;
	int end;

	//All the data 
	float * mVec;
	float * nVec;
	float * LVec;
	float * RVec;
	float * CVec;
	float * FVec;	

	//All the necessary values to get results
	__m128 avgF;	
	__m128 maxF;	
	__m128 minF;	

}threadData_t;

void initializeThreadData(threadData_t * cur, int i, int threads,int n,float * mVec1,float * nVec1,float * LVec1,float * RVec1,float* CVec1,float* FVec1)
{
	cur->threadID=i;
	cur->threadTOTAL=threads;
	cur->threadBARRIER=0;
	cur->threadOPERATION=BUSYWAIT;

	cur->begin=(n/threads)*i; 	
	cur->end  =(n/threads)*(i+1);

	cur->mVec=mVec1;
	cur->nVec=nVec1;
	cur->LVec=LVec1;
	cur->RVec=RVec1;
	cur->CVec=CVec1;
	cur->FVec=FVec1;

	cur->avgF= _mm_set_ps(0,0,0,0);;
	cur->maxF= _mm_set_ps(0,0,0,0);;
	cur->minF= _mm_set_ps(FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX);
}


void updateThreadMMA(threadData_t * threadData)
{
	for (int unsigned i=0;i<(threadData->threadTOTAL);i+=1){
		threadData[i].maxF=_mm_set_ps(0,0,0,0);
		threadData[i].minF=_mm_set_ps(FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX);
		threadData[i].avgF=_mm_set_ps(0,0,0,0);
	}
}



void paralsin(threadData_t * threadData)
{

	__m128 variable,variable1,variable2,variable3,variable4,variable5,variable6;
	__m128 scale1 = _mm_set_ps1(0.01f);
	__m128 scale2 = _mm_set_ps1(1.0f);
	__m128 scale3 = _mm_set_ps1(2.0f);

	int     i= threadData->begin ;
	int     end= threadData->end ;	

	float * mVec=threadData->mVec;
	float * nVec=threadData->nVec;
	float * LVec=threadData->LVec;
	float * RVec=threadData->RVec;
	float * CVec=threadData->CVec;
	float * FVec=threadData->FVec;
	

	for(;i<end;i+=4)
	{

		__m128 LVecss= _mm_set_ps(LVec[i+3], LVec[i+2], LVec[i+1], LVec[i]);
		__m128 RVecss= _mm_set_ps(RVec[i+3], RVec[i+2], RVec[i+1], RVec[i]);
		__m128 mVecss= _mm_set_ps(mVec[i+3], mVec[i+2], mVec[i+1], mVec[i]);
		__m128 nVecss= _mm_set_ps(nVec[i+3], nVec[i+2], nVec[i+1], nVec[i]);
		__m128 CVecss= _mm_set_ps(CVec[i+3], CVec[i+2], CVec[i+1], CVec[i]);
		__m128 FVecss= _mm_set_ps(FVec[i+3], FVec[i+2], FVec[i+1], FVec[i]);

		variable= _mm_add_ps(LVecss, RVecss);
		variable1= _mm_div_ps( _mm_mul_ps(mVecss, _mm_sub_ps(mVecss,scale2))  ,  scale3);
		variable2= _mm_div_ps( _mm_mul_ps(nVecss, _mm_sub_ps(nVecss,scale2))  ,  scale3);
		variable3= _mm_div_ps(variable,_mm_add_ps(variable1,variable2));
		variable4=_mm_sub_ps(CVecss,_mm_sub_ps(LVecss,RVecss));
		variable5=_mm_mul_ps(mVecss,nVecss);
		variable6=_mm_div_ps(variable4,variable5);

		FVecss = _mm_div_ps(variable3, _mm_add_ps(variable6, scale1));

		threadData->avgF= _mm_add_ps(FVecss,threadData->avgF);
		threadData->maxF= _mm_max_ps(FVecss,threadData->maxF);
		threadData->minF= _mm_min_ps(FVecss,threadData->minF);
	}
}

void syncThreadsBARRIER(threadData_t * threadData)
{
	threadData[0].threadOPERATION=BUSYWAIT;
 	pthread_barrier_wait(&barrier);
}

void setThreadOperation(threadData_t * threadData, int operation)
{
	int i, threads=threadData[0].threadTOTAL;
	
	for(i=0;i<threads;i++)
		threadData[i].threadOPERATION = operation;	
}


void startThreadOperations(threadData_t * threadData, int operation)
{
	setThreadOperation(threadData, operation);

	if(operation==LOOP)
		paralsin(&threadData[0]);

	threadData[0].threadBARRIER=1;
	syncThreadsBARRIER(threadData);		
}

void terminateWorkerThreads(pthread_t * workerThreadL, threadData_t * threadData)
{
	int i, threads=threadData[0].threadTOTAL;
	
	for(i=0;i<threads;i++)
		threadData[i].threadOPERATION = EXIT;			

	for(i=1;i<threads;i++)
		pthread_join(workerThreadL[i-1],NULL);
}


void * thread (void * x)
{
	threadData_t * currentThread = (threadData_t *) x;

	int tid = currentThread->threadID;
	int threads = currentThread->threadTOTAL;
	
	while(1)
	{
		__sync_synchronize();

		if(currentThread->threadOPERATION==EXIT)
			return NULL;
		
		if(currentThread->threadOPERATION==LOOP)
		{
			paralsin (currentThread);
			currentThread->threadOPERATION=BUSYWAIT;
 			pthread_barrier_wait(&barrier);		
		}
	}
	
	return NULL;		
}
	 
int main(int argc, char ** argv)
{	
	assert(argc==2);


	float avgF = 0.0f;
	float maxF = 0.0f;
	float minF = FLT_MAX;


	double timeTotalMainStart = gettime();
	unsigned int N = (unsigned int)atoi(argv[1]);
	unsigned int iters = 10;
	srand(1);
	float * mVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(mVec!=NULL);
	float * nVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(nVec!=NULL);
	float * LVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(LVec!=NULL);
	float * RVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(RVec!=NULL);
	float * CVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(CVec!=NULL);
	float * FVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(FVec!=NULL);


	////////////////////////////////////////////////////////////
	//Threads
	int threads = THREADS;	//We pass the number of threads 

	//We are going to use Pthread-Barrier as our choise of synchronization
	int s = pthread_barrier_init(&barrier,NULL,(unsigned int)threads);	//We initialize the barrier
	assert(s == 0);
	

	workerThread=NULL;
	workerThread=(pthread_t *) malloc (sizeof(pthread_t)*((unsigned long)threads-1)); //We write the workerThread as thread-1 cause the one thread is the master

	threadData_t * threadData = (threadData_t *) malloc (sizeof(threadData_t)*((unsigned long)threads));
	assert(threadData!=NULL);

	for (int i = 0; i < threads; i++)
	{
		initializeThreadData(&threadData[i],i,threads,N,mVec,nVec,LVec,RVec,CVec,FVec);
	}

	for (int i = 1; i < threads; i++)
	{
		pthread_create(&workerThread[i-1],NULL,thread,(void*)(&threadData[i]));
	}
	////////////////////////////////////////////////////////////



	for(unsigned int i=0;i<N;i++)
	{
		mVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		nVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		LVec[i] = randpval()*mVec[i];
		RVec[i] = randpval()*nVec[i];
		CVec[i] = randpval()*mVec[i]*nVec[i];
		FVec[i] = 0.0;

		assert(mVec[i]>=MINSNPS_B && mVec[i]<=(MINSNPS_B+MAXSNPS_E));
		assert(nVec[i]>=MINSNPS_B && nVec[i]<=(MINSNPS_B+MAXSNPS_E));
		assert(LVec[i]>=0.0f && LVec[i]<=1.0f*mVec[i]);
		assert(RVec[i]>=0.0f && RVec[i]<=1.0f*nVec[i]);
		assert(CVec[i]>=0.0f && CVec[i]<=1.0f*mVec[i]*nVec[i]);
	}

	double timeOmegaTotalStart = gettime();

	for(unsigned int j=0;j<iters;j++)
	{
		updateThreadMMA(threadData);
		startThreadOperations(threadData, LOOP);		
	}

	float maxl[4];
	float minl[4];
	float suml[4];

	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();

	for (unsigned int k=0;k<threads;k++)
	{
		_mm_store_ps(maxl, (&threadData[k])->maxF);
		_mm_store_ps(minl, (&threadData[k])->minF);
		_mm_store_ps(suml, (&threadData[k])->avgF);
		maxF = max(max(max(max(maxl[0], maxl[1]), maxl[2]), maxl[3]),maxF);
		minF = max(max(max(max(maxl[0], maxl[1]), maxl[2]), maxl[3]),minF);
		avgF = max(max(max(max(maxl[0], maxl[1]), maxl[2]), maxl[3]),avgF);
		
	}

	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",
	timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,(double)avgF/N);

	terminateWorkerThreads(workerThread,threadData);
	pthread_barrier_destroy(&barrier);

	_mm_free(mVec);
	_mm_free(nVec);
	_mm_free(LVec);
	_mm_free(RVec);
	_mm_free(CVec);
	_mm_free(FVec);
}