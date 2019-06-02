#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <pthread.h>
#include "xmmintrin.h"
#include <mpi.h>

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
	__m128 * mVec;
	__m128 * nVec;
	__m128 * LVec;
	__m128 * RVec;
	__m128 * CVec;
	__m128 * FVec;	

	//All the necessary values to get results
	float * maxg;
	float * ming;
	float * sumg;

	__m128 * avgF;	
	__m128 * maxF;	
	__m128 * minF;	

	float thread_max;
	float thread_min;
	float thread_sum;
	



}threadData_t;

void initializeThreadData(int world_rank,int start,int end,threadData_t * cur, int i, int threads,int n,float * mVec1,float * nVec1,float * LVec1,float * RVec1,float* CVec1,float* FVec1)
{

    int tasksperthread=n/threads;
	cur->threadID=i;
	cur->threadTOTAL=threads;
	cur->threadBARRIER=0;
	cur->threadOPERATION=BUSYWAIT;

	///We find the limits of loop
    
	cur->begin=tasksperthread*i; 	
	cur->end  =tasksperthread*i+tasksperthread;

	cur->mVec= (__m128 *) mVec1;
	cur->nVec= (__m128 *) nVec1;
	cur->LVec= (__m128 *) LVec1;
	cur->RVec= (__m128 *) RVec1;
	cur->CVec= (__m128 *) CVec1;
	cur->FVec= (__m128 *) FVec1;

	cur->maxg = (float*)_mm_malloc(sizeof(float),16);
	cur->ming = (float*)_mm_malloc(sizeof(float),16);
	cur->sumg = (float*)_mm_malloc(sizeof(float),16);


	for (int i = 0; i < 4; ++i)
	{
		cur->maxg[i]=0.0f;
		cur->ming[i]=FLT_MAX;
		cur->sumg[i]=0.0f;
	}

	cur->avgF= (__m128 *) cur->sumg;
	cur->maxF= (__m128 *) cur->maxg;
	cur->minF= (__m128 *) cur->ming;

	cur->thread_max=0.0f;
	cur->thread_min=FLT_MAX;
	cur->thread_sum=0.0f;
}


void updateThreadMMA(threadData_t * threadData)
{
	for (int unsigned i=0;i<(threadData->threadTOTAL);i+=1){
		for (int l = 0; l < 4; ++l)
		{
			threadData[i].maxg[l]=0.0f;
			threadData[i].ming[l]=FLT_MAX;
			threadData[i].sumg[l]=0.0f;
		}

		threadData[i].thread_max=0.0f;
		threadData[i].thread_min=FLT_MAX;
		threadData[i].thread_sum=0.0f;	
	
	}

}



void paralsin(threadData_t * threadData)
{

	__m128   variable,variable1,variable2,variable3,variable4,variable5,variable6;
	__m128   scale1 = _mm_set_ps1(0.01f);
	__m128 	 scale2 = _mm_set_ps1(1.0f);
	__m128   scale3 = _mm_set_ps1(2.0f);

	  int    i= threadData->begin ;
	  int    end= threadData->end ;	

	__m128 * mVec=threadData->mVec;
	__m128 * nVec=threadData->nVec;
	__m128 * LVec=threadData->LVec;
	__m128 * RVec=threadData->RVec;
	__m128 * CVec=threadData->CVec;
	__m128 * FVec=threadData->FVec;
	
	__m128 * sumg=threadData->avgF;
	__m128 * maxg=threadData->maxF;
	__m128 * ming=threadData->minF;


	for(i=i/4;i<end/4;i+=1)
	{
		variable= _mm_add_ps(LVec[i], RVec[i]);

		variable1= _mm_sub_ps(mVec[i],scale2);
		variable1= _mm_div_ps(variable1 ,scale3);
		variable1= _mm_mul_ps(mVec[i],variable1);
			

		variable2= _mm_sub_ps(nVec[i],scale2);
		variable2= _mm_div_ps(variable2 ,scale3);
		variable2= _mm_mul_ps(nVec[i],variable2);

		variable3=_mm_add_ps(variable1,variable2);
		variable3= _mm_div_ps(variable,variable3);


		variable4=_mm_sub_ps(CVec[i],LVec[i]);
		variable4=_mm_sub_ps(variable4,RVec[i]);
			
		variable5=_mm_mul_ps(mVec[i],nVec[i]);//!

		variable6=_mm_div_ps(variable4,variable5);//!

		FVec[i]=_mm_add_ps(variable6 ,scale1);
		FVec[i]= _mm_div_ps(variable3,FVec[i]);//!

	

		*maxg= _mm_max_ps(*maxg,FVec[i]);
		*ming= _mm_min_ps(FVec[i],*ming);
		*sumg= _mm_add_ps(FVec[i],*sumg);
	}
		

	//The nessasary redution
	threadData->thread_max = threadData->maxg[0];
   	threadData->thread_max = threadData->maxg[1] > threadData->thread_max ? threadData->maxg[1] : threadData->thread_max;
   	threadData->thread_max = threadData->maxg[2] > threadData->thread_max ? threadData->maxg[2] : threadData->thread_max;
   	threadData->thread_max = threadData->maxg[3] > threadData->thread_max ? threadData->maxg[3] : threadData->thread_max;

   	threadData->thread_min = threadData->ming[0];
   	threadData->thread_min = threadData->ming[1] < threadData->thread_min ? threadData->ming[1] : threadData->thread_min;
   	threadData->thread_min = threadData->ming[2] < threadData->thread_min ? threadData->ming[2] : threadData->thread_min;
   	threadData->thread_min = threadData->ming[3] < threadData->thread_min ? threadData->ming[3] : threadData->thread_min;
	
   	threadData->thread_sum = threadData->sumg[0] + threadData->sumg[1] + threadData->sumg[2] + threadData->sumg[3]; 

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
	assert(argc==3);
	int threads = (int)atoi(argv[2]);
	float avgF = 0.0f;
	float maxF = 0.0f;
	float minF = FLT_MAX;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
  	int name_len;
  	MPI_Get_processor_name(processor_name,&name_len);
	double timeTotalMainStart = gettime();
	unsigned int N = (unsigned int)atoi(argv[1]);

	unsigned int iters = 10;
	int k=N%4;


	int processSize = (N/4) / world_size;
    int start = processSize * world_rank;
    int reamaining_calc=N%world_size;
    int end = start + processSize;


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
	


	//Threads
	////////////////////////////////////////////////////////////

	//We are going to use Pthread-Barrier as our choise of synchronization
	int s = pthread_barrier_init(&barrier,NULL,(unsigned int)threads);	//We initialize the barrier
	assert(s == 0);
	

	workerThread=NULL;
	workerThread=(pthread_t *) malloc (sizeof(pthread_t)*((unsigned long)threads-1)); //We write the workerThread as thread-1 cause the one thread is the master

	threadData_t * threadData = (threadData_t *) malloc (sizeof(threadData_t)*((unsigned long)threads));
	assert(threadData!=NULL);

	for (int i = 0; i < threads; i++)
	{
		initializeThreadData(world_rank,start,end,&threadData[i],i,threads,N,mVec,nVec,LVec,RVec,CVec,FVec);
	}

	for (int i = 1; i < threads; i++)
	{
		pthread_create(&workerThread[i-1],NULL,thread,(void*)(&threadData[i]));
	}
	////////////////////////////////////////////////////////////

	//Initialize data
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
	////////////////////////////////////////////////////////////


	double timeOmegaTotalStart = gettime();


	for(unsigned int j=0;j<iters;j++)
	{
		updateThreadMMA(threadData);
		startThreadOperations(threadData, LOOP);
	}


	for (unsigned int k=0;k<threads;k++)
	{
		maxF =  (&threadData[k])->thread_max>maxF?(&threadData[k])->thread_max:maxF;
		minF =  (&threadData[k])->thread_min<minF?(&threadData[k])->thread_min:minF;
		avgF += (&threadData[k])->thread_sum;	
	}

	// unsigned int leftover= (((&threadData[threads-1])->end) /4) *4;
	// //We fix the left overs
 //   	for(unsigned int i=leftover;i<N;i++)
	// {
	// 	float num_0 = LVec[i]+RVec[i];
	// 	float num_1 = mVec[i]*(mVec[i]-1.0f)/2.0f;
	// 	float num_2 = nVec[i]*(nVec[i]-1.0f)/2.0f;
	// 	float num = num_0/(num_1+num_2);

	// 	float den_0 = CVec[i]-LVec[i]-RVec[i];
	// 	float den_1 = mVec[i]*nVec[i];
	// 	float den = den_0/den_1;

	// 	FVec[i] = num/(den+0.01f);
			
	// 	maxF = FVec[i]>maxF?FVec[i]:maxF;
	// 	minF = FVec[i]<minF?FVec[i]:minF;
	// 	avgF += FVec[i];
	// }	


        if (!world_rank) {
	
        start = N-k;
        end = N;
        for (int i = start; i < end; i++) {
        //use scalar (traditional) way to compute remaining of array ( N%4 iterations )
        float num_0 = LVec[i]+RVec[i];
		float num_1 = mVec[i]*(mVec[i]-1.0f)/2.0f;
		float num_2 = nVec[i]*(nVec[i]-1.0f)/2.0f;
		float num = num_0/(num_1+num_2);

		float den_0 = CVec[i]-LVec[i]-RVec[i];
		float den_1 = mVec[i]*nVec[i];
		float den = den_0/den_1;

		FVec[i] = num/(den+0.01f);
			
		maxF = FVec[i]>maxF?FVec[i]:maxF;
		minF = FVec[i]<minF?FVec[i]:minF;
		avgF += FVec[i];

          }
	}

	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();

	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",
	timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,(double)avgF);

	terminateWorkerThreads(workerThread,threadData);
	pthread_barrier_destroy(&barrier);

    float *processMax = world_rank ? NULL : (float *) malloc(sizeof(float) * world_size);
    MPI_Gather(&maxF, 1, MPI_FLOAT, processMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Finalize the MPI environment.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    if (!world_rank) {
        float globalMax = 0;
        for (int i = 0; i < world_size; i++) {
            globalMax = globalMax < processMax[i] ? processMax[i] : globalMax;
        }
		//printf("Time: %f us, Max %f\n", (timeTotal / iters)*1000000, globalMax);
        // printf("Time %f Max %f\n", timeTotal / iters, globalMax);
    }


    free(processMax);
	_mm_free(mVec);
	_mm_free(nVec);
	_mm_free(LVec);
	_mm_free(RVec);
	_mm_free(CVec);
	_mm_free(FVec);
}
