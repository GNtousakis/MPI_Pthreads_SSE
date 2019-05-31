#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include "xmmintrin.h"

#define MINSNPS_B 5
#define MAXSNPS_E 20

double gettime(void);
float randpval (void);

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

int main(int argc, char ** argv)
{

	assert(argc==2);
	double timeTotalMainStart = gettime();
	float avgF = 0.0f;
	float maxF = 0.0f;
	float minF = FLT_MAX;

	unsigned int N = (unsigned int)atoi(argv[1]);
	unsigned int iters = 10;
	unsigned int leftover= N%4;

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
	float * maxg = (float*)_mm_malloc(sizeof(float),16);
	assert(maxg!=NULL);
	float * ming = (float*)_mm_malloc(sizeof(float),16);
	assert(ming!=NULL);
	float * sumg = (float*)_mm_malloc(sizeof(float),16);
	assert(sumg!=NULL);

	//We give some values to max min sum so we can compare with real values
	for (int i = 0; i < 4; ++i)
	{
		maxg[i]=0.0f;
		ming[i]=FLT_MAX;
		sumg[i]=0.0f;
	}

	//Initialize the data
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


	__m128 variable,variable1,variable2,variable3,variable4,variable5,variable6;


	__m128 scale1 = _mm_set_ps1(0.01f);
	__m128 scale2 = _mm_set_ps1(1.0f);
	__m128 scale3 = _mm_set_ps1(2.0f);


	__m128 *LVecss1= (__m128 *) LVec;
	__m128 *RVecss1= (__m128 *) RVec;
	__m128 *mVecss1= (__m128 *) mVec;
	__m128 *nVecss1= (__m128 *) nVec;
	__m128 *CVecss1= (__m128 *) CVec;
	__m128 *FVecss1= (__m128 *) FVec;

	__m128 *maxg1= (__m128 *) maxg;
	__m128 *ming1= (__m128 *) ming;
	__m128 *sumg1= (__m128 *) sumg;



	double timeOmegaTotalStart = gettime();

	for(unsigned int j=0;j<iters;j++)
	{

		avgF = 0.0f;
		maxF = 0.0f;
		minF = FLT_MAX;

		for(unsigned int i=0;i<N/4; i+=1)//check this later for any changes!!!!!!!!!!!!!!!!!!!!!!!!!!
		{

			variable= _mm_add_ps(LVecss1[i], RVecss1[i]);

			variable1= _mm_sub_ps(mVecss1[i],scale2);
			variable1= _mm_div_ps(variable1 ,scale3);
			variable1= _mm_mul_ps(mVecss1[i],variable1);
			

			variable2= _mm_sub_ps(nVecss1[i],scale2);
			variable2= _mm_div_ps(variable2 ,scale3);
			variable2= _mm_mul_ps(nVecss1[i],variable2);

			variable3=_mm_add_ps(variable1,variable2);
			variable3= _mm_div_ps(variable,variable3);


			variable4=_mm_sub_ps(CVecss1[i],LVecss1[i]);
			variable4=_mm_sub_ps(variable4,RVecss1[i]);
			
			variable5=_mm_mul_ps(mVecss1[i],nVecss1[i]);//!

			variable6=_mm_div_ps(variable4,variable5);//!

			FVecss1[i]=_mm_add_ps(variable6 ,scale1);
			FVecss1[i]= _mm_div_ps(variable3,FVecss1[i]);//!

			*maxg1= _mm_max_ps(*maxg1,FVecss1[i]);
			*ming1= _mm_min_ps(FVecss1[i],*ming1);
			*sumg1= _mm_add_ps(FVecss1[i],*sumg1);
			
		}


		maxF = maxg[0];
   		maxF = maxg[1] > maxF ? maxg[1] : maxF;
   		maxF = maxg[2] > maxF ? maxg[2] : maxF;
   		maxF = maxg[3] > maxF ? maxg[3] : maxF;

   		minF = ming[0];
   		minF = ming[1] < minF ? ming[1] : minF;
   		minF = ming[2] < minF ? ming[2] : minF;
   		minF = ming[3] < minF ? ming[3] : minF;

   		avgF = sumg[0] + sumg[1] + sumg[2] + sumg[3]; 	


   		//We fix the left overs
   		for(unsigned int i=N-leftover;i<N;i++)
		{
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
	timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,(double)avgF/N);
	_mm_free(mVec);
	_mm_free(nVec);
	_mm_free(LVec);
	_mm_free(RVec);
	_mm_free(CVec);
	_mm_free(FVec);

}