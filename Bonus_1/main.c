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

	float max(float a, float b) {
	if (a>b)
		return a;
	return b;
	}

	float min(float a, float b) {
	if (a<b)
		return a;
	return b;
	}
	float sum(float a,float b){
		return(a+b);
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
	 srand(1);

	 float * alldata= (float*)_mm_malloc(sizeof(float)*N*6,16)

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

	__m128 maxg;
	__m128 ming;
	__m128 sumg;

	__m128 *LVecss1= (__m128 *) LVec;
	__m128 *RVecss1= (__m128 *) RVec;
	__m128 *mVecss1= (__m128 *) mVec;
	__m128 *nVecss1= (__m128 *) nVec;
	__m128 *CVecss1= (__m128 *) CVec;
	__m128 *FVecss1= (__m128 *) FVec;

	double timeOmegaTotalStart = gettime();
	for(unsigned int j=0;j<iters;j++)
	{

		maxg= _mm_set_ps(0.0f,0.0f,0.0f,0.0f);
		ming= _mm_set_ps(FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX);
		sumg= _mm_set_ps(0.0f,0.0f,0.0f,0.0f);

		for(unsigned int i=0;i<N/4; i+=1)//check this later for any changes!!!!!!!!!!!!!!!!!!!!!!!!!!
		{

			variable= _mm_add_ps(LVecss1[i], RVecss1[i]);//!
			variable1= _mm_div_ps( _mm_mul_ps(mVecss1[i], _mm_sub_ps(mVecss1[i],scale2))  ,  scale3);//!
			variable2= _mm_div_ps( _mm_mul_ps(nVecss1[i], _mm_sub_ps(nVecss1[i],scale2))  ,  scale3);//!
			variable3= _mm_div_ps(variable,_mm_add_ps(variable1,variable2));//!
			variable4=_mm_sub_ps(CVecss1[i],_mm_sub_ps(LVecss1[i],RVecss1[i]));//!
			variable5=_mm_mul_ps(mVecss1[i],nVecss1[i]);//!
			variable6=_mm_div_ps(variable4,variable5);//!

			FVecss1[i] = _mm_div_ps(variable3, _mm_add_ps(variable6, scale1));//!

			maxg= _mm_max_ps(FVecss1[i],maxg);
			ming= _mm_min_ps(FVecss1[i],ming);
			sumg= _mm_add_ps(FVecss1[i],sumg);
			
		}
	}

	

	float maxl[4];
	float minl[4];
	float suml[4];

	_mm_store_ps(maxl, maxg);
	_mm_store_ps(minl, ming);
	_mm_store_ps(suml, sumg);

	maxF = maxl[0];
    maxF = maxl[1] > maxF ? maxl[1] : maxF;
    maxF = maxl[2] > maxF ? maxl[2] : maxF;
    maxF = maxl[3] > maxF ? maxl[3] : maxF;

    minF = minl[0];
    minF = minl[1] < minF ? minl[1] : minF;
    minF = minl[2] < minF ? minl[2] : minF;
    minF = minl[3] < minF ? minl[3] : minF;

    avgF = suml[0] + suml[1] + suml[2] + suml[3];

/*
	maxF = max(max(max(maxl[0], maxl[1]), maxl[2]), maxl[3]);
	minF = min(min(min(minl[0], minl[1]), minl[2]), minl[3]);
	avgF = sum(sum(sum(suml[0], suml[1]), suml[2]), suml[3]);
*/
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