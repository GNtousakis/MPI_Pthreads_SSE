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




	double timeOmegaTotalStart = gettime();
	for(unsigned int j=0;j<iters;j++)
	{
	avgF = 0.0f;
	maxF = 0.0f;
	minF = FLT_MAX;
		for(unsigned int i=0;i<N; i+=4)//check this later for any changes!!!!!!!!!!!!!!!!!!!!!!!!!!
		{

			__m128 LVecss= _mm_set_ps(LVec[i+3], LVec[i+2], LVec[i+1], LVec[i]);
			__m128 RVecss= _mm_set_ps(RVec[i+3], RVec[i+2], RVec[i+1], RVec[i]);
			__m128 mVecss= _mm_set_ps(mVec[i+3], mVec[i+2], mVec[i+1], mVec[i]);
			__m128 nVecss= _mm_set_ps(nVec[i+3], nVec[i+2], nVec[i+1], nVec[i]);
			__m128 CVecss= _mm_set_ps(CVec[i+3], CVec[i+2], CVec[i+1], CVec[i]);
			__m128 FVecss= _mm_set_ps(FVec[i+3], FVec[i+2], FVec[i+1], FVec[i]);

			variable= _mm_add_ps(LVecss, RVecss);//!
			variable1= _mm_div_ps( _mm_mul_ps(mVecss, _mm_sub_ps(mVecss,scale2))  ,  scale3);//!
			variable2= _mm_div_ps( _mm_mul_ps(nVecss, _mm_sub_ps(nVecss,scale2))  ,  scale3);//!
			variable3= _mm_div_ps(variable,_mm_add_ps(variable1,variable2));//!
			variable4=_mm_sub_ps(CVecss,_mm_sub_ps(LVecss,RVecss));//!
			variable5=_mm_mul_ps(mVecss,nVecss);//!
			variable6=_mm_div_ps(variable4,variable5);//!

			FVecss = _mm_div_ps(variable3, _mm_add_ps(variable6, scale1));//!

			float result[4];
			_mm_store_ps(result, FVecss);
			float newMax = max(max(max(result[0], result[1]), result[2]), result[3]);
			maxF = (newMax>maxF) ? newMax : maxF;
			float newMin = min(min(min(result[0], result[1]), result[2]), result[3]);
			minF = (newMin<minF) ? newMin : minF;
			float sum_all = sum(sum(sum(result[0], result[1]), result[2]), result[3]);
			avgF+=sum_all;


	// float num_0 = LVec[i]+RVec[i];
	// float num_1 = mVec[i]*(mVec[i]-1.0f)/2.0f;
	// float num_2 = nVec[i]*(nVec[i]-1.0f)/2.0f;
	// float num = num_0/(num_1+num_2);
	// float den_0 = CVec[i]-LVec[i]-RVec[i];
	// float den_1 = mVec[i]*nVec[i];
	// float den = den_0/den_1;
	// FVec[i] = num/(den+0.01f);
	// maxF = FVec[i]>maxF?FVec[i]:maxF;
	// minF = FVec[i]<minF?FVec[i]:minF;
	// avgF += FVec[i];
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