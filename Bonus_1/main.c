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


	// mVec nVec LVec RVec CVec FVec
	float * alldata = (float*)_mm_malloc(sizeof(float)*N*6,16);
	assert(alldata!=NULL);

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
	//  mVec 0  nVec 1 LVec 2 RVec 3 CVec 4 FVec 5

	for(unsigned int i=0;i<(N*6);i+=24)
	{
		//mVec
		alldata[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+4] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+12] = randpval()*alldata[i];
		alldata[i+16] = randpval()*alldata[i+4];
		alldata[i+8] = randpval()*alldata[i]*alldata[i+4];
		alldata[i+20] = 0.0;

		alldata[i+1] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+5] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+13] = randpval()*alldata[i+1];
		alldata[i+17] = randpval()*alldata[i+5];
		alldata[i+9] = randpval()*alldata[i+1]*alldata[i+5];
		alldata[i+21] = 0.0;

		alldata[i+2] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+6] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+14] = randpval()*alldata[i+2];
		alldata[i+18] = randpval()*alldata[i+6];
		alldata[i+10] = randpval()*alldata[i+2]*alldata[i+6];
		alldata[i+22] = 0.0;

		alldata[i+3] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+7] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		alldata[i+15] = randpval()*alldata[i+3];
		alldata[i+19] = randpval()*alldata[i+7];
		alldata[i+11] = randpval()*alldata[i+3]*alldata[i+7];
		alldata[i+23] = 0.0;



	}

	

	__m128 variable,variable1,variable2,variable3,variable4,variable5,variable6;


	__m128 scale1 = _mm_set_ps1(0.01f);
	__m128 scale2 = _mm_set_ps1(1.0f);
	__m128 scale3 = _mm_set_ps1(2.0f);


	__m128 * data128 = (__m128 *) alldata;

	__m128 *maxg1= (__m128 *) maxg;
	__m128 *ming1= (__m128 *) ming;
	__m128 *sumg1= (__m128 *) sumg;



	double timeOmegaTotalStart = gettime();

	for(unsigned int j=0;j<iters;j++)
	{

		avgF = 0.0f;
		maxF = 0.0f;
		minF = FLT_MAX;

		for(unsigned int i=0;i<(N*6)/4; i+=6)//check this later for any changes!!!!!!!!!!!!!!!!!!!!!!!!!!
		{

			variable= _mm_add_ps(data128[i+3], data128[i+4]);

			variable1= _mm_sub_ps(data128[i],scale2);
			variable1= _mm_div_ps(variable1 ,scale3);
			variable1= _mm_mul_ps(data128[i],variable1);
			

			variable2= _mm_sub_ps(data128[i+1],scale2);
			variable2= _mm_div_ps(variable2 ,scale3);
			variable2= _mm_mul_ps(data128[i+1],variable2);

			variable3=_mm_add_ps(variable1,variable2);
			variable3= _mm_div_ps(variable,variable3);


			variable4=_mm_sub_ps(data128[i+2],data128[i+3]);
			variable4=_mm_sub_ps(variable4,data128[i+4]);
			
			variable5=_mm_mul_ps(data128[i],data128[i+1]);

			variable6=_mm_div_ps(variable4,variable5);

			data128[i+5]=_mm_add_ps(variable6 ,scale1);
			data128[i+5]= _mm_div_ps(variable3,data128[i+5]);//!

			*maxg1= _mm_max_ps(*maxg1,data128[i+5]);
			*ming1= _mm_min_ps(data128[i+5],*ming1);
			*sumg1= _mm_add_ps(data128[i+5],*sumg1);
			
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
   		for(unsigned int i=N-leftover;i<N;i+=6)
		{
			float num_0 = alldata[i+3]+alldata[i+4];
			float num_1 = alldata[i]*(alldata[i]-1.0f)/2.0f;
			float num_2 = alldata[i+1]*(alldata[i+1]-1.0f)/2.0f;
			float num = num_0/(num_1+num_2);

			float den_0 = alldata[i+2]-alldata[i+3]-alldata[i+4];
			float den_1 = alldata[i]*alldata[i+1];
			float den = den_0/den_1;

			alldata[i+5] = num/(den+0.01f);
			
			maxF = alldata[i+5]>maxF?alldata[i+5]:maxF;
			minF = alldata[i+5]<minF?alldata[i+5]:minF;
			avgF += alldata[i+5];
		}	 
	}
	
   	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();

	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",
	timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,(double)avgF/N);
	_mm_free(alldata);
	

}