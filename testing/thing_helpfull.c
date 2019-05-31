for (int i = 0; i < N*6; i+=24)
	{
		printf("The values of mVec is %e  %e  %e  %e\n",alldata[i],alldata[i+1],alldata[i+2],alldata[i+3] );
		printf("The values of nVec is %e  %e  %e  %e\n",alldata[i+4],alldata[i+5],alldata[i+6],alldata[i+7] );
		printf("The values of CVec is %e  %e  %e  %e\n",alldata[i+8],alldata[i+9],alldata[i+10],alldata[i+11] );
	    printf("The values of LVec is %e  %e  %e  %e\n", alldata[i+12],alldata[i+13],alldata[i+14],alldata[i+15]);
	    printf("The values of RVec is %e  %e  %e  %e\n", alldata[i+16],alldata[i+17],alldata[i+18],alldata[i+19]);
	    printf("The values of FVec is %e  %e  %e  %e\n", alldata[i+20],alldata[i+21],alldata[i+22],alldata[i+23]);
	    printf("----------------------------------------\n\n");

	}



	for (int i = 0; i < N; i+=4)
	{
		printf("The values of mVec is %e  %e  %e  %e\n",mVec[i],mVec[i+1],mVec[i+2],mVec[i+3] );
		printf("The values of nVec is %e  %e  %e  %e\n",nVec[i],nVec[i+1],nVec[i+2],nVec[i+3]);
		printf("The values of CVec is %e  %e  %e  %e\n",CVec[i],CVec[i+1],CVec[i+2],CVec[i+3] );
	    printf("The values of LVec is %e  %e  %e  %e\n",LVec[i],LVec[i+1],LVec[i+2],LVec[i+3]);
	    printf("The values of RVec is %e  %e  %e  %e\n", RVec[i],RVec[i+1],RVec[i+2],RVec[i+3]);
	    printf("The values of FVec is %e  %e  %e  %e\n", FVec[i],FVec[i+1],FVec[i+2],FVec[i+3]);
	    printf("----------------------------------------\n\n");

	}	