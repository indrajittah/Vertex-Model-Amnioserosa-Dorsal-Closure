#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA
#include "selfPropelledCellVertexDynamics.h"
#include "vertexQuadraticEnergy.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"
#include "DatabaseASCAVM.h"
/*!
This file compiles to produce an executable that demonstrates how to use the energy minimization
functionality of cellGPU. Now that energy minimization behaves like any other equation of motion, this
demonstration is pretty straightforward
*/

//! A function of convenience for setting FIRE parameters
void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, Dscalar deltaT, Dscalar alphaStart,
        Dscalar deltaTMax, Dscalar deltaTInc, Dscalar deltaTDec, Dscalar alphaDec, int nMin,
        Dscalar forceCutoff)
    {
    emin->setDeltaT(deltaT);
    emin->setAlphaStart(alphaStart);
    emin->setDeltaTMax(deltaTMax);
    emin->setDeltaTInc(deltaTInc);
    emin->setDeltaTDec(deltaTDec);
    emin->setAlphaDec(alphaDec);
    emin->setNMin(nMin);
    emin->setForceCutoff(forceCutoff);
    };

int serial;
int main(int argc, char*argv[]){

    sscanf(argv[1],"%d",&serial);
    //as in the examples in the main directory, there are a bunch of default parameters that
    //can be changed from the command line
    int numpts = 256;
    int USE_GPU = -1;
    int c;
    int tSteps = 800;
    int initSteps = 60000;
    int steptokill = 15;
    int remainder = 0;
    int Ingress_Step = 440;
    int minimization_step = 1000;
    Dscalar dt = 0.01;
    Dscalar KA = 1.0;
    Dscalar KP = 1.0;
    Dscalar p0 = 4.50;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.0;
    Dscalar Dr = 0.0;
    Dscalar compressrate = 0.01;
    Dscalar stdP0 = 0.50;
    Dscalar stdA0 = 0.50;
    Dscalar SLOPE_PREF_PERI_CHANGE = 0.0005;
    Dscalar STD_CHANGE = 0.0005;
    Dscalar STD_CHANGE_AREA = 0.0005;
    Dscalar slope = 0.00225;
    Dscalar FC =1e-6;
    Dscalar T1_Threshold =0.00001;
    Dscalar Decrease_peri = 0.0034;
    int EQ_AFTER_EACHCOMPRESS = 1000;
    //Dscalar P_Periodic;
    int program_switch = 1;
    while((c=getopt(argc,argv,"n:g:I:Z:O:L:l:S:s:m:w:s:D:r:a:i:P:v:b:x:y:z:p:t:e:k:R:E:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'I': Ingress_Step = atoi(optarg); break;
            case 'm': Decrease_peri = atof(optarg); break;
	    case 'L': steptokill = atoi(optarg); break;
            case 'l': remainder = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'Z': STD_CHANGE = atof(optarg); break;
            case 'O': STD_CHANGE_AREA = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'w': SLOPE_PREF_PERI_CHANGE = atof(optarg); break;
            case 'S': stdP0 = atof(optarg); break;
            case 's': stdA0 = atof(optarg); break;
            case 'k': KA = atof(optarg); break;
            case 'D': Dr = atof(optarg); break;
            case 'E': EQ_AFTER_EACHCOMPRESS = atoi(optarg); break;
            case 'R': compressrate = atof(optarg); break;
            case 'P': KP = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    clock_t t1,t2;
    bool reproducible = false;
    bool initializeGPU = false;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
    else
        initializeGPU = false;
    cout<<"Area Modulus "<<KA<<endl;
    cout<<"Perimeter Modulus "<<KP<<endl;
    cout<<"STD of PERI "<<stdP0<<endl;
    cout<<"STD of AREA "<<stdA0<<endl;
    if(program_switch == 1){
    	char dataname[256];
    	sprintf(dataname,"vertex.nc");
	int Nvert = 2*numpts;
	AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
	bool runSPV = false;
	EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,p0,reproducible,runSPV);
	shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(avm);
	avm->setCellPreferencespoly(a0,p0,stdP0,stdA0);
	avm->setModuliUniform(KA,KP);
	avm->setv0Dr(v0,Dr);
	avm->setT1Threshold(T1_Threshold);
    	vector<int> vi(3*Nvert);
    	vector<int> vf(3*Nvert);

        ArrayHandle<int> vcn(avm->vertexCellNeighbors);
        for (int ii = 0; ii < 3*Nvert; ++ii){
		vi[ii]=vcn.data[ii];
        }
	//combine the equation of motion and the cell configuration in a "Simulation"
    	SimulationPtr sim = make_shared<Simulation>();
    	sim->setConfiguration(avm);
	sim->addUpdater(fireMinimizer,avm);
	sim->setIntegrationTimestep(dt);
    	sim->setCPUOperation(!initializeGPU);
    	sim->setReproducible(reproducible);
        Dscalar mf;
	avm->reportMeanVertexForce();
	vector<int> TYPE;
        for (int i=0; i<numpts; i++){
                TYPE.push_back(i);
        }
        avm->setCellType(TYPE);
        printf("Start Minimizing\n");
        for (int i = 0; i <initSteps;++i){
		setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
                fireMinimizer->setMaximumIterations(50*(i+1));
                sim->performTimestep();
                mf = fireMinimizer->getMaxForce();
	if (mf < FC)
                      break;
        }
        printf("minimized value of q = %f\n",avm->reportq());	
	avm->reportMeanVertexForce();
	//ncdat.WriteState(avm);
	
	avm->setT1Threshold(T1_Threshold);
        Dscalar x11,x12,x21,x22;
	AVMDatabaseASC ascdat;
    	FILE *data1;
	FILE *data2;
	FILE *data3;
	FILE *data4;
	FILE *data5;
    	char dataname1[256];
    	char dataname2[256];
	char dataname3[256];
	char dataname4[256];
	char dataname5[256];
        char dir[128];
        sprintf(dir,"mkdir -p struct000");
        system(dir);
	Dscalar P0 = avm->HydrostaticPressure();
	ArrayHandle<Dscalar2> h_p(avm->AreaPeriPreferences,access_location::host,access_mode::read);
	FILE *fp1;
	fp1=fopen("PREF_AREA_PERI.dat","w");
	for(int k=0;k<numpts;k++){
        	fprintf(fp1,"%f\t%f\n",h_p.data[k].x,h_p.data[k].y);
        }
        vector<double> PREFPERI(numpts);
	vector<double> PREFAREA(numpts);
        for(int k=0;k<numpts;k++){
                PREFPERI[k] = h_p.data[k].y;
		PREFAREA[k] = h_p.data[k].x;
        }
	vector<Dscalar2> AreaPeriPref(numpts);
        for(int k=0;k<numpts;k++){
                AreaPeriPref[k].x = PREFAREA[k];
                AreaPeriPref[k].y = PREFPERI[k];
        }
	fclose(fp1);
	FILE *fp;
	fp=fopen("ALLSTUFF.dat","w");
//Quasistatic compress 
	for(int ii = 0; ii < tSteps; ++ii){
                sim->performTimestep();
		int Nvertices = avm->getNumberOfDegreesOfFreedom();
                numpts = Nvertices/2;
                if(ii%20 == 0){
			sprintf(dataname2,"struct000/instep%07ld.txt",ii);
                        data2 = fopen(dataname2,"w");
			sprintf(dataname1,"struct000/area%07ld.txt",ii);
        		data1 = fopen(dataname1,"w");
			sprintf(dataname3,"struct000/Linetesion%07ld.txt",ii);
                        data3 = fopen(dataname3,"w");
                        ascdat.WriteAreaAVM(avm,data1,numpts);
                        ascdat.WriteStateASCAVM(avm,data2,numpts);
			ascdat.WriteLinetensionAVM(avm,data3,numpts);
			fclose(data3);
                        fclose(data2);
                        fclose(data1);
                }
		/*   This will ingress cell below a certain area threshold value randomly.*/
		if(avm->reportcellingressid() < (numpts-1) && ii>Ingress_Step && ii%steptokill==remainder){ 
			avm->setT1Threshold(T1_Threshold);
                        int deadIdx = avm->reportcellingressid();
                        Nvertices = avm->getNumberOfDegreesOfFreedom();
                        numpts = Nvertices/2;

			ArrayHandle<Dscalar2> h_p(avm->AreaPeriPreferences,access_location::host,access_mode::read);
			ArrayHandle<Dscalar2> h_AP(avm->AreaPeri,access_location::host,access_mode::read);
			vector<Dscalar2> oldAP(numpts); 
			for(int k=0;k<numpts;k++){
				oldAP[k].x = h_p.data[k].x; 
				oldAP[k].y = h_p.data[k].y;
			} 

                        vector<Dscalar2> newPrefs(numpts);
			for(int k=0;k<numpts;k++){
				newPrefs[k].x = oldAP[k].x;
				newPrefs[k].y = oldAP[k].y;
			}

                        newPrefs[deadIdx].x = 0.0;
                        newPrefs[deadIdx].y = 0.0;
			avm->setCellPreferences(newPrefs);
                        int cellVertices = 0;
			for (int tt =0; tt < 50; ++tt){
                                ArrayHandle<int> cn(avm->cellVertexNum,access_location::host,access_mode::read);
                                cellVertices = cn.data[deadIdx];
                                if(cellVertices==3) break;
                                sim->performTimestep();
                        }
			if(cellVertices ==3){
                                avm->cellDeath(deadIdx);
				PREFPERI.erase (PREFPERI.begin()+deadIdx);
                        }
			
			else{
                                for(int k=0;k<numpts;k++){
                                	newPrefs[k].x = oldAP[k].x;
                                	newPrefs[k].y = oldAP[k].y;
                        	}
                                avm->setCellPreferences(newPrefs);
                        }


			Nvertices = avm->getNumberOfDegreesOfFreedom();
                        numpts = Nvertices/2;
                        avm->Box->getBoxDims(x11,x12,x21,x22);
			x22=x22-compressrate;
			x21=x21+compressrate;
			avm->Box->setGeneral(x11,x12,x21,x22);
			Dscalar PREF_AREA = (abs(x11-x12)*abs(x22-x21))/numpts;
                	Dscalar MEANPREFPERO = 0;
                	Dscalar MEANPREFAREA = 0;
			Dscalar STDPREF_PERI = 0;
			for(int k=0;k<numpts;k++){
				AreaPeriPref[k].x = AreaPeriPref[k].x;
				AreaPeriPref[k].y = AreaPeriPref[k].y - Decrease_peri;
				MEANPREFAREA += AreaPeriPref[k].x;
	                        MEANPREFPERO += AreaPeriPref[k].y;

			}

			for(int k=0;k<numpts;k++){
                        	STDPREF_PERI +=  (AreaPeriPref[k].y-(MEANPREFPERO/(Dscalar)numpts))*(AreaPeriPref[k].y-(MEANPREFPERO/(Dscalar)numpts));
                	}
			Dscalar MEANPREFPERI_AFTER_STD_SHIFT = 0;
	                Dscalar MEANPREFAREA_AFTER_STD_SHIFT = 0;
			for(int k=0;k<numpts;k++){
                        if( AreaPeriPref[k].y < (MEANPREFPERO/numpts)){
                                AreaPeriPref[k].y = AreaPeriPref[k].y + STD_CHANGE;
                        }
                        if( AreaPeriPref[k].y > (MEANPREFPERO/numpts)){
                                AreaPeriPref[k].y = AreaPeriPref[k].y - STD_CHANGE;
                        }
                        MEANPREFPERI_AFTER_STD_SHIFT += AreaPeriPref[k].y;

			if( AreaPeriPref[k].x < (MEANPREFAREA/numpts)){
                                AreaPeriPref[k].x = AreaPeriPref[k].x + STD_CHANGE_AREA;
			}
                        if( AreaPeriPref[k].x > (MEANPREFAREA/numpts)){
                                AreaPeriPref[k].x = AreaPeriPref[k].x - STD_CHANGE_AREA;
			}
                        MEANPREFAREA_AFTER_STD_SHIFT += AreaPeriPref[k].x;
                }

			avm->setCellPreferences(AreaPeriPref);

			for (int i = 0; i < minimization_step;++i){
                        	sim->addUpdater(fireMinimizer,avm);
                        	setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,40,1e-12);
                        	fireMinimizer->setMaximumIterations(50*i);
                        	sim->performTimestep();
                        	mf = fireMinimizer->getMaxForce();
                        	if (mf < FC)
                                	break;
                	}

			avm->getCellmajorandminoraxisCPU();
                	ArrayHandle<Dscalar2> h_pos(avm->cellPositions);
                	Dscalar MEANAR=0;
                	Dscalar STDAR=0;
               	 	Dscalar STDPREF_AREA=0;
                	STDPREF_PERI=0;
                	for(int k=0; k <numpts;++k){
                        	int pidx = avm->tagToIdx[k];
                        	MEANAR += sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x);
                	}
                	MEANAR = MEANAR/(double)numpts;

                	for(int k=0; k <numpts;++k){
                        	int pidx = avm->tagToIdx[k];
                        	STDAR += (sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x)-MEANAR)*(sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x)-MEANAR);
                        	STDPREF_PERI +=  (AreaPeriPref[k].y-(MEANPREFPERI_AFTER_STD_SHIFT/(Dscalar)numpts))*(AreaPeriPref[k].y-(MEANPREFPERI_AFTER_STD_SHIFT/(Dscalar)numpts));
                        	STDPREF_AREA +=  (AreaPeriPref[k].x-(MEANPREFAREA_AFTER_STD_SHIFT/(Dscalar)numpts))*(AreaPeriPref[k].x-(MEANPREFAREA_AFTER_STD_SHIFT/(Dscalar)numpts));
                	}
			STDAR = sqrt(STDAR/(Dscalar)numpts);
			Dscalar2 variances = avm->reportVarAP();
			fprintf(fp,"%d\t%d\t%f\t%f\t%.20f\t%.20f\t%.20f\t%.20f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",ii,numpts,(numpts-(abs(x11-x12)*abs(x22-x21)))/numpts,(abs(x11-x12)*abs(x22-x21)),avm->reportMeanA(),avm->reportMeanP(),variances.x,variances.y,avm->reportq(),avm->reportStdq(),avm->computeEnergy(),MEANAR,STDAR,avm->NormalStress(),avm->HydrostaticPressure(),avm->NormalInterstress(),MEANPREFPERO/numpts,MEANPREFAREA/numpts,avm->OOPCPU());
		}

/* Non ingression part */
		else{
			avm->Box->getBoxDims(x11,x12,x21,x22);
                	x22=x22-compressrate;
			x21=x21+compressrate;
			avm->Box->setGeneral(x11,x12,x21,x22);
			Dscalar PREF_AREA = (abs(x11-x12)*abs(x22-x21))/numpts;
                	ArrayHandle<Dscalar2> h_p(avm->AreaPeriPreferences,access_location::host,access_mode::read);
			Dscalar MEANPREFPERO = 0;
			Dscalar MEANPREFAREA = 0;
			Dscalar STDPREF_PERI=0;
			for(int k=0;k<numpts;k++){
				AreaPeriPref[k].x = AreaPeriPref[k].x;
				AreaPeriPref[k].y = AreaPeriPref[k].y - Decrease_peri;
				MEANPREFAREA += AreaPeriPref[k].x;
				MEANPREFPERO += AreaPeriPref[k].y;
        		}
			for(int k=0;k<numpts;k++){
        	                STDPREF_PERI +=  (AreaPeriPref[k].y-(MEANPREFPERO/(Dscalar)numpts))*(AreaPeriPref[k].y-(MEANPREFPERO/(Dscalar)numpts));

                	}
			Dscalar MEANPREFPERI_AFTER_STD_SHIFT = 0;
                	Dscalar MEANPREFAREA_AFTER_STD_SHIFT = 0;

			for(int k=0;k<numpts;k++){
                        	if( AreaPeriPref[k].y < (MEANPREFPERO/numpts)){
                                	AreaPeriPref[k].y = AreaPeriPref[k].y + STD_CHANGE;
                        	}
                        	if( AreaPeriPref[k].y > (MEANPREFPERO/numpts)){
                                	AreaPeriPref[k].y = AreaPeriPref[k].y - STD_CHANGE;
                        	}
                        	MEANPREFPERI_AFTER_STD_SHIFT += AreaPeriPref[k].y;


                        	if( AreaPeriPref[k].x < (MEANPREFAREA/numpts)){
					AreaPeriPref[k].x = AreaPeriPref[k].x + STD_CHANGE_AREA;
				}
                        	if( AreaPeriPref[k].x > (MEANPREFAREA/numpts)){
                         	       AreaPeriPref[k].x = AreaPeriPref[k].x - STD_CHANGE_AREA;
				}
                 	       MEANPREFAREA_AFTER_STD_SHIFT += AreaPeriPref[k].x;
                	}
		
			avm->setCellPreferences(AreaPeriPref);
			for (int i = 0; i < minimization_step;++i){
                        	sim->addUpdater(fireMinimizer,avm);
                        	setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,40,1e-12);
                        	fireMinimizer->setMaximumIterations(50*i);
                        	sim->performTimestep();
                        	mf = fireMinimizer->getMaxForce();
				if (mf < FC)
                                	break;
                	}
	        	avm->getCellmajorandminoraxisCPU();
        		ArrayHandle<Dscalar2> h_pos(avm->cellPositions);
			Dscalar MEANAR=0;
			Dscalar STDAR=0;
			Dscalar STDPREF_AREA=0;
			STDPREF_PERI=0;
			for(int k=0; k <numpts;++k){
				int pidx = avm->tagToIdx[k];
				MEANAR += sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x); 
			}
			MEANAR = MEANAR/(double)numpts;

			for(int k=0; k <numpts;++k){
				int pidx = avm->tagToIdx[k];
				STDAR += (sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x)-MEANAR)*(sqrt(h_pos.data[pidx].y/h_pos.data[pidx].x)-MEANAR);
				STDPREF_PERI +=  (AreaPeriPref[k].y-(MEANPREFPERI_AFTER_STD_SHIFT/(Dscalar)numpts))*(AreaPeriPref[k].y-(MEANPREFPERI_AFTER_STD_SHIFT/(Dscalar)numpts));
				STDPREF_AREA +=  (AreaPeriPref[k].x-(MEANPREFAREA_AFTER_STD_SHIFT/(Dscalar)numpts))*(AreaPeriPref[k].x-(MEANPREFAREA_AFTER_STD_SHIFT/(Dscalar)numpts));
			}
			STDAR = sqrt(STDAR/(Dscalar)numpts);
			Dscalar2 variances = avm->reportVarAP();
			fprintf(fp,"%d\t%d\t%f\t%f\t%.20f\t%.20f\t%.20f\t%.20f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",ii,numpts,(numpts-(abs(x11-x12)*abs(x22-x21)))/numpts,(abs(x11-x12)*abs(x22-x21)),avm->reportMeanA(),avm->reportMeanP(),variances.x,variances.y,avm->reportq(),avm->reportStdq(),avm->computeEnergy(),MEANAR,STDAR,avm->NormalStress(),avm->HydrostaticPressure(),avm->NormalInterstress(),MEANPREFPERO/numpts,MEANPREFAREA/numpts,avm->OOPCPU());
		}
	}
		fclose(fp);


    }
    	if(initializeGPU)
        	cudaDeviceReset();
    		return 0;
}
