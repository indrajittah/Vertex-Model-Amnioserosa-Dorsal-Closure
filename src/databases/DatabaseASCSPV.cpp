#define ENABLE_CUDA
//#include "DatabaseNetCDFSPV.h"
#include "DatabaseASCSPV.h"
//#include "voronoiModelBase.h"
/*! \file DatabaseASCSPV.cpp */

// IT WRITE DATA IN ASCII FORMAT 
void SPVDatabaseASC::WriteStateASC(STATE s, FILE *trajfile, int Nv)
    //:Nv(np) 
    {
    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;	
    //cout<<"Boxlength "<<x11<<"\t"<<x12<<endl;
    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> directordat(Nv);
    std::vector<int> typedat(Nv);
    int idx = 0;
    Dscalar means0=0.0;

    ArrayHandle<Dscalar2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_ex(s->exclusions,access_location::host,access_mode::read);
//	fprintf(trajfile,"%20.14f\t%20.14f\n",x11,x12);//Open if box length required
//	fprintf(trajfile,"%20.14f\t%20.14f\n",x21,x22); 
	fprintf(trajfile,"ITEM: TIMESTEP \n");
	fprintf(trajfile,"%08ld \n", 0);
	fprintf(trajfile,"ITEM: NUMBER OF ATOMS \n");
	fprintf(trajfile,"%08i \n", Nv);
	fprintf(trajfile,"ITEM: BOX BOUNDS xy xz yz pp pp pp \n");
	fprintf(trajfile,"%12.5f %12.5f %12.5f \n", boxdat[1], boxdat[0],0);
	fprintf(trajfile,"%12.5f %12.5f %12.5f \n", boxdat[2], boxdat[3],0);
	fprintf(trajfile,"-0.5 0.5 0.0 \n");

    
	fprintf(trajfile,"ITEM: ATOMS id type x y z radius \n");
	for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        directordat[ii] = h_cd.data[pidx];
	if(h_ex.data[ii] == 0)
		typedat[ii] = h_ct.data[pidx];
        else
            typedat[ii] = h_ct.data[pidx]-5;
	fprintf(trajfile,"%04i\t%04i\t%20.14f\t%20.14f\t%20.14f\t%20.14f\n",ii+1,h_ct.data[pidx],h_p.data[pidx].x,h_p.data[pidx].y,0.1000,0.5000);
        idx +=1;
        };
    }


void SPVDatabaseASC::WriteArea(STATE s, FILE *trajfile, int Nv)
    {
	ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);
	for (int ii = 0; ii < Nv; ++ii){
		int pidx = s->tagToIdx[ii];
		fprintf(trajfile,"%20.14f\t",h_AP.data[pidx].x);
	}
	fprintf(trajfile,"\n");

    }
    
void SPVDatabaseASC::NormalStressValue(STATE s, FILE *trajfile, int Nv){
        ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_APpref(s->AreaPeriPreferences,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_KaKp(s->Moduli,access_location::host,access_mode::read);
        Dscalar TP = 0.0;
        for (int i = 0; i < Nv; ++i){
//		TP = (-2.0*h_KaKp.data[i].x*h_AP.data[i].x*(h_AP.data[i].x  - h_APpref.data[i].x)) - (h_KaKp.data[i].y*h_AP.data[i].y*(h_AP.data[i].y  - h_APpref.data[i].y));
		TP = (-2.0*h_KaKp.data[i].x*h_AP.data[i].x*(h_AP.data[i].x  - h_APpref.data[i].x));
		fprintf(trajfile,"%20.14f\n",TP);
	};
    }

void SPVDatabaseASC::WriteRearASC(STATE s, FILE *trajfile)
    {
	//s->RearrangedIDs;
	//cout<<s->RearrangedIDs.size()<<endl;
	//cout<<s->RearrangedIDs<<endl;
	fprintf(trajfile,"1-indexed atom IDs that have changed neighbors this step\n");
        for(int i=0; i<s->RearrangedIDs.size(); ++i)
        {
		//cout<<"rearr"<<s->RearrangedIDs[i]<<endl;
                fprintf(trajfile, "%i\n", 1+s->RearrangedIDs[i]);
        }
    }
