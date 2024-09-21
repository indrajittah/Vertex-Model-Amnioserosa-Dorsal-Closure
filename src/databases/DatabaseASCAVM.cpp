#define ENABLE_CUDA
#include "DatabaseASCAVM.h"

void AVMDatabaseASC::WriteStateASCAVM(STATE s, FILE *trajfile, int Nv)
    {
    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;	
    std::vector<int> typedat(Nv);
    int idx = 0;
    Dscalar means0=0.0;

    s->getCellCentroidsCPU();
    ArrayHandle<Dscalar2> h_p(s->cellPositions);

    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
	fprintf(trajfile,"ITEM: TIMESTEP \n");
	fprintf(trajfile,"%08ld \n", 0);
	fprintf(trajfile,"ITEM: NUMBER OF ATOMS \n");
	fprintf(trajfile,"%08i \n", Nv);
	fprintf(trajfile,"ITEM: BOX BOUNDS xy xz yz pp pp pp \n");
	fprintf(trajfile,"%12.5f %12.5f %12.5f \n", boxdat[1], boxdat[0],0);
	fprintf(trajfile,"%12.5f %12.5f %12.5f \n", boxdat[2], boxdat[3],0);
	fprintf(trajfile,"-0.5 0.5 0.0 \n");

    
	fprintf(trajfile,"ITEM: ATOMS id type x y z radius\n");
	for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdx[ii];
	fprintf(trajfile,"%04i\t%04i\t%20.14f\t%20.14f\t%20.14f\t%20.14f\n",ii+1,h_ct.data[pidx],h_p.data[pidx].x,h_p.data[pidx].y,0.1000,0.5000);
        idx +=1;
        };
    }


void AVMDatabaseASC::WriteAreaAVM(STATE s, FILE *trajfile, int Nv)
    {
	s->getCellmajorandminoraxisCPU();
	ArrayHandle<Dscalar2> h_p(s->cellPositions);
   	ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
	ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);
	ArrayHandle<Dscalar2> h_APpref(s->AreaPeriPreferences,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_KaKp(s->Moduli,access_location::host,access_mode::read);
	ArrayHandle<int> h_nn(s->cellVertexNum,access_location::host,access_mode::read);
	std::vector<double> Neighs_stress;
        Neighs_stress = s->Return_Normal_stress_due_to_its_neighborcells();
	std::vector<double> Cell_Line_tension;
        Cell_Line_tension = s->Return_Individual_Cell_Line_Tension();
	for (int ii = 0; ii < Nv; ++ii){
                int pidx = s->tagToIdx[ii];
                fprintf(trajfile,"%04i\t%04i\t%20.14f\t%20.14f\t%20.14f\t%20.14f\t%20.14f\t%20.14f\t%20.14f\t%20.14f\n",h_ct.data[pidx],h_nn.data[pidx],h_AP.data[pidx].x,h_AP.data[pidx].y,h_AP.data[pidx].y/sqrt(h_AP.data[pidx].x),(2.0*h_KaKp.data[pidx].x*(h_AP.data[pidx].x-h_APpref.data[pidx].x)), ((1/h_AP.data[pidx].x)*h_KaKp.data[pidx].y*(h_AP.data[pidx].y-h_APpref.data[pidx].y)*h_AP.data[pidx].y),(2.0*h_KaKp.data[pidx].x*(h_AP.data[pidx].x-h_APpref.data[pidx].x)) + ((1/h_AP.data[pidx].x)*h_KaKp.data[pidx].y*(h_AP.data[pidx].y-h_APpref.data[pidx].y)*h_AP.data[pidx].y),Neighs_stress[pidx],Cell_Line_tension[pidx]);
        }

    }


void AVMDatabaseASC::WriteLinetensionAVM(STATE s, FILE *trajfile, int Nv)
    {
	std::vector<double> Linetension;
        Linetension = s->Return_Linetension();
	cout<<"Line tension size = "<<Linetension.size()<<endl;
	for(int i=0;i<Linetension.size();i++){
		fprintf(trajfile,"%20.14f\n",Linetension[i]);		
	}
   
    }
 
void AVMDatabaseASC::NormalStressValueAVM(STATE s, FILE *trajfile, int Nv){
        ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_APpref(s->AreaPeriPreferences,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_KaKp(s->Moduli,access_location::host,access_mode::read);
        Dscalar Hydr_Press = 0.0;
        Dscalar Edge_tension = 0.0;
        for (int i = 0; i < Nv; ++i){
		Hydr_Press = ( 2.0*h_KaKp.data[i].x*(h_AP.data[i].x  - h_APpref.data[i].x) * h_AP.data[i].x );
		Edge_tension = (h_KaKp.data[i].y*(h_AP.data[i].y  - h_APpref.data[i].y) * h_AP.data[i].y);
		fprintf(trajfile,"%20.14f\t%20.14f\t%20.14f\n",Hydr_Press,Edge_tension,(Hydr_Press+Edge_tension));
	};
    }

