#ifndef DATABASE_AVMASC_H
#define DATABASE_AVMASC_H

#include "vertexQuadraticEnergy.h"
//#include "DatabaseNetCDF.h"

/*! \file DatabaseNetCDFSPV.h */
//!Simple databse for reading/writing 2d spv states
/*!
Class for a state database for a 2d delaunay triangulation
the box dimensions are stored, the 2d unwrapped coordinate of the delaunay vertices,
and the shape index parameter for each vertex
*/
class AVMDatabaseASC //: public BaseDatabaseNetCDF
{
private:
    typedef shared_ptr<AVM2D> STATE;
public:
    virtual void WriteStateASCAVM(STATE c, FILE* file, int nc);   //PRINT ASCII FORMAT FILE <IT>
    virtual void WriteAreaAVM(STATE c, FILE* file, int nc);
    virtual void WriteLinetensionAVM(STATE c, FILE* file, int nc);
    virtual void NormalStressValueAVM(STATE c, FILE* file, int nc);
//    virtual void NormalStressValue_by_Neighbourcells_AVM(STATE c, FILE* file, int nc);
};
#endif
