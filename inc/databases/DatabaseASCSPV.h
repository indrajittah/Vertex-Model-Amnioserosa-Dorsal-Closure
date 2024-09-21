#ifndef DATABASE_SPVASC_H
#define DATABASE_SPVASC_H

#include "voronoiQuadraticEnergy.h"
//#include "DatabaseNetCDF.h"

/*! \file DatabaseNetCDFSPV.h */
//!Simple databse for reading/writing 2d spv states
/*!
Class for a state database for a 2d delaunay triangulation
the box dimensions are stored, the 2d unwrapped coordinate of the delaunay vertices,
and the shape index parameter for each vertex
*/
class SPVDatabaseASC //: public BaseDatabaseNetCDF
{
private:
    typedef shared_ptr<Voronoi2D> STATE;
public:
    virtual void WriteStateASC(STATE c, FILE* file, int nc);   //PRINT ASCII FORMAT FILE <IT>
    virtual void WriteRearASC(STATE c, FILE* file);
    virtual void WriteArea(STATE c, FILE* file, int nc);
    virtual void NormalStressValue(STATE c, FILE* file, int nc);
};
#endif
