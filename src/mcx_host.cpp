/*******************************************************************************
**
**  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
**      -- OpenCL edition
**  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
**
**  Reference (Fang2009):
**        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
**        Migration in 3D Turbid Media Accelerated by Graphics Processing 
**        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
**
**  mcx_host.cpp: Host code for OpenCL
**
**  Unpublished work, see LICENSE.txt for details
**
*******************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include "mcx_host.hpp"
#include "tictoc.h"
#include "mcx_const.h"

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MCX_RNG_NAME       "Logistic-Lattice"
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define RO_MEM             (CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR)
#define WO_MEM             (CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR)
#define RW_MEM             (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)
#define RW_PTR             (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR)

extern cl_event kernelevent;

/*
  query GPU info and set active GPU
*/
cl_platform_id mcx_set_gpu(Config *cfg,unsigned int *activedev){

    unsigned int i,j,k,devnum;
    cl_uint numPlatforms,devparam;
    cl_ulong devmem;
    cl_platform_id platform = NULL;
    cl_device_type devtype[]={CL_DEVICE_TYPE_CPU,CL_DEVICE_TYPE_GPU};
    cl_context context;                 // compute context
    const char *devname[]={"CPU","GPU"};
    char pbuf[100];
    cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, NULL, 0};
    cl_int status = 0;
    size_t deviceListSize;

    clGetPlatformIDs(0, NULL, &numPlatforms);
    if(activedev) *activedev=0;

    if (numPlatforms>0) {
        cl_platform_id* platforms =(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
        mcx_assess(clGetPlatformIDs(numPlatforms, platforms, NULL));
        for (i = 0; i < numPlatforms; ++i) {
            platform = platforms[i];
	    if(1){
                mcx_assess(clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_NAME,sizeof(pbuf),pbuf,NULL));
	        if(cfg->isgpuinfo) printf("Platform [%d] Name %s\n",i,pbuf);
                cps[1]=(cl_context_properties)platform;
		if(activedev) *activedev=0;
        	for(j=0; j<2; j++){
		    cl_device_id * devices;
		    context=clCreateContextFromType(cps,devtype[j],NULL,NULL,&status);
		    if(status!=CL_SUCCESS){
		            clReleaseContext(context);
			    continue;
		    }
		    mcx_assess(clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize));
                    devices = (cl_device_id*)malloc(deviceListSize);
                    mcx_assess(clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL));
		    devnum=deviceListSize/sizeof(cl_device_id);
		    if(activedev)
		      for(k=0;k<MAX_DEVICE;k++){
		        if((j==0 && isdigit(cfg->deviceid[k]) && (uint)(cfg->deviceid[k]-'0')<devnum)
			 ||(j>0  && isalpha(cfg->deviceid[k]) && (uint)(cfg->deviceid[k]-'a')<devnum) )
				(*activedev)++;
			if(cfg->deviceid[k]=='\0')
				break;
		      }
		    if(cfg->isgpuinfo)
                      for(k=0;k<devnum;k++){
                	mcx_assess(clGetDeviceInfo(devices[k],CL_DEVICE_NAME,100,(void*)&pbuf,NULL));
                	printf("============ %s device [%d of %d]: %s  ============\n",devname[j],k+1,devnum,pbuf);
			mcx_assess(clGetDeviceInfo(devices[k],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_int),(void*)&devparam,NULL));
                	mcx_assess(clGetDeviceInfo(devices[k],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL));
                	printf(" Compute units :\t%d core(s)\n",(int)devparam);
                	printf(" Global memory :\t%ld B\n",(unsigned long)devmem);
                	mcx_assess(clGetDeviceInfo(devices[k],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL));
                	printf(" Local memory  :\t%ld B\n",(unsigned long)devmem);
                      }
                    free(devices);
                    clReleaseContext(context);
        	}
	    }
        }
        free(platforms);
    }
    if(cfg->isgpuinfo==2) exit(0);
    return platform;
}

char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
}


/*
   assert cuda memory allocation result
*/
void mcx_assess(int cuerr){
     if(cuerr!=CL_SUCCESS){
         mcx_error(-(int)cuerr,print_cl_errstring(cuerr));
     }
}

/*
   master driver code to run MC simulations
*/
void mcx_run_simulation(Config *cfg,int threadid,int activedev,float *fluence,float *totalenergy){

     cl_int i,j,iter;
     cl_float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     cl_float t,twindow0,twindow1;
     cl_float energyloss=0.f,energyabsorbed=0.f,simuenergy,fullload=0.f;
     cl_int deviceload=0;
     cl_float *energy;
     cl_int threadphoton, oddphotons, stopsign=0, detected=0;

     cl_int photoncount=0,printnum;
     cl_int tic,fieldlen;
     uint4 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint4 dimlen;
     cl_float Vvox,scale,absorp,eabsorp;

     cl_context context;                 // compute context
     cl_command_queue commands;          // compute command queue
     cl_program program;                 // compute program
     cl_kernel kernel;                   // compute kernel
     cl_int status = 0;
     size_t deviceListSize;
     cl_platform_id platform = NULL;
     cl_device_type dType;
     cl_device_id* devices;
     size_t kernelWorkGroupSize;
     size_t maxWorkGroupSize;
     int    devid=0;
     cl_mem gmedia,gfield,gPpos,gPdir,gPlen,gPdet,gPseed,genergy,gproperty,gstopsign,gparam,gdetected;

     size_t mcgrid[1], mcblock[1];

     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;

     cl_uchar  *media=(cl_uchar *)(cfg->vol);
     cl_float  *field;

     float4 *Ppos, *Pdir, *Plen;
     cl_uint   *Pseed;
     float  *Pdet;

     MCXParam param={{cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f},
		     {cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f},
		     {cfg->dim.x,cfg->dim.y,cfg->dim.z,0},dimlen,cp0,cp1,cachebox,
		     minstep,0.f,0.f,cfg->tend,R_C0*cfg->unitinmm,cfg->isrowmajor,
                     cfg->issave2pt,cfg->isreflect,cfg->isrefint,cfg->issavedet,1.f/cfg->tstep,
                     cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->maxdetphoton,
                     cfg->medianum-1,cfg->detnum,0,0};

     if(cfg->iscpu){
         dType = CL_DEVICE_TYPE_CPU;
     }else{ //deviceType = "gpu" 
         dType = CL_DEVICE_TYPE_GPU;
     }
     if(!(threadid<MAX_DEVICE && cfg->deviceid[threadid]>0))
	 return;
     printf("threadid=%d\tcfg=%d\n",threadid,cfg->deviceid[threadid]);

     if(isalpha(cfg->deviceid[threadid])){
	     if(cfg->deviceid[threadid]>='a' && cfg->deviceid[threadid]<='z')
		 cfg->deviceid[threadid]-='a';
	     else if(cfg->deviceid[threadid]>='a' && cfg->deviceid[threadid]<='z')
		 cfg->deviceid[threadid]=cfg->deviceid[threadid]-'A'+('z'-'a')+1;
	     dType = CL_DEVICE_TYPE_GPU;
     }else{
	     cfg->deviceid[threadid]-='0';
             dType = CL_DEVICE_TYPE_CPU;
     }
     for(i=0;i<MAX_DEVICE;i++)
     	fullload+=cfg->workload[i];

#pragma omp barrier

     if(fullload<EPS){
	cfg->workload[threadid]=100.f/activedev;
     }
     fullload=(fullload<EPS)?100.f:fullload;
     deviceload=cfg->workload[threadid]/fullload*cfg->nphoton;
#pragma omp critical
{
     platform=mcx_set_gpu(cfg,NULL);

     cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

     /* Use NULL for backward compatibility */
     cl_context_properties* cprops=(platform==NULL)?NULL:cps;

     mcx_assess((context=clCreateContextFromType(cprops,dType,NULL,NULL,&status),status));
     mcx_assess(clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize));
}
     devices = (cl_device_id*)malloc(deviceListSize);
     if(devices == NULL){
         mcx_assess(-1);
     }
     devid=cfg->deviceid[threadid]; // device id starts from 1

     if(devid<0||devid>=(int)(deviceListSize/sizeof(cl_device_id))){
	 fprintf(cfg->flog,"WARNING: maximum device count is %d, specified %d. fall back to default - device 0\n",
             (int)(deviceListSize/sizeof(cl_device_id)),devid+1);
         devid=0; // if out of bound, fall back to the default
     }
     mcx_assess(clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL));

     {
         /* The block is to move the declaration of prop closer to its use */
         cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;
         mcx_assess((commands=clCreateCommandQueue(context,devices[devid],prop,&status),status));
     }
     mcx_assess(clGetDeviceInfo(devices[devid],CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof(size_t),(void*)&maxWorkGroupSize,NULL));

     if(cfg->respin>1){
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate*2); //the second half will be used to accumul$
     }else{
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate);
     }
     if(cfg->nthread%cfg->nblocksize)
        cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;
     threadphoton=cfg->nphoton/cfg->nthread/cfg->respin;
     oddphotons=cfg->nphoton/cfg->respin-threadphoton*cfg->nthread;

     if(cfg->nthread%cfg->nblocksize)
     	cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;

     mcgrid[0]=cfg->nthread;
     mcblock[0]=cfg->nblocksize;

     Ppos=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Pdir=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Plen=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Pseed=(cl_uint*)malloc(sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN);
     energy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*2);
     Pdet=(float*)calloc(cfg->maxdetphoton,sizeof(float)*(cfg->medianum+1));

     cachebox.x=(cp1.x-cp0.x+1);
     cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
     dimlen.x=cfg->dim.x;
     dimlen.y=cfg->dim.x*cfg->dim.y;
     dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;

     memcpy(&(param.dimlen.x),&(dimlen.x),sizeof(uint4));
     memcpy(&(param.cachebox.x),&(cachebox.x),sizeof(uint2));
     param.idx1dorig=(int(floorf(param.ps.z))*dimlen.y+
                      int(floorf(param.ps.y))*dimlen.x+
		      int(floorf(param.ps.x)));
     param.mediaidorig=(cfg->vol[param.idx1dorig] & MED_MASK);

     Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z;

     if(cfg->seed>0)
     	srand(cfg->seed);
     else
        srand(time(0));

     for (i=0; i<cfg->nthread; i++) {
           memcpy(Ppos+i,&param.ps,sizeof(param.ps));
           memcpy(Pdir+i,&param.c0,sizeof(param.c0));
           Plen[i].x=0.f;Plen[i].y=0.f;Plen[i].z=minstep*R_C0;Plen[i].w=0.f;
     }
     for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++) {
	   Pseed[i]=rand()+threadid;
     }
     printf("param.tmax=%e\n",param.tmax);
#pragma omp critical
{
     mcx_assess((gmedia=clCreateBuffer(context,RO_MEM, sizeof(cl_uchar)*(dimxyz),media,&status),status));
     mcx_assess((gfield=clCreateBuffer(context,RW_MEM, sizeof(cl_float)*(dimxyz)*cfg->maxgate,field,&status),status));
     mcx_assess((gPpos=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Ppos,&status),status));
     mcx_assess((gPdir=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Pdir,&status),status));
     mcx_assess((gPlen=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Plen,&status),status));
     mcx_assess((gPseed=clCreateBuffer(context,RW_MEM, sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,Pseed,&status),status));
     mcx_assess((genergy=clCreateBuffer(context,RW_MEM, sizeof(float)*cfg->nthread*2,energy,&status),status));
     mcx_assess((gPdet=clCreateBuffer(context,RW_MEM, sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1),Pdet,&status),status));
     mcx_assess((gproperty=clCreateBuffer(context,RO_MEM, cfg->medianum*sizeof(Medium),cfg->prop,&status),status));
     mcx_assess((gparam=clCreateBuffer(context,RO_MEM,sizeof(MCXParam),&param,&status),status));
     mcx_assess((gstopsign=clCreateBuffer(context,RW_PTR, sizeof(cl_uint),&stopsign,&status),status));
     mcx_assess((gdetected=clCreateBuffer(context,RW_MEM, sizeof(cl_uint),&detected,&status),status));
}
     if(threadid==0) fprintf(cfg->flog,"\
###############################################################################\n\
#                 Monte Carlo eXtreme (MCX) -- OpenCL                         #\n\
###############################################################################\n\
$MCX $Rev::     $ Last Commit:$Date::                     $ by $Author:: fangq$\n\
###############################################################################\n");

     tic=StartTimer();
     fprintf(cfg->flog,"compiled with: [RNG] %s [Seed Length] %d\n",MCX_RNG_NAME,RAND_SEED_LEN);
     fprintf(cfg->flog,"threadph=%d oddphotons=%d np=%d nthread=%d repetition=%d\n",threadphoton,oddphotons,
           deviceload,cfg->nthread,cfg->respin);
     fprintf(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);
     fieldlen=dimxyz*cfg->maxgate;

     fprintf(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

     /*
         if one has to simulate a lot of time gates, using the GPU global memory
	 requires extra caution. If the total global memory is bigger than the total
	 memory to save all the snapshots, i.e. size(field)*(tend-tstart)/tstep, one
	 simply sets cfg->maxgate to the total gate number; this will run GPU kernel
	 once. If the required memory is bigger than the video memory, set cfg->maxgate
	 to a number which fits, and the snapshot will be saved with an increment of 
	 cfg->maxgate snapshots. In this case, the later simulations will restart from
	 photon launching and exhibit redundancies.
	 
	 The calculation of the energy conservation will only reflect the last simulation.
     */
#pragma omp critical
{
     mcx_assess((program=clCreateProgramWithSource(context, 1,(const char **)&(cfg->clsource), NULL, &status),status));
     if(cfg->iscpu && cfg->isverbose){ 
	status=clBuildProgram(program, 0, NULL, "-D __DEVICE_EMULATION__ -D MCX_CPU_ONLY -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
     }else{
	status=clBuildProgram(program, 0, NULL, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);    
     }
}
     if(status!=CL_SUCCESS){
	 size_t len;
	 char msg[2048];
	 // get the details on the error, and store it in buffer
	 clGetProgramBuildInfo(program,devices[devid],CL_PROGRAM_BUILD_LOG,sizeof(msg),msg,&len); 
	 fprintf(cfg->flog,"Kernel build error:\n%s\n", msg);
	 mcx_error(-(int)status,(char*)("Error: Failed to build program executable!"));
     }
     fprintf(cfg->flog,"build program complete : %d ms\n",GetTimeMillis()-tic);

     mcx_assess((kernel = clCreateKernel(program, "mcx_main_loop", &status),status));
     mcx_assess(clGetKernelWorkGroupInfo(kernel,devices[devid],CL_KERNEL_WORK_GROUP_SIZE,
        	sizeof(size_t),&kernelWorkGroupSize,0));
     fprintf(cfg->flog,"create kernel complete : %d ms\n",GetTimeMillis()-tic);

     mcx_assess(clSetKernelArg(kernel, 0, sizeof(cl_uint),(void*)&threadphoton));
     mcx_assess(clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&oddphotons));
     mcx_assess(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&gmedia));
     mcx_assess(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&gfield));
     mcx_assess(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&genergy));
     mcx_assess(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&gPseed));
     mcx_assess(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&gPpos));
     mcx_assess(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&gPdir));
     mcx_assess(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&gPlen));
     mcx_assess(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&gPdet));
     mcx_assess(clSetKernelArg(kernel,10, sizeof(cl_mem), (void*)&gproperty));
     mcx_assess(clSetKernelArg(kernel,11, sizeof(cl_mem), (void*)&gstopsign));
     mcx_assess(clSetKernelArg(kernel,12, sizeof(cl_mem), (void*)&gdetected));

     fprintf(cfg->flog,"set kernel arguments complete : %d ms\n",GetTimeMillis()-tic);

     //simulate for all time-gates in maxgate groups per run
     for(t=cfg->tstart;t<cfg->tend;t+=cfg->tstep*cfg->maxgate){
       twindow0=t;
       twindow1=t+cfg->tstep*cfg->maxgate;

       fprintf(cfg->flog,"lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n"
           ,twindow0*1e9,twindow1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<cfg->respin;iter++){

           fprintf(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);
	   param.twin0=twindow0;
	   param.twin1=twindow1;
	   clReleaseMemObject(gparam);
	   mcx_assess((gparam=clCreateBuffer(context,RO_MEM,sizeof(MCXParam),&param,&status),status));
           mcx_assess(clSetKernelArg(kernel,13, sizeof(cl_mem), (void*)&gparam));

           // launch kernel
           mcx_assess(clEnqueueNDRangeKernel(commands,kernel,1,NULL,mcgrid,mcblock, 0, NULL, 
#ifndef USE_OS_TIMER
                  &kernelevent));
#else
                  NULL));
#endif
           mcx_assess(clEnqueueReadBuffer(commands,gfield,CL_TRUE,0,sizeof(cl_float),
                                            field, 0, NULL, NULL));
           fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",GetTimeMillis()-tic);

	   //handling the 2pt distributions
           if(cfg->issave2pt){
               mcx_assess(clEnqueueReadBuffer(commands,gfield,CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
	                                        field, 0, NULL, NULL));
               fprintf(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

               if(cfg->respin>1){
                   for(i=0;i<fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
               if(iter+1==cfg->respin){ 
                   if(cfg->respin>1)  //copy the accumulated fields back
                       memcpy(field,field+fieldlen,sizeof(cl_float)*fieldlen);

                   if(cfg->isnormalized){

                       fprintf(cfg->flog,"normizing raw data ...\t");
                       mcx_assess(clEnqueueReadBuffer(commands,genergy,CL_TRUE,0,sizeof(cl_float)*cfg->nthread*2,
	                                        energy, 0, NULL, NULL));
                       mcx_assess(clEnqueueReadBuffer(commands,gPlen,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
	                                        Plen, 0, NULL, NULL));
                       eabsorp=0.f;
                       for(i=1;i<cfg->nthread;i++){
                           energy[0]+=energy[i<<1];
       	       	       	   energy[1]+=energy[(i<<1)+1];
                           eabsorp+=Plen[i].z;  // the accumulative absorpted energy near the source
                       }
       	       	       for(i=0;i<dimxyz;i++){
                           absorp=0.f;
                           for(j=0;j<cfg->maxgate;j++)
                              absorp+=field[j*dimxyz+i];
                           eabsorp+=absorp*cfg->prop[media[i]].mua;
       	       	       }
		       simuenergy=energy[0]+energy[1];
                       scale=energy[1]/(simuenergy*Vvox*cfg->tstep*eabsorp);
                       fprintf(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   fprintf(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);
#pragma omp critical
{
                   for(i=0;i<fieldlen;i++)
		   	fluence[i]+=field[i]*simuenergy;

                   *totalenergy+=simuenergy;
}
#pragma omp barrier
#pragma omp master
{
                   fprintf(cfg->flog,"total simulated energy: %f\n",*totalenergy);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
		   fflush(cfg->flog);

                   if(*totalenergy>EPS)
		   	*totalenergy=1.f/(*totalenergy);
                   for(i=0;i<fieldlen;i++)
		   	fluence[i]*=(*totalenergy);

                   fprintf(cfg->flog,"saving data to file ...\t");
                   mcx_savedata(fluence,fieldlen,t>cfg->tstart,cfg);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
                   fflush(cfg->flog);
}
               }
               mcx_assess(clFinish(commands));
           }
	   //initialize the next simulation
	   if(twindow1<cfg->tend && iter<cfg->respin){
                  memset(field,0,sizeof(cl_float)*dimxyz*cfg->maxgate);
                  mcx_assess(clEnqueueWriteBuffer(commands,gfield,CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
                                                field, 0, NULL, NULL));
                  mcx_assess(clEnqueueWriteBuffer(commands,gPpos,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
	                                        Ppos, 0, NULL, NULL));
                  mcx_assess(clEnqueueWriteBuffer(commands,gPdir,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
	                                        Pdir, 0, NULL, NULL));
                  mcx_assess(clEnqueueWriteBuffer(commands,gPlen,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
	                                        Plen, 0, NULL, NULL));
	   }
	   if(cfg->respin>1 && RAND_SEED_LEN>1){
               for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++)
		   Pseed[i]=rand();
               mcx_assess(clEnqueueWriteBuffer(commands,gPseed,CL_TRUE,0,sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,
	                                        Pseed, 0, NULL, NULL));
	   }
       }
       if(twindow1<cfg->tend){
	    cl_float *tmpenergy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*2);
            mcx_assess(clEnqueueWriteBuffer(commands,genergy,CL_TRUE,0,sizeof(cl_float)*cfg->nthread*2,
                                        tmpenergy, 0, NULL, NULL));
	    free(tmpenergy);
            //cudaMemset(genergy,0,sizeof(float)*cfg->nthread*2);
       }
     }
     mcx_assess(clFinish(commands));
     mcx_assess(clEnqueueReadBuffer(commands,gPpos,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
     				   Ppos, 0, NULL, NULL));
     mcx_assess(clEnqueueReadBuffer(commands,gPdir,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
     				   Pdir, 0, NULL, NULL));
     mcx_assess(clEnqueueReadBuffer(commands,gPlen,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
	                           Plen, 0, NULL, NULL));
     mcx_assess(clEnqueueReadBuffer(commands,gPseed,CL_TRUE,0,sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,
	                           Pseed, 0, NULL, NULL));
     mcx_assess(clEnqueueReadBuffer(commands,genergy,CL_TRUE,0,sizeof(cl_float)*cfg->nthread*2,
	                           energy, 0, NULL, NULL));

     for (i=0; i<cfg->nthread; i++) {
	  photoncount+=(int)Plen[i].w;
          energyloss+=energy[i<<1];
          energyabsorbed+=energy[(i<<1)+1];
     }

#ifdef TEST_RACING
     {
       float totalcount=0.f,hitcount=0.f;
       for (i=0; i<fieldlen; i++)
          hitcount+=field[i];
       for (i=0; i<cfg->nthread; i++)
	  totalcount+=Pseed[i];
     
       fprintf(cfg->flog,"expected total recording number: %f, got %f, missed %f\n",
          totalcount,hitcount,(totalcount-hitcount)/totalcount);
     }
#endif

     printnum=cfg->nthread<cfg->printnum?cfg->nthread:cfg->printnum;
     for (i=0; i<printnum; i++) {
           fprintf(cfg->flog,"% 4d[A% f % f % f]C%3d J%5d W% 8f(P%6.3f %6.3f %6.3f)T% 5.3e L% 5.3f %.0f\n", i,
            Pdir[i].x,Pdir[i].y,Pdir[i].z,(int)Plen[i].w,(int)Pdir[i].w,Ppos[i].w, 
            Ppos[i].x,Ppos[i].y,Ppos[i].z,Plen[i].y,Plen[i].x,(float)Pseed[i]);
     }
     // total energy here equals total simulated photons+unfinished photons for all threads
     fprintf(cfg->flog,"simulated %d photons (%d) with %d threads (repeat x%d)\n",
             photoncount,deviceload,cfg->nthread,cfg->respin); fflush(cfg->flog);
     fprintf(cfg->flog,"exit energy:%16.8e + absorbed energy:%16.8e = total: %16.8e\n",
             energyloss,energyabsorbed,energyloss+energyabsorbed);fflush(cfg->flog);
     fflush(cfg->flog);

     clReleaseMemObject(gmedia);
     clReleaseMemObject(gfield);
     clReleaseMemObject(gPpos);
     clReleaseMemObject(gPdir);
     clReleaseMemObject(gPlen);
     clReleaseMemObject(gPseed);
     clReleaseMemObject(genergy);
     clReleaseMemObject(gproperty);
     clReleaseMemObject(gstopsign);
     clReleaseMemObject(gparam);
     clReleaseMemObject(gdetected);

     free(devices);
     clReleaseKernel(kernel);
     clReleaseProgram(program);
     clReleaseCommandQueue(commands);
     clReleaseContext(context);
#ifndef USE_OS_TIMER
     clReleaseEvent(kernelevent);
#endif
     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Pseed);
     free(energy);
     free(field);
}
