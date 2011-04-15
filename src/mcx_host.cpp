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

#define OCL_ASSERT(x)  ocl_assess((x),__FILE__,__LINE__)


extern cl_event kernelevent;


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
void ocl_assess(int cuerr,const char *file,const int linenum){
     if(cuerr!=CL_SUCCESS){
         mcx_error(-(int)cuerr,print_cl_errstring(cuerr),file,linenum);
     }
}


/*
  query GPU info and set active GPU
*/
cl_platform_id mcx_set_gpu(Config *cfg,unsigned int *activedev){

    uint i,j,k,devnum;
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
        OCL_ASSERT((clGetPlatformIDs(numPlatforms, platforms, NULL)));
        for (i = 0; i < numPlatforms; ++i) {
            platform = platforms[i];
	    if(1){
                OCL_ASSERT((clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_NAME,sizeof(pbuf),pbuf,NULL)));
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
		    OCL_ASSERT((clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize)));
                    devices = (cl_device_id*)malloc(deviceListSize);
                    OCL_ASSERT((clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL)));
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
                	OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_NAME,100,(void*)&pbuf,NULL)));
                	printf("============ %s device [%d of %d]: %s  ============\n",devname[j],k+1,devnum,pbuf);
			OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),(void*)&devparam,NULL)));
                	OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL)));
                	printf(" Compute units :\t%d core(s)\n",(uint)devparam);
                	printf(" Global memory :\t%ld B\n",(unsigned long)devmem);
                	OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL)));
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


/*
   master driver code to run MC simulations
*/
void mcx_run_simulation(Config *cfg,int activedev,float *fluence,float *totalenergy){

     cl_uint i,j,iter;
     cl_float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     cl_float t,twindow0,twindow1;
     cl_float energyloss=0.f,energyabsorbed=0.f,simuenergy,fullload=0.f;
     cl_int deviceload=0;
     cl_float *energy;
     cl_int stopsign=0;
     cl_uint detected=0,workdev;

     cl_uint photoncount=0;
     cl_uint tic,fieldlen;
     cl_uint4 cp0={{cfg->crop0.x,cfg->crop0.y,cfg->crop0.z,cfg->crop0.w}};
     cl_uint4 cp1={{cfg->crop1.x,cfg->crop1.y,cfg->crop1.z,cfg->crop1.w}};
     cl_uint2 cachebox;
     cl_uint4 dimlen;

     cl_context mcxcontext;                 // compute mcxcontext
     cl_command_queue *mcxqueue;          // compute command queue
     cl_program mcxprogram;                 // compute mcxprogram
     cl_kernel *mcxkernel;                   // compute mcxkernel
     cl_int status = 0;
     size_t deviceListSize;
     cl_platform_id platform = NULL;
     cl_device_id* devices;
     cl_event * waittoread;

     cl_uint *cucount,totalcucore;
     cl_uint  devid=0;
     cl_mem gmedia,gproperty,gparam;
     cl_mem *gfield,*gdetphoton,*gseed,*genergy;
     cl_mem *gstopsign,*gdetected,*gdetpos;

     size_t mcgrid[1], mcblock[1];

     cl_uint dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;

     cl_uchar  *media=(cl_uchar *)(cfg->vol);
     cl_float  *field;

     cl_uint   *Pseed;
     float  *Pdet;

     MCXParam param={{{cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f}},
		     {{cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f}},
		     {{cfg->dim.x,cfg->dim.y,cfg->dim.z,0}},dimlen,cp0,cp1,cachebox,
		     minstep,0.f,0.f,cfg->tend,R_C0*cfg->unitinmm,cfg->isrowmajor,
                     cfg->issave2pt,cfg->isreflect,cfg->isrefint,cfg->issavedet,1.f/cfg->tstep,
                     cfg->minenergy,
                     cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->maxdetphoton,
                     cfg->medianum-1,cfg->detnum,0,0};

     platform=mcx_set_gpu(cfg,NULL);

     cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

     /* Use NULL for backward compatibility */
     cl_context_properties* cprops=(platform==NULL)?NULL:cps;
     OCL_ASSERT(((mcxcontext=clCreateContextFromType(cprops,CL_DEVICE_TYPE_ALL,NULL,NULL,&status),status)));
     OCL_ASSERT((clGetContextInfo(mcxcontext, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize)));
     workdev=(int)deviceListSize/sizeof(cl_device_id);

     devices = (cl_device_id*)malloc(workdev*sizeof(cl_device_id));
     mcxqueue= (cl_command_queue*)malloc(workdev*sizeof(cl_command_queue));
     waittoread=(cl_event *)malloc(workdev*sizeof(cl_event));
     cucount=(cl_uint *)calloc(workdev,sizeof(cl_uint));

     gseed=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     gfield=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     gdetphoton=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     genergy=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     gstopsign=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     gdetected=(cl_mem *)malloc(workdev*sizeof(cl_mem));
     gdetpos=(cl_mem *)malloc(workdev*sizeof(cl_mem));

     if(devices == NULL){
         OCL_ASSERT(-1);
     }
     OCL_ASSERT((clGetContextInfo(mcxcontext,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL)));
     /* The block is to move the declaration of prop closer to its use */
     cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;

     totalcucore=0;
     for(i=0;i<workdev;i++){
         char pbuf[100]={'\0'};
         OCL_ASSERT(((mcxqueue[i]=clCreateCommandQueue(mcxcontext,devices[i],prop,&status),status)));
         OCL_ASSERT((clGetDeviceInfo(devices[i],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),(void*)(cucount+i),NULL)));
         OCL_ASSERT((clGetDeviceInfo(devices[i],CL_DEVICE_NAME,100,(void*)&pbuf,NULL)));
         if(strstr(pbuf,"ATI")){
            cucount[i]*=(80/5); // an ati core typically has 80 SP, and 80/5=16 VLIW
	 }else if(strstr(pbuf,"GeForce") || strstr(pbuf,"Quadro") || strstr(pbuf,"Tesla")){
            cucount[i]*=8;  // an nvidia MP typically has 8 SP
         }
         totalcucore+=cucount[i];
     }
     fullload=0.f;
     for(i=0;i<workdev;i++)
     	fullload+=cfg->workload[i];

     if(fullload<EPS){
	for(i=0;i<workdev;i++)
     	    cfg->workload[i]=cucount[i];
	fullload=totalcucore;
     }

     if(cfg->respin>1){
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate*2); //the second half will be used to accumul$
     }else{
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate);
     }
     if(cfg->nthread%cfg->nblocksize)
        cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;

     mcgrid[0]=cfg->nthread;
     mcblock[0]=cfg->nblocksize;

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

     if(cfg->seed>0)
     	srand(cfg->seed);
     else
        srand(time(0));

     OCL_ASSERT(((gmedia=clCreateBuffer(mcxcontext,RO_MEM, sizeof(cl_uchar)*(dimxyz),media,&status),status)));
     OCL_ASSERT(((gproperty=clCreateBuffer(mcxcontext,RO_MEM, cfg->medianum*sizeof(Medium),cfg->prop,&status),status)));
     OCL_ASSERT(((gparam=clCreateBuffer(mcxcontext,RO_MEM, sizeof(MCXParam),&param,&status),status)));

     for(i=0;i<workdev;i++){
       for (j=0; j<cfg->nthread*RAND_SEED_LEN;j++)
	   Pseed[j]=rand();
       OCL_ASSERT(((gseed[i]=clCreateBuffer(mcxcontext,RW_MEM, sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,Pseed,&status),status)));
       OCL_ASSERT(((gfield[i]=clCreateBuffer(mcxcontext,RW_MEM, sizeof(cl_float)*(dimxyz)*cfg->maxgate,field,&status),status)));
       OCL_ASSERT(((gdetphoton[i]=clCreateBuffer(mcxcontext,RW_MEM, sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1),Pdet,&status),status)));
       OCL_ASSERT(((genergy[i]=clCreateBuffer(mcxcontext,RW_MEM, sizeof(float)*cfg->nthread*2,energy,&status),status)));
       OCL_ASSERT(((gstopsign[i]=clCreateBuffer(mcxcontext,RW_PTR, sizeof(cl_uint),&stopsign,&status),status)));
       OCL_ASSERT(((gdetected[i]=clCreateBuffer(mcxcontext,RW_MEM, sizeof(cl_uint),&detected,&status),status)));
       OCL_ASSERT(((gdetpos[i]=clCreateBuffer(mcxcontext,RO_MEM, cfg->detnum*sizeof(float4),cfg->detpos,&status),status)));
     }

     fprintf(cfg->flog,"\
===============================================================================\n\
=                     Monte Carlo eXtreme (MCX) -- OpenCL                     =\n\
=     Copyright (c) 2009-2011 Qianqian Fang <fangq at nmr.mgh.harvard.edu>    =\n\
=                                                                             =\n\
=    Martinos Center for Biomedical Imaging, Massachusetts General Hospital   =\n\
===============================================================================\n\
$MCXCL$Rev::    $ Last Commit $Date::                     $ by $Author:: fangq$\n\
===============================================================================\n");

     tic=StartTimer();
     if(cfg->issavedet)
         fprintf(cfg->flog,"- variant name: [%s] compiled with OpenCL version [%d]\n",
             "Detective MCXCL",CL_VERSION_1_0);
     else
         fprintf(cfg->flog,"- code name: [Vanilla MCXCL] compiled with OpenCL version [%d]\n",
             CL_VERSION_1_0);

     fprintf(cfg->flog,"- compiled with: [RNG] %s [Seed Length] %d\n",MCX_RNG_NAME,RAND_SEED_LEN);
     fprintf(cfg->flog,"initializing streams ...\t");
     fflush(cfg->flog);
     fieldlen=dimxyz*cfg->maxgate;

     fprintf(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

     OCL_ASSERT(((mcxprogram=clCreateProgramWithSource(mcxcontext, 1,(const char **)&(cfg->clsource), NULL, &status),status)));
     if(cfg->iscpu && cfg->isverbose){ 
	status=clBuildProgram(mcxprogram, 0, NULL, "-D __DEVICE_EMULATION__ -D MCX_CPU_ONLY -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
     }else{
	if(cfg->issavedet)
		status=clBuildProgram(mcxprogram, 0, NULL, "-D SAVE_DETECTORS -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);    
	else
		status=clBuildProgram(mcxprogram, 0, NULL, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);    
     }

     if(status!=CL_SUCCESS){
	 size_t len;
	 char msg[2048];
	 // get the details on the error, and store it in buffer
	 clGetProgramBuildInfo(mcxprogram,devices[devid],CL_PROGRAM_BUILD_LOG,sizeof(msg),msg,&len); 
	 fprintf(cfg->flog,"Kernel build error:\n%s\n", msg);
	 mcx_error(-(int)status,(char*)("Error: Failed to build program executable!"),__FILE__,__LINE__);
     }
     fprintf(cfg->flog,"build program complete : %d ms\n",GetTimeMillis()-tic);

     mcxkernel=(cl_kernel*)malloc(workdev*sizeof(cl_kernel));

     for(i=0;i<workdev;i++){
         cl_int threadphoton, oddphotons;

         threadphoton=cfg->nphoton*cfg->workload[i]/(fullload*cfg->nthread*cfg->respin);
         oddphotons=cfg->nphoton*cfg->workload[i]/(fullload*cfg->respin)-threadphoton*cfg->nthread;
         fprintf(cfg->flog,"- [device %d] threadph=%d oddphotons=%d np=%.1f nthread=%d repetition=%d\n",i,threadphoton,oddphotons,
               cfg->nphoton*cfg->workload[i]/fullload,cfg->nthread,cfg->respin);

	 OCL_ASSERT(((mcxkernel[i] = clCreateKernel(mcxprogram, "mcx_main_loop", &status),status)));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 0, sizeof(cl_uint),(void*)&threadphoton)));
         OCL_ASSERT((clSetKernelArg(mcxkernel[i], 1, sizeof(cl_uint),(void*)&oddphotons)));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 2, sizeof(cl_mem), (void*)&gmedia)));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 3, sizeof(cl_mem), (void*)(gfield+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 4, sizeof(cl_mem), (void*)(genergy+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 5, sizeof(cl_mem), (void*)(gseed+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 6, sizeof(cl_mem), (void*)(gdetphoton+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 7, sizeof(cl_mem), (void*)&gproperty)));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 8, sizeof(cl_mem), (void*)(gdetpos+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i], 9, sizeof(cl_mem), (void*)(gstopsign+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i],10, sizeof(cl_mem), (void*)(gdetected+i))));
	 OCL_ASSERT((clSetKernelArg(mcxkernel[i],11, cfg->issavedet? sizeof(cl_float)*cfg->nblocksize*param.maxmedia : 1, NULL)));
     }
     fprintf(cfg->flog,"set kernel arguments complete : %d ms\n",GetTimeMillis()-tic);

     //simulate for all time-gates in maxgate groups per run

     cl_float Vvox,scale,absorp,eabsorp;
     Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z;

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
           OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gparam,CL_TRUE,0,sizeof(MCXParam),&param, 0, NULL, NULL)));
           for(devid=0;devid<workdev;devid++){
              OCL_ASSERT((clSetKernelArg(mcxkernel[devid],12, sizeof(cl_mem), (void*)&gparam)));
              // launch mcxkernel
               OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue[devid],mcxkernel[devid],1,NULL,mcgrid,mcblock, 0, NULL, 
#ifndef USE_OS_TIMER
                  &kernelevent)));
#else
                  NULL)));
#endif
               OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gdetected[devid],CL_FALSE,0,sizeof(uint),
                                            &detected, 0, NULL, waittoread+devid)));
           }
           clWaitForEvents(workdev,waittoread);

           fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving flux ... \t",GetTimeMillis()-tic);

           for(devid=0;devid<workdev;devid++){
             if(cfg->issavedet){
                OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gdetphoton[devid],CL_TRUE,0,sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1),
	                                        Pdet, 0, NULL, NULL)));
		if(detected>cfg->maxdetphoton){
			fprintf(cfg->flog,"WARNING: the detected photon (%d) \
is more than what your have specified (%d), please use the -H option to specify a greater number\t"
                           ,detected,cfg->maxdetphoton);
		}else{
			fprintf(cfg->flog,"detected %d photons\t",detected);
		}
		cfg->his.unitinmm=cfg->unitinmm;
		cfg->his.detected=detected;
		cfg->his.savedphoton=MIN(detected,cfg->maxdetphoton);
		if(cfg->exportdetected) //you must allocate the buffer long enough
	                memcpy(cfg->exportdetected,Pdet,cfg->his.savedphoton*(cfg->medianum+1)*sizeof(float));
		else
			mcx_savedata(Pdet,cfg->his.savedphoton*(cfg->medianum+1),
		             photoncount>cfg->his.totalphoton,"mch",cfg);
	     }
	     //handling the 2pt distributions
             if(cfg->issave2pt){
               OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gfield[devid],CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
	                                        field, 0, NULL, NULL)));
               fprintf(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

               if(cfg->respin>1){
                   for(i=0;i<fieldlen;i++)  //accumulate field, can be done in the GPU
                      field[fieldlen+i]+=field[i];
               }
               if(iter+1==cfg->respin){ 
                   if(cfg->respin>1)  //copy the accumulated fields back
                       memcpy(field,field+fieldlen,sizeof(cl_float)*fieldlen);

		   simuenergy=cfg->nphoton*cfg->workload[devid]/fullload;
                   if(cfg->isnormalized){

                       fprintf(cfg->flog,"normizing raw data ...\t");
                       OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],genergy[devid],CL_TRUE,0,sizeof(cl_float)*cfg->nthread*2,
	                                        energy, 0, NULL, NULL)));
                       eabsorp=0.f;
                       for(i=1;i<cfg->nthread;i++){
                           energy[0]+=energy[i<<1];
       	       	       	   energy[1]+=energy[(i<<1)+1];
                           //eabsorp+=Plen[i].z;  // the accumulative absorpted energy near the source
                       }
       	       	       for(i=0;i<dimxyz;i++){
                           absorp=0.f;
                           for(j=0;j<cfg->maxgate;j++)
                              absorp+=field[j*dimxyz+i];
                           eabsorp+=absorp*cfg->prop[media[i]].mua;
       	       	       }
		       energyloss+=energy[0];
		       energyabsorbed+=energy[1];
		       simuenergy=energy[0]+energy[1];
                       scale=energy[1]/(simuenergy*Vvox*cfg->tstep*eabsorp);
                       fprintf(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   fprintf(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);
                   for(i=0;i<fieldlen;i++)
		   	fluence[i]+=field[i]*simuenergy;

                   *totalenergy+=simuenergy;
                   fprintf(cfg->flog,"total simulated energy: %f\n",*totalenergy);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
		   fflush(cfg->flog);

                   if(*totalenergy>EPS)
		   	*totalenergy=1.f/(*totalenergy);
                   for(i=0;i<fieldlen;i++)
		   	fluence[i]*=(*totalenergy);

                   fprintf(cfg->flog,"saving data to file ...\t");
                   mcx_savedata(fluence,fieldlen,t>cfg->tstart,"mc2",cfg);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
                   fflush(cfg->flog);
               }
               OCL_ASSERT((clFinish(mcxqueue[devid])));
             }
	     //initialize the next simulation
	     if(twindow1<cfg->tend && iter<cfg->respin){
                  memset(field,0,sizeof(cl_float)*dimxyz*cfg->maxgate);
                  OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gfield[devid],CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
                                                field, 0, NULL, NULL)));
		  OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 3, sizeof(cl_mem), (void*)(gfield+devid))));
	     }
	     if(cfg->respin>1 && RAND_SEED_LEN>1){
               for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++)
		   Pseed[i]=rand();
               OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gseed[devid],CL_TRUE,0,sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,
	                                        Pseed, 0, NULL, NULL)));
	       OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 5, sizeof(cl_mem), (void*)(gseed+devid))));
	     }
           }// loop over work devices
       }// iteration
       if(twindow1<cfg->tend){
	    cl_float *tmpenergy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*2);
            OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],genergy[devid],CL_TRUE,0,sizeof(cl_float)*cfg->nthread*2,
                                        tmpenergy, 0, NULL, NULL)));
	    OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 4, sizeof(cl_mem), (void*)(genergy+devid))));	
	    free(tmpenergy);
       }
     }// time gates

     // total energy here equals total simulated photons+unfinished photons for all threads
     fprintf(cfg->flog,"simulated %d photons (%d) with %d threads (repeat x%d)\n",
             photoncount,deviceload,cfg->nthread,cfg->respin); fflush(cfg->flog);
     fprintf(cfg->flog,"exit energy:%16.8e + absorbed energy:%16.8e = total: %16.8e\n",
             energyloss,energyabsorbed,energyloss+energyabsorbed);fflush(cfg->flog);
     fflush(cfg->flog);

     clReleaseMemObject(gmedia);
     clReleaseMemObject(gproperty);
     clReleaseMemObject(gparam);

     for(i=0;i<workdev;i++){
         clReleaseMemObject(gfield[i]);
         clReleaseMemObject(gseed[i]);
         clReleaseMemObject(genergy[i]);
         clReleaseMemObject(gstopsign[i]);
         clReleaseMemObject(gdetected[i]);
         clReleaseMemObject(gdetpos[i]);
         clReleaseKernel(mcxkernel[i]);
     }
     free(gfield);
     free(gseed);
     free(genergy);
     free(gstopsign);
     free(gdetected);
     free(gdetpos);
     free(mcxkernel);

     free(devices);
     free(waittoread);

     for(devid=0;devid<workdev;devid++)
        clReleaseCommandQueue(mcxqueue[devid]);

     free(mcxqueue);
     clReleaseProgram(mcxprogram);
     clReleaseContext(mcxcontext);
#ifndef USE_OS_TIMER
     clReleaseEvent(kernelevent);
#endif
     free(Pseed);
     free(energy);
     free(field);
}
