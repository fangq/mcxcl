#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mcx_host.hpp"
#include "tictoc.h"
#include "mcx_const.h"

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MCX_RNG_NAME       "Logistic-Lattice"
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define RO_MEM             (CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR)
#define WO_MEM             (CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR)
#define RW_MEM             (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)

extern cl_event kernelevent;

/*
  query GPU info and set active GPU
*/
cl_platform_id mcx_set_gpu(int printinfo){

#if __DEVICE_EMULATION__
    return 1;
#else
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;

    clGetPlatformIDs(0, NULL, &numPlatforms);

    if (numPlatforms>0) {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        mcx_assess(clGetPlatformIDs(numPlatforms, platforms, NULL));
        for (unsigned i = 0; i < numPlatforms; ++i) {
            platform = platforms[i];
	    if(printinfo){
                char pbuf[100];
                mcx_assess(clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_NAME,sizeof(pbuf),pbuf,NULL));
		printf("Platform [%d] Name %s\n",i,pbuf);
	    }
        }
        delete[] platforms;
    }
    if(printinfo==2) exit(0);
    return platform;
#endif
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
void mcx_run_simulation(Config *cfg){

     cl_int i,j,iter;
     cl_float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
     cl_float4 p0={{cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f}};
     cl_float4 c0={{cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f}};
     cl_float4 maxidx={{cfg->dim.x,cfg->dim.y,cfg->dim.z}};
     cl_float t,twindow0,twindow1;
     cl_float energyloss=0.f,energyabsorbed=0.f,savefreq,bubbler2;
     cl_float *energy;
     cl_int threadphoton, oddphotons;

     cl_int photoncount=0,printnum;
     cl_int tic,fieldlen;
     uint4 cp0=cfg->crop0,cp1=cfg->crop1;
     uint2 cachebox;
     uint4 dimlen;
     //uint4 threaddim;
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
    cl_mem gmedia,gfield,gPpos,gPdir,gPlen,gPseed,genergy,gproperty;

     cl_uint mcgrid, mcblock;
     
     int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     
     cl_uchar  *media=(cl_uchar *)(cfg->vol);
     cl_float  *field;
     if(cfg->respin>1){
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate*2);
     }else{
         field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate); //the second half will be used to accumulate
     }
     threadphoton=cfg->nphoton/cfg->nthread/cfg->respin;
     oddphotons=cfg->nphoton-threadphoton*cfg->nthread*cfg->respin;

     float4 *Ppos;
     float4 *Pdir;
     float4 *Plen;
     cl_uint   *Pseed;
    
    if(cfg->iscpu){
        dType = CL_DEVICE_TYPE_CPU;
    }else{ //deviceType = "gpu" 
        dType = CL_DEVICE_TYPE_GPU;
    }
    platform=mcx_set_gpu(cfg->isgpuinfo);
    
    cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    /* Use NULL for backward compatibility */
    cl_context_properties* cprops=(platform==NULL)?NULL:cps;

    mcx_assess((context=clCreateContextFromType(cprops,dType,NULL,NULL,&status),status));
    mcx_assess(clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize));

    devices = (cl_device_id*)malloc(deviceListSize);
    if(devices == NULL){
        mcx_assess(-1);
    }
    mcx_assess(clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL));

    {
        /* The block is to move the declaration of prop closer to its use */
        cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;
        mcx_assess((commands=clCreateCommandQueue(context,devices[0],prop,&status),status));
    }
    mcx_assess(clGetDeviceInfo(devices[0],CL_DEVICE_MAX_WORK_GROUP_SIZE,
                  sizeof(size_t),(void*)&maxWorkGroupSize,NULL));

     if(cfg->nthread%cfg->nblocksize)
     	cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;
     mcgrid=cfg->nthread/cfg->nblocksize;
     mcblock=cfg->nblocksize;

     Ppos=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Pdir=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Plen=(float4*)malloc(sizeof(cl_float4)*cfg->nthread);
     Pseed=(cl_uint*)malloc(sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN);
     energy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*2);

     if(cfg->isrowmajor){ // if the volume is stored in C array order
	     cachebox.x=(cp1.z-cp0.z+1);
	     cachebox.y=(cp1.y-cp0.y+1)*(cp1.z-cp0.z+1);
	     dimlen.x=cfg->dim.z;
	     dimlen.y=cfg->dim.y*cfg->dim.z;
     }else{               // if the volume is stored in matlab/fortran array order
	     cachebox.x=(cp1.x-cp0.x+1);
	     cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
	     dimlen.x=cfg->dim.x;
	     dimlen.y=cfg->dim.y*cfg->dim.x;
     }
     dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     /*
      threaddim.x=cfg->dim.z;
      threaddim.y=cfg->dim.y*cfg->dim.z;
      threaddim.z=dimlen.z;
     */
     Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z;

     if(cfg->seed>0)
     	srand(cfg->seed);
     else
        srand(time(0));
	
     for (i=0; i<cfg->nthread; i++) {
           memcpy(Ppos+i,&p0,sizeof(p0));
           memcpy(Pdir+i,&c0,sizeof(c0));
	   //Ppos[i]=p0;  // initial position
           //Pdir[i]=c0;
           Plen[i].x=0.f;Plen[i].y=0.f;Plen[i].z=minstep*R_C0;Plen[i].w=0.f;
     }
     for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++) {
	   Pseed[i]=rand();
     }
     
     mcx_assess((gmedia=clCreateBuffer(context,RO_MEM, sizeof(cl_uchar)*(dimxyz),media,&status),status));
     mcx_assess((gfield=clCreateBuffer(context,RW_MEM, sizeof(cl_float)*(dimxyz)*cfg->maxgate,field,&status),status));
     mcx_assess((gPpos=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Ppos,&status),status));
     mcx_assess((gPdir=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Pdir,&status),status));
     mcx_assess((gPlen=clCreateBuffer(context,RW_MEM, sizeof(cl_float4)*cfg->nthread,Plen,&status),status));
     mcx_assess((gPseed=clCreateBuffer(context,RW_MEM, sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,Pseed,&status),status));
     mcx_assess((genergy=clCreateBuffer(context,RW_MEM, sizeof(float)*cfg->nthread*2,energy,&status),status));
     mcx_assess((gproperty=clCreateBuffer(context,RO_MEM, cfg->medianum*sizeof(Medium),cfg->prop,&status),status));

     fprintf(cfg->flog,"\
###############################################################################\n\
#                 Monte Carlo Extreme (MCX) -- OpenCL                         #\n\
###############################################################################\n\
$MCX $Rev::     $ Last Commit:$Date::                     $ by $Author:: fangq$\n\
###############################################################################\n");

     tic=StartTimer();
     fprintf(cfg->flog,"compiled with: [RNG] %s [Seed Length] %d\n",MCX_RNG_NAME,RAND_SEED_LEN);
     fprintf(cfg->flog,"threadph=%d oddphotons=%d np=%d nthread=%d repetition=%d\n",threadphoton,oddphotons,
           cfg->nphoton,cfg->nthread,cfg->respin);
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

    mcx_assess((program=clCreateProgramWithSource(context, 1,(const char **)&(cfg->clsource), NULL, &status),status));
    if(cfg->iscpu && cfg->isverbose){ 
       status=clBuildProgram(program, 0, NULL, "-D __DEVICE_EMULATION__ -cl-fast-relaxed-math", NULL, NULL);
    }else{
       status=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);    
    }
    if(status!=CL_SUCCESS){
	size_t len;
	char msg[2048];
	// get the details on the error, and store it in buffer
	clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,sizeof(msg),msg,&len); 
	fprintf(cfg->flog,"Kernel build error:\n%s\n", msg);
	mcx_error(-(int)status,(char*)("Error: Failed to build program executable!"));
    }
    mcx_assess((kernel = clCreateKernel(program, "mcx_main_loop", &status),status));
    mcx_assess(clGetKernelWorkGroupInfo(kernel,devices[0],CL_KERNEL_WORK_GROUP_SIZE,
               sizeof(size_t),&kernelWorkGroupSize,0));
    oddphotons=0;
    savefreq=1.f/cfg->tstep;
    bubbler2=cfg->sradius*cfg->sradius;
    
    mcx_assess(clSetKernelArg(kernel, 0, sizeof(cl_uint), (void*)&(cfg->nphoton)));
    mcx_assess(clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&(oddphotons)));
    mcx_assess(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&gmedia));
    mcx_assess(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&gfield));
    mcx_assess(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&genergy));
    mcx_assess(clSetKernelArg(kernel, 5, sizeof(cl_float4), (void*)&(cfg->steps)));
    mcx_assess(clSetKernelArg(kernel, 6, sizeof(cl_float), (void*)&(minstep)));
    mcx_assess(clSetKernelArg(kernel, 9, sizeof(cl_float), (void*)&(cfg->tend)));
    mcx_assess(clSetKernelArg(kernel,10, sizeof(cl_uint4), (void*)&(dimlen)));
    mcx_assess(clSetKernelArg(kernel,11, sizeof(cl_uchar), (void*)&(cfg->isrowmajor)));
    mcx_assess(clSetKernelArg(kernel,12, sizeof(cl_uchar), (void*)&(cfg->issave2pt)));
    mcx_assess(clSetKernelArg(kernel,13, sizeof(cl_float), (void*)&(savefreq)));
    mcx_assess(clSetKernelArg(kernel,14, sizeof(cl_float4), (void*)&(p0)));
    mcx_assess(clSetKernelArg(kernel,15, sizeof(cl_float4), (void*)&(c0)));
    mcx_assess(clSetKernelArg(kernel,16, sizeof(cl_float4), (void*)&(maxidx)));
    mcx_assess(clSetKernelArg(kernel,17, sizeof(cl_uint4), (void*)&(cp0)));
    mcx_assess(clSetKernelArg(kernel,18, sizeof(cl_uint4), (void*)&(cp1)));
    mcx_assess(clSetKernelArg(kernel,19, sizeof(cl_uint2), (void*)&(cachebox)));
    mcx_assess(clSetKernelArg(kernel,20, sizeof(cl_uchar), (void*)&(cfg->isreflect)));
    mcx_assess(clSetKernelArg(kernel,21, sizeof(cl_uchar), (void*)&(cfg->isref3)));
    mcx_assess(clSetKernelArg(kernel,22, sizeof(cl_float), (void*)&(cfg->minenergy)));
    mcx_assess(clSetKernelArg(kernel,23, sizeof(cl_float), (void*)&(bubbler2)));
    mcx_assess(clSetKernelArg(kernel,24, sizeof(cl_mem),   (void*)&(gPseed)));
    mcx_assess(clSetKernelArg(kernel,25, sizeof(cl_mem), (void*)&gPpos));
    mcx_assess(clSetKernelArg(kernel,26, sizeof(cl_mem), (void*)&gPdir));
    mcx_assess(clSetKernelArg(kernel,27, sizeof(cl_mem), (void*)&gPlen));
    mcx_assess(clSetKernelArg(kernel,28, sizeof(cl_mem), (void*)&gproperty));

     //simulate for all time-gates in maxgate groups per run
     for(t=cfg->tstart;t<cfg->tend;t+=cfg->tstep*cfg->maxgate){
       twindow0=t;
       twindow1=t+cfg->tstep*cfg->maxgate;

       fprintf(cfg->flog,"lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n"
           ,twindow0*1e9,twindow1*1e9);

       //total number of repetition for the simulations, results will be accumulated to field
       for(iter=0;iter<cfg->respin;iter++){

           fprintf(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);

           mcx_assess(clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&(twindow0)));
           mcx_assess(clSetKernelArg(kernel, 8, sizeof(cl_float), (void*)&(twindow1)));

           // launch kernel
           mcx_assess(clEnqueueNDRangeKernel(commands,kernel,1,NULL,(size_t*)(&(cfg->nthread)), 
	          (size_t*)(&(cfg->nblocksize)), 0, NULL, &kernelevent));

	   //handling the 2pt distributions
           if(cfg->issave2pt){
               mcx_assess(clEnqueueReadBuffer(commands,gfield,CL_TRUE,0,sizeof(cl_float),
                                                field, 0, NULL, NULL));
               fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving fields ... \t",GetTimeMillis()-tic);
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
                       scale=energy[1]/(energy[0]+energy[1])/Vvox/cfg->tstep/eabsorp;
                       fprintf(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
                       mcx_normalize(field,scale,fieldlen);
                   }
                   fprintf(cfg->flog,"data normalization complete : %d ms\n",GetTimeMillis()-tic);

                   fprintf(cfg->flog,"saving data to file ...\t");
                   mcx_savedata(field,fieldlen,cfg);
                   fprintf(cfg->flog,"saving data complete : %d ms\n",GetTimeMillis()-tic);
                   fflush(cfg->flog);
               }
           }
	   //initialize the next simulation
	   if(twindow1<cfg->tend && iter+1<cfg->respin){
//                  cudaMemset(gfield,0,sizeof(float)*fieldlen); // cost about 1 ms
//                  mcx_assess(clEnqueueWriteBuffer(commands,gfield,CL_TRUE,0,sizeof(cl_float4)*cfg->nthread,
//	                                        field, 0, NULL, NULL));
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
//            cudaMemset(genergy,0,sizeof(float)*cfg->nthread*2);
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
             photoncount,cfg->nphoton,cfg->nthread,cfg->respin); fflush(cfg->flog);
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

    free(devices);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

     free(Ppos);
     free(Pdir);
     free(Plen);
     free(Pseed);
     free(energy);
     free(field);
}
