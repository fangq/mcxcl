/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration \
**      -- OpenCL edition
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**         "Scalable and massively parallel Monte Carlo photon transport simulations 
**         for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504 (2018)
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_utils.c
@brief   mcconfiguration and command line option processing unit
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

#ifndef WIN32
  #include <sys/ioctl.h>
#endif
#include <sys/stat.h>

#include "mcx_utils.h"
#include "mcx_shapes.h"
#include "mcx_const.h"

#ifdef MCX_EMBED_CL
    #include "mcx_core.clh"
#endif

#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? fallback : tmp->val) \
                     : tmp->val)

#define FIND_JSON_OBJ(id,idfull,parent) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? NULL : tmp) \
                     : tmp)

/**
 * Short command line options
 * If a short command line option is '-' that means it only has long/verbose option.
 * Array terminates with '\0'.
 */

char shortopt[]={'h','i','f','n','m','t','T','s','a','g','b','B','D','-','G','W','z',
                 'd','r','S','p','e','U','R','l','L','M','I','-','o','c','k','v','J',
                 'A','P','E','F','H','-','u','-','x','X','\0'};

/**
 * Long command line options
 * The length of this array must match the length of shortopt[], terminates with ""
 */

const char *fullopt[]={"--help","--interactive","--input","--photon","--move",
                 "--thread","--blocksize","--session","--array","--gategroup",
                 "--reflect","--reflect3","--debug","--devicelist","--gpu","--workload","--srcfrom0",
		 "--savedet","--repeat","--save2pt","--printlen","--minenergy",
                 "--normalize","--skipradius","--log","--listgpu","--dumpmask",
                 "--printgpu","--root","--optlevel","--cpu","--kernel","--verbose","--compileropt",
                 "--autopilot","--shapes","--seed","--outputformat","--maxdetphoton",
		 "--mediabyte","--unitinmm","--atomic","--saveexit","--saveref",""};

/**
 * Debug flags
 * R: debug random number generator
 * M: record photon movement and trajectory
 * P: show progress bar
 */

const char debugflag[]={'R','M','P','\0'};

/**
 * Source type specifier
 * User can specify the source type using a string
 */

const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar",
    "pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian",
    "line","slit","pencilarray","pattern3d",""};

#ifdef WIN32
         char pathsep='\\';
#else
         char pathsep='/';
#endif

const char outputtype[]={'x','f','e','j','p','\0'};
const char *outputformat[]={"mc2","nii","hdr","ubj",""};

void mcx_initcfg(Config *cfg){
     cfg->medianum=0;
     cfg->mediabyte=1;
     cfg->detnum=0;
     cfg->dim.x=0;
     cfg->dim.y=0;
     cfg->dim.z=0;
     cfg->steps.x=1.f;
     cfg->steps.y=1.f;
     cfg->steps.z=1.f;
     cfg->nblocksize=64;
     cfg->nphoton=0;
     cfg->nthread=(1<<14);
     cfg->seed=0x623F9A9E;
     cfg->isrowmajor=0; /* default is Matlab array*/
     cfg->maxgate=0;
     cfg->isreflect=1;
     cfg->isref3=0;
     cfg->isnormalized=1;
     cfg->issavedet=1;
     cfg->respin=1;
     cfg->issave2pt=1;
     cfg->isgpuinfo=0;
     cfg->unitinmm=1.f;
     cfg->isrefint=0;
     cfg->prop=NULL;
     cfg->detpos=NULL;
     cfg->vol=NULL;
     cfg->session[0]='\0';
     cfg->printnum=0;
     cfg->minenergy=0.f;
     cfg->flog=stdout;
     cfg->sradius=0.f;
     cfg->rootpath[0]='\0';
     cfg->iscpu=0;
     cfg->isverbose=0;

     cfg->srctype=0;;         /** use pencil beam as default source type */
     cfg->maxvoidstep=1000;
     cfg->voidtime=1;
     memset(&(cfg->srcparam1),0,sizeof(float4));
     memset(&(cfg->srcparam2),0,sizeof(float4));
#ifdef MCX_EMBED_CL
     cfg->clsource=(char *)mcx_core_cl;
#else
     cfg->clsource=NULL;
#endif
     cfg->maxdetphoton=1000000; 
     cfg->isdumpmask=0;
     cfg->autopilot=0;
     cfg->shapedata=NULL;
     cfg->optlevel=3;

     memset(cfg->deviceid,0,MAX_DEVICE);
     memset(cfg->compileropt,0,MAX_PATH_LENGTH);
     memset(cfg->workload,0,MAX_DEVICE*sizeof(float));
     cfg->deviceid[0]='1'; /*use the first GPU device by default*/
     memset(cfg->kernelfile,0,MAX_SESSION_LENGTH);
     cfg->issrcfrom0=0;

     cfg->exportfield=NULL;
     cfg->exportdetected=NULL;
     cfg->exportdebugdata=NULL;
     cfg->maxjumpdebug=1000000;

     cfg->seeddata=NULL;
     cfg->reseedlimit=10000000;
     cfg->issaveseed=0;
     cfg->issaveexit=0;
     cfg->issaveref=0;
     cfg->isatomic=1;

     cfg->replay.seed=NULL;
     cfg->replay.weight=NULL;
     cfg->replay.tof=NULL;
     cfg->replaydet=0;
     cfg->seedfile[0]='\0';

     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     cfg->runtime=0;
     cfg->debuglevel=0;
     cfg->gpuid=0;

     memset(&cfg->his,0,sizeof(History));
     memcpy(cfg->his.magic,"MCXH",4);
     cfg->his.version=1;
     cfg->his.unitinmm=1.f;
     cfg->his.normalizer=1.f;
     cfg->exportfield=NULL;
     cfg->exportdetected=NULL;
     cfg->detectedcount=0;
     cfg->runtime=0;
#ifdef MCX_CONTAINER
     cfg->parentid=mpMATLAB;
#else
     cfg->parentid=mpStandalone;
#endif
     cfg->seeddata=NULL;
     cfg->outputtype=otFlux;
     cfg->outputformat=ofMC2;
     cfg->srcdir.w=0.f;
}

void mcx_clearcfg(Config *cfg){
     if(cfg->medianum)
     	free(cfg->prop);
     if(cfg->detnum)
     	free(cfg->detpos);
     if(cfg->dim.x && cfg->dim.y && cfg->dim.z)
        free(cfg->vol);
#ifndef MCX_EMBED_CL
     if(cfg->clsource && cfg->clsource!=(char *)mcx_core_cl)
        free(cfg->clsource);
#endif
     if(cfg->exportfield)
        free(cfg->exportfield);
     if(cfg->exportdetected)
        free(cfg->exportdetected);
     if(cfg->seeddata)
        free(cfg->seeddata);

     mcx_initcfg(cfg);
}


/**
 * @brief Reset and clear the GPU information data structure
 *
 * Clearing the GPU information data structure
 */

void mcx_cleargpuinfo(GPUInfo **gpuinfo){
    if(*gpuinfo){
	free(*gpuinfo);
	*gpuinfo=NULL;
    }
}

void mcx_savenii(float *dat, int len, char* name, int type32bit, int outputformatid, Config *cfg){
     FILE *fp;
     char fname[MAX_PATH_LENGTH]={'\0'};
     nifti_1_header hdr;
     nifti1_extender pad={{0,0,0,0}};
     float *logval=dat;
     int i;

     memset((void *)&hdr, 0, sizeof(hdr));
     hdr.sizeof_hdr = MIN_HEADER_SIZE;
     hdr.dim[0] = 4;
     hdr.dim[1] = cfg->dim.x;
     hdr.dim[2] = cfg->dim.y;
     hdr.dim[3] = cfg->dim.z;
     hdr.dim[4] = len/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
     hdr.datatype = type32bit;
     hdr.bitpix = 32;
     hdr.pixdim[1] = cfg->unitinmm;
     hdr.pixdim[2] = cfg->unitinmm;
     hdr.pixdim[3] = cfg->unitinmm;
     hdr.intent_code=NIFTI_INTENT_NONE;
     logval=(float *)malloc(sizeof(float)*len);

     if(type32bit==NIFTI_TYPE_FLOAT32){
	 for(i=0;i<len;i++)
	    logval[i]=log10f(dat[i]);
	 hdr.intent_code=NIFTI_INTENT_LOG10PVAL;
         hdr.pixdim[4] = cfg->tstep*1e6f;
     }else{
         short *mask=(short*)logval;
	 for(i=0;i<len;i++){
	    mask[i]    =(((unsigned int *)dat)[i] & MED_MASK);
	    mask[i+len]=(((unsigned int *)dat)[i] & DET_MASK)>>16;
	 }
	 hdr.datatype = NIFTI_TYPE_UINT16;
	 hdr.bitpix = 16;
         hdr.dim[4] = 2;
         hdr.pixdim[4] = 1.f;
     }
     if (outputformatid==ofNifti){
	strncpy(hdr.magic, "n+1\0", 4);
	hdr.vox_offset = (float) NII_HEADER_SIZE;
     }else{
	strncpy(hdr.magic, "ni1\0", 4);
	hdr.vox_offset = (float)0;
     }
     hdr.scl_slope = 0.f;
     hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_USEC;

     sprintf(fname,"%s.%s",name,outputformat[outputformatid]);

     if (( fp = fopen(fname,"wb")) == NULL)
             mcx_error(-9, "Error opening header file for write",__FILE__,__LINE__);

     if (fwrite(&hdr, MIN_HEADER_SIZE, 1, fp) != 1)
             mcx_error(-9, "Error writing header file",__FILE__,__LINE__);

     if (outputformatid==ofNifti) {
         if (fwrite(&pad, 4, 1, fp) != 1)
             mcx_error(-9, "Error writing header file extension pad",__FILE__,__LINE__);

         if (fwrite(logval, (size_t)(hdr.bitpix>>3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) !=
	          hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
             mcx_error(-9, "Error writing data to file",__FILE__,__LINE__);
	 fclose(fp);
     }else if(outputformatid==ofAnalyze){
         fclose(fp);  /* close .hdr file */

         sprintf(fname,"%s.img",name);

         fp = fopen(fname,"wb");
         if (fp == NULL)
             mcx_error(-9, "Error opening img file for write",__FILE__,__LINE__);
         if (fwrite(logval, (size_t)(hdr.bitpix>>3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) != 
	       hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
             mcx_error(-9, "Error writing img file",__FILE__,__LINE__);

         fclose(fp);
     }else
         mcx_error(-9, "Output format is not supported",__FILE__,__LINE__);
     free(logval);
}

void mcx_savedata(float *dat, int len, Config *cfg){
     FILE *fp;
     char name[MAX_PATH_LENGTH];
     char fname[MAX_PATH_LENGTH];

     if(cfg->rootpath[0])
         sprintf(name,"%s%c%s",cfg->rootpath,pathsep,cfg->session);
     else
         sprintf(name,"%s",cfg->session);

     if(cfg->outputformat==ofNifti || cfg->outputformat==ofAnalyze){
         mcx_savenii(dat, len, name, NIFTI_TYPE_FLOAT32, cfg->outputformat, cfg);
         return;
     }
     sprintf(fname,"%s.%s",name,outputformat[(int)cfg->outputformat]);
     fp=fopen(fname,"wb");

     if(fp==NULL){
	mcx_error(-2,"can not save data to disk",__FILE__,__LINE__);
     }
     fwrite(dat,sizeof(float),len,fp);
     fclose(fp);
}

void mcx_savedetphoton(float *ppath, void *seeds, int count, int doappend, Config *cfg){
	FILE *fp;
	char fhistory[MAX_PATH_LENGTH];
        if(cfg->rootpath[0])
                sprintf(fhistory,"%s%c%s.mch",cfg->rootpath,pathsep,cfg->session);
        else
                sprintf(fhistory,"%s.mch",cfg->session);
	if(doappend){
           fp=fopen(fhistory,"ab");
	}else{
           fp=fopen(fhistory,"wb");
	}
	if(fp==NULL){
	   mcx_error(-2,"can not save data to disk",__FILE__,__LINE__);
        }
	fwrite(&(cfg->his),sizeof(History),1,fp);
	fwrite(ppath,sizeof(float),count*cfg->his.colcount,fp);
	fclose(fp);
}

void mcx_printlog(Config *cfg, const char *str){
     if(cfg->flog!=NULL){ /*stdout is 1*/
         MCX_FPRINTF(cfg->flog,"%s\n",str);
     }
}

void mcx_normalize(float field[], float scale, int fieldlen, int option){
     int i;
     for(i=0;i<fieldlen;i++){
         if(option==2 && field[i]<0.f)
	     continue;
         field[i]*=scale;
     }
}

void mcx_assess(const int err,const char *msg,const char *file,const int linenum){
     if(!err){
         mcx_error(err,msg,file,linenum);
     }
}

void mcx_error(const int id,const char *msg,const char *file,const int linenum){
     fprintf(stdout,"\nMCXCL ERROR(%d):%s in unit %s:%d\n",id,msg,file,linenum);
#ifdef MCX_CONTAINER
     mcx_throw_exception(id,msg,file,linenum);
#else
     exit(id);
#endif
}

/**
 * @brief Function to recursively create output folder
 *
 * Source: https://stackoverflow.com/questions/2336242/recursive-mkdir-system-call-on-unix
 * @param[in] dir_path: folder name to be created
 * @param[in] mode: mode of the created folder
 */

int mkpath(char* dir_path, int mode){
    char* p=dir_path;
    p[strlen(p)+1]='\0';
    p[strlen(p)]=pathsep;
    for (p=strchr(dir_path+1, pathsep); p; p=strchr(p+1, pathsep)) {
      *p='\0';
#ifdef __MINGW32__
      if (mkdir(dir_path)==-1) {
#else
      if (mkdir(dir_path, mode)==-1) {
#endif
          if (errno!=EEXIST) { *p=pathsep; return -1; }
      }
      *p=pathsep;
    }
    if(dir_path[strlen(p)-1]==pathsep)
        dir_path[strlen(p)-1]='\0';
    return 0;
}

void mcx_createfluence(float **fluence, Config *cfg){
     mcx_clearfluence(fluence);
     *fluence=(float*)calloc(cfg->dim.x*cfg->dim.y*cfg->dim.z,cfg->maxgate*sizeof(float));
}

void mcx_clearfluence(float **fluence){
     if(*fluence) free(*fluence);
}

void mcx_readconfig(char *fname, Config *cfg){
     if(fname[0]==0){
     	mcx_loadconfig(stdin,cfg);
     }else{
        FILE *fp=fopen(fname,"rt");
        if(fp==NULL && fname[0]!='{') mcx_error(-2,"can not load the specified config file",__FILE__,__LINE__);
        if(strstr(fname,".json")!=NULL || fname[0]=='{'){
            char *jbuf;
            int len;
            cJSON *jroot;

            if(fp!=NULL){
                fclose(fp);
                fp=fopen(fname,"rb");
                fseek (fp, 0, SEEK_END);
                len=ftell(fp)+1;
                jbuf=(char *)malloc(len);
                rewind(fp);
                if(fread(jbuf,len-1,1,fp)!=1)
                    mcx_error(-2,"reading input file is terminated",__FILE__,__LINE__);
                jbuf[len-1]='\0';
            }else
		jbuf=fname;
            jroot = cJSON_Parse(jbuf);
            if(jroot){
                mcx_loadjson(jroot,cfg);
                cJSON_Delete(jroot);
            }else{
                char *ptrold=NULL, *ptr=(char*)cJSON_GetErrorPtr();
                if(ptr) ptrold=strstr(jbuf,ptr);
                if(fp!=NULL) fclose(fp);
                if(ptr && ptrold){
                   char *offs=(ptrold-jbuf>=50) ? ptrold-50 : jbuf;
                   while(offs<ptrold){
                      MCX_FPRINTF(stderr,"%c",*offs);
                      offs++;
                   }
                   MCX_FPRINTF(stderr,"<error>%.50s\n",ptrold);
                }
                if(fp!=NULL) free(jbuf);
                mcx_error(-9,"invalid JSON input file",__FILE__,__LINE__);
            }
            if(fp!=NULL) free(jbuf);
        }else{
	    mcx_loadconfig(fp,cfg); 
        }
        if(fp!=NULL) fclose(fp);
	if(cfg->session[0]=='\0'){
	    strncpy(cfg->session,fname,MAX_SESSION_LENGTH);
	}
     }
     if(cfg->rootpath[0]!='\0'){
	struct stat st = {0};
	if (stat((const char *)cfg->rootpath, &st) == -1) {
	    if(mkpath(cfg->rootpath, 0755))
	       mcx_error(-9,"can not create output folder",__FILE__,__LINE__);
	}
     }
}

void mcx_writeconfig(const char *fname, Config *cfg){
     if(fname[0]==0)
     	mcx_saveconfig(stdout,cfg);
     else{
     	FILE *fp=fopen(fname,"wt");
	if(fp==NULL) mcx_error(-2,"can not write to the specified config file",__FILE__,__LINE__);
	mcx_saveconfig(fp,cfg);     
	fclose(fp);
     }
}

void mcx_loadconfig(FILE *in, Config *cfg){
     int i;
     unsigned int gates,itmp;
     float dtmp;
     char filename[MAX_PATH_LENGTH]={0}, strtypestr[MAX_SESSION_LENGTH], comment[MAX_PATH_LENGTH],*comm;
     
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of photons: [1000000]\n\t");
     MCX_ASSERT(fscanf(in,"%d", &(i) )==1); 
     if(cfg->nphoton==0) cfg->nphoton=i;
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the random number generator seed: [1234567]\n\t",cfg->nphoton);
     MCX_ASSERT(fscanf(in,"%d", &(cfg->seed) )==1);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the position of the source: [10 10 5]\n\t",cfg->seed);
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->srcpos.x),&(cfg->srcpos.y),&(cfg->srcpos.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(cfg->issrcfrom0==0 && comm!=NULL && sscanf(comm,"%u",&itmp)==1)
         cfg->issrcfrom0=itmp;

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the normal direction of the source fiber: [0 0 1]\n\t",
                                   cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
     if(!cfg->issrcfrom0){
     	cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
     }
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->srcdir.x),&(cfg->srcdir.y),&(cfg->srcdir.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(comm!=NULL && sscanf(comm,"%f",&dtmp)==1)
         cfg->srcdir.w=dtmp;

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the time gates in seconds (start end and step) [0.0 1e-9 1e-10]\n\t",
                                   cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z);
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->tstart),&(cfg->tend),&(cfg->tstep) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the path to the volume binary file:\n\t",
                                   cfg->tstart,cfg->tend,cfg->tstep);
     if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
         mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);
     }
     gates=(unsigned int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(cfg->maxgate==0)
	 cfg->maxgate=gates;
     else if(cfg->maxgate>gates)
	 cfg->maxgate=gates;

     MCX_ASSERT(fscanf(in,"%s", filename)==1);
     if(cfg->rootpath[0]){
#ifdef WIN32
         sprintf(comment,"%s\\%s",cfg->rootpath,filename);
#else
         sprintf(comment,"%s/%s",cfg->rootpath,filename);
#endif
         strncpy(filename,comment,MAX_PATH_LENGTH);
     }
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t",filename);
     MCX_ASSERT(fscanf(in,"%f %u %u %u", &(cfg->steps.x),&(cfg->dim.x),&(cfg->crop0.x),&(cfg->crop1.x))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %u %u %u\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.x,cfg->dim.x,cfg->crop0.x,cfg->crop1.x);
     MCX_ASSERT(fscanf(in,"%f %u %u %u", &(cfg->steps.y),&(cfg->dim.y),&(cfg->crop0.y),&(cfg->crop1.y))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %u %u %u\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.y,cfg->dim.y,cfg->crop0.y,cfg->crop1.y);
     MCX_ASSERT(fscanf(in,"%f %u %u %u", &(cfg->steps.z),&(cfg->dim.z),&(cfg->crop0.z),&(cfg->crop1.z))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %u %u %u\nPlease specify the total types of media:\n\t",
                                  cfg->steps.z,cfg->dim.z,cfg->crop0.z,cfg->crop1.z);
     MCX_ASSERT(fscanf(in,"%u", &(cfg->medianum))==1);
     cfg->medianum++;
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%d\n",cfg->medianum);
     cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
     cfg->prop[0].mua=0.f; /*property 0 is already air*/
     cfg->prop[0].mus=0.f;
     cfg->prop[0].g=0.f;
     cfg->prop[0].n=1.f;
     for(i=1;i<(int)cfg->medianum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define medium #%d: mus(1/mm), anisotropy, mua(1/mm) and refractive index: [1.01 0.01 0.04 1.37]\n\t",i);
     	MCX_ASSERT(fscanf(in, "%f %f %f %f", &(cfg->prop[i].mus),&(cfg->prop[i].g),&(cfg->prop[i].mua),&(cfg->prop[i].n))==4);
        comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
		fprintf(stdout,"%f %f %f %f\n",cfg->prop[i].mus,cfg->prop[i].g,cfg->prop[i].mua,cfg->prop[i].n);
     }
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of detectors and fiber diameter (in mm):\n\t");
     MCX_ASSERT(fscanf(in,"%u %f", &(cfg->detnum), &(cfg->detradius))==2);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d %f\n",cfg->detnum,cfg->detradius);
     cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
     if(cfg->issavedet && cfg->detnum==0) 
      	cfg->issavedet=0;
     for(i=0;i<(int)cfg->detnum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define detector #%d: x,y,z (in mm): [5 5 5 1]\n\t",i);
     	MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->detpos[i].x),&(cfg->detpos[i].y),&(cfg->detpos[i].z))==3);
	cfg->detpos[i].w=cfg->detradius*cfg->detradius;
        if(!cfg->issrcfrom0){
	   cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	}
        comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
		fprintf(stdout,"%f %f %f\n",cfg->detpos[i].x,cfg->detpos[i].y,cfg->detpos[i].z);
     }
     mcx_prepdomain(filename,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+1+(cfg->issaveexit)*6; /*column count=maxmedia+2*/

     if(in==stdin)
     	fprintf(stdout,"Please specify the source type[pencil|cone|gaussian]:\n\t");
     if(fscanf(in,"%s", strtypestr)==1 && strtypestr[0]){
        int srctype=mcx_keylookup(strtypestr,srctypeid);
	if(srctype==-1)
	   MCX_ERROR(-6,"the specified source type is not supported");
        if(srctype>=0){
           comm=fgets(comment,MAX_PATH_LENGTH,in);
	   cfg->srctype=srctype;
	   if(in==stdin)
     	      fprintf(stdout,"Please specify the source parameters set 1 (4 floating-points):\n\t");
           MCX_ASSERT(fscanf(in, "%f %f %f %f", &(cfg->srcparam1.x),
	          &(cfg->srcparam1.y),&(cfg->srcparam1.z),&(cfg->srcparam1.w))==4);
	   if(in==stdin)
     	      fprintf(stdout,"Please specify the source parameters set 2 (4 floating-points):\n\t");
           MCX_ASSERT(fscanf(in, "%f %f %f %f", &(cfg->srcparam2.x),
	          &(cfg->srcparam2.y),&(cfg->srcparam2.z),&(cfg->srcparam2.w))==4);
           if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcparam1.w*cfg->srcparam2.w>0){
               char patternfile[MAX_PATH_LENGTH];
               FILE *fp;
               if(cfg->srcpattern) free(cfg->srcpattern);
               cfg->srcpattern=(float*)calloc((cfg->srcparam1.w*cfg->srcparam2.w),sizeof(float));
               MCX_ASSERT(fscanf(in, "%s", patternfile)==1);
               fp=fopen(patternfile,"rb");
               if(fp==NULL)
                     MCX_ERROR(-6,"pattern file can not be opened");
               MCX_ASSERT(fread(cfg->srcpattern,cfg->srcparam1.w*cfg->srcparam2.w,sizeof(float),fp)==sizeof(float));
               fclose(fp);
           }else if(cfg->srctype==MCX_SRC_PATTERN3D && cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z>0){
               char patternfile[MAX_PATH_LENGTH];
               FILE *fp;
               if(cfg->srcpattern) free(cfg->srcpattern);
               cfg->srcpattern=(float*)calloc((int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z),sizeof(float));
               MCX_ASSERT(fscanf(in, "%s", patternfile)==1);
               fp=fopen(patternfile,"rb");
               if(fp==NULL)
                     MCX_ERROR(-6,"pattern file can not be opened");
               MCX_ASSERT(fread(cfg->srcpattern,cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z,sizeof(float),fp)==sizeof(float));
               fclose(fp);
           }
	}else
	   return;
     }else
        return;
}

/**
 * @brief Preprocess user input and prepare the volumetric domain for simulation
 *
 * This function preprocess the user input and prepare the domain for the simulation.
 * It loads the media index array from file, add detector masks for easy detection, and
 * check inconsistency between the user specified inputs.
 *
 * @param[in] filename: the name of the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_prepdomain(char *filename, Config *cfg){
     if(filename[0] || cfg->vol){
        if(cfg->vol==NULL){
	     mcx_loadvolume(filename,cfg);
	     if(cfg->shapedata && strstr(cfg->shapedata,":")!=NULL){
	          int status;
     		  Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
        	  if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
		  status=mcx_parse_shapestring(&grid,cfg->shapedata);
		  if(status){
		      MCX_ERROR(status,mcx_last_shapeerror());
		  }
	     }
	}
	if(cfg->isrowmajor){
		/*from here on, the array is always col-major*/
		mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
		cfg->isrowmajor=0;
	}
	if(cfg->issavedet)
		mcx_maskdet(cfg);
	if(cfg->isdumpmask)
	        mcx_dumpmask(cfg);
     }else{
     	mcx_error(-4,"one must specify a binary volume file in order to run the simulation",__FILE__,__LINE__);
     }
/*
     if(cfg->seed==SEED_FROM_FILE && cfg->seedfile[0]){
        if(cfg->respin>1){
	   cfg->respin=1;
	   fprintf(stderr,"Warning: respin is disabled in the replay mode\n");
	}
        mcx_loadseedfile(cfg);
     }
*/  
     if(cfg->medianum){
        for(int i=0;i<cfg->medianum;i++)
             if(cfg->prop[i].mus==0.f)
	         cfg->prop[i].mus=EPS;
     }
     for(int i=0;i<MAX_DEVICE;i++)
        if(cfg->deviceid[i]=='0')
           cfg->deviceid[i]='\0';
}

int mcx_loadjson(cJSON *root, Config *cfg){
     unsigned int i;
     cJSON *Domain, *Optode, *Forward, *Session, *Shapes, *tmp, *subitem;
     char filename[MAX_PATH_LENGTH]={'\0'};
     Domain  = cJSON_GetObjectItem(root,"Domain");
     Optode  = cJSON_GetObjectItem(root,"Optode");
     Session = cJSON_GetObjectItem(root,"Session");
     Forward = cJSON_GetObjectItem(root,"Forward");
     Shapes  = cJSON_GetObjectItem(root,"Shapes");

     if(Domain){
        char volfile[MAX_PATH_LENGTH];
	cJSON *meds,*val;
	val=FIND_JSON_OBJ("VolumeFile","Domain.VolumeFile",Domain);
	if(val){
          strncpy(volfile, val->valuestring, MAX_PATH_LENGTH);
          if(cfg->rootpath[0]){
#ifdef WIN32
           sprintf(filename,"%s\\%s",cfg->rootpath,volfile);
#else
           sprintf(filename,"%s/%s",cfg->rootpath,volfile);
#endif
          }else{
	     strncpy(filename,volfile,MAX_PATH_LENGTH);
	  }
	}
        if(cfg->unitinmm==1.f)
	    cfg->unitinmm=FIND_JSON_KEY("LengthUnit","Domain.LengthUnit",Domain,1.f,valuedouble);
        meds=FIND_JSON_OBJ("Media","Domain.Media",Domain);
        if(meds){
           cJSON *med=meds->child;
           if(med){
             cfg->medianum=cJSON_GetArraySize(meds);
             if(cfg->medianum>MAX_PROP)
                 MCX_ERROR(-4,"input media types exceed the maximum (255)");
             cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
             for(i=0;i<cfg->medianum;i++){
               cJSON *val=FIND_JSON_OBJ("mua",(MCX_ERROR(-1,"You must specify absorption coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mua=val->valuedouble;
	       val=FIND_JSON_OBJ("mus",(MCX_ERROR(-1,"You must specify scattering coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mus=val->valuedouble;
	       val=FIND_JSON_OBJ("g",(MCX_ERROR(-1,"You must specify anisotropy [0-1]"),""),med);
               if(val) cfg->prop[i].g=val->valuedouble;
	       val=FIND_JSON_OBJ("n",(MCX_ERROR(-1,"You must specify refractive index"),""),med);
	       if(val) cfg->prop[i].n=val->valuedouble;

               med=med->next;
               if(med==NULL) break;
             }
	     if(cfg->unitinmm!=1.f){
        	 for(i=0;i<cfg->medianum;i++){
			cfg->prop[i].mus*=cfg->unitinmm;
			cfg->prop[i].mua*=cfg->unitinmm;
        	 }
	     }
           }
        }
	val=FIND_JSON_OBJ("Dim","Domain.Dim",Domain);
	if(val && cJSON_GetArraySize(val)>=3){
	   cfg->dim.x=val->child->valueint;
           cfg->dim.y=val->child->next->valueint;
           cfg->dim.z=val->child->next->next->valueint;
	}else{
	   if(!Shapes)
	      MCX_ERROR(-1,"You must specify the dimension of the volume");
	}
	val=FIND_JSON_OBJ("Step","Domain.Step",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->steps.x=val->child->valuedouble;
               cfg->steps.y=val->child->next->valuedouble;
               cfg->steps.z=val->child->next->next->valuedouble;
           }else{
	       MCX_ERROR(-1,"Domain::Step has incorrect element numbers");
           }
	}
	if(cfg->steps.x!=cfg->steps.y || cfg->steps.y!=cfg->steps.z)
           mcx_error(-9,"MCX currently does not support anisotropic voxels",__FILE__,__LINE__);

	if(cfg->steps.x!=1.f && cfg->unitinmm==1.f)
           cfg->unitinmm=cfg->steps.x;

	if(cfg->unitinmm!=1.f){
           cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
	}
	val=FIND_JSON_OBJ("CacheBoxP0","Domain.CacheBoxP0",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop0.x=val->child->valueint;
               cfg->crop0.y=val->child->next->valueint;
               cfg->crop0.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP0 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("CacheBoxP1","Domain.CacheBoxP1",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop1.x=val->child->valueint;
               cfg->crop1.y=val->child->next->valueint;
               cfg->crop1.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP1 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("OriginType","Domain.OriginType",Domain);
	if(val && cfg->issrcfrom0==0) cfg->issrcfrom0=val->valueint;

	if(cfg->sradius>0.f){
     	   cfg->crop0.x=MAX((uint)(cfg->srcpos.x-cfg->sradius),0);
     	   cfg->crop0.y=MAX((uint)(cfg->srcpos.y-cfg->sradius),0);
     	   cfg->crop0.z=MAX((uint)(cfg->srcpos.z-cfg->sradius),0);
     	   cfg->crop1.x=MIN((uint)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	   cfg->crop1.y=MIN((uint)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	   cfg->crop1.z=MIN((uint)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
	}else if(cfg->sradius==0.f){
     	   memset(&(cfg->crop0),0,sizeof(uint3));
     	   memset(&(cfg->crop1),0,sizeof(uint3));
	}else{
           /*
              if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
           */
           if(!cfg->issrcfrom0){
               cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
               cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
           }
	}
     }
     if(Optode){
        cJSON *dets, *src=FIND_JSON_OBJ("Source","Optode.Source",Optode);
        if(src){
           subitem=FIND_JSON_OBJ("Pos","Optode.Source.Pos",src);
           if(subitem){
              cfg->srcpos.x=subitem->child->valuedouble;
              cfg->srcpos.y=subitem->child->next->valuedouble;
              cfg->srcpos.z=subitem->child->next->next->valuedouble;
           }
           subitem=FIND_JSON_OBJ("Dir","Optode.Source.Dir",src);
           if(subitem){
              cfg->srcdir.x=subitem->child->valuedouble;
              cfg->srcdir.y=subitem->child->next->valuedouble;
              cfg->srcdir.z=subitem->child->next->next->valuedouble;
	      if(subitem->child->next->next->next)
	         cfg->srcdir.w=subitem->child->next->next->next->valuedouble;
           }
	   if(!cfg->issrcfrom0){
              cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
	   }
           cfg->srctype=mcx_keylookup((char*)FIND_JSON_KEY("Type","Optode.Source.Type",src,"pencil",valuestring),srctypeid);
           subitem=FIND_JSON_OBJ("Param1","Optode.Source.Param1",src);
           if(subitem){
              cfg->srcparam1.x=subitem->child->valuedouble;
              cfg->srcparam1.y=subitem->child->next->valuedouble;
              cfg->srcparam1.z=subitem->child->next->next->valuedouble;
              cfg->srcparam1.w=subitem->child->next->next->next->valuedouble;
           }
           subitem=FIND_JSON_OBJ("Param2","Optode.Source.Param2",src);
           if(subitem){
              cfg->srcparam2.x=subitem->child->valuedouble;
              cfg->srcparam2.y=subitem->child->next->valuedouble;
              cfg->srcparam2.z=subitem->child->next->next->valuedouble;
              cfg->srcparam2.w=subitem->child->next->next->next->valuedouble;
           }
           subitem=FIND_JSON_OBJ("Pattern","Optode.Source.Pattern",src);
           if(subitem){
              int nx=FIND_JSON_KEY("Nx","Optode.Source.Pattern.Nx",subitem,0,valueint);
              int ny=FIND_JSON_KEY("Ny","Optode.Source.Pattern.Ny",subitem,0,valueint);
              int nz=FIND_JSON_KEY("Nz","Optode.Source.Pattern.Nz",subitem,1,valueint);
              if(nx>0 && ny>0){
                 cJSON *pat=FIND_JSON_OBJ("Data","Optode.Source.Pattern.Data",subitem);
                 if(pat && pat->child){
                     int i;
                     pat=pat->child;
                     if(cfg->srcpattern) free(cfg->srcpattern);
                     cfg->srcpattern=(float*)calloc(nx*ny*nz,sizeof(float));
                     for(i=0;i<nx*ny*nz;i++){
                         cfg->srcpattern[i]=pat->valuedouble;
                         if((pat=pat->next)==NULL){
                             MCX_ERROR(-1,"Incomplete pattern data");
                         }
                     }
                 }
              }
           }
        }
        dets=FIND_JSON_OBJ("Detector","Optode.Detector",Optode);
        if(dets){
           cJSON *det=dets->child;
           if(det){
             cfg->detnum=cJSON_GetArraySize(dets);
             cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
	     if(cfg->issavedet && cfg->detnum==0) 
      		cfg->issavedet=0;
             for(i=0;i<cfg->detnum;i++){
               cJSON *pos=dets, *rad=NULL;
               rad=FIND_JSON_OBJ("R","Optode.Detector.R",det);
               if(cJSON_GetArraySize(det)==2){
                   pos=FIND_JSON_OBJ("Pos","Optode.Detector.Pos",det);
               }
               if(pos){
	           cfg->detpos[i].x=pos->child->valuedouble;
                   cfg->detpos[i].y=pos->child->next->valuedouble;
	           cfg->detpos[i].z=pos->child->next->next->valuedouble;
               }
               if(rad){
                   cfg->detpos[i].w=rad->valuedouble;
               }
               if(!cfg->issrcfrom0){
		   cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	       }
               det=det->next;
               if(det==NULL) break;
             }
           }
        }
     }
     if(Session){
        char val[1];
	if(cfg->seed==0)      cfg->seed=FIND_JSON_KEY("RNGSeed","Session.RNGSeed",Session,-1,valueint);
        if(cfg->nphoton==0)   cfg->nphoton=FIND_JSON_KEY("Photons","Session.Photons",Session,0,valuedouble);
        if(cfg->session[0]=='\0')  strncpy(cfg->session, FIND_JSON_KEY("ID","Session.ID",Session,"default",valuestring), MAX_SESSION_LENGTH);
        if(cfg->rootpath[0]=='\0') strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath","Session.RootPath",Session,"",valuestring), MAX_PATH_LENGTH);

        if(!cfg->isreflect)   cfg->isreflect=FIND_JSON_KEY("DoMismatch","Session.DoMismatch",Session,cfg->isreflect,valueint);
        if(cfg->issave2pt)    cfg->issave2pt=FIND_JSON_KEY("DoSaveVolume","Session.DoSaveVolume",Session,cfg->issave2pt,valueint);
        if(cfg->isnormalized) cfg->isnormalized=FIND_JSON_KEY("DoNormalize","Session.DoNormalize",Session,cfg->isnormalized,valueint);
        if(!cfg->issavedet)   cfg->issavedet=FIND_JSON_KEY("DoPartialPath","Session.DoPartialPath",Session,cfg->issavedet,valueint);
        if(!cfg->issaveexit)  cfg->issaveexit=FIND_JSON_KEY("DoSaveExit","Session.DoSaveExit",Session,cfg->issaveexit,valueint);
/*
	if(!cfg->issaveseed)  cfg->issaveseed=FIND_JSON_KEY("DoSaveSeed","Session.DoSaveSeed",Session,cfg->issaveseed,valueint);
	cfg->reseedlimit=FIND_JSON_KEY("ReseedLimit","Session.ReseedLimit",Session,cfg->reseedlimit,valueint);
*/
        if(!cfg->outputformat)  cfg->outputformat=mcx_keylookup((char *)FIND_JSON_KEY("OutputFormat","Session.OutputFormat",Session,"mc2",valuestring),outputformat);
        if(cfg->outputformat<0)
                mcx_error(-2,"the specified output format is not recognized",__FILE__,__LINE__);

	strncpy(val,FIND_JSON_KEY("OutputType","Session.OutputType",Session,outputtype+cfg->outputtype,valuestring),1);
	if(mcx_lookupindex(val, outputtype)){
		mcx_error(-2,"the specified output data type is not recognized",__FILE__,__LINE__);
	}
	cfg->outputtype=val[0];
     }
     if(Forward){
        uint gates;
        cfg->tstart=FIND_JSON_KEY("T0","Forward.T0",Forward,0.0,valuedouble);
        cfg->tend  =FIND_JSON_KEY("T1","Forward.T1",Forward,0.0,valuedouble);
        cfg->tstep =FIND_JSON_KEY("Dt","Forward.Dt",Forward,0.0,valuedouble);
	if(cfg->tstart>cfg->tend || cfg->tstep==0.f)
            mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);

        gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
        if(cfg->maxgate==0)
            cfg->maxgate=gates;
        else if(cfg->maxgate>gates)
            cfg->maxgate=gates;
     }
     if(filename[0]=='\0'){
         if(Shapes){
             int status;
             Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
             if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
	     status=mcx_parse_jsonshapes(root, &grid);
	     if(status){
	         MCX_ERROR(status,mcx_last_shapeerror());
	     }
	 }else{
	     MCX_ERROR(-1,"You must either define Domain.VolumeFile, or define a Shapes section");
	 }
     }else if(Shapes){
         MCX_ERROR(-1,"You can not specify both Domain.VolumeFile and Shapes sections");
     }
     mcx_prepdomain(filename,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+1+(cfg->issaveexit)*6; /*column count=maxmedia+2*/
     return 0;
}

void mcx_saveconfig(FILE *out, Config *cfg){
     unsigned int i;

     fprintf(out,"%d\n", (cfg->nphoton) ); 
     fprintf(out,"%d\n", (cfg->seed) );
     fprintf(out,"%f %f %f\n", (cfg->srcpos.x),(cfg->srcpos.y),(cfg->srcpos.z) );
     fprintf(out,"%f %f %f\n", (cfg->srcdir.x),(cfg->srcdir.y),(cfg->srcdir.z) );
     fprintf(out,"%f %f %f\n", (cfg->tstart),(cfg->tend),(cfg->tstep) );
     fprintf(out,"%f %d %d %d\n", (cfg->steps.x),(cfg->dim.x),(cfg->crop0.x),(cfg->crop1.x));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.y),(cfg->dim.y),(cfg->crop0.y),(cfg->crop1.y));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.z),(cfg->dim.z),(cfg->crop0.z),(cfg->crop1.z));
     fprintf(out,"%d", (cfg->medianum));
     for(i=0;i<cfg->medianum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->prop[i].mus),(cfg->prop[i].g),(cfg->prop[i].mua),(cfg->prop[i].n));
     }
     fprintf(out,"%d", (cfg->detnum));
     for(i=0;i<cfg->detnum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->detpos[i].x),(cfg->detpos[i].y),(cfg->detpos[i].z),(cfg->detpos[i].w));
     }
}

void mcx_loadvolume(char *filename,Config *cfg){
     unsigned int i,datalen,res;
     unsigned char *inputvol=NULL;
     FILE *fp;
     
     if(strstr(filename,".json")!=NULL){
         int status;
         Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
	 if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
         status=mcx_load_jsonshapes(&grid,filename);
	 if(status){
	     MCX_ERROR(status,mcx_last_shapeerror());
	 }
	 return;
     }
     fp=fopen(filename,"rb");
     if(fp==NULL){
     	     mcx_error(-5,"the specified binary volume file does not exist",__FILE__,__LINE__);
     }
     if(cfg->vol){
     	     free(cfg->vol);
     	     cfg->vol=NULL;
     }
     datalen=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     cfg->vol=(unsigned int*)malloc(sizeof(unsigned int)*datalen);
     if(cfg->mediabyte==4)
         inputvol=(unsigned char*)(cfg->vol);
     else
         inputvol=(unsigned char*)malloc(sizeof(unsigned char)*cfg->mediabyte*datalen);
     res=fread(inputvol,sizeof(unsigned char)*cfg->mediabyte,datalen,fp);
     fclose(fp);
     if(res!=datalen){
     	 mcx_error(-6,"file size does not match specified dimensions",__FILE__,__LINE__);
     }
     if(cfg->mediabyte==1){  /*convert all format into 4-byte int index*/
       unsigned char *val=inputvol;
       for(i=0;i<datalen;i++)
         cfg->vol[i]=val[i];
     }else if(cfg->mediabyte==2){
       unsigned short *val=(unsigned short *)inputvol;
       for(i=0;i<datalen;i++)
         cfg->vol[i]=val[i];
     }
     for(i=0;i<datalen;i++){
         if(cfg->vol[i]>=cfg->medianum)
            mcx_error(-6,"medium index exceeds the specified medium types",__FILE__,__LINE__);
     }
     if(cfg->mediabyte<4)
         free(inputvol);
}

void  mcx_convertrow2col(unsigned int **vol, uint4 *dim){
     uint x,y,z;
     unsigned int dimxy,dimyz;
     unsigned int *newvol=NULL;
     
     if(*vol==NULL || dim->x==0 || dim->y==0 || dim->z==0){
        return;
     }     
     newvol=(unsigned int*)malloc(sizeof(unsigned int)*dim->x*dim->y*dim->z);
     dimxy=dim->x*dim->y;
     dimyz=dim->y*dim->z;
     for(x=0;x<dim->x;x++)
      for(y=0;y<dim->y;y++)
       for(z=0;z<dim->z;z++){
                newvol[z*dimxy+y*dim->x+x]=*vol[x*dimyz+y*dim->z+z];
       }
     free(*vol);
     *vol=newvol;
}

void  mcx_maskdet(Config *cfg){
     uint d,dx,dy,dz,idx1d,zi,yi,c,count;
     float x,y,z,ix,iy,iz,rx,ry,rz,d2,mind2,d2max;
     unsigned int *padvol;
     const float corners[8][3]={{0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f},
                                {1.f,1.f,0.f},{1.f,0.f,1.f},{0.f,1.f,1.f},{1.f,1.f,1.f}};
     
     dx=cfg->dim.x+2;
     dy=cfg->dim.y+2;
     dz=cfg->dim.z+2;
     
     /*handling boundaries in a volume search is tedious, I first pad vol by a layer of zeros,
       then I don't need to worry about boundaries any more*/

     padvol=(unsigned int*)calloc(dx*dy*sizeof(unsigned int),dz);

     for(zi=1;zi<=cfg->dim.z;zi++)
        for(yi=1;yi<=cfg->dim.y;yi++)
	        memcpy(padvol+zi*dy*dx+yi*dx+1,cfg->vol+(zi-1)*cfg->dim.y*cfg->dim.x+(yi-1)*cfg->dim.x,cfg->dim.x*sizeof(int));

     /**
        The goal here is to find a set of voxels for each 
	detector so that the intersection between a sphere
	of R=cfg->detradius,c0=cfg->detpos[d] and the object 
	surface (or bounding box) is fully covered.
     */
     for(d=0;d<cfg->detnum;d++){                             /*loop over each detector*/
        count=0;
        d2max=(cfg->detpos[d].w+1.7321f)*(cfg->detpos[d].w+1.7321f);
        for(z=-cfg->detpos[d].w-1.f;z<=cfg->detpos[d].w+1.f;z+=0.5f){   /*search in a cube with edge length 2*R+3*/
           iz=z+cfg->detpos[d].z;
           for(y=-cfg->detpos[d].w-1.f;y<=cfg->detpos[d].w+1.f;y+=0.5f){
              iy=y+cfg->detpos[d].y;
              for(x=-cfg->detpos[d].w-1.f;x<=cfg->detpos[d].w+1.f;x+=0.5f){
	         ix=x+cfg->detpos[d].x;

		 if(iz<0||ix<0||iy<0||ix>=cfg->dim.x||iy>=cfg->dim.y||iz>=cfg->dim.z||
		    x*x+y*y+z*z > (cfg->detpos[d].w+1.f)*(cfg->detpos[d].w+1.f))
		     continue;
		 mind2=VERY_BIG;
                 for(c=0;c<8;c++){ /*test each corner of a voxel*/
			rx=(int)ix-cfg->detpos[d].x+corners[c][0];
			ry=(int)iy-cfg->detpos[d].y+corners[c][1];
			rz=(int)iz-cfg->detpos[d].z+corners[c][2];
			d2=rx*rx+ry*ry+rz*rz;
		 	if(d2>d2max){ /*R+sqrt(3) to make sure the circle is fully corvered*/
				mind2=VERY_BIG;
		     		break;
			}
			if(d2<mind2) mind2=d2;
		 }
		 if(mind2==VERY_BIG || mind2>=cfg->detpos[d].w*cfg->detpos[d].w) continue;
		 idx1d=((int)(iz+1.f)*dy*dx+(int)(iy+1.f)*dx+(int)(ix+1.f)); /*1.f comes from the padded layer*/

		 if(padvol[idx1d])  /*looking for a voxel on the interface or bounding box*/
                  if(!(padvol[idx1d+1]&&padvol[idx1d-1]&&padvol[idx1d+dx]&&padvol[idx1d-dx]&&padvol[idx1d+dy*dx]&&padvol[idx1d-dy*dx]&&
		     padvol[idx1d+dx+1]&&padvol[idx1d+dx-1]&&padvol[idx1d-dx+1]&&padvol[idx1d-dx-1]&&
		     padvol[idx1d+dy*dx+1]&&padvol[idx1d+dy*dx-1]&&padvol[idx1d-dy*dx+1]&&padvol[idx1d-dy*dx-1]&&
		     padvol[idx1d+dy*dx+dx]&&padvol[idx1d+dy*dx-dx]&&padvol[idx1d-dy*dx+dx]&&padvol[idx1d-dy*dx-dx]&&
		     padvol[idx1d+dy*dx+dx+1]&&padvol[idx1d+dy*dx+dx-1]&&padvol[idx1d+dy*dx-dx+1]&&padvol[idx1d+dy*dx-dx-1]&&
		     padvol[idx1d-dy*dx+dx+1]&&padvol[idx1d-dy*dx+dx-1]&&padvol[idx1d-dy*dx-dx+1]&&padvol[idx1d-dy*dx-dx-1])){
		          cfg->vol[((int)iz*cfg->dim.y*cfg->dim.x+(int)iy*cfg->dim.x+(int)ix)] |= ((d+1)<<16);/*set the highest bit to 1*/
                          count++;
	          }
	       }
	   }
        }
        if(cfg->issavedet && count==0)
              fprintf(stderr,"MCX WARNING: detector %d is not located on an interface, please check coordinates.\n",d+1);
     }

     free(padvol);
}

void mcx_dumpmask(Config *cfg){
     /**
         To test the results, you should use -M to dump the det-mask, load 
	 it in matlab, and plot the interface containing the detector with
	 pcolor() (has the matching index), and then draw a circle with the
	 radius and center set in the input file. the pixels should completely
	 cover the circle.
     */

     char fname[MAX_PATH_LENGTH];
     if(cfg->rootpath[0])
         sprintf(fname,"%s%c%s_vol",cfg->rootpath,pathsep,cfg->session);
     else
         sprintf(fname,"%s_vol",cfg->session);

     mcx_savenii((float *)cfg->vol, cfg->dim.x*cfg->dim.y*cfg->dim.z, fname, NIFTI_TYPE_UINT32, ofNifti, cfg);
     if(cfg->isdumpmask==1){ /*if dumpmask>1, simulation will also run*/
         MCX_FPRINTF(cfg->flog,"volume mask is saved as uint16 format in %s",fname);
         exit(0);
     }
}

/**
 * @brief Print a progress bar
 *
 * When -D P is specified, this function prints and update a progress bar.
 *
 * @param[in] percent: the percentage value from 1 to 100
 * @param[in] cfg: simulation configuration
 */

void mcx_progressbar(float percent, Config *cfg){
    unsigned int percentage, j,colwidth=79;
    static unsigned int oldmarker=0xFFFFFFFF;

#ifndef MCX_CONTAINER
  #ifdef TIOCGWINSZ
    struct winsize ttys={0,0,0,0};
    ioctl(0, TIOCGWINSZ, &ttys);
    colwidth=ttys.ws_col;
    if(colwidth==0)
         colwidth=79;
  #endif
#endif
    percent=MIN(percent,1.f);

    percentage=percent*(colwidth-18);

    if(percentage != oldmarker){
        if(percent!=-0.f)
	    for(j=0;j<colwidth;j++)     MCX_FPRINTF(stdout,"\b");
        oldmarker=percentage;
        MCX_FPRINTF(stdout,"Progress: [");
        for(j=0;j<percentage;j++)      MCX_FPRINTF(stdout,"=");
        MCX_FPRINTF(stdout,(percentage<colwidth-18) ? ">" : "=");
        for(j=percentage;j<colwidth-18;j++) MCX_FPRINTF(stdout," ");
        MCX_FPRINTF(stdout,"] %3d%%",(int)(percent*100));
#ifdef MCX_CONTAINER
        mcx_matlab_flush();
#else
        fflush(stdout);
#endif
    }
}


int mcx_readarg(int argc, char *argv[], int id, void *output,const char *type){
     /*
         when a binary option is given without a following number (0~1), 
         we assume it is 1
     */
     if(strcmp(type,"char")==0 && (id>=argc-1||(argv[id+1][0]<'0'||argv[id+1][0]>'9'))){
	*((char*)output)=1;
	return id;
     }
     if(id<argc-1){
         if(strcmp(type,"char")==0)
             *((char*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"int")==0)
             *((int*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"float")==0)
             *((float*)output)=atof(argv[id+1]);
	 else if(strcmp(type,"string")==0)
	     strcpy((char *)output,argv[id+1]);
	 else if(strcmp(type,"bytenumlist")==0){
	     char *nexttok,*numlist=(char *)output;
	     int len=0,i;
	     nexttok=strtok(argv[id+1]," ,;");
	     while(nexttok){
    		 numlist[len++]=(char)(atoi(nexttok)); /*device id<256*/
		 for(i=0;i<len-1;i++) /* remove duplicaetd ids */
		    if(numlist[i]==numlist[len-1]){
		       numlist[--len]='\0';
		       break;
		    }
		 nexttok=strtok(NULL," ,;");
		 /*if(len>=MAX_DEVICE) break;*/
	     }
	 }else if(strcmp(type,"floatlist")==0){
	     char *nexttok;
	     float *numlist=(float *)output;
	     int len=0;
	     nexttok=strtok(argv[id+1]," ,;");
	     while(nexttok){
    		 numlist[len++]=atof(nexttok); /*device id<256*/
		 nexttok=strtok(NULL," ,;");
	     }
	 }
     }else{
     	 mcx_error(-1,"incomplete input",__FILE__,__LINE__);
     }
     return id+1;
}
int mcx_remap(char *opt){
    int i=0;
    while(shortopt[i]!='\0'){
	if(strcmp(opt,fullopt[i])==0){
		opt[1]=shortopt[i];
		if(shortopt[i]!='-')
                    opt[2]='\0';
		return 0;
	}
	i++;
    }
    return 1;
}
void mcx_parsecmd(int argc, char* argv[], Config *cfg){
     int i=1,isinteractive=1,issavelog=0;
     char filename[MAX_PATH_LENGTH]={0}, *jsoninput=NULL;
     char logfile[MAX_PATH_LENGTH]={0};
     float np=0.f;

     if(argc<=1){
     	mcx_usage(cfg,argv[0]);
     	exit(0);
     }
     while(i<argc){
     	    if(argv[i][0]=='-'){
		if(argv[i][1]=='-'){
			if(mcx_remap(argv[i])){
				MCX_FPRINTF(cfg->flog,"unknown verbose option %s\n", argv[i]);
				i+=2;
				continue;
			}
		}
	        switch(argv[i][1]){
		     case 'h': 
		                mcx_usage(cfg,argv[0]);
				exit(0);
		     case 'i':
				if(filename[0]){
					mcx_error(-2,"you can not specify both interactive mode and config file",__FILE__,__LINE__);
				}
		     		isinteractive=1;
				break;
		     case 'f': 
		     		isinteractive=0;
				if(argc>i && argv[i+1][0]=='{'){
					jsoninput=argv[i+1];
					i++;
				}else
		     	        	i=mcx_readarg(argc,argv,i,filename,"string");
				break;
		     case 'm':
				mcx_error(-2,"specifying photon move is not supported any more, please use -n",__FILE__,__LINE__);
		     	        i=mcx_readarg(argc,argv,i,&(cfg->nphoton),"int");
		     	        break;
		     case 'n':
		     	        i=mcx_readarg(argc,argv,i,&(np),"float");
				cfg->nphoton=(int)np;
		     	        break;
		     case 't':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->nthread),"int");
		     	        break;
                     case 'T':
                               	i=mcx_readarg(argc,argv,i,&(cfg->nblocksize),"int");
                               	break;
		     case 's':
		     	        i=mcx_readarg(argc,argv,i,cfg->session,"string");
		     	        break;
		     case 'a':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isrowmajor),"char");
		     	        break;
		     case 'g':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxgate),"int");
		     	        break;
		     case 'b':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isreflect),"char");
		     	        break;
                     case 'B':
                                i=mcx_readarg(argc,argv,i,&(cfg->isref3),"char");
                               	break;
		     case 'd':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issavedet),"char");
		     	        break;
		     case 'r':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->respin),"int");
		     	        break;
		     case 'S':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issave2pt),"char");
		     	        break;
		     case 'p':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->printnum),"int");
		     	        break;
                     case 'e':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->minenergy),"float");
                                break;
		     case 'U':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isnormalized),"char");
		     	        break;
                     case 'u':
                                i=mcx_readarg(argc,argv,i,&(cfg->unitinmm),"float");
                                break;
                     case 'R':
                                i=mcx_readarg(argc,argv,i,&(cfg->sradius),"float");
                                break;
                     case 'A':
                                i=mcx_readarg(argc,argv,i,&(cfg->autopilot),"char");
                                break;
                     case 'l':
                                issavelog=1;
                                break;
		     case 'L':  
		                cfg->isgpuinfo=2;
		                break;
		     case 'I':  
		                cfg->isgpuinfo=1;
		                break;
		     case 'c':  
		                cfg->iscpu=1;
		                break;
		     case 'v':  
		                cfg->isverbose=1;
		                break;
		     case 'o':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->optlevel),"int");
		     	        break;
		     case 'k': 
		     	        i=mcx_readarg(argc,argv,i,cfg->kernelfile,"string");
				break;
		     case 'J': 
		     	        cfg->compileropt[strlen(cfg->compileropt)]=' ';
				i=mcx_readarg(argc,argv,i,cfg->compileropt+strlen(cfg->compileropt),"string");
				break;
                     case 'D':
                                if(i+1<argc && isalpha(argv[i+1][0]) )
                                        cfg->debuglevel=mcx_parsedebugopt(argv[++i],debugflag);
                                else
                                        i=mcx_readarg(argc,argv,i,&(cfg->debuglevel),"int");
                                break;
		     case 'F':
                                if(i>=argc)
                                        mcx_error(-1,"incomplete input",__FILE__,__LINE__);
                                if((cfg->outputformat=mcx_keylookup(argv[++i], outputformat))<0)
                                        mcx_error(-2,"the specified output data type is not recognized",__FILE__,__LINE__);
                                break;
		     case 'x':
 		                i=mcx_readarg(argc,argv,i,&(cfg->issaveexit),"char");
 				if (cfg->issaveexit) cfg->issavedet=1;
 				break;
		     case 'X':
 		                i=mcx_readarg(argc,argv,i,&(cfg->issaveref),"char");
 				if (cfg->issaveref) cfg->issaveref=1;
 				break;
                     case 'G':
                                if(mcx_isbinstr(argv[i+1])){
                                    i=mcx_readarg(argc,argv,i,cfg->deviceid,"string");
                                    break;
                                }else{
				    int gpuid;
                                    i=mcx_readarg(argc,argv,i,&gpuid,"int");
                                    memset(cfg->deviceid,'0',MAX_DEVICE);
                                    if(gpuid<MAX_DEVICE)
                                         cfg->deviceid[gpuid-1]='1';
                                    else
                                         mcx_error(-2,"GPU id can not be more than 256",__FILE__,__LINE__);
                                    break;
                                }
                     case 'W':
			        i=mcx_readarg(argc,argv,i,cfg->workload,"floatlist");
                                break;
                     case 'z':
                                i=mcx_readarg(argc,argv,i,&(cfg->issrcfrom0),"char");
                                break;
		     case 'M':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isdumpmask),"char");
		     	        break;
                     case 'P':
                                cfg->shapedata=argv[++i];
                                break;
                     case 'E':
				i=mcx_readarg(argc,argv,i,&(cfg->seed),"int");
		     	        break;
		     case 'H':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxdetphoton),"int");
		     	        break;
		     case '-':  /*additional verbose parameters*/
                                if(strcmp(argv[i]+2,"devicelist")==0)
                                     i=mcx_readarg(argc,argv,i,cfg->deviceid,"string");
                                else if(strcmp(argv[i]+2,"mediabyte")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->mediabyte),"int");
                                else if(strcmp(argv[i]+2,"root")==0)
                                     i=mcx_readarg(argc,argv,i,cfg->rootpath,"string");
                                else if(strcmp(argv[i]+2,"atomic")==0)
		                     i=mcx_readarg(argc,argv,i,&(cfg->isatomic),"int");
                                else
                                     MCX_FPRINTF(cfg->flog,"unknown verbose option: --%s\n",argv[i]+2);
		     	        break;
		}
	    }
	    i++;
     }
     if(issavelog && cfg->session[0]){
          sprintf(logfile,"%s.log",cfg->session);
          cfg->flog=fopen(logfile,"wt");
          if(cfg->flog==NULL){
		cfg->flog=stdout;
		MCX_FPRINTF(cfg->flog,"unable to save to log file, will print from stdout\n");
          }
     }
     if(cfg->kernelfile[0]!='\0' && cfg->isgpuinfo!=2){
     	  FILE *fp=fopen(cfg->kernelfile,"rb");
	  int srclen;
	  if(fp==NULL){
	  	mcx_error(-10,"the specified OpenCL kernel file does not exist!",__FILE__,__LINE__);
	  }
	  fseek(fp,0,SEEK_END);
	  srclen=ftell(fp);
	  if(cfg->clsource!=(char *)mcx_core_cl)
	      free(cfg->clsource);
	  cfg->clsource=(char *)malloc(srclen+1);
	  fseek(fp,0,SEEK_SET);
	  MCX_ASSERT((fread(cfg->clsource,srclen,1,fp)==1));
	  cfg->clsource[srclen]='\0';
	  fclose(fp);
     }
     if(cfg->isgpuinfo!=2){ /*print gpu info only*/
	  if(isinteractive){
             mcx_readconfig((char*)"",cfg);
	  }else if(jsoninput){
     	     mcx_readconfig(jsoninput,cfg);
	  }else{
             mcx_readconfig(filename,cfg);
          }
     }
}

int mcx_parsedebugopt(char *debugopt,const char *debugflag){
    char *c=debugopt,*p;
    int debuglevel=0;

    while(*c){
       p=(char *)strchr(debugflag, ((*c<='z' && *c>='a') ? *c-'a'+'A' : *c) );
       if(p!=NULL)
          debuglevel |= (1 << (p-debugflag));
       c++;
    }
    return debuglevel;
}

int mcx_keylookup(char *origkey, const char *table[]){
    int i=0;
    char *key=(char *)malloc(strlen(origkey)+1);
    memcpy(key,origkey,strlen(origkey)+1);
    while(key[i]){
        key[i]=tolower(key[i]);
	i++;
    }
    i=0;
    while(table[i]!='\0'){
	if(strcmp(key,table[i])==0){
		return i;
	}
	i++;
    }
    free(key);
    return -1;
}

int mcx_lookupindex(char *key, const char *index){
    int i=0;
    while(index[i]!='\0'){
        if(tolower(*key)==index[i]){
                *key=i;
                return 0;
        }
        i++;
    }
    return 1;
}

void mcx_version(Config *cfg){
    const char ver[]="$Rev::4fdc45$";
    int v=0;
    sscanf(ver,"$Rev::%d",&v);
    MCX_FPRINTF(cfg->flog, "MCXCL Revision %d\n",v);
    exit(0);
}

int mcx_isbinstr(const char * str){
    int i, len=strlen(str);
    if(len==0)
        return 0;
    for(i=0;i<len;i++)
        if(str[i]!='0' && str[i]!='1')
	   return 0;
    return 1;
}


/**
 * @brief Force flush the command line to print the message
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_flush(Config *cfg){
#ifdef MCX_CONTAINER
    mcx_matlab_flush();
#else
    fflush(cfg->flog);
#endif
}

void mcx_printheader(Config *cfg){
    MCX_FPRINTF(cfg->flog,"\
==============================================================================\n\
=                       Monte Carlo eXtreme (MCX) -- OpenCL                  =\n\
=          Copyright (c) 2010-2018 Qianqian Fang <q.fang at neu.edu>         =\n\
=                             http://mcx.space/                              =\n\
=                                                                            =\n\
= Computational Optics&Translational Imaging (COTI) Lab - http://fanglab.org =\n\
=            Department of Bioengineering, Northeastern University           =\n\
==============================================================================\n\
=    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365     =\n\
==============================================================================\n\
$Rev::4fdc45 $ Last $Date::2018-03-29 00:35:53 -04$ by $Author::Qianqian Fang$\n\
==============================================================================\n");
}

void mcx_usage(Config *cfg,char *exename){
     mcx_printheader(cfg);
     printf("\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first value in [*|*] is the default)\n\
\n\
== Required option ==\n\
 -f config     (--input)       read an input file in .json or .inp format\n\
\n\
== MC options ==\n\
\n\
 -n [0|int]    (--photon)      total photon number (exponential form accepted)\n\
 -r [1|int]    (--repeat)      divide photons into r groups (1 per GPU call)\n\
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit\n\
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge\n\
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw\n\
 -E [0|int]    (--seed)        set random-number-generator seed, -1 to generate\n\
 -z [0|1]      (--srcfrom0)    1 volume origin is [0 0 0]; 0: origin at [1 1 1]\n\
 -k [1|0]      (--voidtime)    when src is outside, 1 enables timer inside void\n\
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid\n\
 -e [0.|float] (--minenergy)   minimum energy level to terminate a photon\n\
 -g [1|int]    (--gategroup)   number of time gates per run\n\
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array\n\
\n\
== GPU options ==\n\
 -L            (--listgpu)     print GPU information only\n\
 -t [16384|int](--thread)      total thread number\n\
 -T [64|int]   (--blocksize)   thread number per block\n\
 -A [0|int]    (--autopilot)   auto thread config:1 enable;0 disable\n\
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto\n\
      or\n\
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)\n\
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum\n\
 -I            (--printgpu)    print GPU information and run program\n\
 -o [3|int]    (--optlevel)    optimization level 0-no opt;1,2,3 more optimized\n\
 -J '-D MCX'   (--compileropt) specify additional JIT compiler options\n\
 -k my_simu.cl (--kernel)      user specified OpenCL kernel source file\n\
\n\
== Output options ==\n\
 -s sessionid  (--session)     a string to label all output file names\n\
 -d [1|0]      (--savedet)     1 to save photon info at detectors; 0 not save\n\
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions\n\
                               setting -x to 1 also implies setting '-d' to 1\n\
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels\n\
                               right outside of the domain; if non-zero voxels\n\
			       appear at the boundary, pad 0s before using -X\n\
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save\n\
 -H [1000000] (--maxdetphoton) max number of detected photons\n\
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save\n\
 -F [mc2|...] (--outputformat) fluence data output format:\n\
                               mc2 - MCX mc2 format (binary 32bit float)\n\
                               nii - Nifti format\n\
                               hdr - Analyze 7.5 hdr/img format\n\
 -O [X|XFEJP]  (--outputtype)  X - output flux, F - fluence, E - energy deposit\n\
                               J - Jacobian (replay mode),   P - scattering\n\
                               event counts at each voxel (replay mode only)\n\
\n\
== User IO options ==\n\
 -h            (--help)        print this message\n\
 -v            (--version)     print MCX revision number\n\
 -l            (--log)         print messages to a log file instead\n\
 -i 	       (--interactive) interactive mode\n\
\n\
== Debug options ==\n\
 -D [0|int]    (--debug)       print debug information (you can use an integer\n\
  or                           or a string by combining the following flags)\n\
 -D [''|RMP]                   4 P  print progress bar\n\
      combine multiple items by using a string, or add selected numbers together\n\
\n\
== Additional options ==\n\
 --atomic       [1|0]          1: use atomic operations; 0: do not use atomics\n\
 --root         [''|string]    full path to the folder storing the input files\n\
 --maxvoidstep  [1000|int]     maximum distance (in voxel unit) of a photon that\n\
                               can travel before entering the domain, if \n\
                               launched outside (i.e. a widefield source)\n\
\n\
== Example ==\n\
example: (autopilot mode)\n\
  %s -A -n 1e7 -f input.inp -G 1 \n\
or (manual mode)\n\
  %s -t 16384 -T 64 -n 1e7 -f input.inp -s test -r 1 -b 0 -G 1010 -W '50,50'\n",exename,exename,exename);
}
