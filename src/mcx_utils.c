/*******************************************************************************
**
**  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
**      -- OpenCL edition
**  Author: Qianqian Fang <q.fang at neu.edu>
**
**  Reference (Fang2009):
**        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
**        Migration in 3D Turbid Media Accelerated by Graphics Processing 
**        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
**
**  mcx_utils.c: configuration and command line option processing unit
**
**  Unpublished work, see LICENSE.txt for details
**
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "mcx_utils.h"
#include "mcx_shapes.h"
#include "mcx_const.h"


#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? fallback : tmp->val) \
                     : tmp->val)

#define FIND_JSON_OBJ(id,idfull,parent) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? NULL : tmp) \
                     : tmp)


char shortopt[]={'h','i','f','n','m','t','T','s','a','g','b','B','D','-','G','W','z',
                 'd','r','S','p','e','U','R','l','L','M','I','o','c','k','v','J',
                 'A','P','E','F','H','-','u','\0'};
const char *fullopt[]={"--help","--interactive","--input","--photon","--move",
                 "--thread","--blocksize","--session","--array","--gategroup",
                 "--reflect","--reflect3","--device","--devicelist","--gpu","--workload","--srcfrom0",
		 "--savedet","--repeat","--save2pt","--printlen","--minenergy",
                 "--normalize","--skipradius","--log","--listgpu","--dumpmask",
                 "--printgpu","--root","--cpu","--kernel","--verbose","--compileropt",
                 "--autopilot","--shapes","--seed","--outputformat","--maxdetphoton",
		 "--mediabyte","--unitinmm",""};
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
     cfg->maxgate=1;
     cfg->isreflect=1;
     cfg->isref3=0;
     cfg->isnormalized=1;
     cfg->issavedet=0;
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
     cfg->clsource=NULL;;
     cfg->maxdetphoton=1000000; 
     cfg->isdumpmask=0;
     cfg->autopilot=0;
     cfg->shapedata=NULL;

     memset(cfg->deviceid,0,MAX_DEVICE);
     memset(cfg->compileropt,0,MAX_PATH_LENGTH);
     memset(cfg->workload,0,MAX_DEVICE*sizeof(float));
     cfg->deviceid[0]='1'; /*use the first GPU device by default*/
     strcpy(cfg->kernelfile,"mcx_core.cl");
     cfg->issrcfrom0=0;

     cfg->exportfield=NULL;
     cfg->exportdetected=NULL;
     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     cfg->runtime=0;

     memset(&cfg->his,0,sizeof(History));
     memcpy(cfg->his.magic,"MCXH",4);
     cfg->his.version=1;
     cfg->his.unitinmm=1.f;
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
}

void mcx_clearcfg(Config *cfg){
     if(cfg->medianum)
     	free(cfg->prop);
     if(cfg->detnum)
     	free(cfg->detpos);
     if(cfg->dim.x && cfg->dim.y && cfg->dim.z)
        free(cfg->vol);
     if(cfg->clsource)
        free(cfg->clsource);
     if(cfg->exportfield)
        free(cfg->exportfield);
     if(cfg->exportdetected)
        free(cfg->exportdetected);
     if(cfg->seeddata)
        free(cfg->seeddata);

     mcx_initcfg(cfg);
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
         fprintf(cfg->flog,"%s\n",str);
     }
}

void mcx_normalize(float field[], float scale, int fieldlen){
     int i;
     for(i=0;i<fieldlen;i++){
         field[i]*=scale;
     }
}

void mcx_assess(const int err,const char *msg,const char *file,const int linenum){
     if(!err){
         mcx_error(err,msg,file,linenum);
     }
}

void mcx_error(const int id,const char *msg,const char *file,const int linenum){
     fprintf(stdout,"\nMCX ERROR(%d):%s in unit %s:%d\n",id,msg,file,linenum);
#ifdef MCX_CONTAINER
     mcx_throw_exception(id,msg,file,linenum);
#else
     exit(id);
#endif
}

void mcx_createfluence(float **fluence, Config *cfg){
     mcx_clearfluence(fluence);
     *fluence=(float*)calloc(cfg->dim.x*cfg->dim.y*cfg->dim.z,cfg->maxgate*sizeof(float));
}

void mcx_clearfluence(float **fluence){
     if(*fluence) free(*fluence);
}

void mcx_readconfig(const char *fname, Config *cfg){
     if(fname[0]==0){
     	mcx_loadconfig(stdin,cfg);
        if(cfg->session[0]=='\0'){
		strcpy(cfg->session,"default");
	}
     }
     else{
     	FILE *fp=fopen(fname,"rt");
	if(fp==NULL) mcx_error(-2,"can not load the specified config file",__FILE__,__LINE__);
        if(strstr(fname,".json")!=NULL){
            char *jbuf;
            int len;
            cJSON *jroot;

            fclose(fp);
            fp=fopen(fname,"rb");
            fseek (fp, 0, SEEK_END);
            len=ftell(fp)+1;
            jbuf=(char *)malloc(len);
            rewind(fp);
            if(fread(jbuf,len-1,1,fp)!=1)
                mcx_error(-2,"reading input file is terminated",__FILE__,__LINE__);
            jbuf[len-1]='\0';
            jroot = cJSON_Parse(jbuf);
            if(jroot){
                mcx_loadjson(jroot,cfg);
                cJSON_Delete(jroot);
            }else{
                char *ptrold, *ptr=(char*)cJSON_GetErrorPtr();
                if(ptr) ptrold=strstr(jbuf,ptr);
                fclose(fp);
                if(ptr && ptrold){
                   char *offs=(ptrold-jbuf>=50) ? ptrold-50 : jbuf;
                   while(offs<ptrold){
                      fprintf(stderr,"%c",*offs);
                      offs++;
                   }
                   fprintf(stderr,"<error>%.50s\n",ptrold);
                }
                free(jbuf);
                mcx_error(-9,"invalid JSON input file",__FILE__,__LINE__);
            }
            free(jbuf);
        }else{
	    mcx_loadconfig(fp,cfg); 
        }
	fclose(fp);
        if(cfg->session[0]=='\0'){
		strcpy(cfg->session,fname);
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

void mcx_prepdomain(char *filename, Config *cfg){
     int i;
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
     for(i=0;i<MAX_DEVICE;i++)
        if(cfg->deviceid[i]=='0')
           cfg->deviceid[i]='\0';
}

void mcx_loadconfig(FILE *in, Config *cfg){
     int i,idx1d;
     unsigned int gates,itmp;
     char filename[MAX_PATH_LENGTH]={0}, comment[MAX_PATH_LENGTH],*comm;
     
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
     if(cfg->maxgate>gates)
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
     if(filename[0]){
        mcx_loadvolume(filename,cfg);
        if(cfg->isrowmajor){
                /*from here on, the array is always col-major*/
                mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
                cfg->isrowmajor=0;
        }
	if(cfg->issavedet)
		mcx_maskdet(cfg);
        if(cfg->srcpos.x<0.f || cfg->srcpos.y<0.f || cfg->srcpos.z<0.f ||
            cfg->srcpos.x>=cfg->dim.x || cfg->srcpos.y>=cfg->dim.y || cfg->srcpos.z>=cfg->dim.z)
                mcx_error(-4,"source position is outside of the volume",__FILE__,__LINE__);

	idx1d=((int)floor(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+(int)floor(cfg->srcpos.y)*cfg->dim.x+(int)floor(cfg->srcpos.x));

        /* if the specified source position is outside the domain, move the source
	   along the initial vector until it hit the domain */
	if(cfg->vol && cfg->vol[idx1d]==0){
                printf("source (%f %f %f) is located outside the domain, vol[%d]=%d\n",
		      cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,idx1d,cfg->vol[idx1d]);
		while(cfg->vol[idx1d]==0){
			cfg->srcpos.x+=cfg->srcdir.x;
			cfg->srcpos.y+=cfg->srcdir.y;
			cfg->srcpos.z+=cfg->srcdir.z;
			printf("fixing source position to (%f %f %f)\n",cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
			idx1d=cfg->isrowmajor?(int)(floor(cfg->srcpos.x)*cfg->dim.y*cfg->dim.z+floor(cfg->srcpos.y)*cfg->dim.z+floor(cfg->srcpos.z)):\
                		      (int)(floor(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+floor(cfg->srcpos.y)*cfg->dim.x+floor(cfg->srcpos.x));
		}
	}
        cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
        cfg->his.detnum=cfg->detnum;
        cfg->his.colcount=cfg->medianum+1; /*column count=maxmedia+2*/
     }else{
     	mcx_error(-4,"one must specify a binary volume file in order to run the simulation",__FILE__,__LINE__);
     }
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
/*
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
              if(nx>0 || ny>0){
                 cJSON *pat=FIND_JSON_OBJ("Data","Optode.Source.Pattern.Data",subitem);
                 if(pat && pat->child){
                     int i;
                     pat=pat->child;
                     if(cfg->srcpattern) free(cfg->srcpattern);
                     cfg->srcpattern=(float*)calloc(nx*ny,sizeof(float));
                     for(i=0;i<nx*ny;i++){
                         cfg->srcpattern[i]=pat->valuedouble;
                         if((pat=pat->next)==NULL){
                             MCX_ERROR(-1,"Incomplete pattern data");
                         }
                     }
                 }
              }
           }
*/
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
/*
	if(!cfg->issaveseed)  cfg->issaveseed=FIND_JSON_KEY("DoSaveSeed","Session.DoSaveSeed",Session,cfg->issaveseed,valueint);
	cfg->reseedlimit=FIND_JSON_KEY("ReseedLimit","Session.ReseedLimit",Session,cfg->reseedlimit,valueint);
*/
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
     cfg->his.colcount=cfg->medianum+1; /*column count=maxmedia+2*/
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
     if(cfg->isdumpmask==1) /*if dumpmask>1, simulation will also run*/
         exit(0);
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
     char filename[MAX_PATH_LENGTH]={0};
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
				fprintf(cfg->flog,"unknown verbose option %s\n", argv[i]);
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
		     	        i=mcx_readarg(argc,argv,i,cfg->rootpath,"string");
		     	        break;
		     case 'k': 
		     	        i=mcx_readarg(argc,argv,i,cfg->kernelfile,"string");
				break;
		     case 'J': 
		     	        cfg->compileropt[strlen(cfg->compileropt)]=' ';
				i=mcx_readarg(argc,argv,i,cfg->compileropt+strlen(cfg->compileropt),"string");
				break;
                     case 'D':
			        i=mcx_readarg(argc,argv,i,cfg->deviceid,"bytenumlist");
                                break;
                     case 'G':
                                i=mcx_readarg(argc,argv,i,cfg->deviceid,"string");
                                break;
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
                                else
                                     fprintf(cfg->flog,"unknown verbose option: --%s\n",argv[i]+2);
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
		fprintf(cfg->flog,"unable to save to log file, will print from stdout\n");
          }
     }
     if(cfg->clsource==NULL && cfg->isgpuinfo!=2){
     	  FILE *fp=fopen(cfg->kernelfile,"rb");
	  int srclen;
	  if(fp==NULL){
	  	mcx_error(-10,"the specified OpenCL kernel file does not exist!",__FILE__,__LINE__);
	  }
	  fseek(fp,0,SEEK_END);
	  srclen=ftell(fp);
	  cfg->clsource=(char *)malloc(srclen+1);
	  fseek(fp,0,SEEK_SET);
	  MCX_ASSERT((fread(cfg->clsource,srclen,1,fp)==1));
	  cfg->clsource[srclen]='\0';
	  fclose(fp);
     }
     if(cfg->isgpuinfo!=2){ /*print gpu info only*/
       if(isinteractive){
          mcx_readconfig("",cfg);
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

void mcx_printheader(Config *cfg){
    fprintf(cfg->flog,"\
==============================================================================\n\
=                      Monte Carlo eXtreme (MCX) -- OpenCL                   =\n\
=            Copyright (c) 2010-2017 Qianqian Fang <q.fang at neu.edu>       =\n\
=                                 http://mcx.space/                          =\n\
=                                                                            =\n\
= Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.or =\n\
=               Department of Bioengineering, Northeastern University        =\n\
==============================================================================\n\
=        The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365 =\n\
==============================================================================\n\
$Rev::6e839e $ Last $Date::2017-07-20 12:46:23 -04$ by $Author::Qianqian Fang$\n\
==============================================================================\n");
}

void mcx_usage(Config *cfg,char *exename){
     mcx_printheader(cfg);     
     printf("\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first item in [] is the default value)\n\
 -i 	        (--interactive) interactive mode\n\
 -f config      (--input)	read config from a file\n\
 -A [0|int]     (--autopilot)   auto thread config:1 dedicated GPU;2 non-dedica.\n\
 -t [16384|int] (--thread)	total thread number\n\
 -T [64|int]    (--blocksize)	thread number per block\n\
 -n [0|int]     (--photon)	total photon number\n\
 -r [1|int]     (--repeat)	number of repeations\n\
 -a [0|1]       (--array)	0 for Matlab array, 1 for C array\n\
 -u [1.|float]  (--unitinmm)    defines the length unit for the grid edge\n\
 -z [0|1]       (--srcfrom0)    src/detector coordinates start from 0, otherwise from 1\n\
 -g [1|int]     (--gategroup)	number of time gates per run\n\
 -b [1|0]       (--reflect)	1 to reflect the photons at the boundary, 0 to exit\n\
 -B [0|1]       (--reflect3)	1 to consider maximum 3 reflections, 0 consider only 2\n\
 -E [0|int]     (--seed)        set random-number-generator seed, -1 to generate\n\
 -e [0.|float]  (--minenergy)	minimum energy level to propagate a photon\n\
 -R [0.|float]  (--skipradius)  minimum distance to source to start accumulation\n\
 -U [1|0]       (--normalize)	1 to normailze the fluence to unitary, 0 to save raw fluence\n\
 -d [0|1]       (--savedet)	1 to save photon info at detectors, 0 not to save\n\
 -S [1|0]       (--save2pt)	1 to save the fluence field, 0 do not save\n\
 -s sessionid   (--session)	a string to identify this specific simulation (and output files)\n\
 -p [0|int]     (--printlen)	number of threads to print (debug)\n\
 -h             (--help)	print this message\n\
 -l             (--log) 	print messages to a log file instead\n\
 -L             (--listgpu)	print GPU information only\n\
 -I             (--printgpu)	print GPU information and run program\n\
 -c             (--cpu) 	use CPU as the platform for OpenCL backend\n\
 -k mcx_core.cl (--kernel)      specify path to OpenCL kernel source file\n\
 -G '0111'      (--gpu)         specify the active OpenCL devices (1 enable, 0 disable)\n\
 -W '50,30,20'  (--workload)    specify relative workload for each device; total is the sum\n\
 -J '-D MCX'    (--compileropt) specify additional JIT compiler options\n\
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid\n\
 -F [mc2|...] (--outputformat) fluence data output format:\n\
                               mc2 - MCX mc2 format (binary 32bit float)\n\
                               nii - Nifti format\n\
                               hdr - Analyze 7.5 hdr/img format\n\
 -H [1000000] (--maxdetphoton) max number of detected photons\n\
example: (autopilot mode)\n\
  %s -A -n 1e7 -f input.inp -G 1 \n\
or (manual mode)\n\
  %s -t 16384 -T 64 -n 1e7 -f input.inp -s test -r 1 -b 0 -G 1010 -W '50,50' -k ../../src/mcx_core.cl\n",exename,exename,exename);
}
