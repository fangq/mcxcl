////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
//      -- OpenCL edition
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//
//  Reference (Fang2009):
//        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
//        Migration in 3D Turbid Media Accelerated by Graphics Processing 
//        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
//
//  mcx_core.cl: OpenCL kernels
//
//  Unpublished work, see LICENSE.txt for details
//
////////////////////////////////////////////////////////////////////////////////

#ifdef MCX_SAVE_DETECTORS
  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef MCX_GPU_DEBUG
  #define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
  //#pragma OPENCL EXTENSION cl_amd_printf : enable
#else
  #define GPUDEBUG(x)
#endif

#define R_PI               0.318309886183791f
#define RAND_MAX           4294967295

#define ONE_PI             3.1415926535897932f     //pi
#define TWO_PI             6.28318530717959f       //2*pi

#define C0                 299792458000.f          //speed of light in mm/s
#define R_C0               3.335640951981520e-12f  //1/C0 in s/mm

#define EPS                FLT_EPSILON             //round-off limit
#define VERY_BIG           (1.f/FLT_EPSILON)       //a big number
#define JUST_ABOVE_ONE     1.0001f                 //test for boundary
#define SAME_VOXEL         -9999.f                 //scatter within a voxel
#define NO_LAUNCH          9999                    //when fail to launch, for debug
#define MAX_PROP           256                     //maximum property number

#define DET_MASK           0x80
#define MED_MASK           0x7F
#define NULL               0

typedef struct KernelParams {
  float4 ps,c0;
  float4 maxidx;
  uint4  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minstep;
  float  twin0,twin1,tmax;
  float  oneoverc0;
  unsigned int isrowmajor,save2pt,doreflect,dorefint,savedet;
  float  Rtstep;
  float  minenergy;
  float  skipradius2;
  float  minaccumtime;
  unsigned int maxdetphoton;
  unsigned int maxmedia;
  unsigned int detnum;
  unsigned int idx1dorig;
  unsigned int mediaidorig;
} MCXParam __attribute__ ((aligned (16)));

#ifndef USE_POSIX_RAND

#define RAND_BUF_LEN       5        //register arrays
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define INIT_LOGISTIC      100

#ifndef DOUBLE_PREC_LOGISTIC
  typedef float RandType;
  #define FUN(x)               (4.f*(x)*(1.f-(x)))
  #define FUN4(x)              ((float4)4.f*(x)*((float4)1.f-(x)))
  #define NU 1e-7f
  #define NU2 (1.f-2.f*NU)
  #define MIN_INVERSE_LIMIT 1e-7f
  #define logistic_uniform(v)  (acos(1.f-2.f*(v))*R_PI)
  #define R_MAX_C_RAND       (1.f/RAND_MAX)
  #define LOG_MT_MAX         22.1807097779182f
#else
  typedef double RandType;
  #define FUN(x)               (4.0*(x)*(1.0-(x)))
  #define FUN4(x)              ((double4)4.f*(x)*((double4)1.f-(x)))
  #define NU 1e-14
  #define NU2 (1.0-2.0*NU)
  #define MIN_INVERSE_LIMIT 1e-12
  #define logistic_uniform(v)  (acos(1.0-2.0*(v))*R_PI)
  #define R_MAX_C_RAND       (1./RAND_MAX)
  #define LOG_MT_MAX         22.1807097779182
#endif

#define RING_FUN(x,y,z)      (NU2*(x)+NU*((y)+(z)))

void logistic_step(__private RandType *t, __private RandType *tnew, int len_1){
    t[0]=FUN(t[0]);
    t[1]=FUN(t[1]);
    t[2]=FUN(t[2]);
    t[3]=FUN(t[3]);
    t[4]=FUN(t[4]);
    tnew[4]=RING_FUN(t[0],t[4],t[1]);   /* shuffle the results by separation of 2*/
    tnew[0]=RING_FUN(t[1],t[0],t[2]);
    tnew[1]=RING_FUN(t[2],t[1],t[3]);
    tnew[2]=RING_FUN(t[3],t[2],t[4]);
    tnew[3]=RING_FUN(t[4],t[3],t[0]);
}
// generate random number for the next zenith angle
void rand_need_more(__private RandType t[RAND_BUF_LEN]){
    RandType tnew[RAND_BUF_LEN]={0.f};
    logistic_step(t,tnew,RAND_BUF_LEN-1);
    logistic_step(tnew,t,RAND_BUF_LEN-1);
}

void logistic_init(__private RandType *t,__global uint seed[],uint idx){
     int i;
     for(i=0;i<RAND_BUF_LEN;i++)
           t[i]=(RandType)seed[idx*RAND_BUF_LEN+i]*R_MAX_C_RAND;

     for(i=0;i<INIT_LOGISTIC;i++)  /*initial randomization*/
           rand_need_more(t);
}
// transform into [0,1] random number
RandType rand_uniform01(__private RandType t[RAND_BUF_LEN]){
    rand_need_more(t);
    return logistic_uniform(t[0]);
}
void gpu_rng_init(__private RandType t[RAND_BUF_LEN],__global uint *n_seed,int idx){
    logistic_init(t,n_seed,idx);
}

// generate [0,1] random number for the next scattering length
float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]){
    RandType ran=rand_uniform01(t);
    if(ran==0.f) ran=logistic_uniform(t[1]);
    return ((ran==0.f)?LOG_MT_MAX:(-log(ran)));
}


#else

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define RAND_BUF_LEN       4        //register arrays
#define RAND_SEED_LEN      4        //48 bit packed with 64bit length
#define LOG_MT_MAX         22.1807097779182f
#define LCG_MULTIPLIER     0x5deece66dul
#define LCG_INCREMENT      0xb
#define IEEE754_DOUBLE_BIAS     0x3ff /* Added to exponent.  */

typedef unsigned short     RandType;

int __drand48_iterate (__private RandType t[RAND_BUF_LEN]){
  ulong X;
  ulong result;

  X = (ulong) t[2] << 32 | (uint) t[1] << 16 | t[0];

  result = X * LCG_MULTIPLIER + LCG_INCREMENT;

  t[0] = result & 0xffff;
  t[1] = (result >> 16) & 0xffff;
  t[2] = (result >> 32) & 0xffff;

  return 0;
}

float __erand48_r (__private RandType t[RAND_BUF_LEN]){
  union{
    double d;
    ulong  i;
  } temp;

  temp.i=0ul;
  /* Compute next state.  */
  __drand48_iterate (t);

  /* Construct a positive double with the 48 random bits distributed over
     its fractional part so the resulting FP number is [0.0,1.0).  */
  temp.i |= ((ulong)((IEEE754_DOUBLE_BIAS <<4) | (t[0] & 0xf))<<48);
  temp.i |= (ulong)t[1] << 32;
  temp.i |= (ulong)t[2] << 16;
  temp.i |= (ulong)(t[0] & 0xfff0);

  /* Please note the lower 4 bits of mantissa1 are always 0.  */
  return (float)temp.d - 1.0f;
}

void copystate(__private RandType t[RAND_BUF_LEN], __private RandType tnew[RAND_BUF_LEN]){
    tnew[0]=t[0];
    tnew[1]=t[1];
    tnew[2]=t[2];
}

// generate random number for the next zenith angle
void rand_need_more(__private RandType t[RAND_BUF_LEN]){
}

float rand_uniform01(__private RandType t[RAND_BUF_LEN]){
    return __erand48_r(t);
}

void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint *n_seed, int idx){
    t[0] = n_seed[idx*RAND_BUF_LEN]   & 0xffff;
    t[1] = n_seed[idx*RAND_BUF_LEN+1] & 0xffff;
    t[2] = n_seed[idx*RAND_BUF_LEN+2] & 0xffff;
}
void gpu_rng_reseed(__private RandType t[RAND_BUF_LEN],__global uint cpuseed[],uint idx,float reseed){
}
// generate [0,1] random number for the next scattering length
float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]){
    float ran=__erand48_r(t);
    return ((ran==0.f)?LOG_MT_MAX:(-log(ran)));
}

#endif

// generate [0,1] random number for the next arimuthal angle
float rand_next_aangle(__private RandType t[RAND_BUF_LEN]){
    return rand_uniform01(t);
}

#define rand_next_zangle(t1)  rand_next_aangle(t1)
#define rand_next_reflect(t1) rand_next_aangle(t1)
#define rand_do_roulette(t1)  rand_next_aangle(t1) 


#ifdef USE_ATOMIC
// OpenCL float atomicadd hack:
// http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html

inline void atomicadd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
#endif

void clearpath(__local float *p, __constant MCXParam gcfg[]){
      uint i;
      for(i=0;i<gcfg->maxmedia;i++)
      	   p[i]=0.f;
}

#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4 p0[],__constant float4 gdetpos[],__constant MCXParam gcfg[]){
      uint i;
      for(i=0;i<gcfg->detnum;i++){
      	if((gdetpos[i].x-p0[0].x)*(gdetpos[i].x-p0[0].x)+
	   (gdetpos[i].y-p0[0].y)*(gdetpos[i].y-p0[0].y)+
	   (gdetpos[i].z-p0[0].z)*(gdetpos[i].z-p0[0].z) < gdetpos[i].w){
	        return i+1;
	   }
      }
      return 0;
}

void savedetphoton(__global float n_det[],__global uint *detectedphoton,float nscat,
                   __local float *ppath,float4 p0[],__constant float4 gdetpos[],__constant MCXParam gcfg[]){
      uint detid;
      detid=finddetector(p0,gdetpos,gcfg);
      if(detid){
	 uint baseaddr=atomic_inc(detectedphoton);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    baseaddr*=gcfg->maxmedia+2;
	    n_det[baseaddr++]=detid;
	    n_det[baseaddr++]=nscat;
	    for(i=0;i<gcfg->maxmedia;i++){
		n_det[baseaddr+i]=ppath[i]; // save partial pathlength to the memory
	    }
	 }
      }
}
#endif

float mcx_nextafterf(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a+1000.f;
      num.i+=dir ^ (num.i & 0x80000000U);
      return num.f-1000.f;
}

float hitgrid(float4 p0[], float4 v[], float4 htime[], int *id){
      float dist;

      //time-of-flight to hit the wall in each direction

      htime[0]=fabs(floor(p0[0])+convert_float4(isgreater(v[0],((float4)(0.f))))-p0[0]);
      htime[0]=fabs(native_divide(htime[0]+(float4)EPS,v[0]));

      //get the direction with the smallest time-of-flight
      dist=fmin(fmin(htime[0].x,htime[0].y),htime[0].z);
      (*id)=(dist==htime[0].x?0:(dist==htime[0].y?1:2));

      htime[0]=p0[0]+(float4)(dist)*v[0];

      (*id==0) ?
          (htime[0].x=mcx_nextafterf(convert_int_rte(htime[0].x), (v[0].x > 0.f)-(v[0].x < 0.f))) :
	  ((*id==1) ? 
	  	(htime[0].y=mcx_nextafterf(convert_int_rte(htime[0].y), (v[0].y > 0.f)-(v[0].y < 0.f))) :
		(htime[0].z=mcx_nextafterf(convert_int_rte(htime[0].z), (v[0].z > 0.f)-(v[0].z < 0.f))) );
      return dist;
}


void rotatevector(float4 v[], float stheta, float ctheta, float sphi, float cphi){
      if( v[0].z>-1.f+EPS && v[0].z<1.f-EPS ) {
   	  float tmp0=1.f-v[0].z*v[0].z;
   	  float tmp1=stheta*rsqrt(tmp0);
   	  *((float4*)v)=(float4)(
   	       tmp1*(v[0].x*v[0].z*cphi - v[0].y*sphi) + v[0].x*ctheta,
   	       tmp1*(v[0].y*v[0].z*cphi + v[0].x*sphi) + v[0].y*ctheta,
   	      -tmp1*tmp0*cphi                          + v[0].z*ctheta,
   	       v[0].w
   	  );
      }else{
   	  v[0]=(float4)(stheta*cphi,stheta*sphi,(v[0].z>0.f)?ctheta:-ctheta,v[0].w);
      }
      GPUDEBUG(((__constant char*)"new dir: %10.5e %10.5e %10.5e\n",v[0].x,v[0].y,v[0].z));
}

int launchnewphoton(float4 p[],float4 v[],float4 f[],float4 prop[],uint *idx1d,
           uint *mediaid,float *w0, float *Lmove,uchar isdet, __local float ppath[],float *energyloss,float *energylaunched,
	   __global float n_det[],__global uint *dpnum, __constant float4 gproperty[],
	   __constant float4 gdetpos[],__constant MCXParam gcfg[],int threadid, int threadphoton, int oddphotons){
      
      if(p[0].w>=0.f){
          *energyloss+=p[0].w;  // sum all the remaining energy
#ifdef MCX_SAVE_DETECTORS
          // let's handle detectors here
          if(gcfg->savedet){
             if(*mediaid==0 && isdet){
	          savedetphoton(n_det,dpnum,v[0].w,ppath,p,gdetpos,gcfg);
	     }
	     clearpath(ppath,gcfg);
          }
#endif
      }

      if(f[0].w>=(threadphoton+(threadid<oddphotons)))
         return 1; // all photons complete 
      p[0]=gcfg->ps;
      v[0]=gcfg->c0;
      f[0]=(float4)(0.f,0.f,gcfg->minaccumtime,f[0].w+1);
      *idx1d=gcfg->idx1dorig;
      *mediaid=gcfg->mediaidorig;
      prop[0]=gproperty[*mediaid & MED_MASK]; //always use mediaid to read gproperty[]
      *energylaunched+=p[0].w;
      *w0=p[0].w;
      *Lmove=0.f;
      return 0;
}

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
__kernel void mcx_main_loop(const int nphoton, const int ophoton,__global const uchar media[],
     __global float field[], __global float genergy[], __global uint n_seed[],
     __global float n_det[],__constant float4 gproperty[],
     __constant float4 gdetpos[], __global uint stopsign[1],__global uint detectedphoton[1],
     __local float *sharedmem, __constant MCXParam gcfg[]){

     int idx= get_global_id(0);

     float4 p={0.f,0.f,0.f,-1.f};  //{x,y,z}: x,y,z coordinates,{w}:packet weight
     float4 v=gcfg->c0;  //{x,y,z}: ix,iy,iz unitary direction vector, {w}:total scat event
     float4 f={0.f,0.f,gcfg->minaccumtime,0.f};  //f.w can be dropped to save register
     float  energyloss=0.f;
     float  energylaunched=0.f;

     uint idx1d, idx1dold;   //idx1dold is related to reflection

     uint   mediaid=gcfg->mediaidorig,mediaidold=0;
     float  w0;
     float  n1;   //reflection var
     float4 htime;            //reflection var
     int flipdir=0;

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN];
     float4 prop;    //can become float2 if no reflection

     float len,cphi,sphi,theta,stheta,ctheta,tmp0,tmp1;
     float accumweight=0.f;
     float slen;
     float Lmove = 0.f;

     __local float *ppath=sharedmem+get_local_id(0)*gcfg->maxmedia;

#ifdef  MCX_SAVE_DETECTORS
     if(gcfg->savedet) clearpath(ppath,gcfg);
#endif

     gpu_rng_init(t,n_seed,idx);

     if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,&w0,&Lmove,0,ppath,
		      &energyloss,&energylaunched,n_det,detectedphoton,gproperty,gdetpos,gcfg,idx,nphoton,ophoton)){
         n_seed[idx]=NO_LAUNCH;
         return;
     }

     while(f.w<=nphoton + (idx<ophoton)) {

          GPUDEBUG(((__constant char*)"photonid [%d] L=%f w=%e medium=%d\n",(int)f.w,f.x,p.w,mediaid));

	  if(f.x<=0.f) {  // if this photon has finished the current jump
   	       f.x=rand_next_scatlen(t);

               GPUDEBUG(((__constant char*)"scat L=%f RNG=[%e %e %e] \n",f.x,t[0],t[1],t[2]));

	       if(p.w<1.f){ //weight
                       //random arimuthal angle
                       tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                       sphi=sincos(tmp0,&cphi);
                       GPUDEBUG(((__constant char*)"scat phi=%f\n",tmp0));

                       //Henyey-Greenstein Phase Function, "Handbook of Optical 
                       //Biomedical Diagnostics",2002,Chap3,p234, also see Boas2002

                       if(prop.z>EPS){  //if prop.z is too small, the distribution of theta is bad
		           tmp0=(1.f-prop.z*prop.z)/(1.f-prop.z+2.f*prop.z*rand_next_zangle(t));
		           tmp0*=tmp0;
		           tmp0=(1.f+prop.z*prop.z-tmp0)/(2.f*prop.z);

                           // when ran=1, CUDA will give me 1.000002 for tmp0 which produces nan later
                           // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                           tmp0=max(-1.f, min(1.f, tmp0));

		           theta=acos(tmp0);
		           stheta=sin(theta);
		           ctheta=tmp0;
                       }else{
			   theta=acos(2.f*rand_next_zangle(t)-1.f);
                           stheta=sincos(theta,&ctheta);
                       }
                       GPUDEBUG(((__constant char*)"scat theta=%f\n",theta));
                       rotatevector(&v,stheta,ctheta,sphi,cphi);
                       v.w+=1.f;
	       }
	  }

	  n1=prop.w;
	  prop=gproperty[mediaid & MED_MASK];
	  
	  len=hitgrid(&p, &v, &htime, &flipdir);
	  slen=len*prop.y;
	  slen=fmin(slen,f.x);
	  len=slen/prop.y;

          GPUDEBUG(((__constant char*)"p=[%f %f %f] -> <%f %f %f>*%f -> hit=[%f %f %f] flip=%d\n",p.x,p.y,p.z,v.x,v.y,v.z,len,htime.x,htime.y,htime.z,flipdir));

	  p.xyz = (slen==f.x) ? p.xyz+(float3)(len)*v.xyz : htime.xyz;
	  p.w*=exp(-prop.x*len);
	  f.x-=slen;
	  f.y+=len*prop.w*gcfg->oneoverc0;
	  Lmove+=len;

          GPUDEBUG(((__constant char*)"update p=[%f %f %f] -> len=%f\n",p.x,p.y,p.z,len));

#ifdef MCX_SAVE_DETECTORS
          if(gcfg->savedet)
	      ppath[(mediaid & MED_MASK)-1]+=len; //(unit=grid)
#endif

          mediaidold=media[idx1d];
          idx1dold=idx1d;
          idx1d=((int)floor(p.z)*gcfg->dimlen.y+(int)floor(p.y)*gcfg->dimlen.x+(int)floor(p.x));
          GPUDEBUG(((__constant char*)"idx1d [%d]->[%d]\n",idx1dold,idx1d));
          if(any(isless(p.xyz,(float3)(0.f))) || any(isgreater(p.xyz,(gcfg->maxidx.xyz)))){
	      mediaid=0;	
	  }else{
              mediaid=media[idx1d] & MED_MASK;
          }
          GPUDEBUG(((__constant char*)"medium [%d]->[%d]\n",mediaidold,mediaid));

	  if(idx1d!=idx1dold && idx1dold>0 && mediaidold){
             GPUDEBUG(((__constant char*)"field add to %d->%f(%d)\n",idx1dold,w0-p.w,(int)f.w));
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(gcfg->save2pt && f.y>=gcfg->twin0 && f.y<gcfg->twin1){
                  GPUDEBUG(((__constant char*)"deposit to [%d] %e, w=%f\n",idx1dold,w0-p.w,p.w));
#ifndef USE_ATOMIC
                  // set gcfg->skipradius2 to only start depositing energy when dist^2>gcfg->skipradius2 
                  if(gcfg->skipradius2>EPS){
                      if((p.x-gcfg->ps.x)*(p.x-gcfg->ps.x)+(p.y-gcfg->ps.y)*(p.y-gcfg->ps.y)+(p.z-gcfg->ps.z)*(p.z-gcfg->ps.z)>gcfg->skipradius2){
                          field[idx1dold+(int)(floor((f.y-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=w0-p.w;
                      }else{
                          accumweight+=p.w*prop.x; // weight*absorption
                      }
                  }else{
                      field[idx1dold+(int)(floor((f.y-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=w0-p.w;
                  }
#else
		  atomicadd(& field[idx1dold+(int)(floor((f.y-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z], w0-p.w);
                  GPUDEBUG(((__constant char*)"atomic write to [%d] %e, w=%f\n",idx1dold,weight,p.w));
#endif
	     }
	     w0=p.w;
	     Lmove=0.f;
	  }

          if((mediaid==0 && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[mediaid].w))) || f.y>gcfg->twin1){
                  GPUDEBUG(((__constant char*)"direct relaunch at idx=[%d] mediaid=[%d], ref=[%d]\n",idx1d,mediaid,gcfg->doreflect));
		  if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),ppath,
		      &energyloss,&energylaunched,n_det,detectedphoton,gproperty,gdetpos,gcfg,idx,nphoton,ophoton)){ 
                         break;
		  }
                  continue;
          }
#ifdef MCX_DO_REFLECTION
          //if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one
          if(gcfg->doreflect && n1!=gproperty[mediaid].w){
	          float Rtotal=1.f;

                  *((float4*)(&prop))=gproperty[mediaid]; // optical property across the interface

                  tmp0=n1*n1;
                  tmp1=prop.w*prop.w;
                  cphi=fabs( (flipdir==0) ? v.x : (flipdir==1 ? v.y : v.z)); // cos(si)
                  sphi=1.f-cphi*cphi;            // sin(si)^2

                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2
	          GPUDEBUG(((__constant char*)"ref total ref=%f\n",len));

                  if(len>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.w*cphi*sqrt(len);
                     Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
		     GPUDEBUG(((__constant char*)"Rtotal=%f\n",Rtotal));
                  }
	          if(Rtotal<1.f && rand_next_reflect(t)>Rtotal){ // do transmission
                        if(mediaid==0){ // transmission to external boundary
                            GPUDEBUG(((__constant char*)"transmit to air, relaunch\n"));
		    	    if(launchnewphoton(&p,&v,&f,&prop,&idx1d,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),
			        ppath,&energyloss,&energylaunched,n_det,detectedphoton,gproperty,gdetpos,gcfg,idx,nphoton,ophoton)){
                                    break;
			    }
			    continue;
			}
	                GPUDEBUG(((__constant char*)"do transmission\n"));
			tmp0=n1/prop.w;
                	if(flipdir==2) { //transmit through z plane
                	   v.xy=tmp0*v.xy;
			   v.z=sqrt(1.f - v.y*v.y - v.x*v.x);
                	}else if(flipdir==1){ //transmit through y plane
                	   v.xz=tmp0*v.xz;
			   v.y=sqrt(1.f - v.x*v.x - v.z*v.z);
                	}else if(flipdir==0){ //transmit through x plane
                	   v.yz=tmp0*v.yz;
			   v.x=sqrt(1.f - v.y*v.y - v.z*v.z);
                	}
		  }else{ //do reflection
	                GPUDEBUG(((__constant char*)"do reflection\n"));
	                GPUDEBUG(((__constant char*)"ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n",flipdir,p.x,p.y,p.z,v.x,v.y,v.z));
                	(flipdir==0) ? (v.x=-v.x) : ((flipdir==1) ? (v.y=-v.y) : (v.z=-v.z)) ;
			(flipdir==0) ?
        		    (p.x=nextafter(convert_int_rte(p.x), p.x+(v.x > 0.f)-0.5f)) :
			    ((flipdir==1) ? 
				(p.y=nextafter(convert_int_rte(p.y), p.y+(v.y > 0.f)-0.5f)) :
				(p.z=nextafter(convert_int_rte(p.z), p.z+(v.z > 0.f)-0.5f)) );
	                GPUDEBUG(((__constant char*)"ref p_new=[%f %f %f] v_new=[%f %f %f]\n",p.x,p.y,p.z,v.x,v.y,v.z));
                	idx1d=idx1dold;
		 	mediaid=(media[idx1d] & MED_MASK);
			prop=gproperty[mediaid];
			n1=prop.w;
		  }
              }
#endif
     }
     // accumweight saves the total absorbed energy in the sphere r<sradius.
     // in non-atomic mode, accumweight is more accurate than saving to the grid
     // as it is not influenced by race conditions.
     // now I borrow f.z to pass this value back

     f.z=accumweight;

     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energylaunched;
}

