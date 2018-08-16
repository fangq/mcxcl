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

#ifdef MCX_SAVE_DETECTORS
  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef USE_HALF
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #define FLOAT4VEC half4
  #define TOFLOAT4  convert_half4
#else
  #define FLOAT4VEC float4
  #define TOFLOAT4
#endif

#ifdef MCX_USE_NATIVE
  #define MCX_MATHFUN(fun)              native_##fun
  #define MCX_SINCOS(theta,osin,ocos)   {(osin)=native_sin(theta);(ocos)=native_cos(theta);} 
#else
  #define MCX_MATHFUN(fun)              fun
  #define MCX_SINCOS(theta,osin,ocos)   (ocos)=sincos((theta),&(osin))
#endif

#ifdef MCX_GPU_DEBUG
  #define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
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
#define MAX_PROP           2000                     /*maximum property number*/
#define OUTSIDE_VOLUME     0xFFFFFFFF              /**< flag indicating the index is outside of the volume */

#define DET_MASK           0xFFFF0000
#define MED_MASK           0x0000FFFF
#define NULL               0
#define MAX_ACCUM          1000.f

#define MCX_DEBUG_RNG       1                   /**< MCX debug flags */
#define MCX_DEBUG_MOVE      2
#define MCX_DEBUG_PROGRESS  4

#define MIN(a,b)           ((a)<(b)?(a):(b))

typedef struct KernelParams {
  float4 ps,c0;
  float4 maxidx;
  uint4  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minstep;
  float  twin0,twin1,tmax;
  float  oneoverc0;
  uint   isrowmajor,save2pt,doreflect,dorefint,savedet;
  float  Rtstep;
  float  minenergy;
  float  skipradius2;
  float  minaccumtime;
  uint   maxdetphoton;
  uint   maxmedia;
  uint   detnum;
  uint   idx1dorig;
  uint   mediaidorig;
  uint   blockphoton;
  uint   blockextra;
  int    voidtime;
  int    srctype;                    /**< type of the source */
  float4 srcparam1;                  /**< source parameters set 1 */
  float4 srcparam2;                  /**< source parameters set 2 */
  uint   maxvoidstep;
  uint   issaveexit;    /**<1 save the exit position and dir of a detected photon, 0 do not save*/
  uint   issaveref;     /**<1 save diffuse reflectance at the boundary voxels, 0 do not save*/
  uint   maxgate;
  uint   threadphoton;                  /**< how many photons to be simulated in a thread */
  uint   debuglevel;           /**< debug flags */
} MCXParam __attribute__ ((aligned (32)));


//#ifndef USE_XORSHIFT128P_RAND     // xorshift128+ is the default RNG
#ifdef USE_LL5_RAND                 //enable the legacy Logistic Lattic RNG

#define RAND_BUF_LEN       5        //register arrays
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define INIT_LOGISTIC      100

typedef float RandType;

#define FUN(x)               (4.f*(x)*(1.f-(x)))
#define NU                   1e-7f
#define NU2                  (1.f-2.f*NU)
#define MIN_INVERSE_LIMIT    1e-7f
#define logistic_uniform(v)  (acos(1.f-2.f*(v))*R_PI)
#define R_MAX_C_RAND         (1.f/RAND_MAX)
#define LOG_MT_MAX           22.1807097779182f

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

void logistic_init(__private RandType *t,__global uint *seed,uint idx){
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

#else

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define RAND_BUF_LEN       2        //register arrays
#define RAND_SEED_LEN      4        //48 bit packed with 64bit length
#define LOG_MT_MAX         22.1807097779182f
#define IEEE754_DOUBLE_BIAS     0x3FF0000000000000ul /* Added to exponent.  */

typedef ulong  RandType;

static float xorshift128p_nextf (__private RandType t[RAND_BUF_LEN]){
   union {
        ulong  i;
	float f[2];
	uint  u[2];
   } s1;
   const ulong s0 = t[1];
   s1.i = t[0];
   t[0] = s0;
   s1.i ^= s1.i << 23; // a
   t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
   s1.i = t[1] + s0;
   s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);

   return s1.f[0] - 1.0f;
}

static void copystate(__private RandType t[RAND_BUF_LEN], __private RandType tnew[RAND_BUF_LEN]){
    tnew[0]=t[0];
    tnew[1]=t[1];
}

// generate random number for the next zenith angle
static void rand_need_more(__private RandType t[RAND_BUF_LEN]){
}

static float rand_uniform01(__private RandType t[RAND_BUF_LEN]){
    return xorshift128p_nextf(t);
}

static void xorshift128p_seed (__global uint *seed,RandType t[RAND_BUF_LEN])
{
    t[0] = (ulong)seed[0] << 32 | seed[1] ;
    t[1] = (ulong)seed[2] << 32 | seed[3];
}

static void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint *n_seed, int idx){
    xorshift128p_seed((n_seed+idx*RAND_SEED_LEN),t);
}
static void gpu_rng_reseed(__private RandType t[RAND_BUF_LEN],__global uint *cpuseed,uint idx,float reseed){
}

#endif

float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]);

float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]){
    return -MCX_MATHFUN(log)(rand_uniform01(t)+EPS);
}

#define rand_next_aangle(t)  rand_uniform01(t)
#define rand_next_zangle(t)  rand_uniform01(t)
#define rand_next_reflect(t) rand_uniform01(t)
#define rand_do_roulette(t)  rand_uniform01(t) 


/* function prototypes */

void clearpath(__local float *p, __constant MCXParam *gcfg);
float mcx_nextafterf(float a, int dir);
float hitgrid(float4 *p0, float4 *v, float4 *htime, int *id);
void rotatevector(float4 *v, float stheta, float ctheta, float sphi, float cphi);
void transmit(float4 *v, float n1, float n2,int flipdir);
float reflectcoeff(float4 *v, float n1, float n2, int flipdir);
int skipvoid(float4 *p,float4 *v,float4 *f,__global const uint *media, __constant float4 *gproperty, __constant MCXParam *gcfg);
#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4 *p0,__constant float4 *gdetpos,__constant MCXParam *gcfg);
void savedetphoton(__global float *n_det,__global uint *detectedphoton,float nscat,
                   __local float *ppath,float4 *p0,float4 *v,__constant float4 *gdetpos,__constant MCXParam *gcfg);
#endif
int launchnewphoton(float4 *p,float4 *v,float4 *f,FLOAT4VEC *prop,uint *idx1d,
           __global float *field, uint *mediaid,float *w0,float *Lmove,uint isdet, 
	   __local float *ppath,float *energyloss,float *energylaunched,
	   __global float *n_det,__global uint *dpnum, __private RandType t[RAND_BUF_LEN],
	   __constant float4 *gproperty, __global const uint *media, __global float *srcpattern,
	   __constant float4 *gdetpos,__constant MCXParam *gcfg,int threadid, int threadphoton, 
	   int oddphotons, __local int *blockphoton, volatile __global uint *gprogress);

#ifdef USE_ATOMIC
// OpenCL float atomicadd hack:
// http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html
// https://devtalk.nvidia.com/default/topic/458062/atomicadd-float-float-atomicmul-float-float-/

inline float atomicadd(volatile __global float* address, const float value){
    float old = value;
    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
    return old;
}
#endif

void clearpath(__local float *p, __constant MCXParam *gcfg){
      uint i;
      for(i=0;i<gcfg->maxmedia;i++)
      	   p[i]=0.f;
}

#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4 *p0,__constant float4 *gdetpos,__constant MCXParam *gcfg){
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

void savedetphoton(__global float *n_det,__global uint *detectedphoton,float nscat,
                   __local float *ppath,float4 *p0,float4 *v,__constant float4 *gdetpos,__constant MCXParam *gcfg){
      uint detid=finddetector(p0,gdetpos,gcfg);
      if(detid){
	 uint baseaddr=atomic_inc(detectedphoton);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    baseaddr*=gcfg->maxmedia+2+gcfg->issaveexit*6;
	    n_det[baseaddr++]=detid;
	    n_det[baseaddr++]=nscat;
	    for(i=0;i<gcfg->maxmedia;i++)
		n_det[baseaddr+i]=ppath[i]; // save partial pathlength to the memory
	    if(gcfg->issaveexit){
                baseaddr+=gcfg->maxmedia;
	        n_det[baseaddr++]=p0->x;
		n_det[baseaddr++]=p0->y;
		n_det[baseaddr++]=p0->z;
		n_det[baseaddr++]=v->x;
		n_det[baseaddr++]=v->y;
		n_det[baseaddr++]=v->z;
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

#ifndef USE_HALF

float hitgrid(float4 *p0, float4 *v, float4 *htime, int *id){
      float dist;

      //time-of-flight to hit the wall in each direction

      htime[0]=fabs(floor(p0[0])-convert_float4(isgreater(v[0],((float4)(0.f))))-p0[0]);
      htime[0]=fabs(native_divide(htime[0]+(float4)EPS,v[0]));

      //get the direction with the smallest time-of-flight
      dist=fmin(fmin(htime[0].x,htime[0].y),htime[0].z);
      (*id)=(dist==htime[0].x?0:(dist==htime[0].y?1:2));

      htime[0]=p0[0]+(float4)(dist)*v[0];

#ifdef MCX_VECTOR_INDEX
      ((float*)htime)[*id]=mcx_nextafterf(convert_float_rte(((float*)htime)[*id]),  (((float*)v)[*id] > 0.f)-(((float*)v)[*id] < 0.f));
#else
      (*id==0) ?
          (htime[0].x=mcx_nextafterf(convert_float_rte(htime[0].x), (v[0].x > 0.f)-(v[0].x < 0.f))) :
	  ((*id==1) ? 
	  	(htime[0].y=mcx_nextafterf(convert_float_rte(htime[0].y), (v[0].y > 0.f)-(v[0].y < 0.f))) :
		(htime[0].z=mcx_nextafterf(convert_float_rte(htime[0].z), (v[0].z > 0.f)-(v[0].z < 0.f))) );
#endif
      return dist;
}

#else

half mcx_nextafter_half(const half a, short dir){
      union{
          half f;
          short i;
      } num;
      num.f=a;
      ((num.i & 0x7FFFU)==0) ? num.i =(((dir & 0x8000U) ) | 1) : ((num.i & 0x8000U) ? (num.i-=dir) : (num.i+=dir) );
      return num.f;
}

float hitgrid(float4 *p, float4 *v0, half4 *htime, int *id){
      half dist;
      half4 p0, v;

      p0=convert_half4(p[0]);
      v=convert_half4(v0[0]);

      //time-of-flight to hit the wall in each direction

      htime[0]=fabs(floor(p0)-convert_half4(isgreater(v,((half4)(0.f))))-p0);
      htime[0]=fabs((htime[0]+(half4)EPS)/v);

      //get the direction with the smallest time-of-flight
      dist=fmin(fmin(htime[0].x,htime[0].y),htime[0].z);
      (*id)=(dist==htime[0].x?0:(dist==htime[0].y?1:2));

      htime[0]=p0+(half4)(dist)*v;

#ifdef MCX_VECTOR_INDEX
      ((half*)htime)[*id]=mcx_nextafter_half(round(((half*)htime)[*id]), (((half*)&v)[*id] > 0.f)-(((half*)&v)[*id] < 0.f));
#else
      (*id==0) ?
          (htime[0].x=mcx_nextafter_half(round(htime[0].x), (v.x > 0.f)-(v.x < 0.f))) :
	  ((*id==1) ? 
	  	(htime[0].y=mcx_nextafter_half(round(htime[0].y), (v.y > 0.f)-(v.y < 0.f))) :
		(htime[0].z=mcx_nextafter_half(round(htime[0].z), (v.z > 0.f)-(v.z < 0.f))) );
#endif
      return convert_float(dist);
}

#endif

void rotatevector(float4 *v, float stheta, float ctheta, float sphi, float cphi){
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
      v[0].xyz=v[0].xyz*rsqrt(v[0].x*v[0].x+v[0].y*v[0].y+v[0].z*v[0].z);
      GPUDEBUG(((__constant char*)"new dir: %10.5e %10.5e %10.5e\n",v[0].x,v[0].y,v[0].z));
}

void transmit(float4 *v, float n1, float n2,int flipdir){
      float tmp0=n1/n2;
      v[0].xyz*=tmp0;

      (flipdir==0) ?
          (v[0].x=sqrt(1.f - v[0].y*v[0].y - v[0].z*v[0].z)*((v[0].x>0.f)-(v[0].x<0.f))):
	  ((flipdir==1) ? 
	      (v[0].y=sqrt(1.f - v[0].x*v[0].x - v[0].z*v[0].z)*((v[0].y>0.f)-(v[0].y<0.f))):
	      (v[0].z=sqrt(1.f - v[0].x*v[0].x - v[0].y*v[0].y)*((v[0].z>0.f)-(v[0].z<0.f))));
}

/**
 * @brief Calculating the reflection coefficient at an interface
 *
 * This function calculates the reflection coefficient at
 * an interface of different refrective indicex (n1/n2)
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

float reflectcoeff(float4 *v, float n1, float n2, int flipdir){
      float Icos=fabs((flipdir==0) ? v[0].x : (flipdir==1 ? v[0].y : v[0].z));
      float tmp0=n1*n1;
      float tmp1=n2*n2;
      float tmp2=1.f-tmp0/tmp1*(1.f-Icos*Icos); /** 1-[n1/n2*sin(si)]^2 = cos(ti)^2*/
      if(tmp2>0.f){ ///< partial reflection
          float Re,Im,Rtotal;
	  Re=tmp0*Icos*Icos+tmp1*tmp2;
	  tmp2=sqrt(tmp2); /** to save one sqrt*/
	  Im=2.f*n1*n2*Icos*tmp2;
	  Rtotal=(Re-Im)/(Re+Im);     /** Rp*/
	  Re=tmp1*Icos*Icos+tmp0*tmp2*tmp2;
	  Rtotal=(Rtotal+(Re-Im)/(Re+Im))*0.5f; /** (Rp+Rs)/2*/
	  return Rtotal;
      }else{ ///< total reflection
          return 1.f;
      }
}

/**
 * @brief Advance photon to the 1st non-zero voxel if launched in the backgruond 
 *
 * This function advances the photon to the 1st non-zero voxel along the direction
 * of v if the photon is launched outside of the cubic domain or in a zero-voxel.
 * To avoid large overhead, photon can only advance gcfg->minaccumtime steps, which
 * can be set using the --maxvoidstep flag; by default, this limit is 1000.
 *
 * @param[in] v: the direction vector of the photon
 * @param[in] n1: the refrective index of the voxel the photon leaves
 * @param[in] n2: the refrective index of the voxel the photon enters
 * @param[in] flipdir: 0: transmit through x=x0 plane; 1: through y=y0 plane; 2: through z=z0 plane
 * @return the reflection coefficient R=(Rs+Rp)/2, Rs: R of the perpendicularly polarized light, Rp: parallelly polarized light
 */

int skipvoid(float4 *p,float4 *v,float4 *f,__global const uint *media, __constant float4 *gproperty, __constant MCXParam *gcfg){
      int count=1,idx1d;
      while(1){
          if(!(any(isless(p[0].xyz,(float3)(0.f))) || any(isgreaterequal(p[0].xyz,(gcfg->maxidx.xyz))))){
	    idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));
	    if(media[idx1d] & MED_MASK){ ///< if enters a non-zero voxel
                GPUDEBUG(("inside volume [%f %f %f] v=<%f %f %f>\n",p[0].x,p[0].y,p[0].z,v[0].x,v[0].y,v[0].z));
	        float4 htime;
                int flipdir;
		p[0].xyz-=v[0].xyz;
                f[0].y-=gcfg->minaccumtime;
                idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));

                //GPUDEBUG(("look for entry p0=[%f %f %f] rv=[%f %f %f]\n",p[0].x,p[0].y,p[0].z,rv[0].x,rv[0].y,rv[0].z));
		count=0;
		while((any(isless(p[0].xyz,(float3)(0.f))) || any(isgreaterequal(p[0].xyz,(gcfg->maxidx.xyz)))) || !(media[idx1d] & MED_MASK)){ // at most 3 times
	            f[0].y+=gcfg->minaccumtime*hitgrid(p,v,&htime,&flipdir);
                    p[0]=(float4)(htime.x,htime.y,htime.z,p[0].w);
                    idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));
                    GPUDEBUG(("entry p=[%f %f %f] flipdir=%d\n",p[0].x,p[0].y,p[0].z,flipdir));

		    if(count++>3){
		       GPUDEBUG(("fail to find entry point after 3 iterations, something is wrong, abort!!"));
		       break;
		    }
		}
                f[0].y = (gcfg->voidtime) ? f[0].y : 0.f;

		if(gproperty[media[idx1d] & MED_MASK].w!=gproperty[0].w){
	            p[0].w*=1.f-reflectcoeff(v, gproperty[0].w,gproperty[media[idx1d] & MED_MASK].w,flipdir);
                    GPUDEBUG(("transmitted intensity w=%e\n",p[0].w));
	            if(p[0].w>EPS){
		        transmit(v, gproperty[0].w,gproperty[media[idx1d] & MED_MASK].w,flipdir);
                        GPUDEBUG(("transmit into volume v=<%f %f %f>\n",v[0].x,v[0].y,v[0].z));
                    }
		}
		return idx1d;
	    }
          }
	  if( ((p[0].x<0.f) && (v[0].x<=0.f)) || ((p[0].x >= gcfg->maxidx.x) && (v[0].x>=0.f))
	   || ((p[0].y<0.f) && (v[0].y<=0.f)) || ((p[0].y >= gcfg->maxidx.y) && (v[0].y>=0.f))
	   || ((p[0].z<0.f) && (v[0].z<=0.f)) || ((p[0].z >= gcfg->maxidx.z) && (v[0].z>=0.f)))
	      return -1;
	  p[0]=(float4)(p[0].x+v[0].x,p[0].y+v[0].y,p[0].z+v[0].z,p[0].w);
          GPUDEBUG(("inside void [%f %f %f]\n",p[0].x,p[0].y,p[0].z));
          f[0].y+=gcfg->minaccumtime;
	  if((uint)count++>gcfg->maxvoidstep)
	      return -1;
      }
}


/**
 * @brief Terminate a photon and launch a new photon according to specified source form
 *
 * This function terminates the current photon and launches a new photon according 
 * to the source type selected. Currently, over a dozen source types are supported, 
 * including pencil, isotropic, planar, fourierx, gaussian, zgaussian etc.
 *
 * @param[in,out] p: the 3D position and weight of the photon
 * @param[in,out] v: the direction vector of the photon
 * @param[in,out] f: the parameter vector of the photon
 * @param[in,out] rv: the reciprocal direction vector of the photon (rv[i]=1/v[i])
 * @param[out] prop: the optical properties of the voxel the photon is launched into
 * @param[in,out] idx1d: the linear index of the voxel containing the photon at launch
 * @param[in] field: the 3D array to store photon weights
 * @param[in,out] mediaid: the medium index at the voxel at launch
 * @param[in,out] w0: initial weight, reset here after launch
 * @param[in,out] Lmove: photon movement length variable, reset to 0 after launch
 * @param[in] isdet: whether the previous photon being terminated lands at a detector
 * @param[in,out] ppath: pointer to the shared-mem buffer to store photon partial-path data
 * @param[in,out] energyloss: register variable to accummulate the escaped photon energy
 * @param[in,out] energylaunched: register variable to accummulate the total launched photon energy
 * @param[in,out] n_det: array in the constant memory where detector positions are stored
 * @param[in,out] dpnum: global-mem variable where the count of detected photons are stored
 * @param[in] t: RNG state
 * @param[in,out] photonseed: RNG state stored at photon's launch time if replay is needed
 * @param[in] media: domain medium index array, read-only
 * @param[in] srcpattern: user-specified source pattern array if pattern source is used
 * @param[in] threadid: the global index of the current thread
 * @param[in] rngseed: in the replay mode, pointer to the saved detected photon seed data
 * @param[in,out] seeddata: pointer to the buffer to save detected photon seeds
 * @param[in,out] gdebugdata: pointer to the buffer to save photon trajectory positions
 * @param[in,out] gprogress: pointer to the host variable to update progress bar
 */

int launchnewphoton(float4 *p,float4 *v,float4 *f,FLOAT4VEC *prop,uint *idx1d,
           __global float *field, uint *mediaid,float *w0,float *Lmove,uint isdet, 
	   __local float *ppath,float *energyloss,float *energylaunched,
	   __global float *n_det,__global uint *dpnum, __private RandType t[RAND_BUF_LEN],
	   __constant float4 *gproperty, __global const uint *media, __global float *srcpattern,
	   __constant float4 *gdetpos,__constant MCXParam *gcfg,int threadid, int threadphoton, 
	   int oddphotons, __local int *blockphoton, volatile __global uint *gprogress){

/*
__device__ inline int launchnewphoton(float4 *p,float4 *v,float4 *f,float3* rv,float4 *prop,uint *idx1d, float *field,
           uint *mediaid,float *w0,float *Lmove,int isdet, float ppath[],float energyloss[],float energylaunched[],float n_det[],uint *dpnum,
	   RandType t[RAND_BUF_LEN],RandType photonseed[RAND_BUF_LEN],
	   uint media[],float srcpattern[],int threadid,RandType rngseed[],RandType seeddata[],float gdebugdata[],volatile int gprogress[]){
*/
      *w0=1.f;     ///< reuse to count for launchattempt
      *Lmove=-1.f; ///< reuse as "canfocus" flag for each source: non-zero: focusable, zero: not focusable
      *prop=(float4)(gcfg->ps.x,gcfg->ps.y,gcfg->ps.z,0); ///< reuse as the origin of the src, needed for focusable sources

      /**
       * First, let's terminate the current photon and perform detection calculations
       */
      if(p[0].w>=0.f){
          *energyloss+=p[0].w;  ///< sum all the remaining energy
#ifdef GROUP_LOAD_BALANCE
          if(f[0].w<0.f || atomic_sub(blockphoton,1)<=1)
              return 1;
          if(blockphoton[0]<get_local_size(0) && get_local_id(0))
              f[0].w=-2.f;
#endif
#ifdef MCX_SAVE_DETECTORS
      // let's handle detectors here
          if(gcfg->savedet){
             if(isdet>0 && *mediaid==0)
	         savedetphoton(n_det,dpnum,v[0].w,ppath,p,v,gdetpos,gcfg);
             clearpath(ppath,gcfg);
          }
#endif
          if(gcfg->issaveref && *mediaid==0 && *idx1d!=OUTSIDE_VOLUME){
	       int tshift=MIN((int)gcfg->maxgate-1,(int)(floor((f[0].y-gcfg->twin0)*gcfg->Rtstep)));
#ifdef USE_ATOMIC
               atomicadd(& field[*idx1d+tshift*gcfg->dimlen.z],-p[0].w);	       
#else
	       field[*idx1d+tshift*gcfg->dimlen.z]+=-p[0].w;
#endif
	  }
      }
      /**
       * If the thread completes all assigned photons, terminate this thread.
       */
#ifndef GROUP_LOAD_BALANCE
      if(f[0].w>=(threadphoton+(threadid<oddphotons)))
         return 1; // all photons complete 
#endif
      
      /**
       * If this is a replay of a detected photon, initilize the RNG with the stored seed here.
       */
/*
      if(gcfg->seed==SEED_FROM_FILE){
          int seedoffset=(threadid*gcfg->threadphoton+min(threadid,gcfg->oddphotons-1)+max(0,(int)f[0].w+1))*RAND_BUF_LEN;
          for(int i=0;i<RAND_BUF_LEN;i++)
	      t[i]=rngseed[seedoffset+i];
      }
*/
      /**
       * Attempt to launch a new photon until success
       */
      do{
	  p[0]=gcfg->ps;
	  v[0]=gcfg->c0;
	  f[0]=(float4)(0.f,0.f,gcfg->minaccumtime,f[0].w);
          *idx1d=gcfg->idx1dorig;
          *mediaid=gcfg->mediaidorig;
/*
	  if(gcfg->issaveseed)
              copystate(t,photonseed);
*/
          /**
           * Only one branch is taken because of template, this can reduce thread divergence
           */

#if defined(MCX_SRC_PLANAR) || defined(MCX_SRC_PATTERN) || defined(MCX_SRC_PATTERN3D) || defined(MCX_SRC_FOURIER) || defined(MCX_SRC_PENCILARRAY) /*a rectangular grid over a plane*/
	      float rx=rand_uniform01(t);
	      float ry=rand_uniform01(t);
	      float rz;
    #if defined(MCX_SRC_PATTERN3D) 
	            rz=rand_uniform01(t);
	            p[0]=(float4)(p[0].x+rx*gcfg->srcparam1.x,
		 		         p[0].y+ry*gcfg->srcparam1.y,
				         p[0].z+rz*gcfg->srcparam1.z,
				         p[0].w);
    #else
	            p[0]=(float4)(p[0].x+rx*gcfg->srcparam1.x+ry*gcfg->srcparam2.x,
				       p[0].y+rx*gcfg->srcparam1.y+ry*gcfg->srcparam2.y,
				       p[0].z+rx*gcfg->srcparam1.z+ry*gcfg->srcparam2.z,
				       p[0].w);
    #endif
    #if defined(MCX_SRC_PATTERN)  // need to prevent rx/ry=1 here
		  p[0].w=srcpattern[(int)(ry*JUST_BELOW_ONE*gcfg->srcparam2.w)*(int)(gcfg->srcparam1.w)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.w)];
    #elif defined(MCX_SRC_PATTERN3D)  // need to prevent rx/ry=1 here
	          p[0].w=srcpattern[(int)(rz*JUST_BELOW_ONE*gcfg->srcparam1.z)*(int)(gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+
	                          (int)(ry*JUST_BELOW_ONE*gcfg->srcparam1.y)*(int)(gcfg->srcparam1.x)+(int)(rx*JUST_BELOW_ONE*gcfg->srcparam1.x)];
    #elif defined(MCX_SRC_FOURIER)  // need to prevent rx/ry=1 here
		  p[0].w=(MCX_MATHFUN(cos)((floor(gcfg->srcparam1.w)*rx+floor(gcfg->srcparam2.w)*ry
			  +gcfg->srcparam1.w-floor(gcfg->srcparam1.w))*TWO_PI)*(1.f-gcfg->srcparam2.w+floor(gcfg->srcparam2.w))+1.f)*0.5f; //between 0 and 1
    #elif defined(MCX_SRC_PENCILARRAY)  // need to prevent rx/ry=1 here
		  p[0].x=gcfg->ps.x+ floor(rx*gcfg->srcparam1.w)*gcfg->srcparam1.x/(gcfg->srcparam1.w-1.f)+floor(ry*gcfg->srcparam2.w)*gcfg->srcparam2.x/(gcfg->srcparam2.w-1.f);
		  p[0].y=gcfg->ps.y+ floor(rx*gcfg->srcparam1.w)*gcfg->srcparam1.y/(gcfg->srcparam1.w-1.f)+floor(ry*gcfg->srcparam2.w)*gcfg->srcparam2.y/(gcfg->srcparam2.w-1.f);
		  p[0].z=gcfg->ps.z+ floor(rx*gcfg->srcparam1.w)*gcfg->srcparam1.z/(gcfg->srcparam1.w-1.f)+floor(ry*gcfg->srcparam2.w)*gcfg->srcparam2.z/(gcfg->srcparam2.w-1.f);
    #endif
	      *idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));
	      if(p[0].x<0.f || p[0].y<0.f || p[0].z<0.f || p[0].x>=gcfg->maxidx.x || p[0].y>=gcfg->maxidx.y || p[0].z>=gcfg->maxidx.z){
		  *mediaid=0;
	      }else{
		  *mediaid=media[*idx1d];
	      }
              *prop=(float4)(prop[0].x+(gcfg->srcparam1.x+gcfg->srcparam2.x)*0.5f,
	                 prop[0].y+(gcfg->srcparam1.y+gcfg->srcparam2.y)*0.5f,
			 prop[0].z+(gcfg->srcparam1.z+gcfg->srcparam2.z)*0.5f,0.f);
#elif defined(MCX_SRC_FOURIERX) || defined(MCX_SRC_FOURIERX2D) // [v1x][v1y][v1z][|v2|]; [kx][ky][phi0][M], unit(v0) x unit(v1)=unit(v2)
	      float rx=rand_uniform01(t);
	      float ry=rand_uniform01(t);
	      float4 v2=gcfg->srcparam1;
	      // calculate v2 based on v2=|v2| * unit(v0) x unit(v1)
	      v2.w*=rsqrt(gcfg->srcparam1.x*gcfg->srcparam1.x+gcfg->srcparam1.y*gcfg->srcparam1.y+gcfg->srcparam1.z*gcfg->srcparam1.z);
	      v2.x=v2.w*(gcfg->c0.y*gcfg->srcparam1.z - gcfg->c0.z*gcfg->srcparam1.y);
	      v2.y=v2.w*(gcfg->c0.z*gcfg->srcparam1.x - gcfg->c0.x*gcfg->srcparam1.z); 
	      v2.z=v2.w*(gcfg->c0.x*gcfg->srcparam1.y - gcfg->c0.y*gcfg->srcparam1.x);
	      p[0]=(float4)(p[0].x+rx*gcfg->srcparam1.x+ry*v2.x,
				   p[0].y+rx*gcfg->srcparam1.y+ry*v2.y,
				   p[0].z+rx*gcfg->srcparam1.z+ry*v2.z,
				   p[0].w);
    #if defined(MCX_SRC_FOURIERX2D)
		 p[0].w=(MCX_MATHFUN(sin)((gcfg->srcparam2.x*rx+gcfg->srcparam2.z)*TWO_PI)*MCX_MATHFUN(sin)((gcfg->srcparam2.y*ry+gcfg->srcparam2.w)*TWO_PI)+1.f)*0.5f; //between 0 and 1
    #else
		 p[0].w=(MCX_MATHFUN(cos)((gcfg->srcparam2.x*rx+gcfg->srcparam2.y*ry+gcfg->srcparam2.z)*TWO_PI)*(1.f-gcfg->srcparam2.w)+1.f)*0.5f; //between 0 and 1
    #endif

	      *idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));
	      if(p[0].x<0.f || p[0].y<0.f || p[0].z<0.f || p[0].x>=gcfg->maxidx.x || p[0].y>=gcfg->maxidx.y || p[0].z>=gcfg->maxidx.z){
		  *mediaid=0;
	      }else{
		  *mediaid=media[*idx1d];
	      }
              *prop=(float4)(prop[0].x+(gcfg->srcparam1.x+v2.x)*0.5f,
	                 prop[0].y+(gcfg->srcparam1.y+v2.y)*0.5f,
			 prop[0].z+(gcfg->srcparam1.z+v2.z)*0.5f,0.f);
#elif defined(MCX_SRC_DISK) || defined(MCX_SRC_GAUSSIAN) // uniform disk distribution or Gaussian-beam
	      // Uniform disk point picking
	      // http://mathworld.wolfram.com/DiskPointPicking.html
	      float sphi, cphi;
	      float phi=TWO_PI*rand_uniform01(t);
	      MCX_SINCOS(phi,sphi,cphi);
	      float r;
    #if defined(MCX_SRC_DISK)
		  r=sqrt(rand_uniform01(t))*gcfg->srcparam1.x;
    #else
	      if(fabs(gcfg->c0.w) < 1e-5f || fabs(gcfg->srcparam1.y) < 1e-5f)
	          r=sqrt(-0.5f*log(rand_uniform01(t)))*gcfg->srcparam1.x;
	      else{
	          r=gcfg->srcparam1.x*gcfg->srcparam1.x*M_PI/gcfg->srcparam1.y; //Rayleigh range
	          r=sqrt(-0.5f*log(rand_uniform01(t))*(1.f+(gcfg->c0.w*gcfg->c0.w/(r*r))))*gcfg->srcparam1.x;
              }
    #endif
	      if( v[0].z>-1.f+EPS && v[0].z<1.f-EPS ) {
		  float tmp0=1.f-v[0].z*v[0].z;
		  float tmp1=r*rsqrt(tmp0);
		  p[0]=(float4)(
		       p[0].x+tmp1*(v[0].x*v[0].z*cphi - v[0].y*sphi),
		       p[0].y+tmp1*(v[0].y*v[0].z*cphi + v[0].x*sphi),
		       p[0].z-tmp1*tmp0*cphi                   ,
		       p[0].w
		  );
		  GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",v[0].x,v[0].y,v[0].z));
	      }else{
		  p[0].x+=r*cphi;
		  p[0].y+=r*sphi;
		  GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",v[0].x,v[0].y,v[0].z));
	      }
	      *idx1d=((int)(floor(p[0].z))*gcfg->dimlen.y+(int)(floor(p[0].y))*gcfg->dimlen.x+(int)(floor(p[0].x)));
	      if(p[0].x<0.f || p[0].y<0.f || p[0].z<0.f || p[0].x>=gcfg->maxidx.x || p[0].y>=gcfg->maxidx.y || p[0].z>=gcfg->maxidx.z){
		  *mediaid=0;
	      }else{
		  *mediaid=media[*idx1d];
	      }
#elif defined(MCX_SRC_CONE) || defined(MCX_SRC_ISOTROPIC) || defined(MCX_SRC_ARCSINE) 
	      // Uniform point picking on a sphere 
	      // http://mathworld.wolfram.com/SpherePointPicking.html
	      float ang,stheta,ctheta,sphi,cphi;
	      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
	      MCX_SINCOS(ang,sphi,cphi);
    #if defined(MCX_SRC_CONE) // a solid-angle section of a uniform sphere
		  do{
		      ang=(gcfg->srcparam1.y>0) ? TWO_PI*rand_uniform01(t) : MCX_MATHFUN(cos)(2.f*rand_uniform01(t)-1.f); //sine distribution
		  }while(ang>gcfg->srcparam1.x);
    #else
		  if(gcfg->srctype==MCX_SRC_ISOTROPIC) // uniform sphere
		      ang=MCX_MATHFUN(cos)(2.f*rand_uniform01(t)-1.f); //sine distribution
		  else
		      ang=ONE_PI*rand_uniform01(t); //uniform distribution in zenith angle, arcsine
    #endif
	      MCX_SINCOS(ang,stheta,ctheta);
	      rotatevector(v,stheta,ctheta,sphi,cphi);
              *Lmove=0.f;
#elif defined(MCX_SRC_ZGAUSSIAN)
	      float ang,stheta,ctheta,sphi,cphi;
	      ang=TWO_PI*rand_uniform01(t); //next arimuth angle
	      MCX_SINCOS(ang,sphi,cphi);
	      ang=sqrt(-2.f*log(rand_uniform01(t)))*(1.f-2.f*rand_uniform01(t))*gcfg->srcparam1.x;
	      MCX_SINCOS(ang,stheta,ctheta);
	      rotatevector(v,stheta,ctheta,sphi,cphi);
              *Lmove=0.f;
#elif defined(MCX_SRC_LINE) || defined(MCX_SRC_SLIT) 
	      float r=rand_uniform01(t);
	      p[0]=(float4)(p[0].x+r*gcfg->srcparam1.x,
				   p[0].y+r*gcfg->srcparam1.y,
				   p[0].z+r*gcfg->srcparam1.z,
				   p[0].w);
    #if defined(MCX_SRC_LINE)
		      float s,q;
		      r=1.f-2.f*rand_uniform01(t);
		      s=1.f-2.f*rand_uniform01(t);
		      q=sqrt(1.f-v[0].x*v[0].x-v[0].y*v[0].y)*(rand_uniform01(t)>0.5f ? 1.f : -1.f);
		      v[0]=(float4)(v[0].y*q-v[0].z*s,v[0].z*r-v[0].x*q,v[0].x*s-v[0].y*r,v[0].w);
              *Lmove=0.f;
    #else
              *Lmove=-1.f;
    #endif
              *prop=(float4)(prop[0].x+(gcfg->srcparam1.x)*0.5f,
	                 prop[0].y+(gcfg->srcparam1.y)*0.5f,
			 prop[0].z+(gcfg->srcparam1.z)*0.5f,0.f);
#endif
          /**
           * If beam focus is set, determine the incident angle
           */
          if(*Lmove<0.f && gcfg->c0.w!=0.f){
	        float Rn2=(gcfg->c0.w > 0.f) - (gcfg->c0.w < 0.f);
	        prop[0].x+=gcfg->c0.w*v[0].x;
		prop[0].y+=gcfg->c0.w*v[0].y;
		prop[0].z+=gcfg->c0.w*v[0].z;
                v[0].x=Rn2*(prop[0].x-p[0].x);
                v[0].y=Rn2*(prop[0].y-p[0].y);
                v[0].z=Rn2*(prop[0].z-p[0].z);
		Rn2=rsqrt(v[0].x*v[0].x+v[0].y*v[0].y+v[0].z*v[0].z); // normalize
                v[0].x*=Rn2;
                v[0].y*=Rn2;
                v[0].z*=Rn2;
	  }

          /**
           * Compute the reciprocal of the velocity vector
           */
//          *rv=float3(native_divide(1.f,v[0].x),native_divide(1.f,v[0].y),native_divide(1.f,v[0].z));

          /**
           * If a photon is launched outside of the box, or inside a zero-voxel, move it until it hits a non-zero voxel
           */
	  if((*mediaid & MED_MASK)==0){
             int idx=skipvoid(p, v, f, media, gproperty, gcfg); /** specular reflection of the bbx is taken care of here*/
             if(idx>=0){
		 *idx1d=idx;
		 *mediaid=media[*idx1d];
	     }
	  }
	  *w0+=1.f;
	  
	  /**
           * if launch attempted for over 1000 times, stop trying and return
           */
	  if(*w0>gcfg->maxvoidstep)
	     return -1;  // launch failed
      }while((*mediaid & MED_MASK)==0 || p[0].w<=gcfg->minenergy);
      
      /**
       * Now a photon is successfully launched, perform necssary initialization for a new trajectory
       */
      f[0].w+=1.f; 
      prop[0]=TOFLOAT4(gproperty[*mediaid & MED_MASK]); //always use mediaid to read gproperty[]
/*
      if(gcfg->debuglevel & MCX_DEBUG_MOVE)
          savedebugdata(p,(uint)f[0].w+threadid*gcfg->threadphoton+umin(threadid,(threadid<gcfg->oddphotons)*threadid),gdebugdata);
*/
      /**
        total energy enters the volume. for diverging/converting 
        beams, this is less than nphoton due to specular reflection 
        loss. This is different from the wide-field MMC, where the 
        total launched energy includes the specular reflection loss
       */
      *energylaunched+=p[0].w;
      *w0=p[0].w;
      v[0].w=EPS;
      *Lmove=0.f;
      
      /**
       * If a progress bar is needed, only sum completed photons from the 1st, last and middle threads to determine progress bar
       */

      if((gcfg->debuglevel & MCX_DEBUG_PROGRESS) && ((int)(f[0].w) & 1) && (threadid==0 || threadid==(int)(get_global_size(0) - 1)
          || threadid==(int)(get_global_size(0)>>1))) { ///< use the 1st, middle and last thread for progress report
          gprogress[0]++;
      }
      return 0;
}

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
__kernel void mcx_main_loop(const int nphoton, const int ophoton,__global const uint *media,
     __global float *field, __global float *genergy, __global uint *n_seed,
     __global float *n_det,__constant float4 *gproperty,__global float *srcpattern,
     __constant float4 *gdetpos, volatile __global uint *gprogress,__global uint *detectedphoton,
     __local float *sharedmem, __constant MCXParam *gcfg){

     int idx= get_global_id(0);

     float4 p={0.f,0.f,0.f,-1.f};  //{x,y,z}: x,y,z coordinates,{w}:packet weight
     float4 v=gcfg->c0;  //{x,y,z}: ix,iy,iz unitary direction vector, {w}:total scat event
     float4 f={0.f,0.f,0.f,0.f};  //f.w can be dropped to save register
     float  energyloss=genergy[idx<<1];
     float  energylaunched=genergy[(idx<<1)+1];

     uint idx1d, idx1dold;   //idx1dold is related to reflection

     uint   mediaid=gcfg->mediaidorig,mediaidold=0,isdet=0;
     float  w0, Lmove;
     float  n1;   //reflection var
     int flipdir=0;

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN];
     FLOAT4VEC prop;    //can become float2 if no reflection

     __local float *ppath=sharedmem+get_local_id(0)*gcfg->maxmedia;
     __local int   blockphoton[1];

#ifdef GROUP_LOAD_BALANCE
     if(get_local_id(0) == 0)
	blockphoton[0] = gcfg->blockphoton + ((int)get_group_id(0) < gcfg->blockextra);
     barrier(CLK_LOCAL_MEM_FENCE);
#endif

#ifdef  MCX_SAVE_DETECTORS
     if(gcfg->savedet) clearpath(ppath,gcfg);
#endif

     gpu_rng_init(t,n_seed,idx);

     if(launchnewphoton(&p,&v,&f,&prop,&idx1d,field,&mediaid,&w0,&Lmove,0,ppath,
		      &energyloss,&energylaunched,n_det,detectedphoton,t,gproperty,
		      media,srcpattern,gdetpos,gcfg,idx,nphoton,ophoton,blockphoton,gprogress)){
         n_seed[idx]=NO_LAUNCH;
         return;
     }
#ifdef GROUP_LOAD_BALANCE
     while(blockphoton[0]>0 || f.w<0.f) {
#else
     while(f.w<=nphoton + (idx<ophoton)) {
#endif
          GPUDEBUG(((__constant char*)"photonid [%d] L=%f w=%e medium=%d\n",(int)f.w,f.x,p.w,mediaid));

	  if(f.x<=0.f) {  // if this photon has finished the current jump
   	       f.x=rand_next_scatlen(t);

               GPUDEBUG(((__constant char*)"scat L=%f RNG=[%e %e %e] \n",f.x,rand_next_aangle(t),rand_next_zangle(t),rand_uniform01(t)));

	       if(v.w!=EPS){ //weight
                       //random arimuthal angle
                       float cphi,sphi,theta,stheta,ctheta;
                       float tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                       MCX_SINCOS(tmp0,sphi,cphi);
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
		           stheta=MCX_MATHFUN(sin)(theta);
		           ctheta=tmp0;
                       }else{
			   theta=acos(2.f*rand_next_zangle(t)-1.f);
                           MCX_SINCOS(theta,stheta,ctheta);
                       }
                       GPUDEBUG(((__constant char*)"scat theta=%f\n",theta));
                       rotatevector(&v,stheta,ctheta,sphi,cphi);
                       v.w+=1.f;
	       }
	       v.w=(int)v.w;
	  }

	  n1=prop.w;
	  prop=TOFLOAT4(gproperty[mediaid & MED_MASK]);

          FLOAT4VEC htime;            //reflection var
	  f.z=hitgrid(&p, &v, &htime, &flipdir);
	  float slen=f.z*prop.y;
	  slen=fmin(slen,f.x);
	  f.z=native_divide(slen,prop.y);

          GPUDEBUG(((__constant char*)"p=[%f %f %f] -> <%f %f %f>*%f -> hit=[%f %f %f] flip=%d\n",p.x,p.y,p.z,v.x,v.y,v.z,f.z,htime.x,htime.y,htime.z,flipdir));

#ifdef USE_HALF
	  p.xyz = (slen==f.x) ? p.xyz+(float3)(f.z)*v.xyz : convert_float3(htime.xyz);
#else
	  p.xyz = (slen==f.x) ? p.xyz+(float3)(f.z)*v.xyz : htime.xyz;
#endif
	  p.w*=MCX_MATHFUN(exp)(-prop.x*f.z);
	  f.x-=slen;
	  f.y+=f.z*prop.w*gcfg->oneoverc0;

          GPUDEBUG(((__constant char*)"update p=[%f %f %f] -> f.z=%f\n",p.x,p.y,p.z,f.z));

#ifdef MCX_SAVE_DETECTORS
          if(gcfg->savedet)
	      ppath[(mediaid & MED_MASK)-1]+=f.z; //(unit=grid)
#endif

          mediaidold=mediaid | isdet;
          idx1dold=idx1d;
          idx1d=((int)floor(p.z)*gcfg->dimlen.y+(int)floor(p.y)*gcfg->dimlen.x+(int)floor(p.x));
          GPUDEBUG(((__constant char*)"idx1d [%d]->[%d]\n",idx1dold,idx1d));
#ifdef MCX_SIMPLIFY_BRANCH
	  mediaid=(any(isless(p.xyz,(float3)(0.f))) || any(isgreaterequal(p.xyz,(gcfg->maxidx.xyz))));
	  idx1d=mediaid ? OUTSIDE_VOLUME : idx1d;
	  mediaid=mediaid ? 0 : media[idx1d];
          isdet=mediaid & DET_MASK;
          mediaid &= MED_MASK;
#else
          if(any(isless(p.xyz,(float3)(0.f))) || any(isgreaterequal(p.xyz,(gcfg->maxidx.xyz)))){
	      mediaid=0;
	      idx1d=OUTSIDE_VOLUME;
	  }else{
              mediaid=media[idx1d];
              isdet=mediaid & DET_MASK;
              mediaid &= MED_MASK;
          }
#endif

          GPUDEBUG(((__constant char*)"medium [%d]->[%d]\n",mediaidold,mediaid));

	  if(idx1d!=idx1dold && idx1dold>0 && mediaidold){
             GPUDEBUG(((__constant char*)"field add to %d->%f(%d)\n",idx1dold,w0-p.w,(int)f.w));
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(gcfg->save2pt && f.y>=gcfg->twin0 && f.y<gcfg->twin1){
                  GPUDEBUG(((__constant char*)"deposit to [%d] %e, w=%f\n",idx1dold,w0-p.w,p.w));
#ifndef USE_ATOMIC
                  field[idx1dold+(int)(floor((f.y-gcfg->twin0)*gcfg->Rtstep))*gcfg->dimlen.z]+=w0-p.w;
#else
                  int tshift=(int)(floor((f.y-gcfg->twin0)*gcfg->Rtstep));
		  float oldval=atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], w0-p.w);
		  if(oldval>MAX_ACCUM){
			if(atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], -oldval)<0.f)
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z], oldval);
			else
			    atomicadd(& field[idx1dold+tshift*gcfg->dimlen.z+gcfg->dimlen.w], oldval);
		  }
                  GPUDEBUG(((__constant char*)"atomic write to [%d] %e, w=%f\n",idx1dold,w0,p.w));
#endif
	     }
	     w0=p.w;
	  }

          if((mediaid==0 && (!gcfg->doreflect || (gcfg->doreflect && n1==gproperty[mediaid & MED_MASK].w))) || f.y>gcfg->twin1){
                  GPUDEBUG(((__constant char*)"direct relaunch at idx=[%d] mediaid=[%d], ref=[%d]\n",idx1d,mediaid,gcfg->doreflect));
		  if(launchnewphoton(&p,&v,&f,&prop,&idx1d,field,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),ppath,
		      &energyloss,&energylaunched,n_det,detectedphoton,t,gproperty,media,srcpattern,gdetpos,gcfg,idx,nphoton,ophoton,blockphoton,gprogress)){ 
                         break;
		  }
                  isdet=mediaid & DET_MASK;
                  mediaid &= MED_MASK;
                  continue;
          }
#ifdef MCX_DO_REFLECTION
          //if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one
          if(gcfg->doreflect && n1!=gproperty[mediaid & MED_MASK].w){
	          float Rtotal=1.f;
                  float cphi,sphi,stheta,ctheta;

                  *((float4*)(&prop))=gproperty[mediaid & MED_MASK]; // optical property across the interface

                  float tmp0=n1*n1;
                  float tmp1=prop.w*prop.w;
                  cphi=fabs( (flipdir==0) ? v.x : (flipdir==1 ? v.y : v.z)); // cos(si)
                  sphi=1.f-cphi*cphi;            // sin(si)^2

                  f.z=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2
	          GPUDEBUG(((__constant char*)"ref total ref=%f\n",f.z));

                  if(f.z>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*f.z;
                     stheta=2.f*n1*prop.w*cphi*sqrt(f.z);
                     Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*f.z;
       	       	     Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
		     GPUDEBUG(((__constant char*)"Rtotal=%f\n",Rtotal));
                  }
	          if(Rtotal<1.f && rand_next_reflect(t)>Rtotal){ // do transmission
                        transmit(&v,n1,prop.w,flipdir);
                        if(mediaid==0){ // transmission to external boundary
                            GPUDEBUG(((__constant char*)"transmit to air, relaunch\n"));
		    	    if(launchnewphoton(&p,&v,&f,&prop,&idx1d,field,&mediaid,&w0,&Lmove,(mediaidold & DET_MASK),
			        ppath,&energyloss,&energylaunched,n_det,detectedphoton,t,gproperty,media,srcpattern,gdetpos,gcfg,idx,nphoton,ophoton,blockphoton,gprogress)){
                                    break;
			    }
                            isdet=mediaid & DET_MASK;
                            mediaid &= MED_MASK;
			    continue;
			}
	                GPUDEBUG(((__constant char*)"do transmission\n"));
		  }else{ //do reflection
	                GPUDEBUG(((__constant char*)"do reflection\n"));
	                GPUDEBUG(((__constant char*)"ref faceid=%d p=[%f %f %f] v_old=[%f %f %f]\n",flipdir,p.x,p.y,p.z,v.x,v.y,v.z));
                	(flipdir==0) ? (v.x=-v.x) : ((flipdir==1) ? (v.y=-v.y) : (v.z=-v.z)) ;
			(flipdir==0) ?
        		    (p.x=mcx_nextafterf(convert_float_rte(p.x), (v.x > 0.f)-0.5f)) :
			    ((flipdir==1) ? 
				(p.y=mcx_nextafterf(convert_float_rte(p.y), (v.y > 0.f)-0.5f)) :
				(p.z=mcx_nextafterf(convert_float_rte(p.z), (v.z > 0.f)-0.5f)) );
	                GPUDEBUG(((__constant char*)"ref p_new=[%f %f %f] v_new=[%f %f %f]\n",p.x,p.y,p.z,v.x,v.y,v.z));
                	idx1d=idx1dold;
		 	mediaid=(media[idx1d] & MED_MASK);
			prop=TOFLOAT4(gproperty[mediaid]);
			n1=prop.w;
		  }
              }
#endif
     }
     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energylaunched;
}

