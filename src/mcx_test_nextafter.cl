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

#ifdef USE_HALF
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
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
} MCXParam __attribute__ ((aligned (32)));

float mcx_nextafter1(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a+1000.f;
      num.i+=dir ^ (num.i & 0x80000000U);
      return num.f-1000.f;
}

float mcx_nextafter2(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a;
      ((num.i & 0x7FFFFFFFU)==0) ? num.i=((dir & 0x80000000U) | 1) : ((num.i & 0x80000000U) ? (num.i-=dir) : (num.i+=dir) );
      return num.f;
}

#ifdef USE_HALF
half mcx_nextafter2_half(const half a, short dir){
      union{
          half f;
          short i;
      } num;
      num.f=a;
      ((num.i & 0x7FFFU)==0) ? num.i =(((dir & 0x8000U) ) | 1) : ((num.i & 0x8000U) ? (num.i-=dir) : (num.i+=dir) );
      return num.f;
}
#endif

void test_nextafter(float v){
      printf("input: %e\n",v);
      printf("lib v+1.f: %e\tv-1.f: %e\tv+0: %e\n", nextafter(v,v+1.f),nextafter(v,v-1.f),nextafter(v,v+0.f));
      printf("my1 v+1.f: %e\tv-1.f: %e\tv+0: %e\n", mcx_nextafter1(v,1),mcx_nextafter1(v,-1),mcx_nextafter1(v,0));
      printf("my2 v+1.f: %e\tv-1.f: %e\tv+0: %e\n", mcx_nextafter2(v,1),mcx_nextafter2(v,-1),mcx_nextafter2(v,0));
#ifdef USE_HALF
      printf("m2h v+1.f: %e\tv-1.f: %e\tv+0: %e\n", convert_float(mcx_nextafter2_half(convert_half(v),1)),
                                                    convert_float(mcx_nextafter2_half(convert_half(v),-1)),
						    convert_float(mcx_nextafter2_half(convert_half(v),0)));
#endif
}


__kernel void mcx_main_loop(const int nphoton, const int ophoton,__global const uint *media,
     __global float *field, __global float *genergy, __global uint *n_seed,
     __global float *n_det,__constant float4 *gproperty,
     __constant float4 *gdetpos, __global uint *stopsign,__global uint *detectedphoton,
     __local float *sharedmem, __constant MCXParam *gcfg){
     
     int idx= get_global_id(0);

     if(idx==0){
        test_nextafter(15.2f);
	test_nextafter(1.f);
	test_nextafter(1.2f);
	test_nextafter(0.2f);
	test_nextafter(0.f);
	test_nextafter(-0.f);
	test_nextafter(-0.1f);
	test_nextafter(-2.2f);
	test_nextafter(-100.f);   
     }
}