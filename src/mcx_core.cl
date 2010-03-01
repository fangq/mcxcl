////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//
//  Reference (Fang2009):
//        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
//        Migration in 3D Turbid Media Accelerated by Graphics Processing 
//        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
//
//  mcx_core.cl: OpenCL kernels
//
//  License: GNU General Public License v3, see LICENSE.txt for details
//
////////////////////////////////////////////////////////////////////////////////

#ifdef __DEVICE_EMULATION__
#define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
#else
#define GPUDEBUG(x)
#endif


#define RAND_BUF_LEN       5        //register arrays
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define R_PI               0.318309886183791f
#define INIT_LOGISTIC      100

#define RAND_MAX 4294967295

#define ONE_PI             3.1415926535897932f     //pi
#define TWO_PI             6.28318530717959f       //2*pi
#define EPS                1e-10f                  //round-off limit

#define C0                 299792458000.f          //speed of light in mm/s
#define R_C0               3.335640951981520e-12f  //1/C0 in s/mm

#define VERY_BIG           1e10f                   //a big number
#define JUST_ABOVE_ONE     1.0001f                 //test for boundary
#define SAME_VOXEL         -9999.f                 //scatter within a voxel
#define MAX_PROP           256                     //maximum property number



#ifndef DOUBLE_PREC_LOGISTIC
  typedef float RandType;
  #define FUN(x)               (4.f*(x)*(1.f-(x)))
  #define NU 1e-8f
  #define NU2 (1.f-2.f*NU)
  #define MIN_INVERSE_LIMIT 1e-7f
  #define logistic_uniform(v)  (acos(1.f-2.f*(v))*R_PI)
  #define R_MAX_C_RAND       (1.f/RAND_MAX)
  #define LOG_MT_MAX         22.1807097779182f
#else
  typedef double RandType;
  #define FUN(x)               (4.0*(x)*(1.0-(x)))
  #define NU 1e-14
  #define NU2 (1.0-2.0*NU)
  #define MIN_INVERSE_LIMIT 1e-12
  #define logistic_uniform(v)  (acos(1.0-2.0*(v))*R_PI)
  #define R_MAX_C_RAND       (1./RAND_MAX)
  #define LOG_MT_MAX         22.1807097779182
#endif

#define RING_FUN(x,y,z)      (NU2*(x)+NU*((y)+(z)))


void logistic_step(RandType *t, RandType *tnew, int len_1){
    RandType tmp;
    t[0]=FUN(t[0]);
    t[1]=FUN(t[1]);
    t[2]=FUN(t[2]);
    t[3]=FUN(t[3]);
    t[4]=FUN(t[4]);
    tnew[3]=RING_FUN(t[0],t[4],t[1]);   /* shuffle the results by separation of 2*/
    tnew[4]=RING_FUN(t[1],t[0],t[2]);
    tnew[0]=RING_FUN(t[2],t[1],t[3]);
    tnew[1]=RING_FUN(t[3],t[2],t[4]);
    tnew[2]=RING_FUN(t[4],t[3],t[0]);
    tmp =t[0];
    t[0]=t[2];
    t[2]=t[4];
    t[4]=t[1];
    t[1]=t[3];
    t[3]=tmp;
}
// generate random number for the next zenith angle
void rand_need_more(RandType t[RAND_BUF_LEN],RandType tbuf[RAND_BUF_LEN]){
    logistic_step(t,tbuf,RAND_BUF_LEN-1);
    logistic_step(tbuf,t,RAND_BUF_LEN-1);
}

void logistic_init(RandType *t,RandType *tnew,__global uint seed[],uint idx){
     int i;
     for(i=0;i<RAND_BUF_LEN;i++)
           t[i]=(RandType)seed[idx*RAND_BUF_LEN+i]*R_MAX_C_RAND;

     for(i=0;i<INIT_LOGISTIC;i++)  /*initial randomization*/
           rand_need_more(t,tnew);
}
// transform into [0,1] random number
RandType rand_uniform01(RandType v){
    return logistic_uniform(v);
}
void gpu_rng_init(RandType t[RAND_BUF_LEN], RandType tnew[RAND_BUF_LEN],__global uint *n_seed,int idx){
    logistic_init(t,tnew,n_seed,idx);
}
// generate [0,1] random number for the next scattering length
float rand_next_scatlen(RandType t[RAND_BUF_LEN]){
    RandType ran=rand_uniform01(t[0]);
    return ((ran==0.f)?LOG_MT_MAX:(-log(ran)));
}
// generate [0,1] random number for the next arimuthal angle
float rand_next_aangle(RandType t[RAND_BUF_LEN]){
    return rand_uniform01(t[2]);
}
// generate random number for the next zenith angle
float rand_next_zangle(RandType t[RAND_BUF_LEN]){
    return rand_uniform01(t[4]);
}

/*
the following is the definition for a photon state:

typedef struct PhotonData {
  float4 pos;  //{x,y,z}: x,y,z coordinates,{w}:packet weight
  float4 dir;  //{x,y,z}: ix,iy,iz unitary direction vector,{w}:total scat event
  float4 len;  //{x}:remaining probability
               //{y}:accumulative time (on-photon timer)
               //{z}:next accum. time (accum. happens at a fixed time interval)
               //{w}:total completed photons
  uint   seed; // random seed, length may different for different RNGs
} Photon;

*/


// optical properties saved in the constant memory
// {x}:mua,{y}:mus,{z}:anisotropy (g),{w}:refraction index (n)
//__constant float4 gproperty[MAX_PROP];


// tested with texture memory for media, only improved 1% speed
// to keep code portable, use global memory for now
// also need also change all media[idx1d] to tex1Dfetch() below
//texture<uchar, 1, cudaReadModeElementType> texmedia;


//need to move these arguments to the constant memory, as they use shared memory

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
__kernel void mcx_main_loop( const int nphoton, const int ophoton,__global const uchar media[],
     __global float field[], __global float genergy[], const float minstep, 
     float twin0, float twin1,  float tmax,  uint4 dimlen, 
     uchar isrowmajor,  uchar save2pt,  float Rtstep,
     const float4 p0, const float4 c0, const float4 maxidx,
     const uchar doreflect, const uchar doreflect3, 
     const float minenergy,  const float sradius2, __global uint n_seed[],__global float4 n_pos[],
     __global float4 n_dir[],__global float4 n_len[],__constant float4 gproperty[],__global uint stopsign[1]){

     int idx= get_global_id(0);

     float4 npos=n_pos[idx];  //{x,y,z}: x,y,z coordinates,{w}:packet weight
     float4 ndir=n_dir[idx];  //{x,y,z}: ix,iy,iz unitary direction vector, {w}:total scat event
                              //ndir.w can be dropped to save register
     float4 nlen=n_len[idx];  //nlen.w can be dropped to save register
     float4 npos0;            //reflection var, to save pre-reflection npos state
     float4 htime;            //reflection var
     float  minaccumtime=minstep*R_C0;   //can be moved to constant memory
     float  energyloss=genergy[idx<<1];
     float  energyabsorbed=genergy[(idx<<1)+1];

     int i,idx1d, idx1dold,idxorig;   //idx1dold is related to reflection
     //int np=nphoton+((idx==get_local_size(0)*get_num_groups(0)-1) ? ophoton: 0);

#ifdef TEST_RACING
     int cc=0;
#endif
     uchar  mediaid, mediaidorig;
     char   medid=-1;
     float  atten;         //can be taken out to minimize registers
     float  flipdir,n1,Rtotal;   //reflection var

     //for MT RNG, these will be zero-length arrays and be optimized out
     RandType t[RAND_BUF_LEN],tnew[RAND_BUF_LEN];
     float4 prop;    //can become float2 if no reflection

     float len,cphi,sphi,theta,stheta,ctheta,tmp0,tmp1;
     float accumweight=0.f;

     GPUDEBUG(("(%f %f %f %d)\n",n_pos[0].x,n_pos[0].y,n_pos[0].z,n_seed[0]));

     gpu_rng_init(t,tnew,n_seed,idx);

     // assuming the initial position is within the domain (mcx_config is supposed to ensure)
     idx1d=isrowmajor?(int)(floor(npos.x)*dimlen.y+floor(npos.y)*dimlen.x+floor(npos.z)):\
                      (int)(floor(npos.z)*dimlen.y+floor(npos.y)*dimlen.x+floor(npos.x));
     idxorig=idx1d;
     mediaid=media[idx1d];
     mediaidorig=mediaid;
	  
     if(mediaid==0) {
          return; // the initial position is not within the medium
     }
     prop=gproperty[mediaid];

     /*
      using a while-loop to terminate a thread by np will cause MT RNG to be 3.5x slower
      LL5 RNG will only be slightly slower than for-loop with photon-move criterion
     */
     //while(nlen.w<np) {

     for(i=0;i<nphoton;i++){ // here nphoton actually means photon moves

          GPUDEBUG(("*i= (%d) L=%f w=%e a=%f\n",(int)nlen.w,nlen.x,npos.w,nlen.y));
	  if(nlen.x<=0.f) {  // if this photon has finished the current jump
               rand_need_more(t,tnew);
   	       nlen.x=rand_next_scatlen(t);

               GPUDEBUG(("next scat len=%20.16e \n",nlen.x));
	       if(npos.w<1.f){ //weight
                       //random arimuthal angle
                       tmp0=TWO_PI*rand_next_aangle(t); //next arimuth angle
                       sphi=sincos(tmp0,&cphi);
                       GPUDEBUG(("next angle phi %20.16e\n",tmp0));

                       //Henyey-Greenstein Phase Function, "Handbook of Optical Biomedical Diagnostics",2002,Chap3,p234
                       //see Boas2002

                       if(prop.w>EPS){  //if prop.w is too small, the distribution of theta is bad
		           tmp0=(1.f-prop.w*prop.w)/(1.f-prop.w+2.f*prop.w*rand_next_zangle(t));
		           tmp0*=tmp0;
		           tmp0=(1.f+prop.w*prop.w-tmp0)/(2.f*prop.w);

                           // when ran=1, CUDA will give me 1.000002 for tmp0 which produces nan later
                           // detected by Ocelot,thanks to Greg Diamos,see http://bit.ly/cR2NMP
                           tmp0=max(-1.f, min(1.f, tmp0));

		           theta=acos(tmp0);
		           stheta=sin(theta);
		           ctheta=tmp0;
                       }else{  //Wang1995 has acos(2*ran-1), rather than 2*pi*ran, need to check
			   theta=ONE_PI*rand_next_zangle(t);
                           stheta=sincos(theta,&ctheta);
                       }
                       GPUDEBUG(("next scat angle theta %20.16e\n",theta));

		       if( ndir.z>-1.f+EPS && ndir.z<1.f-EPS ) {
		           tmp0=1.f-ndir.z*ndir.z;   //reuse tmp to minimize registers
		           tmp1=rsqrt(tmp0);
		           tmp1=stheta*tmp1;
		           ndir=(float4)(
				tmp1*(ndir.x*ndir.z*cphi - ndir.y*sphi) + ndir.x*ctheta,
				tmp1*(ndir.y*ndir.z*cphi + ndir.x*sphi) + ndir.y*ctheta,
				-tmp1*tmp0*cphi                         + ndir.z*ctheta,
				ndir.w
			   );
                           GPUDEBUG(("new dir: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z));
		       }else{
			   ndir=(float4)(stheta*cphi,stheta*sphi,(ndir.z>0.f)?ctheta:-ctheta,ndir.w);
                           GPUDEBUG(("new dir-z: %10.5e %10.5e %10.5e\n",ndir.x,ndir.y,ndir.z));
 		       }
                       ndir.w++;
	       }
	  }

          n1=prop.z;
	  prop=gproperty[mediaid];
	  len=minstep*prop.y; //Wang1995: minstep*(prop.x+prop.y)

          npos0=npos;
	  if(len>nlen.x){  //scattering ends in this voxel: mus*minstep > s 
               tmp0=nlen.x/prop.y;
	       energyabsorbed+=npos.w;
	       npos.xyz+=ndir.xyz*tmp0;
               npos.w=npos.w*exp(-prop.x*tmp0);
   	       //npos=(float4)(npos.x+ndir.x*tmp0,npos.y+ndir.y*tmp0,npos.z+ndir.z*tmp0,
               //            npos.w*exp(-prop.x*tmp0));
	       energyabsorbed-=npos.w;
	       nlen.x=SAME_VOXEL;
	       nlen.y+=tmp0*prop.z*R_C0;  // accumulative time
               GPUDEBUG((">>ends in voxel %f<%f %f [%d]\n",nlen.x,len,prop.y,idx1d));
	  }else{                      //otherwise, move minstep
	       energyabsorbed+=npos.w;
               if(mediaid!=medid){
                  atten=exp(-prop.x*minstep);
               }
               npos.xyz+=ndir.xyz;
               npos.w*=atten;
   	       //npos=(float4)(npos.x+ndir.x,npos.y+ndir.y,npos.z+ndir.z,npos.w*atten);
               medid=mediaid;
	       energyabsorbed-=npos.w;
	       nlen.x-=len;     //remaining probability: sum(s_i*mus_i)
	       nlen.y+=minaccumtime*prop.z; //total time
               GPUDEBUG((">>keep going %f<%f %f [%d] %e %e\n",nlen.x,len,prop.y,idx1d,nlen.y,nlen.z));
	  }

          idx1dold=idx1d;
          idx1d=isrowmajor?(int)(floor(npos.x)*dimlen.y+floor(npos.y)*dimlen.x+floor(npos.z)):\
                           (int)(floor(npos.z)*dimlen.y+floor(npos.y)*dimlen.x+floor(npos.x));
          GPUDEBUG(("old and new voxel: %d<->%d\n",idx1dold,idx1d));
          if(npos.x<0||npos.y<0||npos.z<0||npos.x>=maxidx.x||npos.y>=maxidx.y||npos.z>=maxidx.z){
	      mediaid=0;
	  }else{
              mediaid=media[idx1d];
          }

          //if hit the boundary, exceed the max time window or exit the domain, rebound or launch a new one
	  if(mediaid==0||nlen.y>tmax||nlen.y>twin1){
              flipdir=0.f;
              if(doreflect) {
                //time-of-flight to hit the wall in each direction
                htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floor(npos0.x)+(ndir.x>0.f)-npos0.x)/ndir.x:VERY_BIG;
                htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floor(npos0.y)+(ndir.y>0.f)-npos0.y)/ndir.y:VERY_BIG;
                htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floor(npos0.z)+(ndir.z>0.f)-npos0.z)/ndir.z:VERY_BIG;
                //get the direction with the smallest time-of-flight
                tmp0=fmin(fmin(htime.x,htime.y),htime.z);
                flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));

                //move to the 1st intersection pt
                tmp0*=JUST_ABOVE_ONE;
                htime.x=floor(npos0.x+tmp0*ndir.x);
       	        htime.y=floor(npos0.y+tmp0*ndir.y);
       	        htime.z=floor(npos0.z+tmp0*ndir.z);

                if(htime.x>=0&&htime.y>=0&&htime.z>=0&&htime.x<maxidx.x&&htime.y<maxidx.y&&htime.z<maxidx.z){
                    if( media[isrowmajor?(int)(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           (int)(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)]){ //hit again

                     GPUDEBUG((" first try failed: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime.x,htime.y,htime.z,
                           media[isrowmajor?(int)(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                           (int)(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)], maxidx.x, maxidx.y,maxidx.z));

                     htime.x=(ndir.x>EPS||ndir.x<-EPS)?(floor(npos.x)+(ndir.x<0.f)-npos.x)/(-ndir.x):VERY_BIG;
                     htime.y=(ndir.y>EPS||ndir.y<-EPS)?(floor(npos.y)+(ndir.y<0.f)-npos.y)/(-ndir.y):VERY_BIG;
                     htime.z=(ndir.z>EPS||ndir.z<-EPS)?(floor(npos.z)+(ndir.z<0.f)-npos.z)/(-ndir.z):VERY_BIG;
                     tmp0=fmin(fmin(htime.x,htime.y),htime.z);
                     tmp1=flipdir;   //save the previous ref. interface id
                     flipdir=(tmp0==htime.x?1.f:(tmp0==htime.y?2.f:(tmp0==htime.z&&idx1d!=idx1dold)?3.f:0.f));

                     if(doreflect3){
                       tmp0*=JUST_ABOVE_ONE;
                       htime.x=floor(npos.x-tmp0*ndir.x); //move to the last intersection pt
                       htime.y=floor(npos.y-tmp0*ndir.y);
                       htime.z=floor(npos.z-tmp0*ndir.z);

                       if(tmp1!=flipdir&&htime.x>=0&&htime.y>=0&&htime.z>=0&&htime.x<maxidx.x&&htime.y<maxidx.y&&htime.z<maxidx.z){
                           if(! media[isrowmajor?(int)(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                                  (int)(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)]){ //this is an air voxel

                               GPUDEBUG((" second try failed: [%.1f %.1f,%.1f] %d (%.1f %.1f %.1f)\n",htime.x,htime.y,htime.z,
                                   media[isrowmajor?(int)(htime.x*dimlen.y+htime.y*dimlen.x+htime.z):\
                                   (int)(htime.z*dimlen.y+htime.y*dimlen.x+htime.x)], maxidx.x, maxidx.y,maxidx.z));

                               /*to compute the remaining interface, we used the following fact to accelerate: 
                                 if there exist 3 intersections, photon must pass x/y/z interface exactly once,
                                 we solve the coeff of the following equation to find the last interface:
                                    a*1+b*2+c=3
       	       	       	       	    a*1+b*3+c=2 -> [a b c]=[-1 -1 6], this will give the remaining interface id
       	       	       	       	    a*2+b*3+c=1
                               */
                               flipdir=-tmp1-flipdir+6.f;
                           }
                       }
                     }
                  }
                }
              }

              prop=gproperty[mediaid];

              GPUDEBUG(("->ID%d J%d C%d tlen %e flip %d %.1f!=%.1f dir=%f %f %f pos=%f %f %f\n",idx,(int)ndir.w,
                  (int)nlen.w,nlen.y, (int)flipdir, n1,prop.z,ndir.x,ndir.y,ndir.z,npos.x,npos.y,npos.z));

              //recycled some old register variables to save memory
	      //if hit boundary within the time window and is n-mismatched, rebound

              if(doreflect&&nlen.y<tmax&&nlen.y<twin1&& flipdir>0.f && n1!=prop.z&&npos.w>minenergy){
                  tmp0=n1*n1;
                  tmp1=prop.z*prop.z;
                  if(flipdir>=3.f) { //flip in z axis
                     cphi=fabs(ndir.z);
                     sphi=ndir.x*ndir.x+ndir.y*ndir.y;
                     ndir.z=-ndir.z;
                  }else if(flipdir>=2.f){ //flip in y axis
                     cphi=fabs(ndir.y);
       	       	     sphi=ndir.x*ndir.x+ndir.z*ndir.z;
                     ndir.y=-ndir.y;
                  }else if(flipdir>=1.f){ //flip in x axis
                     cphi=fabs(ndir.x);                //cos(si)
                     sphi=ndir.y*ndir.y+ndir.z*ndir.z; //sin(si)^2
                     ndir.x=-ndir.x;
                  }
		  energyabsorbed+=npos.w-npos0.w;
                  npos=npos0;   //move back
                  idx1d=idx1dold;
                  len=1.f-tmp0/tmp1*sphi;   //1-[n1/n2*sin(si)]^2
	          GPUDEBUG((" ref len=%f %f+%f=%f w=%f\n",len,cphi,sphi,cphi*cphi+sphi,npos.w));

                  if(len>0.f) {
                     ctheta=tmp0*cphi*cphi+tmp1*len;
                     stheta=2.f*n1*prop.z*cphi*sqrt(len);
                     Rtotal=(ctheta-stheta)/(ctheta+stheta);
       	       	     ctheta=tmp1*cphi*cphi+tmp0*len;
       	       	     Rtotal=(Rtotal+(ctheta-stheta)/(ctheta+stheta))*0.5f;
	             GPUDEBUG(("  dir=%f %f %f htime=%f %f %f Rs=%f\n",ndir.x,ndir.y,ndir.z,htime.x,htime.y,htime.z,Rtotal));
	             GPUDEBUG(("  ID%d J%d C%d flip=%3f (%d %d) cphi=%f sphi=%f npos=%f %f %f npos0=%f %f %f\n",
                         idx,(int)ndir.w,(int)nlen.w,
	                 flipdir,idx1dold,idx1d,cphi,sphi,npos.x,npos.y,npos.z,npos0.x,npos0.y,npos0.z));
		     energyloss+=(1.f-Rtotal)*npos.w; //energy loss due to reflection
                     npos.w*=Rtotal;
                  } // else, total internal reflection, no loss
                  mediaid=media[idx1d];
                  prop=gproperty[mediaid];
                  n1=prop.z;
                  //ndir.w++;
              }else{  // launch a new photon
#ifdef MCX_CPU_ONLY
		  if(stopsign[0]) break;
#endif
                  energyloss+=npos.w;  // sum all the remaining energy
	          npos=p0;
	          ndir=c0;
	          nlen=(float4)(0.f,0.f,minaccumtime,nlen.w+1);
                  idx1d=idxorig;
		  mediaid=mediaidorig;
              }
	  }else if(nlen.y>=nlen.z){
             GPUDEBUG(("field add to %d->%f(%d)  t(%e)>t0(%e)\n",idx1d,npos.w,(int)nlen.w,nlen.y,nlen.z));
             // if t is within the time window, which spans cfg->maxgate*cfg->tstep wide
             if(save2pt&&nlen.y>=twin0 & nlen.y<twin1){
#ifdef TEST_RACING
                  // enable TEST_RACING to determine how many missing accumulations due to race
                  if( (npos.x-p0.x)*(npos.x-p0.x)+(npos.y-p0.y)*(npos.y-p0.y)+(npos.z-p0.z)*(npos.z-p0.z)>sradius2) {
                      field[idx1d+(int)(floor((nlen.y-twin0)*Rtstep))*dimlen.z]+=1.f;
		      cc++;
                  }
#else
  #ifndef USE_ATOMIC
                  // set sradius2 to only start depositing energy when dist^2>sradius2 
                  if(sradius2>EPS){
                      if((npos.x-p0.x)*(npos.x-p0.x)+(npos.y-p0.y)*(npos.y-p0.y)+(npos.z-p0.z)*(npos.z-p0.z)>sradius2){
                          field[idx1d+(int)(floor((nlen.y-twin0)*Rtstep))*dimlen.z]+=npos.w;
                      }else{
                          accumweight+=npos.w*prop.x; // weight*absorption
                      }
                  }else{
                      field[idx1d+(int)(floor((nlen.y-twin0)*Rtstep))*dimlen.z]+=npos.w;
                  }
  #else
                  // ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
//		  atomicFloatAdd(& field[idx1d+(int)(floor((nlen.y-twin0)*Rtstep))*dimlen.z], npos.w);
  #endif
#endif
	     }
             nlen.z+=minaccumtime; // fluence is a temporal-integration
	  }
     }
     // accumweight saves the total absorbed energy in the sphere r<sradius.
     // in non-atomic mode, accumweight is more accurate than saving to the grid
     // as it is not influenced by race conditions.
     // now I borrow nlen.z to pass this value back

     nlen.z=accumweight;

     genergy[idx<<1]=energyloss;
     genergy[(idx<<1)+1]=energyabsorbed;

#ifdef TEST_RACING
     n_seed[idx]=cc;
#endif
     n_pos[idx]=npos;
     n_dir[idx]=ndir;
     n_len[idx]=nlen;
}

