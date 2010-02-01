#ifndef _MCEXTREME_AUTO_GENERATED_H_
#define _MCEXTREME_AUTO_GENERATED_H_

/**************************************************************************** 
                                                                              
Copyright (c) 2003, Stanford University                                       
All rights reserved.                                                          
                                                                              
Copyright (c) 2008, Advanced Micro Devices, Inc.                              
All rights reserved.                                                          
                                                                              
                                                                              
The BRCC portion of BrookGPU is derived from the cTool project                
(http://ctool.sourceforge.net) and distributed under the GNU Public License.  
                                                                              
Additionally, see LICENSE.ctool for information on redistributing the         
contents of this directory.                                                   
                                                                              
****************************************************************************/ 

#include "brook/Stream.h" 
#include "brook/KernelInterface.h" 

//! Kernel declarations
class __mcx_main_loop
{
    public:
        void operator()(const int  totalmove, const ::brook::Stream< uchar >& media, const ::brook::Stream<  float >& field, const ::brook::Stream< float3 >& gproperty, const float3  vsize, const float  minstep, const float  lmax, const float  gg, const float  gg2, const float  ggx2, const float  one_add_gg2, const float  one_sub_gg2, const float  one_sub_gg, const float4  p0, const float4  c0, const float3  maxidx, const uint3  cp0, const uint3  cp1, const float4  cachebox, const uchar  doreflect, const ::brook::Stream< float3 >& seed, const ::brook::Stream< float4 >& pos, const ::brook::Stream< float4 >& dir, const ::brook::Stream< float3 >& len, const ::brook::Stream<  float3 >& n_seed, const ::brook::Stream<  float4 >& n_pos, const ::brook::Stream<  float4 >& n_dir, const ::brook::Stream<  float3 >& n_len);
        EXTENDCLASS();
};
extern __THREAD__ __mcx_main_loop mcx_main_loop;

#endif // _MCEXTREME_AUTO_GENERATED_H_

