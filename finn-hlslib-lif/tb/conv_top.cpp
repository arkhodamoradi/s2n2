/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *
 *  \file conv_top.cpp
 *
 *  HLS Top function with a single convolutional layer for unit testing
 *
 *****************************************************************************/
#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"

#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
#include "memdataM.h"
#include "configM.h"
#include "fclayer.h"

/* Changes:
 * 1- we use smaller SIMD size: SIMDSP
 * 2- we have new weight class and it loads sparse weights based on input
 * 3- we use a new acc class that does not need the input */

void Testbench_conv(stream<ap_uint<IFM_Channels11*INPUT_PRECISION> > & in, stream<ap_uint<OFM_Channels16*ACTIVATION_PRECISION> > & out, unsigned int numReps){
#pragma HLS DATAFLOW
	stream<ap_uint<OFM_Channels11*ACTIVATION_PRECISION> > out1;
	stream<ap_uint<OFM_Channels12*ACTIVATION_PRECISION> > out2;
	stream<ap_uint<OFM_Channels13*ACTIVATION_PRECISION> > out3;
	//stream<ap_uint<OFM_Channels13*ACTIVATION_PRECISION> > out4;
	//stream<ap_uint<OFM_Channels13*ACTIVATION_PRECISION> > out5;
	ConvLayer_Batch<KERNEL_DIM1, IFM_Channels11, IFMDim11, OFM_Channels11, OFMDim11, SIMD11, SIMDSP1, PE11, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(in, out1, PARAM::weights1, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM2, IFM_Channels12, IFMDim12, OFM_Channels12, OFMDim12, SIMD12, SIMDSP1, PE12, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out1, out2, PARAM::weights2, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM3, IFM_Channels13, IFMDim13, OFM_Channels13, OFMDim13, SIMD13, SIMDSP1, PE13, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out2, out3, PARAM::weights3, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM4, IFM_Channels14, IFMDim14, OFM_Channels14, OFMDim14, SIMD14, SIMDSP1, PE14, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out3, out, PARAM::weights4, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	//ConvLayer_Batch<KERNEL_DIM5, IFM_Channels15, IFMDim15, OFM_Channels15, OFMDim15, SIMD15, SIMDSP1, PE15, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out4, out5, PARAM::weights5, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	//ConvLayer_Batch<KERNEL_DIM6, IFM_Channels16, IFMDim16, OFM_Channels16, OFMDim16, SIMD16, SIMDSP1, PE16, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out5, out, PARAM::weights6, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
}


/*
// rf1
void Testbench_conv(stream<ap_uint<IFM_Channels11*INPUT_PRECISION> > & in, stream<ap_uint<OFM_Channels14*ACTIVATION_PRECISION> > & out, unsigned int numReps){
#pragma HLS DATAFLOW
	stream<ap_uint<OFM_Channels11*ACTIVATION_PRECISION> > out1;
	stream<ap_uint<OFM_Channels12*ACTIVATION_PRECISION> > out2;
	stream<ap_uint<OFM_Channels13*ACTIVATION_PRECISION> > out3;
	ConvLayer_Batch<KERNEL_DIM1, IFM_Channels11, IFMDim11, OFM_Channels11, OFMDim11, SIMD11, SIMDSP1, PE11, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(in, out1, PARAM::weights1, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM2, IFM_Channels12, IFMDim12, OFM_Channels12, OFMDim12, SIMD12, SIMDSP1, PE12, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out1, out2, PARAM::weights2, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM3, IFM_Channels13, IFMDim13, OFM_Channels13, OFMDim13, SIMD13, SIMDSP1, PE13, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out2, out3, PARAM::weights3, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
	ConvLayer_Batch<KERNEL_DIM4, IFM_Channels14, IFMDim14, OFM_Channels14, OFMDim14, SIMD14, SIMDSP1, PE14, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity, ap_fixed<WIDTH, _I>, WIDTH, _I >(out3, out, PARAM::weights4, ThresholdActivation<ap_fixed<WIDTH, _I>>(0), numReps, DECAY1, ap_resource_dsp());
}

*/
