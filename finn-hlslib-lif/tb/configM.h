#define KERNEL_DIM1 7
#define IFM_Channels11 1
#define IFMDim11 28
#define OFM_Channels11 16
#define OFMDim11 13
#define SIMD11 1
#define PE11 8
#define TILE11 98 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define KERNEL_DIM2 7
#define IFM_Channels12 16
#define IFMDim12 13
#define OFM_Channels12 24
#define OFMDim12 11
#define SIMD12 4
#define PE12 4
#define TILE12 1176 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define KERNEL_DIM3 7
#define IFM_Channels13 24
#define IFMDim13 11
#define OFM_Channels13 32
#define OFMDim13 4
#define SIMD13 4
#define PE13 8
#define TILE13 1176 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define KERNEL_DIM4 3
#define IFM_Channels14 32
#define IFMDim14 4
#define OFM_Channels14 2
#define OFMDim14 2
#define SIMD14 16
#define PE14 2
#define TILE14 18 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define KERNEL_DIM5 3
#define IFM_Channels15 128
#define IFMDim15 10
#define OFM_Channels15 16
#define OFMDim15 8
#define SIMD15 16
#define PE15 4
#define TILE15 288 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define KERNEL_DIM6 7
#define IFM_Channels16 16
#define IFMDim16 8
#define OFM_Channels16 6
#define OFMDim16 2
#define SIMD16 16
#define PE16 3
#define TILE16 98 // = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

#define SIMDSP1 1
#define MMV1 1 
#define WIDTH 32
#define _I 4 
#define STRIDE 1 
#define INPUT_PRECISION 1 
#define ACTIVATION_PRECISION 1 
#define DECAY1 0.9
