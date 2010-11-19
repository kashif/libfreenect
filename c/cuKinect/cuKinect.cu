#include <stdio.h>
#include <cutil_inline.h>

#include "libfreenect.h"

uint16_t t_gamma[2048];
uint8_t* gl_depth[2];

void depth_cb(freenect_device *dev, freenect_depth *depth, uint32_t timestamp)
{
    printf("called depth_cb()\n");
	int i;

	for (i=0; i<FREENECT_FRAME_PIX; i++) {
		int pval = t_gamma[depth[i]];
		int lb = pval & 0xff;
		switch (pval>>8) {
			case 0:
				gl_depth[0][3*i+0] = 255;
				gl_depth[0][3*i+1] = 255-lb;
				gl_depth[0][3*i+2] = 255-lb;
				break;
			case 1:
				gl_depth[0][3*i+0] = 255;
				gl_depth[0][3*i+1] = lb;
				gl_depth[0][3*i+2] = 0;
				break;
			case 2:
				gl_depth[0][3*i+0] = 255-lb;
				gl_depth[0][3*i+1] = 255;
				gl_depth[0][3*i+2] = 0;
				break;
			case 3:
				gl_depth[0][3*i+0] = 0;
				gl_depth[0][3*i+1] = 255;
				gl_depth[0][3*i+2] = lb;
				break;
			case 4:
				gl_depth[0][3*i+0] = 0;
				gl_depth[0][3*i+1] = 255-lb;
				gl_depth[0][3*i+2] = 255;
				break;
			case 5:
				gl_depth[0][3*i+0] = 0;
				gl_depth[0][3*i+1] = 0;
				gl_depth[0][3*i+2] = 255-lb;
				break;
			default:
				gl_depth[0][3*i+0] = 0;
				gl_depth[0][3*i+1] = 0;
				gl_depth[0][3*i+2] = 0;
				break;
		}
	}
}

int main(int argc, char** argv) 
{
    int devID = 0;
    cudaDeviceProp deviceProps;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
        cutilDeviceInit(argc, argv);
    } else {
        devID = cutGetMaxGflopsDeviceId();
        cutilSafeCall(cudaSetDevice(devID));
    }

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", 
           deviceProps.name, deviceProps.multiProcessorCount);
    
    
    freenect_context *f_ctx;
    freenect_device *f_dev;
	  
	if (freenect_init(&f_ctx, NULL) < 0) {
        printf("freenect_init() failed\n");
		exit(0);
	}

	int nr_devices = freenect_num_devices (f_ctx);
	printf ("Number of Kinect devices found: %d\n", nr_devices);

	if (nr_devices < 1)
		exit(0);

	if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
		printf("Could not open Kinect device\n");
		exit(0);
	}
	
	int i;
	for (i=0; i<2048; i++) {
		float v = i/2048.0;
		v = powf(v, 3)* 6;
		t_gamma[i] = v*6*256;
	}
	
    // allocate pinned buffers
	cudaHostAlloc(&(gl_depth[0]), sizeof(uint8_t)*640*480*4, 0);
	cudaHostAlloc(&(gl_depth[1]), sizeof(uint8_t)*640*480*4, 0);
	
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_depth_format(f_dev, FREENECT_FORMAT_11_BIT);
    
    freenect_start_depth(f_dev);
    
    int bufNum = 0;
    void *pGPUbuf[2];
    
    // allocate buffers
    cudaMalloc(&(pGPUbuf[0]), sizeof(uint8_t)*640*480*4);
    cudaMalloc(&(pGPUbuf[1]), sizeof(uint8_t)*640*480*4);
    
    // inf. loop
    while (freenect_process_events(f_ctx) >= 0) {
        printf("copy gl_depth[%d] to pGPUbuf[%d]\n", (bufNum+1)%2, bufNum);
        cudaMemcpyAsync(pGPUbuf[bufNum], gl_depth[(bufNum+1)%2], sizeof(uint8_t)*640*480*4, cudaMemcpyHostToDevice, 0);
        
        //kernel call here
        
        // get next frame... 
        printf("copy gl_depth[0] to gl_depth[%d]\n", bufNum);
        cudaMemcpy(gl_depth[bufNum], gl_depth[0], sizeof(uint8_t)*640*480*4, cudaMemcpyHostToHost);
        
        cudaThreadSynchronize();
        bufNum++; bufNum %=2;
    }
    
    // free buffers
    cudaFreeHost(gl_depth[0]);
    cudaFreeHost(gl_depth[1]);
    cudaFree(pGPUbuf[0]);
    cudaFree(pGPUbuf[1]);
    
    cudaThreadExit();
    cutilExit(argc, argv);
    return 0;
}
