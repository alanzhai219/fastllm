#include <sycl/sycl.hpp>
#include "fastllm.h"

#ifdef  __cplusplus
extern "C" {
#endif
// void FastllmInitCublas(void);

void FastllmSyclMallocBigBuffer(size_t size);
void FastllmSyclClearBigBuffer();
void* FastllmSyclMalloc(size_t size);
void FastllmSyclFree(void *ret);
void* FastllmSyclDirectMalloc(size_t size);
void FastllmSyclDirectFree(void *ret);

void FastllmSyclCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmSyclCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmSyclCopyFromDeviceToDevice(void *dst, void *src, size_t size);
void FastllmSyclMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);

void FastllmSyclMemcpy2DDeviceToDevice(void* dst, size_t dpitch, const void* src,
                                       size_t spitch, size_t width, size_t height);
void FastllmSyclMemcpy2DDeviceToDeviceBatch(void** dsts, size_t* dpitchs, void** srcs,
                                       size_t* spitchs, size_t* widths, size_t*	heights,
                                       int batch);
bool FastllmSyclAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale);
bool FastllmSyclGeluNew(const fastllm::Data &input, fastllm::Data &output);
bool FastllmSyclSilu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmSyclSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmSyclMul(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmSyclSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
bool FastllmSyclAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmSyclMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmSyclAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmSyclAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmSyclRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmSyclLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmSyclTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmSyclPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmSyclMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmSyclMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmSyclMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmSyclMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmSyclMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmSyclBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool FastllmSyclBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool FastllmSyclRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmSyclNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmSyclLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmSyclApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds);

bool FastllmSyclAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                          fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch);
bool FastllmSyclSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis);
bool FastllmSyclCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis);
bool FastllmSyclMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs);
bool FastllmSyclSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch);
bool FastllmSyclBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                      int *ns, int *ms, int *ks,
                                      int *i0Strides, int *i1Strides, float alpha, int batch);
bool FastllmSyclBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch);
void FastllmSyclSetPlatform(int plt_id);
void FastllmSyclSetDevice(int gpu_id);
void FastllmSyclSetQueue();
sycl::queue FastllmSyclGetQueue();

void* FastllmSyclPrepareInput(const fastllm::Data& input);
void FastllmSyclFinishInput(const fastllm::Data& input, void* data);
void* FastllmSyclPrepareOutput(const fastllm::Data& output);
void FastllmSyclFinishOutput(const fastllm::Data& output, void* data);
#ifdef  __cplusplus
}
#endif

