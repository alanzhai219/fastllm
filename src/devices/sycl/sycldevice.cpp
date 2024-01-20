//
// Created by alanzhai219 on 1/16/24.
//

#include "device/cpu/cpudevice.h"
#include "device/sycl/sycldevice.h"

#include "fastllm-sycl.h"
#include "utils.h"

namespace fastllm {
    SyclDevice::SyclDevice() {
        this->deviceType = "sycl";
        this->ops["Attension"] = (BaseOperator*)(new SyclAttension());
        this->ops["CopyKVCache"] = (BaseOperator*)(new SyclCopyKVCacheOp());
        this->ops["LayerNorm"] = (BaseOperator*)(new SyclLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new SyclRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new SyclLinearOp());
        this->ops["Split"] = (BaseOperator*)(new SyclSplitOp());
        this->ops["CatDirect"] = (BaseOperator*)(new SyclCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new SyclMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new SyclMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new SyclSoftMaxOp());
        this->ops["GeluNew"] = (BaseOperator*)(new SyclGeluNewOp());
        this->ops["Silu"] = (BaseOperator*)(new SyclSiluOp());
        this->ops["Swiglu"] = (BaseOperator*)(new SyclSwigluOp());
        this->ops["Mul"] = (BaseOperator*)(new SyclMulOp());
        this->ops["AddTo"] = (BaseOperator*)(new SyclAddToOp());
        this->ops["MulTo"] = (BaseOperator*)(new SyclMulToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new SyclAttentionMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new SyclAlibiMaskOp());
        this->ops["TopK"] = (BaseOperator*)(new SyclTopKOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new SyclPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new SyclRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new SyclNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new SyclLlamaRotatePosition2DOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new SyclApplyLognAttnOp());

        this->ops["SplitBatch"] = (BaseOperator*)(new SyclSplitBatchOp());
        this->ops["CatBatch"] = (BaseOperator*)(new SyclCatBatchOp());
        this->ops["MulBatch"] = (BaseOperator*)(new SyclMulBatchOp());
        this->ops["MatMulBatch"] = (BaseOperator*)(new SyclMatMulBatchOp());
        this->ops["MatMulTransBBatch"] = (BaseOperator*)(new SyclMatMulTransBBatchOp());
        this->ops["SoftMaxBatch"] = (BaseOperator*)(new SyclSoftmaxBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator*)(new SyclCatDirectBatchOp());
        this->ops["AttentionBatch"] = (BaseOperator*)(new SyclAttentionBatchOp());
    }

    bool SyclDevice::Malloc(void **ret, size_t size) {

    }
}
