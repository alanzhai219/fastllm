#include <sycl/sycl.hpp>
#include <vector>
#include <stdio.h>
#include <chrono>

#include "fastllm-sycl.h"
#include "fastllm.h"

auto sycl_error_handler = [](sycl::exception_list exceptions){
    for (const std::exception_ptr &e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch(const sycl::exception &e) {
            std::cout << "Caught SYCL exception: " << e.what() << "\n";
        }
        catch(const std::exception &e) {
            std::cout << "Caught std exception: " << e.what() << "\n";
        }
        catch(...) {
            std::cout << "Caught unknown exception\n";
        }
    }
};

struct SyclMemoryBuffer {
    SyclMemoryBuffer() {}
    SyclMemoryBuffer(void* data, size_t size, bool busy)
        :data(data), size(size), busy(busy) {}

    void *data;
    size_t size;
    bool busy;
};

std::map<int, std::vector<SyclMemoryBuffer>> syclBuffersMap;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector<SyclMemoryBuffer>> bigBuffersMap;

void* FastllmSyclDirectMalloc(size_t size) {
    void *ret;
    const auto cur_queue = FastllmSyclGetQueue();
    ret = sycl::malloc_device(size, cur_queue);
    if (!ret) {
        std::cout << "Error: SYCL error when allocating " << (size >> 10) << " kB memory! Maybe there is no enough memory left on device.\n";
        return nullptr;
    }
    return ret;
}

void FastllmSyclDirectFree(void* ret) {
    auto cur_queue = FastllmSyclGetQueue();
    sycl::free(ret, cur_queue);
}

void* FastllmSyclMalloc(size_t size) {
    auto cur_queue = FastllmSyclGetQueue();
    sycl::device dev = cur_queue.get_device();
    auto id = dev.get_info<sycl::ext::intel::info::device::device_id>();
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffersMap.size(); ++i) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 1 * 1024 * 1024) {
                    if (selId == -1 && bigBuffers[selId].size > bigBuffers[i].size) {
                        selId = i;
                    }
                }
        }

        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void* ret = sycl::malloc_device(size, cur_queue);
        if (!ret) {
            std::cout << "Error: SYCL error when allocating " << (size >> 10) << " kB memory! Maybe there is no enough memory left on device.\n";
            return nullptr;
        }
        bigBuffers.push_back(SyclMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &syclBuffers = syclBuffersMap[id];
    for (int i = 0; i < syclBuffers.size(); ++i) {
        if (syclBuffers[i].size >= size && !syclBuffers[i].busy) {
            syclBuffers[i].busy = true;
            noBusyCnt[id] -= syclBuffers[i].size;
            return syclBuffers[i].data;
        }
    }
    void* ret = sycl::malloc_device(size, cur_queue);
    if (!ret) {
        std::cout << "Error: SYCL error when allocating " << (size >> 10) << " kB memory! Maybe there is no enough memory left on device.\n";
        return nullptr;
    }
    syclBuffers.push_back(SyclMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmSyclFree(void* ret) {
    if (ret == nullptr) {
        return;
    }
    if (syclBuffersMap.empty()) {
        return;
    }
    auto cur_queue = FastllmSyclGetQueue();
    sycl::device dev = cur_queue.get_device();
    int id = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::device_id>());
    for (auto &it : syclBuffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &syclBuffers = it.second;
            std::vector<SyclMemoryBuffer> temp;
            for (int i = 0; i < syclBuffers.size(); ++i) {
                if (!syclBuffers[i].busy) {
                    sycl::free(syclBuffers[i].data, cur_queue);
                }
                else {
                    temp.push_back(syclBuffers[i]);
                }
            }
            syclBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }
    for (auto &it: syclBuffersMap) {
        auto &syclBuffers = it.second;
        for (int i = 0; i < syclBuffers.size(); i++) {
            if (syclBuffers[i].data == ret) {
                noBusyCnt[it.first] += syclBuffers[i].size;
                syclBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    sycl::free(ret, cur_queue);
}

void FastllmSyclMallocBigBuffer(size_t size) {
    void* ret;
    auto cur_queue = FastllmSyclGetQueue();
    sycl::device dev = cur_queue.get_device();
    int id = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::device_id>());
    auto &bigBuffers = bigBuffersMap[id];
    ret = sycl::malloc_device(size, cur_queue);
    if (!ret) {
        std::cout << "Error: SYCL error when allocating " << (size >> 10) << " kB memory! Maybe there is no enough memory left on device.\n";
    }
    bigBuffers.push_back(SyclMemoryBuffer(ret, size, false));
}

void FastllmSyclClearBigBuffer() {
    auto cur_queue = FastllmSyclGetQueue();
    sycl::device dev = cur_queue.get_device();
    int id = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::device_id>());
    if (bigBuffersMap.empty()) {
        return;
    }
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector<SyclMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); ++i) {
            if (!bigBuffers[i].busy) {
                sycl::free(bigBuffers[i].data, cur_queue);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
}

void FastllmSyclCopyFromHostToDevice(void* dst, void* src, size_t size) {
    auto cur_queue = FastllmSyclGetQueue();
    cur_queue.memcpy(dst, src, size);
}

void FastllmSyclCopyFromDeviceToHost(void* dst, void* src, size_t size) {
    auto cur_queue = FastllmSyclGetQueue();
    cur_queue.memcpy(dst, src, size);
}

// TODO
void FastllmSyclCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    printf("No supported\n");
    return;
}

// TODO
void FastllmSyclMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    printf("No supported\n");
    return;
}

// TODO
void FastllmCudaMemcpy2DDeviceToDevice(void* dst, size_t dpitch, const void* src,
                                       size_t spitch, size_t width, size_t height) {
    printf("No supported\n");
    return;
}

// TODO
void FastllmSyclMemcpy2DDeviceToDeviceBatch(void** dsts, size_t* dpitchs, void** srcs,
                                       size_t* spitchs, size_t* widths, size_t*	heights,
                                       int batch) {
    printf("No supported\n");
    return;
}

void* FastllmSyclPrepareInput(const fastllm::Data &input) {
    void* ret;
    if (input.dataDevice == fastllm::DataDevice::SYCL) {
        ret = (void*)input.syclData;
    } else {
        ret = (void*)(input.expansionBytes);
        auto cur_queue = FastllmSyclGetQueue();
        cur_queue.memcpy(ret, input.cpuData, input.expansionBytes);
    }
    return ret;
}

void FastllmSyclFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::SYCL) {
        FastllmSyclFree(data);
    }
}

void *FastllmSyclPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::SYCL) {
        ret = (float*)output.syclData;
    } else {
        ret = (float*)FastllmSyclMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmSyclFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::SYCL) {
        auto cur_queue = FastllmSyclGetQueue();
        cur_queue.memcpy(output.cpuData, data, output.expansionBytes);
        FastllmSyclFree(data);
    }
}

void FastllmSiluKernel(float* a, float* b, int len, sycl::nd_item<3> it) {
    size_t work_item_idx = it.get_local_id(2); 
    size_t work_group_idx = it.get_group(2);
    size_t work_group_size = it.get_local_range().get(2);
    size_t idx = work_item_idx + work_group_idx * work_group_size;
    if (idx < len) {
        float x = a[idx];
        // TODO
        // b[idx] = x / (1.0 * sycl::native::expf(-x));
        b[idx] = x / (1.0 * expf(-x));
    }
}

bool FastllmSyclSilu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *syclInput = (float *)FastllmSyclPrepareInput(input);
    float *syclOutput = (float *)FastllmSyclPrepareOutput(output);

    int threadPerBlock = std::min(256, len);
    // FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    // TODO
    auto loc_num = (len - 1) / threadPerBlock + 1;
    sycl::range<3> glb(loc_num * threadPerBlock, 1, 1);
    sycl::range<3> loc(threadPerBlock, 1, 1);
    sycl::nd_range<3> nd_parallel(glb, loc); 
    auto cur_queue = FastllmSyclGetQueue();
    cur_queue.parallel_for(nd_parallel, [=](sycl::nd_item<3> it){ FastllmSiluKernel(syclInput, syclOutput, len, it); });

    FastllmSyclFinishInput(input, syclInput);
    FastllmSyclFinishOutput(output, syclOutput);
    return true;
}
class sycl_device_mgr {
public:
    // 获取单实例对象
    static sycl_device_mgr &getInstance() {
        // 局部静态特性的方式实现单实例
        static sycl_device_mgr inst;
        return inst;
    }

    void set_platform(const sycl::platform& plt) {
        m_plt = plt;
    }

    sycl::platform get_platform() const {
        return m_plt;
    }
    
    void set_device(const sycl::device& dev) {
        m_dev = dev;
    }

    sycl::device get_device() const {
        return m_dev;
    }
    
    void set_queue(const sycl::queue& queue) {
        m_queue = queue;
    }

    sycl::queue get_queue() const {
        return m_queue;
    }
    
private:
    // 禁止外部构造
    sycl_device_mgr();

    // 禁止外部析构
    ~sycl_device_mgr();

    // 禁止外部复制构造
    sycl_device_mgr(const sycl_device_mgr &inst);

    // 禁止外部赋值操作
    const sycl_device_mgr &operator=(const sycl_device_mgr &inst);
private:
    sycl::platform m_plt;
    sycl::device m_dev;
    sycl::queue m_queue;
};

void FastllmSyclSetPlatform(int plt_id) {
   auto platforms = sycl::platform::get_platforms(); 
   sycl_device_mgr::getInstance().set_platform(platforms[plt_id]);
}

void FastllmSyclSetDevice(int gpu_id) {
    auto plt = sycl_device_mgr::getInstance().get_platform();
    auto devices = plt.get_devices();
    sycl_device_mgr::getInstance().set_device(devices[gpu_id]);
}

void FastllmSyclSetQueue() {
    auto dev = sycl_device_mgr::getInstance().get_device();
    sycl::queue q(dev, sycl_error_handler);
    sycl_device_mgr::getInstance().set_queue(q);
}

sycl::queue FastllmSyclGetQueue() {
    return sycl_device_mgr::getInstance().get_queue();
}