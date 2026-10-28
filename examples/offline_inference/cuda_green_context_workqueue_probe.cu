#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

void check(cudaError_t status, const char* operation) {
  std::cout << operation << ": " << cudaGetErrorName(status) << " ("
            << static_cast<int>(status) << ") - "
            << cudaGetErrorString(status) << '\n';
  if (status != cudaSuccess) {
    std::exit(1);
  }
}

__global__ void write_value(int* output, int value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *output = value;
  }
}

}  // namespace

int main() {
  constexpr int device = 0;
  check(cudaSetDevice(device), "cudaSetDevice");
  check(cudaFree(nullptr), "initialize primary context");

  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
  std::cout << "device: " << properties.name << '\n';
  std::cout << "compute capability: " << properties.major << '.'
            << properties.minor << '\n';

  cudaDevResource all_sms{};
  check(cudaDeviceGetDevResource(device, &all_sms, cudaDevResourceTypeSm),
        "query device SM resource");
  std::cout << "SM resource: count=" << all_sms.sm.smCount
            << " min_partition=" << all_sms.sm.minSmPartitionSize
            << " coscheduled_alignment="
            << all_sms.sm.smCoscheduledAlignment << '\n';

  cudaDevResource device_workqueues{};
  check(cudaDeviceGetDevResource(device, &device_workqueues,
                                 cudaDevResourceTypeWorkqueueConfig),
        "query device workqueue config");
  std::cout << "device WQ config: concurrency_limit="
            << device_workqueues.wqConfig.wqConcurrencyLimit
            << " sharing_scope="
            << static_cast<int>(device_workqueues.wqConfig.sharingScope)
            << '\n';

  cudaDevSmResourceGroupParams split_params{};
  split_params.smCount = all_sms.sm.smCoscheduledAlignment;

  cudaDevResource latency_sms{};
  cudaDevResource bulk_sms{};
  check(cudaDevSmResourceSplit(&latency_sms, 1, &all_sms, &bulk_sms, 0,
                               &split_params),
        "split SM resource");
  std::cout << "SM partition: latency=" << latency_sms.sm.smCount
            << " bulk=" << bulk_sms.sm.smCount << '\n';

  cudaDevResource balanced_workqueue{};
  balanced_workqueue.type = cudaDevResourceTypeWorkqueueConfig;
  balanced_workqueue.wqConfig.device = device;
  balanced_workqueue.wqConfig.sharingScope =
      cudaDevWorkqueueConfigScopeGreenCtxBalanced;
  balanced_workqueue.wqConfig.wqConcurrencyLimit = 2;

  cudaDevResource latency_pack[2] = {latency_sms, balanced_workqueue};
  cudaDevResource bulk_pack[2] = {bulk_sms, balanced_workqueue};
  cudaDevResourceDesc_t latency_desc{};
  cudaDevResourceDesc_t bulk_desc{};
  check(cudaDevResourceGenerateDesc(&latency_desc, latency_pack, 2),
        "generate latency SM+WQ descriptor");
  check(cudaDevResourceGenerateDesc(&bulk_desc, bulk_pack, 2),
        "generate bulk SM+WQ descriptor");

  cudaExecutionContext_t latency_ctx{};
  cudaExecutionContext_t bulk_ctx{};
  check(cudaGreenCtxCreate(&latency_ctx, latency_desc, device, 0),
        "create latency green context");
  check(cudaGreenCtxCreate(&bulk_ctx, bulk_desc, device, 0),
        "create bulk green context");

  cudaDevResource latency_wq{};
  cudaDevResource bulk_wq{};
  check(cudaExecutionCtxGetDevResource(
            latency_ctx, &latency_wq, cudaDevResourceTypeWorkqueueConfig),
        "query latency GC workqueue config");
  check(cudaExecutionCtxGetDevResource(
            bulk_ctx, &bulk_wq, cudaDevResourceTypeWorkqueueConfig),
        "query bulk GC workqueue config");
  std::cout << "latency GC WQ: concurrency_limit="
            << latency_wq.wqConfig.wqConcurrencyLimit << " sharing_scope="
            << static_cast<int>(latency_wq.wqConfig.sharingScope) << '\n';
  std::cout << "bulk GC WQ: concurrency_limit="
            << bulk_wq.wqConfig.wqConcurrencyLimit << " sharing_scope="
            << static_cast<int>(bulk_wq.wqConfig.sharingScope) << '\n';

  cudaStream_t latency_stream{};
  cudaStream_t bulk_stream{};
  check(cudaExecutionCtxStreamCreate(&latency_stream, latency_ctx,
                                     cudaStreamNonBlocking, 0),
        "create latency GC stream");
  check(cudaExecutionCtxStreamCreate(&bulk_stream, bulk_ctx,
                                     cudaStreamNonBlocking, 0),
        "create bulk GC stream");

  int* latency_output = nullptr;
  int* bulk_output = nullptr;
  check(cudaMalloc(&latency_output, sizeof(int)), "allocate latency output");
  check(cudaMalloc(&bulk_output, sizeof(int)), "allocate bulk output");
  write_value<<<1, 1, 0, latency_stream>>>(latency_output, 13);
  check(cudaPeekAtLastError(), "launch latency GC kernel");
  write_value<<<1, 1, 0, bulk_stream>>>(bulk_output, 31);
  check(cudaPeekAtLastError(), "launch bulk GC kernel");
  check(cudaStreamSynchronize(latency_stream), "synchronize latency GC stream");
  check(cudaStreamSynchronize(bulk_stream), "synchronize bulk GC stream");

  int latency_value = 0;
  int bulk_value = 0;
  check(cudaMemcpy(&latency_value, latency_output, sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy latency output");
  check(cudaMemcpy(&bulk_value, bulk_output, sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy bulk output");
  std::cout << "kernel outputs: latency=" << latency_value
            << " bulk=" << bulk_value << '\n';

  check(cudaFree(latency_output), "free latency output");
  check(cudaFree(bulk_output), "free bulk output");
  check(cudaStreamDestroy(latency_stream), "destroy latency GC stream");
  check(cudaStreamDestroy(bulk_stream), "destroy bulk GC stream");
  check(cudaExecutionCtxDestroy(latency_ctx), "destroy latency GC");
  check(cudaExecutionCtxDestroy(bulk_ctx), "destroy bulk GC");

  if (latency_value != 13 || bulk_value != 31) {
    std::cerr << "FAIL: unexpected kernel output\n";
    return 2;
  }
  std::cout << "PASS: CUDA 13.1 balanced workqueue Green Contexts are usable.\n";
  return 0;
}
