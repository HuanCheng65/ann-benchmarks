#include <cuda_runtime.h>

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kKiB = 1024ULL;
constexpr std::size_t kMiB = 1024ULL * kKiB;
constexpr std::size_t kGiB = 1024ULL * kMiB;

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t status__ = (call);                                               \
    if (status__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s failed at %s:%d: %s\n", #call, __FILE__,         \
                   __LINE__, cudaGetErrorString(status__));                      \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct Region {
  void *host_ptr = nullptr;
  void *device_ptr = nullptr;
  std::size_t bytes = 0;
};

struct Options {
  int device = 0;
  std::size_t max_bytes = 0;
  std::vector<std::size_t> step_bytes = {8ULL * kGiB, 1ULL * kGiB, 256ULL * kMiB,
                                         64ULL * kMiB};
};

std::string format_bytes(std::size_t bytes) {
  char buf[128];
  if (bytes >= kGiB) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB",
                  static_cast<double>(bytes) / static_cast<double>(kGiB));
  } else if (bytes >= kMiB) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB",
                  static_cast<double>(bytes) / static_cast<double>(kMiB));
  } else if (bytes >= kKiB) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB",
                  static_cast<double>(bytes) / static_cast<double>(kKiB));
  } else {
    std::snprintf(buf, sizeof(buf), "%zu B", bytes);
  }
  return std::string(buf);
}

std::size_t parse_size_arg(const char *text) {
  char *end = nullptr;
  errno = 0;
  double value = std::strtod(text, &end);
  if (errno != 0 || end == text || value <= 0.0) {
    std::fprintf(stderr, "Invalid size argument: %s\n", text);
    std::exit(2);
  }

  while (*end == ' ') {
    ++end;
  }

  double multiplier = 1.0;
  if (*end == '\0' || std::strcmp(end, "b") == 0 || std::strcmp(end, "B") == 0) {
    multiplier = 1.0;
  } else if (std::strcmp(end, "k") == 0 || std::strcmp(end, "kb") == 0 ||
             std::strcmp(end, "K") == 0 || std::strcmp(end, "KB") == 0) {
    multiplier = static_cast<double>(kKiB);
  } else if (std::strcmp(end, "m") == 0 || std::strcmp(end, "mb") == 0 ||
             std::strcmp(end, "M") == 0 || std::strcmp(end, "MB") == 0) {
    multiplier = static_cast<double>(kMiB);
  } else if (std::strcmp(end, "g") == 0 || std::strcmp(end, "gb") == 0 ||
             std::strcmp(end, "G") == 0 || std::strcmp(end, "GB") == 0) {
    multiplier = static_cast<double>(kGiB);
  } else if (std::strcmp(end, "kib") == 0 || std::strcmp(end, "KiB") == 0) {
    multiplier = static_cast<double>(kKiB);
  } else if (std::strcmp(end, "mib") == 0 || std::strcmp(end, "MiB") == 0) {
    multiplier = static_cast<double>(kMiB);
  } else if (std::strcmp(end, "gib") == 0 || std::strcmp(end, "GiB") == 0) {
    multiplier = static_cast<double>(kGiB);
  } else {
    std::fprintf(stderr, "Unsupported size suffix in: %s\n", text);
    std::exit(2);
  }

  long double bytes = static_cast<long double>(value) * multiplier;
  if (bytes <= 0.0L ||
      bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
    std::fprintf(stderr, "Size out of range: %s\n", text);
    std::exit(2);
  }
  return static_cast<std::size_t>(bytes);
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--device") {
      opts.device = std::atoi(need_value("--device"));
    } else if (arg == "--max") {
      opts.max_bytes = parse_size_arg(need_value("--max"));
    } else if (arg == "--steps") {
      opts.step_bytes.clear();
      std::string text = need_value("--steps");
      std::size_t start = 0;
      while (start < text.size()) {
        std::size_t comma = text.find(',', start);
        std::string token = text.substr(start, comma == std::string::npos
                                                   ? std::string::npos
                                                   : comma - start);
        if (!token.empty()) {
          opts.step_bytes.push_back(parse_size_arg(token.c_str()));
        }
        if (comma == std::string::npos) {
          break;
        }
        start = comma + 1;
      }
      if (opts.step_bytes.empty()) {
        std::fprintf(stderr, "--steps produced no valid sizes\n");
        std::exit(2);
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: cuda_host_register_probe [--device N] [--max SIZE]\n"
          << "                               [--steps SIZE,SIZE,...]\n"
          << "Examples:\n"
          << "  ./cuda_host_register_probe --device 2\n"
          << "  ./cuda_host_register_probe --device 2 --max 128GiB\n"
          << "  ./cuda_host_register_probe --steps 4GiB,1GiB,256MiB,64MiB\n";
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  return opts;
}

void print_memlock_limit() {
  struct rlimit limit {};
  if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
    std::perror("getrlimit(RLIMIT_MEMLOCK)");
    return;
  }
  if (limit.rlim_cur == RLIM_INFINITY) {
    std::cout << "RLIMIT_MEMLOCK soft: unlimited\n";
  } else {
    std::cout << "RLIMIT_MEMLOCK soft: " << format_bytes(limit.rlim_cur) << '\n';
  }
  if (limit.rlim_max == RLIM_INFINITY) {
    std::cout << "RLIMIT_MEMLOCK hard: unlimited\n";
  } else {
    std::cout << "RLIMIT_MEMLOCK hard: " << format_bytes(limit.rlim_max) << '\n';
  }
}

__global__ void touch_pages_kernel(std::uint8_t *base, std::size_t num_pages,
                                   std::size_t page_size,
                                   unsigned long long *checksum) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_pages) {
    return;
  }
  std::uint8_t value = base[idx * page_size];
  atomicAdd(checksum, static_cast<unsigned long long>(value));
}

bool try_register_chunk(std::size_t chunk_bytes, std::size_t page_size,
                        std::vector<Region> *regions,
                        std::size_t *running_total) {
  void *host_ptr = mmap(nullptr, chunk_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_ptr == MAP_FAILED) {
    std::cout << "mmap(" << format_bytes(chunk_bytes)
              << ") failed: " << std::strerror(errno) << '\n';
    return false;
  }

  auto *bytes = static_cast<std::uint8_t *>(host_ptr);
  const std::size_t num_pages = chunk_bytes / page_size;
  const unsigned int thread_count_hint = std::thread::hardware_concurrency();
  const std::size_t thread_count = std::max<std::size_t>(1, thread_count_hint);
  std::vector<std::thread> init_threads;
  init_threads.reserve(thread_count);
  for (std::size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    init_threads.emplace_back([=]() {
      for (std::size_t i = thread_idx; i < num_pages; i += thread_count) {
        bytes[i * page_size] = static_cast<std::uint8_t>(i);
      }
    });
  }
  for (auto &thread : init_threads) {
    thread.join();
  }

  cudaError_t status = cudaHostRegister(
      host_ptr, chunk_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
  if (status != cudaSuccess) {
    std::cout << "cudaHostRegister(" << format_bytes(chunk_bytes)
              << ") failed: " << cudaGetErrorString(status) << '\n';
    munmap(host_ptr, chunk_bytes);
    return false;
  }

  void *device_ptr = nullptr;
  status = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (status != cudaSuccess) {
    std::cout << "cudaHostGetDevicePointer(" << format_bytes(chunk_bytes)
              << ") failed: " << cudaGetErrorString(status) << '\n';
    cudaHostUnregister(host_ptr);
    munmap(host_ptr, chunk_bytes);
    return false;
  }

  unsigned long long *checksum = nullptr;
  CHECK_CUDA(cudaMalloc(&checksum, sizeof(*checksum)));
  CHECK_CUDA(cudaMemset(checksum, 0, sizeof(*checksum)));

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((num_pages + kThreads - 1) / kThreads);
  touch_pages_kernel<<<blocks, kThreads>>>(
      static_cast<std::uint8_t *>(device_ptr), num_pages, page_size, checksum);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }

  unsigned long long checksum_value = 0;
  if (status == cudaSuccess) {
    CHECK_CUDA(cudaMemcpy(&checksum_value, checksum, sizeof(checksum_value),
                          cudaMemcpyDeviceToHost));
  }
  CHECK_CUDA(cudaFree(checksum));

  if (status != cudaSuccess) {
    std::cout << "GPU touch failed for " << format_bytes(chunk_bytes)
              << ": " << cudaGetErrorString(status) << '\n';
    cudaHostUnregister(host_ptr);
    munmap(host_ptr, chunk_bytes);
    return false;
  }

  regions->push_back(Region{host_ptr, device_ptr, chunk_bytes});
  *running_total += chunk_bytes;
  std::cout << "registered +" << format_bytes(chunk_bytes) << ", total "
            << format_bytes(*running_total) << ", checksum " << checksum_value
            << '\n';
  return true;
}

void cleanup_regions(const std::vector<Region> &regions) {
  for (auto it = regions.rbegin(); it != regions.rend(); ++it) {
    cudaError_t status = cudaHostUnregister(it->host_ptr);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaHostUnregister failed during cleanup: %s\n",
                   cudaGetErrorString(status));
    }
    if (munmap(it->host_ptr, it->bytes) != 0) {
      std::perror("munmap");
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);

  Options opts = parse_args(argc, argv);
  const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));

  std::cout << "page size: " << format_bytes(page_size) << '\n';
  print_memlock_limit();

  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CHECK_CUDA(cudaSetDevice(opts.device));

  cudaDeviceProp prop {};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, opts.device));
  std::cout << "device " << opts.device << ": " << prop.name << '\n';
  std::cout << "canMapHostMemory=" << prop.canMapHostMemory
            << ", unifiedAddressing=" << prop.unifiedAddressing << '\n';
  if (!prop.canMapHostMemory) {
    std::cerr << "This GPU cannot map host memory into the device address space.\n";
    return 1;
  }

  std::vector<Region> regions;
  std::size_t total = 0;

  for (std::size_t step : opts.step_bytes) {
    step = (step / page_size) * page_size;
    if (step == 0) {
      continue;
    }
    while (true) {
      if (opts.max_bytes != 0 && total >= opts.max_bytes) {
        break;
      }

      std::size_t request = step;
      if (opts.max_bytes != 0 && total + request > opts.max_bytes) {
        request = ((opts.max_bytes - total) / page_size) * page_size;
      }
      if (request == 0) {
        break;
      }

      if (!try_register_chunk(request, page_size, &regions, &total)) {
        break;
      }
    }
  }

  std::cout << "max simultaneously registered + mapped host memory: "
            << format_bytes(total) << '\n';

  cleanup_regions(regions);
  return 0;
}
