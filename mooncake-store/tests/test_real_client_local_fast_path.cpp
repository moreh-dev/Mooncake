// Unit tests for RealClient::try_local_disk_fast_path (T1, Plan 017).
//
// Tests the same-node LOCAL_DISK fast path helper in isolation without a
// live Master or TransferEngine.
//
// Key design choice:
//   A single FileStorage instance is shared between the "writer" (uses its
//   internal storage_backend_ via the LocalFastPathTest friend declaration
//   added to file_storage.h) and the fast-path reader.  This avoids the
//   BucketIdGenerator divergence that occurs when two independent backends
//   are created at the same directory.
//
//   RealClient is constructed with the default ctor and its file_storage_
//   and local_rpc_addr members (both accessible without friend access) are
//   injected before calling try_local_disk_fast_path.

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "file_storage.h"
#include "real_client.h"
#include "replica.h"
#include "storage_backend.h"
#include "types.h"

namespace mooncake {
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const std::string kLocalAddr = "127.0.0.1:59001";

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

// LocalFastPathTest is declared a friend of FileStorage (file_storage.h),
// allowing direct access to the private storage_backend_ member for writing
// test data through the same internal backend that BatchGet will read from.
class LocalFastPathTest : public ::testing::Test {
   protected:
    std::string data_dir_;
    std::shared_ptr<FileStorage> file_storage_;

    void SetUp() override {
        google::InitGoogleLogging("LocalFastPathTest");
        FLAGS_logtostderr = true;

        data_dir_ = (fs::temp_directory_path() /
                     ("lfp_test_" + std::to_string(getpid())))
                        .string();
        fs::create_directories(data_dir_);

        auto cfg = FileStorageConfig::FromEnvironment();
        cfg.storage_filepath = data_dir_;
        cfg.local_buffer_size = 128 * 1024 * 1024;
        // client_ == nullptr: no master RPCs. BatchGet / ReleaseBuffer are
        // safe. We skip Init() to avoid background heartbeat threads.
        file_storage_ = std::make_shared<FileStorage>(cfg, nullptr, kLocalAddr);

        // Initialise the storage backend (metadata tables, file scanner).
        file_storage_->storage_backend_->Init();
    }

    void TearDown() override {
        file_storage_.reset();
        fs::remove_all(data_dir_);
        google::ShutdownGoogleLogging();
    }

    // Write a key/value pair directly into FileStorage's own storage_backend_.
    // Because LocalFastPathTest is a friend of FileStorage we can access the
    // private storage_backend_ field.
    void WriteKey(const std::string& key, const std::string& value) {
        std::vector<char> buf(value.begin(), value.end());
        std::vector<Slice> slices;
        slices.push_back(Slice{buf.data(), buf.size()});

        std::unordered_map<std::string, std::vector<Slice>> batched;
        batched[key] = slices;

        auto& backend = file_storage_->storage_backend_;
        backend->BatchOffload(batched,
                              [](const std::vector<std::string>&,
                                 const std::vector<StorageObjectMetadata>&) {
                                  return ErrorCode::OK;
                              });
    }

    // Build a LocalDiskDescriptor with the given transport_endpoint.
    static LocalDiskDescriptor MakeLD(const std::string& endpoint,
                                      size_t sz = 0) {
        LocalDiskDescriptor ld;
        ld.client_id = generate_uuid();
        ld.object_size = sz;
        ld.transport_endpoint = endpoint;
        return ld;
    }

    // Invoke try_local_disk_fast_path with an injected RealClient.
    bool CallFastPath(const std::string& key, const LocalDiskDescriptor& ld,
                      void* dst, size_t size,
                      std::vector<tl::expected<int64_t, ErrorCode>>& results,
                      size_t idx,
                      const std::string& client_rpc_addr = kLocalAddr,
                      bool use_null_fs = false,
                      std::shared_ptr<FileStorage> fs_override = nullptr) {
        RealClient client;
        client.local_rpc_addr = client_rpc_addr;
        if (use_null_fs) {
            client.file_storage_ = nullptr;
        } else {
            client.file_storage_ = fs_override ? fs_override : file_storage_;
        }
        Slice dst_slice{dst, size};
        return client.try_local_disk_fast_path(key, ld, dst_slice, size,
                                               results, idx);
    }
};

// ---------------------------------------------------------------------------
// Test 1: same-node + host buffer + valid file_storage -> succeeds, bit-exact
// ---------------------------------------------------------------------------

TEST_F(LocalFastPathTest, test__same_node_host_buffer_succeeds) {
    const std::string key = "lfp_key_1";
    const std::string value = "hello_local_disk_fast_path";

    WriteKey(key, value);

    std::vector<char> dst(value.size(), '\0');
    std::vector<tl::expected<int64_t, ErrorCode>> results(1);

    auto ld = MakeLD(kLocalAddr, value.size());
    bool handled = CallFastPath(key, ld, dst.data(), value.size(), results, 0);

    ASSERT_TRUE(handled) << "fast path should handle same-node host buffer";
    ASSERT_TRUE(results[0].has_value())
        << "result must be a success value, not an error";
    EXPECT_EQ(results[0].value(), static_cast<int64_t>(value.size()));

    // Bit-exact verify.
    std::string got(dst.data(), dst.size());
    EXPECT_EQ(got, value) << "bit-exact mismatch: fast path copied wrong data";
}

// ---------------------------------------------------------------------------
// Test 2: same hostname but different port -> fallback (§3.1 multi-process)
// ---------------------------------------------------------------------------

TEST_F(LocalFastPathTest, test__different_port_falls_back) {
    // The LOCAL_DISK replica was created by a different process on this host
    // with a different auto-assigned port (PR #1995 scenario).
    auto ld = MakeLD("127.0.0.1:59999", 64);  // different port

    std::vector<char> dst(64, '\0');
    std::vector<tl::expected<int64_t, ErrorCode>> results(1);

    bool handled = CallFastPath("any_key", ld, dst.data(), 64, results, 0,
                                kLocalAddr /* client port = 59001 */);

    EXPECT_FALSE(handled)
        << "different port must trigger fallback to regular RPC path";
}

// ---------------------------------------------------------------------------
// Test 3: file_storage_ == nullptr -> fallback (enable_ssd_offload = false)
// ---------------------------------------------------------------------------

TEST_F(LocalFastPathTest, test__file_storage_null_falls_back) {
    auto ld = MakeLD(kLocalAddr, 64);

    std::vector<char> dst(64, '\0');
    std::vector<tl::expected<int64_t, ErrorCode>> results(1);

    bool handled = CallFastPath("any_key", ld, dst.data(), 64, results, 0,
                                kLocalAddr, /* use_null_fs */ true);

    EXPECT_FALSE(handled)
        << "null file_storage_ must trigger fallback (offload disabled)";
}

// ---------------------------------------------------------------------------
// Test 4: GPU device pointer destination -> fallback (§3.2)
// ---------------------------------------------------------------------------

TEST_F(LocalFastPathTest, test__gpu_pointer_falls_back) {
#if !defined(USE_CUDA) && !defined(USE_HIP) && !defined(USE_ASCEND) && \
    !defined(USE_ASCEND_DIRECT) && !defined(USE_UBSHMEM) &&            \
    !defined(USE_MUSA) && !defined(USE_MACA)
    GTEST_SKIP()
        << "No GPU runtime compiled in. is_host_pointer() always returns true "
           "in this build; the GPU-pointer fallback cannot be exercised "
           "without a mock GPU allocator.";
#else
    // Allocate a small device buffer.
#if defined(USE_HIP)
    void* gpu_ptr = nullptr;
    if (hipMalloc(&gpu_ptr, 64) != hipSuccess || gpu_ptr == nullptr) {
        GTEST_SKIP() << "hipMalloc failed; no GPU device available";
    }
    struct Cleanup {
        void* p;
        ~Cleanup() { hipFree(p); }
    } _c{gpu_ptr};
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA)
    void* gpu_ptr = nullptr;
    if (cudaMalloc(&gpu_ptr, 64) != cudaSuccess || gpu_ptr == nullptr) {
        GTEST_SKIP() << "cudaMalloc failed; no GPU device available";
    }
    struct Cleanup {
        void* p;
        ~Cleanup() { cudaFree(p); }
    } _c{gpu_ptr};
#else
    GTEST_SKIP() << "GPU runtime detected but alloc API unknown; skipping";
    void* gpu_ptr = nullptr;
#endif

    const std::string key = "lfp_gpu_key";
    const std::string value(64, 'G');
    WriteKey(key, value);

    auto ld = MakeLD(kLocalAddr, 64);
    std::vector<tl::expected<int64_t, ErrorCode>> results(1);

    bool handled = CallFastPath(key, ld, gpu_ptr, 64, results, 0);

    EXPECT_FALSE(handled)
        << "GPU device pointer must fall through to RPC+RDMA path";
#endif
}

// ---------------------------------------------------------------------------
// Test 5: concurrent reads from multiple threads -> no data corruption (§3.3)
// ---------------------------------------------------------------------------

TEST_F(LocalFastPathTest, test__concurrent_calls_no_corruption) {
    constexpr int kNumThreads = 8;
    constexpr int kKeysPerThread = 4;
    constexpr size_t kValueSize = 512;

    // Prepare reference data and write to disk via the shared backend.
    std::unordered_map<std::string, std::string> reference;
    for (int t = 0; t < kNumThreads; ++t) {
        for (int k = 0; k < kKeysPerThread; ++k) {
            std::string key =
                "concurrent_t" + std::to_string(t) + "_k" + std::to_string(k);
            char fill = static_cast<char>('A' + (t * kKeysPerThread + k) % 26);
            std::string value(kValueSize, fill);
            WriteKey(key, value);
            reference[key] = value;
        }
    }

    std::atomic<int> pass_count{0};
    std::atomic<int> miss_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int k = 0; k < kKeysPerThread; ++k) {
                std::string key = "concurrent_t" + std::to_string(t) + "_k" +
                                  std::to_string(k);
                std::vector<char> dst(kValueSize, '\0');
                std::vector<tl::expected<int64_t, ErrorCode>> results(1);

                auto ld = MakeLD(kLocalAddr, kValueSize);

                // Each thread creates its own RealClient to avoid shared
                // client state, but all share the same file_storage_.
                RealClient client;
                client.local_rpc_addr = kLocalAddr;
                client.file_storage_ = file_storage_;

                Slice dst_slice{dst.data(), kValueSize};
                bool handled = client.try_local_disk_fast_path(
                    key, ld, dst_slice, kValueSize, results, 0);

                if (!handled) {
                    miss_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (!results[0].has_value()) {
                    miss_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // Bit-exact check: any mismatch is a data corruption.
                std::string got(dst.data(), dst.size());
                if (got == reference.at(key)) {
                    pass_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ADD_FAILURE() << "Data corruption for key=" << key
                                  << " thread=" << t << " k=" << k
                                  << " expected_fill='" << reference.at(key)[0]
                                  << "' got_fill='" << got[0] << "'";
                    miss_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    int total = kNumThreads * kKeysPerThread;
    LOG(INFO) << "concurrent test: pass=" << pass_count.load()
              << " miss=" << miss_count.load() << " total=" << total;

    EXPECT_GT(pass_count.load(), 0)
        << "At least one concurrent fast-path call should succeed";
}

}  // namespace mooncake
