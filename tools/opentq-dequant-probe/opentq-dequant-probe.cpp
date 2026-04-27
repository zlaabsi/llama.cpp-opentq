#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::cerr << "usage:\n";
    std::cerr << "  " << argv0 << " dequant TYPE block.bin expected.f32\n";
    std::cerr << "  " << argv0 << " dot TYPE block.bin activation.f32 expected_scalar.f32\n";
}

template <typename T>
static int run_probe(const std::vector<uint8_t> & block, const std::vector<float> & expected, void (*fn)(const T *, float *, int64_t)) {
    if (block.size() != sizeof(T)) {
        std::cerr << "block size mismatch: got " << block.size() << ", expected " << sizeof(T) << "\n";
        return 2;
    }
    if (expected.size() != QK_OPENTQ) {
        std::cerr << "expected vector size mismatch: got " << expected.size() << ", expected " << QK_OPENTQ << "\n";
        return 2;
    }

    std::vector<float> decoded(QK_OPENTQ);
    fn(reinterpret_cast<const T *>(block.data()), decoded.data(), QK_OPENTQ);

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < QK_OPENTQ; ++i) {
        const float diff = std::fabs(decoded[i] - expected[i]);
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
    }
    mean_abs /= QK_OPENTQ;

    std::cout << "max_abs=" << max_abs << "\n";
    std::cout << "mean_abs=" << mean_abs << "\n";
    if (max_abs > 2.0e-4f) {
        for (int i = 0; i < std::min(8, QK_OPENTQ); ++i) {
            std::cout << i << ": decoded=" << decoded[i] << " expected=" << expected[i] << "\n";
        }
        return 1;
    }
    return 0;
}

template <typename T>
static int run_dot_probe(
        const std::vector<uint8_t> & block,
        const std::vector<float> & activation,
        const std::vector<float> & expected,
        void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int)) {
    if (block.size() != sizeof(T)) {
        std::cerr << "block size mismatch: got " << block.size() << ", expected " << sizeof(T) << "\n";
        return 2;
    }
    if (activation.size() != QK_OPENTQ || expected.size() != 1) {
        std::cerr << "activation/expected size mismatch\n";
        return 2;
    }

    std::vector<block_q8_0> q8(QK_OPENTQ / QK8_0);
    quantize_row_q8_0_ref(activation.data(), q8.data(), QK_OPENTQ);

    float got = 0.0f;
    vec_dot(QK_OPENTQ, &got, 0, block.data(), 0, q8.data(), 0, 1);
    const float diff = std::fabs(got - expected[0]);
    std::cout << "got=" << got << "\n";
    std::cout << "expected=" << expected[0] << "\n";
    std::cout << "abs=" << diff << "\n";
    if (diff > 2.0e-3f) {
        return 1;
    }
    return 0;
}

static std::vector<uint8_t> read_bytes(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(2);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static std::vector<float> read_f32(const char * path) {
    std::vector<uint8_t> bytes = read_bytes(path);
    if (bytes.size() % sizeof(float) != 0) {
        std::cerr << "expected file is not f32 aligned\n";
        std::exit(2);
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        usage(argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    const std::string type = argv[2];
    const std::vector<uint8_t> block = read_bytes(argv[3]);

    if (mode == "dequant") {
        if (argc != 5) {
            usage(argv[0]);
            return 2;
        }
        const std::vector<float> payload = read_f32(argv[4]);
        if (type == "TQ3_SB4") {
            return run_probe<block_opentq_tq3_sb4>(block, payload, dequantize_row_opentq_tq3_sb4);
        }
        if (type == "TQ4_SB2") {
            return run_probe<block_opentq_tq4_sb2>(block, payload, dequantize_row_opentq_tq4_sb2);
        }
        if (type == "TQ4_SB4") {
            return run_probe<block_opentq_tq4_sb4>(block, payload, dequantize_row_opentq_tq4_sb4);
        }
        if (type == "TQ4R2") {
            return run_probe<block_opentq_tq4r2>(block, payload, dequantize_row_opentq_tq4r2);
        }
        if (type == "TQ4R4") {
            return run_probe<block_opentq_tq4r4>(block, payload, dequantize_row_opentq_tq4r4);
        }
    }

    if (mode == "dot") {
        if (argc != 6) {
            usage(argv[0]);
            return 2;
        }
        const std::vector<float> activation = read_f32(argv[4]);
        const std::vector<float> expected = read_f32(argv[5]);
        if (type == "TQ3_SB4") {
            return run_dot_probe<block_opentq_tq3_sb4>(block, activation, expected, ggml_vec_dot_opentq_tq3_sb4_q8_0);
        }
        if (type == "TQ4_SB2") {
            return run_dot_probe<block_opentq_tq4_sb2>(block, activation, expected, ggml_vec_dot_opentq_tq4_sb2_q8_0);
        }
        if (type == "TQ4_SB4") {
            return run_dot_probe<block_opentq_tq4_sb4>(block, activation, expected, ggml_vec_dot_opentq_tq4_sb4_q8_0);
        }
        if (type == "TQ4R2") {
            return run_dot_probe<block_opentq_tq4r2>(block, activation, expected, ggml_vec_dot_opentq_tq4r2_q8_0);
        }
        if (type == "TQ4R4") {
            return run_dot_probe<block_opentq_tq4r4>(block, activation, expected, ggml_vec_dot_opentq_tq4r4_q8_0);
        }
    }

    std::cerr << "unknown mode/type: " << mode << " " << type << "\n";
    return 2;
}
