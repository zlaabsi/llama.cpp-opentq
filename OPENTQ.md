# OpenTQ Runtime Patch

This fork adds reference `llama.cpp` support for GGUF files emitted by OpenTQ.

## Supported GGUF Tensor Types

| GGML type | OpenTQ variant | Block size | Type bytes |
| --- | --- | ---: | ---: |
| `opentq_tq3_sb4` | `TQ3_SB4` | 128 | 84 |
| `opentq_tq4_sb2` | `TQ4_SB2` | 128 | 84 |
| `opentq_tq4_sb4` | `TQ4_SB4` | 128 | 100 |
| `opentq_tq4r2` | `TQ4R2` | 128 | 164 |
| `opentq_tq4r4` | `TQ4R4` | 128 | 196 |

## Build On Apple Silicon

```bash
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```

## Validate A GGUF

```bash
./build/bin/llama-gguf /path/to/model.gguf r n
./build/bin/llama-cli -m /path/to/model.gguf -ngl 99 -fa -c 8192 -p "Hello" -n 64
```

`llama-gguf ... r n` checks that the GGUF metadata and custom tensor payloads are readable. `llama-cli` requires a full model file, not a smoke GGUF with only a few tensors.

## Current Status

- GGUF type registration is implemented.
- CPU reference dequantization is implemented.
- CPU `Q8_0` vec-dot fallback is implemented.
- Metal builds with the patch.
- Optimized compressed-domain Metal matmul kernels are not implemented yet.
- Stock upstream `llama.cpp` will not load these GGUF files until equivalent type registrations and kernels are upstreamed.
