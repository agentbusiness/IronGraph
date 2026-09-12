#ifndef IRONGRAPH_H
#define IRONGRAPH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private binary boundary. Handles are opaque, non-reused uint64_t JSON values.
 * Calls on the same handle serialize; different handles operate independently.
 * Callers must coordinate close with in-flight operations on that handle. */
typedef struct {
    uint8_t *data;
    size_t len;
} irongraph_buffer;

uint32_t irongraph_abi_version(void);
/* Static NUL-terminated storage. Never free or modify. */
const char *irongraph_package_version_v1(void);

/* Request is UTF-8 JSON. Output is an owned UTF-8 JSON envelope.
 * 0: success {ok:true,version,result}; 1: {ok:false,version,error};
 * 2: invalid pointer arguments, output untouched.
 * Request must be readable for len bytes; output must be writable and aligned.
 * They must not overlap. Every non-null output must be freed exactly once. */
int32_t irongraph_call_v1(const uint8_t *request, size_t len, irongraph_buffer *output);
void irongraph_buffer_free_v1(irongraph_buffer buffer);

#ifdef __cplusplus
}
#endif
#endif
