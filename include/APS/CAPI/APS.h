#ifndef APS_CAPI_APS_H
#define APS_CAPI_APS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(APS, aps);

#ifdef __cplusplus
}
#endif

#endif // APS_CAPI_APS_H