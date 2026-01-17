#include "APS/CAPI/APS.h"
#include "APS/APSDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(APS, aps, aps::APSDialect)