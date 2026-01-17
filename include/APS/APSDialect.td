#ifndef APS_DIALECT_TD
#define APS_DIALECT_TD

include "mlir/IR/OpBase.td"

def APSDialect : Dialect {
  let name = "aps";
  let cppNamespace = "::aps";
  let summary = "APS Dialect";
}

#endif
