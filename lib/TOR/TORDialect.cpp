#include "TOR/TORDialect.h"
#include "TOR/TOR.h"
#include "TOR/TORTypes.h"
#include "TOR/TORAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"


using namespace mlir;
using namespace mlir::tor;



void TORDialect::initialize()
{
  registerTypes();

  registerAttributes();
  
  addOperations<
#define GET_OP_LIST
#include "TOR/TOR.cpp.inc"
      >();
  // addInterfaces<TORInlinerInterface>();
}

// Provide implementations for the enums we use.
#include "TOR/TOREnums.cpp.inc"

#include "TOR/TORDialect.cpp.inc"
