def AddIOp : TOROp<"addi", [Pure]> {
  let summary = "addi operation";
  let description = [{
    The "addi" operation represents the addition of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def SubIOp : TOROp<"subi", [Pure]> {
  let summary = "subi operation";
  let description = [{
    The "subi" operation represents the substraction of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def MulIOp : TOROp<"muli", [Pure]> {
  let summary = "muli operation";
  let description = [{
    The "muli" operation represents the mulplication of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def MulIConstOp : TOROp<"muli_const", [Pure]> {
  let summary = "muli_const operation";
  let description = [{
    The "muli_const" operation represents the mulplication of two variables.
    One of them must be a constant.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs,
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)`
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs, "ArrayRef<NamedAttribute>" : $attr)>,
  ];
}

def MulSIOp : TOROp<"mulsi", [Pure]> {
  let summary = "mulsi operation";
  let description = [{
    The "mulsi" operation represents the mulplication of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def MulUIOp : TOROp<"mului", [Pure]> {
  let summary = "mului operation";
  let description = [{
    The "mului" operation represents the mulplication of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def MacIOp : TOROp<"maci", [Pure]> {
  let summary = "maci operation";
  let description = [{
    he "maci" operation performs a multiply-accumulate operation.
  }];

  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs, 
                   AnySignlessInteger:$accumulator, I32Attr : $starttime, I32Attr : $endtime);

  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{$lhs $rhs $accumulator `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `,` type($accumulator)`)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs, "Value" : $accumulator)>];
}

def AddFOp : TOROp<"addf", [Pure, 
                           SameOperandsAndResultType]> {
  let summary = "addf operation";
  let description = [{
    The "addf" operation represents the addition of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def SubFOp : TOROp<"subf", [Pure, 
                           SameOperandsAndResultType]> {
  let summary = "subf operation";
  let description = [{
    The "subf" operation represents the substraction of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def MulFOp : TOROp<"mulf", [Pure, 
                           SameOperandsAndResultType]> {
  let summary = "mulf operation";
  let description = [{
    The "mulf" operation represents the mulplication of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def DivFOp : TOROp<"divf", [Pure, 
                           SameOperandsAndResultType]> {
  let summary = "divf operation";
  let description = [{
    The "divf" operation represents the mulplication of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def MacFOp : TOROp<"macf", [Pure,
                            SameOperandsAndResultType]> {
  let summary = "macf operation";
  let description = [{
    he "macf" operation performs a multiply-accumulate operation.
  }];

  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs, 
                   FloatLike:$accumulator, I32Attr : $starttime, I32Attr : $endtime);

  let results = (outs FloatLike:$result);

  let assemblyFormat = [{$lhs $rhs $accumulator `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs, "Value" : $accumulator), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs, accumulator},
                   ArrayRef<NamedAttribute>{});
    }]>];
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def CMPF_P_FALSE   : I64EnumAttrCase<"AlwaysFalse", 0, "false">;
def CMPF_P_OEQ     : I64EnumAttrCase<"OEQ", 1, "oeq">;
def CMPF_P_OGT     : I64EnumAttrCase<"OGT", 2, "ogt">;
def CMPF_P_OGE     : I64EnumAttrCase<"OGE", 3, "oge">;
def CMPF_P_OLT     : I64EnumAttrCase<"OLT", 4, "olt">;
def CMPF_P_OLE     : I64EnumAttrCase<"OLE", 5, "ole">;
def CMPF_P_ONE     : I64EnumAttrCase<"ONE", 6, "one">;
def CMPF_P_ORD     : I64EnumAttrCase<"ORD", 7, "ord">;
def CMPF_P_UEQ     : I64EnumAttrCase<"UEQ", 8, "ueq">;
def CMPF_P_UGT     : I64EnumAttrCase<"UGT", 9, "ugt">;
def CMPF_P_UGE     : I64EnumAttrCase<"UGE", 10, "uge">;
def CMPF_P_ULT     : I64EnumAttrCase<"ULT", 11, "ult">;
def CMPF_P_ULE     : I64EnumAttrCase<"ULE", 12, "ule">;
def CMPF_P_UNE     : I64EnumAttrCase<"UNE", 13, "une">;
def CMPF_P_UNO     : I64EnumAttrCase<"UNO", 14, "uno">;
def CMPF_P_TRUE    : I64EnumAttrCase<"AlwaysTrue", 15, "true">;


def CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [CMPF_P_FALSE, CMPF_P_OEQ, CMPF_P_OGT, CMPF_P_OGE, CMPF_P_OLT, CMPF_P_OLE,
     CMPF_P_ONE, CMPF_P_ORD, CMPF_P_UEQ, CMPF_P_UGT, CMPF_P_UGE, CMPF_P_ULT,
     CMPF_P_ULE, CMPF_P_UNE, CMPF_P_UNO, CMPF_P_TRUE]> {
  let cppNamespace = "::mlir::tor";
}

def CmpFOp : TOROp<"cmpf", [Pure, SameTypeOperands]> {
  let summary = "cmpf operation";
  let description = [{
    The "cmpf" operation represents the comparison of two variables.
  }];

  let arguments = (ins 
      CmpFPredicateAttr : $predicate,
      FloatLike : $lhs, FloatLike : $rhs, 
      I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs BoolLike : $result);

  let assemblyFormat = [{
    $predicate
    $lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($lhs) `->` type($result)
  }];

}


//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def CMPI_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMPI_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMPI_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMPI_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMPI_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE, CMPI_P_ULT, CMPI_P_ULE, CMPI_P_UGT, CMPI_P_UGE]> {
  let cppNamespace = "::mlir::tor";
}

def CmpIOp : TOROp<"cmpi", [Pure]> {
  let summary = "cmpi operation";
  let description = [{
    The "cmpf" operation represents the comparison of two variables.
  }];

  let arguments = (ins 
      CmpIPredicateAttr : $predicate,
      SignlessIntegerLike : $lhs, SignlessIntegerLike : $rhs, 
      I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs BoolLike : $result);

  let assemblyFormat = [{
    $predicate
    $lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

}

class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<Trait> traits = []> :
    TOROp<mnemonic,
    !listconcat([
     /* AttrSizedOperandSegments */
    ], traits)> {

  let results = (outs Res<TORAnyMemRef, "", [MemAlloc<resource>]>:$memref);
  /*
  let extraClassDeclaration = [{
    MemRefType getType() { return dyn_cast<MemRefType>(getResult().getType()); }
  }];
  */

  let assemblyFormat = [{
     attr-dict `:` type($memref)
  }];

}

def AllocOp : AllocLikeOp<"alloc", DefaultResource> {
  let summary = "memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = tor.alloc() : memref<8x64xf32, 1>
    ```
  }];
  let arguments = (ins 
              OptionalAttr<AnyAttr>:$local_type, 
              OptionalAttr<AnyAttr>:$value_d,
              OptionalAttr<AnyAttr>:$value_h);
}

def StreamReadOp : TOROp<"stream_read"> {
  let summary = "stream read operation";
  let description = [{
    The `stream_read` operation read from a stream.

    Example:

    ```mlir
    %0 = tor.stream_read %1 : <f32> -> f32
    ```
  }];

  let arguments = (ins TORAnyStream: $stream);
  let results = (outs AnyType: $data);

  let assemblyFormat = [{
     $stream attr-dict `:` type($stream) `->` type($data)
  }];
}

def StreamWriteOp : TOROp<"stream_write"> {
  let summary = "stream write operation";
  let description = [{
    The `stream_write` operation write from a stream.

    Example:

    ```mlir
    tor.stream_write %1 to %2 : f32, <f32>
    ```
  }];

  let arguments = (ins AnyType: $data, TORAnyStream: $stream);

  let assemblyFormat = [{
     $data `to` $stream attr-dict `:` type($data) `,` type($stream)
  }];
}

def StreamCreateOp : TOROp<"stream_create"> {
  let summary = "stream allocation operation";
  let description = [{
    The `stream_create` operation allocates a stream, as specified by its
    stream type.

    Example:

    ```mlir
    %0 = tor.stream_create : <f32>[2]
    ```
  }];

  let arguments = (ins I32Attr : $depth);
  let results = (outs TORAnyStream:$stream);

  let assemblyFormat = [{
     attr-dict `:` type($stream) `[`  $depth `]`
  }];
}

def AXICreateOp : AllocLikeOp<"axi_create", DefaultResource> {
  let summary = "memory allocation operation";
  let description = [{
    The `axi_create` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = tor.axi_create : memref<8x64xf32, 1>
    ```
  }];

  let arguments = (ins 
              OptionalAttr<AnyAttr>:$ARLEN, 
              OptionalAttr<AnyAttr>:$AWLEN,
              OptionalAttr<AnyAttr>:$bus,
              OptionalAttr<AnyAttr>:$value_d,
              OptionalAttr<AnyAttr>:$value_h);
}

def AXIReadOp : TOROp<"axi_read",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "axi read operation";
  let description = [{
    The `axi_read` op reads an element from a memref specified by an index. The
    output of load is a new value with the same type as the elements of the
    memref. 
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime,
               CArg<"::mlir::ValueRange", "">:$indices), [{   
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
      attr-dict `:` type($memref)`[`type($indices)`]`}];
}

def AXIWriteOp : TOROp<"axi_write",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "axi write operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<TORAnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref,
                   "IntegerAttr":$starttime, "IntegerAttr":$endtime), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

    }]>];

  let assemblyFormat = [{
    $value `to` $memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:`  type($memref)`[`type($indices) `]`
  }];
}

def AXIReadRequestOp : TOROp<"axi_read_request", [MemRefsNormalizable]> {
  let summary = "axi read request operation";
  let description = [{
    The `axi_read_request` op perform a read request to a m_axi memref
    with specified offset and length.
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to have a read request from",
                           [MemRead]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       AnyTypeOf<[AnySignlessInteger, Index]>:$offset,
                       AnySignlessInteger:$length);
  let results = (outs TOR_Request:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime,
               CArg<"::mlir::ValueRange", "">:$offset, "::mlir::ValueRange":$length), [{
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(offset);
      $_state.addOperands(length);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `[` $offset`,` $length `]` `on` ` ` `(` $starttime `to` $endtime `)`
      attr-dict `:` type($memref)`[`type($offset)`,` type($length)`]`}];
}

def AXIWriteRequestOp : TOROp<"axi_write_request", [MemRefsNormalizable]> {
  let summary = "axi write request operation";
  let description = [{
    The `axi_write_request` op perform a write request to a m_axi memref
    with specified offset and length.
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to have a write request from",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       AnyTypeOf<[AnySignlessInteger, Index]>:$offset,
                       AnySignlessInteger:$length);
  let results = (outs TOR_Request:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime,
               CArg<"::mlir::ValueRange", "">:$offset, "::mlir::Value":$length), [{
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(offset);
      $_state.addOperands(length);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `[` $offset`,` $length `]` `on` ` ` `(` $starttime `to` $endtime `)`
      attr-dict `:` type($memref)`[`type($offset)`,` type($length)`]`}];
}

def AXIBurstReadOp : TOROp<"axi_burst_read",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "axi burst read operation";
  let description = [{
    The `axi_burst_read` op reads an element from a memref via a request. The
    output of load is a new value with the same type as the elements of the
    memref.
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to read from",
                           [MemRead]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       TOR_Request:$request);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime,
               "::mlir::Value":$request), [{
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(request);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `(` $request `)` `on` ` ` `(` $starttime `to` $endtime `)`
      attr-dict `:` type($memref)}];
}

def AXIBurstWriteOp : TOROp<"axi_burst_write",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "value",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "axi burst write operation";
  let description = [{
    The `axi_burst_write` op writes an element to a memref via a request.
    The value stored should have the same type as the elemental type of
    the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<TORAnyMemRef, "the reference to write to",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       TOR_Request:$request);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "::mlir::Value":$memref, "::mlir::Value":$request,
                   "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime), [{
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
      $_state.addOperands(request);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{
    $value `to` $memref `(` $request `)` `on` ` ` `(` $starttime `to` $endtime `)`
    attr-dict `:` type($memref)
  }];
}

def AXIWriteResponseOp : TOROp<"axi_write_response", [MemRefsNormalizable]> {
  let summary = "axi write response operation";
  let description = [{
    The `axi_write_response` op is used to await response from
    write request op.
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to have a read request from",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       TOR_Request:$request);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::Value":$request,
               "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime), [{
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(request);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `(` $request `)` `on` ` ` `(` $starttime `to` $endtime `)`
      attr-dict `:` type($memref)
  }];
}

def LoadOp : TOROp<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index. The
    output of load is a new value with the same type as the elements of the
    memref. 
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$memref, "::mlir::IntegerAttr":$starttime, "::mlir::IntegerAttr":$endtime,
               CArg<"::mlir::ValueRange", "">:$indices), [{   
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let extraClassDeclaration = [{
    // Value getMemRef() { return getOperand(0); }
    // void setMemRef(Value value) { setOperand(0, value); }
    // MemRefType getMemRefType() {
    //   return getMemRef().getType().cast<MemRefType>();
    // }

    // operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let assemblyFormat = [{$memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
      attr-dict `:` type($memref)`[`type($indices)`]`}];
}


def StoreOp : TOROp<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<TORAnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref,
                   "IntegerAttr":$starttime, "IntegerAttr":$endtime), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

    }]>];

  let extraClassDeclaration = [{
      // Value getValueToStore() { return getOperand(0); }

      // Value getMemRef() { return getOperand(1); }
      // void setMemRef(Value value) { setOperand(1, value); }
      // MemRefType getMemRefType() {
      //   return getMemRef().getType().cast<MemRefType>();
      // }

      // operand_range getIndices() {
      //   return {operand_begin() + 2, operand_end()};
      // }
  }];

  let assemblyFormat = [{
    $value `to` $memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($value) `,` type($memref)`[`type($indices) `]` `)`
  }];
}

def GuardedStoreOp : TOROp<"guarded_store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "dyn_cast<MemRefType>($_self).getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "guarded store operation";
  let description = [{
    Store a value to a memref location given by indices, only if the guard
    condition is true. The value stored should have the same type as the
    elemental type of the memref. The number of arguments provided within
    brackets need to match the rank of the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<TORAnyMemRef, "the reference to store to", [MemWrite]>:$memref,
                       I1:$guard,
                       I32Attr:$starttime,
                       I32Attr:$endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref, "Value":$guard,
                   "IntegerAttr":$starttime, "IntegerAttr":$endtime), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
      $_state.addOperands(guard);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);
    }]>];

  let assemblyFormat = [{
    $value `to` $memref `[` $indices `]` `if` $guard `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($value) `,` type($memref)`[`type($indices) `]` `)`
  }];
}
