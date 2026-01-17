#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <unordered_set>

#include "Schedule/CDFG.h"
#include "TOR/TOR.h"
#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "Schedule/DbgHelper.h"
#include "Schedule/ScheduleAlgo.h"
#include "TOR/TORAttrs.h"
#include "TOR/Utils.h"

#include "TOR/DependenceAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "schedule-algo"

namespace scheduling {

    using namespace mlir;

/// @return std::pair<BasicBlock*, BasicBlock*> **first** is the entry BB,
/// **second** is the exit BB
    std::pair<BasicBlock *, BasicBlock *> ScheduleBase::buildCFG(Block &block,
                                                                 Loop *parentLoop) {

        BasicBlocks.push_back(createBasicBlock(parentLoop));

        BasicBlock *lastHead = BasicBlocks.back().get();

        BasicBlock *exitBlock = lastHead;

        for (auto &op: llvm::reverse(block)) {
            if (auto ifOp = llvm::dyn_cast<tor::IfOp>(op)) {

                if (!ifOp.getElseRegion().empty()) {

                    auto thenBB = buildCFG(ifOp.getThenRegion().front(), parentLoop);
                    auto elseBB = buildCFG(ifOp.getElseRegion().front(), parentLoop);

                    BasicBlock::addControlDependency(
                            {thenBB.second, lastHead, ControlEdge::FORWARD});
                    BasicBlock::addControlDependency(
                            {elseBB.second, lastHead, ControlEdge::FORWARD});

                    auto thenYieldOp = ifOp.getThenRegion().back().getTerminator();
                    auto elseYieldOp = ifOp.getElseRegion().back().getTerminator();

                    for (unsigned i = 0, n = ifOp.getNumResults(); i < n; ++i) {
                        Value x = ifOp.getResult(i);
                        OpAbstract *newOpA =
                                createOp(&op, parentLoop, lastHead, {x},
                                         {thenYieldOp->getOperand(i), elseYieldOp->getOperand(i),
                                          ifOp.getOperand() /* control dependency */},
                                         OpAbstract::OpType::PHI_OP);
                        ValueMap.insert(std::make_pair(x, newOpA));
                        lastHead->addOperation(newOpA);
                    }

                    BasicBlocks.push_back(
                            createBasicBlock(parentLoop));

                    lastHead = BasicBlocks.back().get();

                    BasicBlock::addControlDependency(
                            {lastHead, thenBB.first, ControlEdge::COND});
                    BasicBlock::addControlDependency(
                            {lastHead, elseBB.first, ControlEdge::COND});
                } else {

                    auto thenBB = buildCFG(ifOp.getThenRegion().front(), parentLoop);

                    BasicBlocks.push_back(
                            std::make_unique<BasicBlock>(BasicBlock(parentLoop)));

                    BasicBlock *condHead = BasicBlocks.back().get();

                    if (parentLoop != nullptr)
                        parentLoop->addBasicBlock(condHead);

                    BasicBlock::addControlDependency(
                            {thenBB.second, lastHead, ControlEdge::FORWARD});
                    BasicBlock::addControlDependency(
                            {condHead, lastHead, ControlEdge::COND});
                    BasicBlock::addControlDependency(
                            {condHead, thenBB.first, ControlEdge::COND});

                    lastHead = condHead;
                }

                lastHead->setBranchValue(ifOp.getOperand());

            } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
                bool pipelineFlag = false;
                int targetII = -1;

                if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("pipeline"))
                    pipelineFlag = attr.getInt();
                if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("II"))
                    targetII = attr.getInt();
                op.removeAttr("pipeline");
                op.removeAttr("II");

                Loops.push_back(std::make_unique<Loop>(
                        Loop(parentLoop, &op, pipelineFlag, targetII)));

                Loop *curLoop = Loops.back().get();
                if (parentLoop != nullptr)
                    parentLoop->addChildLoop(curLoop);

                LoopMap[&op] = curLoop;

                auto bodyBB = buildCFG(whileOp.getAfter().front(), curLoop);
                auto condBB = buildCFG(whileOp.getBefore().front(), curLoop);

                BasicBlock::addControlDependency(
                        {condBB.second, lastHead, ControlEdge::COND});
                BasicBlock::addControlDependency(
                        {condBB.second, bodyBB.first, ControlEdge::COND});
                BasicBlock::addControlDependency(
                        {bodyBB.second, condBB.first, ControlEdge::LOOPBACK});

                auto yieldOp = whileOp.getAfter().back().getTerminator();

                for (unsigned i = 0, n = whileOp.getBefore().getNumArguments(); i < n;
                     ++i) {
                    /// phi in cond argument
                    Value x = whileOp.getBefore().getArgument(i);

                    OpAbstract *newOpA =
                            createOp(&op, curLoop, condBB.first, {x},
                                     {whileOp.getOperand(i), yieldOp->getOperand(i)},
                                     OpAbstract::OpType::PHI_OP); // PHI happens inside the loop

                    ValueMap.insert(std::make_pair(x, newOpA));
                    condBB.first->addOperation(newOpA);
                }

                auto condOp =
                        whileOp.getBefore()
                                .back()
                                .getTerminator(); /// the first operand of condop is condition

                condBB.second->setBranchValue(condOp->getOperand(0));

                for (unsigned i = 0, n = whileOp.getAfter().getNumArguments(); i < n;
                     ++i) {
                    /// assign in body arguments
                    Value x = whileOp.getAfter().getArgument(i);
                    OpAbstract *newOpA = createOp(
                            &op, curLoop, bodyBB.first, {x}, {condOp->getOperand(i + 1)},
                            OpAbstract::OpType::ASSIGN_OP); // assign happens inside the loop
                    ValueMap.insert(std::make_pair(x, newOpA));
                    bodyBB.first->addOperation(newOpA);
                }

                for (unsigned i = 0, n = whileOp.getNumResults(); i < n; ++i) {
                    /// assign in results
                    Value x = whileOp.getResult(i);
                    OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, {x},
                                                  {condOp->getOperand(i + 1)},
                                                  OpAbstract::OpType::ASSIGN_OP);
                    ValueMap.insert(std::make_pair(x, newOpA));
                    lastHead->addOperation(newOpA);
                }

                BasicBlocks.push_back(
                        createBasicBlock(parentLoop));

                lastHead = BasicBlocks.back().get();

                BasicBlock::addControlDependency(
                        {lastHead, condBB.first, ControlEdge::FORWARD});

            } else if (auto allocOp = llvm::dyn_cast<tor::AllocOp>(op)) {
                /// omited

            } else if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(op)) {

                Value result = loadOp.getResult();

                ArrayRef<int> signatures;
                ArrayRef<int> distances;

                if (loadOp.getOperation()->hasAttr("dependence")) {
                    auto deps = dyn_cast<tor::DependenceAttr>(loadOp.getOperation()
                            ->getAttr("dependence"));
                    signatures = deps.getSignatures();
                    distances = deps.getDistances();
                } else {
                    signatures = SmallVector<int, 2>();
                    distances = SmallVector<int, 2>();
                }
                auto allocOp = llvm::dyn_cast<tor::AllocOp>(loadOp.getMemref().getDefiningOp());
                std::string storageType = "";
                if (allocOp && allocOp->hasAttr("bind_storage_type")) {
                    storageType = dyn_cast<StringAttr>(allocOp->getAttr("bind_storage_type")).getValue();
                    setPragmaStructureAttrStatusByOp(allocOp, "bind_storage");
                }
                OpAbstract *newOpA =
                        createMemOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                    std::vector<Value>{loadOp.getIndices().begin(),
                                                       loadOp.getIndices().end()},
                                    OpAbstract::OpType::LOAD_OP, signatures, distances, storageType);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto readRfOp = llvm::dyn_cast<aps::CpuRfRead>(op)) {
                // Handle APS CPU register file read as a regular operation
                Value result = readRfOp.getResult();
                SmallVector<Value, 4> operands;
                operands.push_back(readRfOp.getRs());
                
                // Create as a regular operation, not a memory operation
                OpAbstract *newOpA = createOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{result},
                    operands,
                    OpAbstract::OpType::DEFINED_OP
                );
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(op)) {

                Value mem = storeOp.getMemref();
                std::vector<Value> operands{storeOp.getIndices().begin(),
                                            storeOp.getIndices().end()};
                operands.push_back(storeOp.getValue());

                ArrayRef<int> signatures;
                ArrayRef<int> distances;

                if (storeOp.getOperation()->hasAttr("dependence")) {
                    auto deps = dyn_cast<tor::DependenceAttr>(storeOp.getOperation()
                            ->getAttr("dependence")
                            );
                    signatures = deps.getSignatures();
                    distances = deps.getDistances();
                } else {
                    signatures = SmallVector<int, 2>();
                    distances = SmallVector<int, 2>();
                }
                auto allocOp = llvm::dyn_cast<tor::AllocOp>(storeOp.getMemref().getDefiningOp());
                std::string storageType = "";
                if (allocOp && allocOp->hasAttr("bind_storage_type")) {
                    setPragmaStructureAttrStatusByOp(allocOp, "bind_storage");
                    storageType = dyn_cast<StringAttr>(allocOp->getAttr("bind_storage_type")).getValue();
                }
                OpAbstract *newOpA = createMemOp(
                        &op, parentLoop, lastHead, std::vector<Value>{mem}, operands,
                        OpAbstract::OpType::STORE_OP, signatures, distances, storageType);
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto storeOp = llvm::dyn_cast<tor::GuardedStoreOp>(op)) {

                Value mem = storeOp.getMemref();
                std::vector<Value> operands{storeOp.getIndices().begin(),
                                            storeOp.getIndices().end()};
                operands.push_back(storeOp.getValue());
                operands.push_back(storeOp.getGuard());

                ArrayRef<int> signatures;
                ArrayRef<int> distances;

                if (storeOp.getOperation()->hasAttr("dependence")) {
                    auto deps = dyn_cast<tor::DependenceAttr>(storeOp.getOperation()
                            ->getAttr("dependence"));
                    signatures = deps.getSignatures();
                    distances = deps.getDistances();
                } else {
                    signatures = SmallVector<int, 2>();
                    distances = SmallVector<int, 2>();
                }
                auto allocOp = llvm::dyn_cast<tor::AllocOp>(storeOp.getMemref().getDefiningOp());
                std::string storageType = "";
                if (allocOp && allocOp->hasAttr("bind_storage_type")) {
                    setPragmaStructureAttrStatusByOp(allocOp, "bind_storage");
                    storageType = dyn_cast<StringAttr>(allocOp->getAttr("bind_storage_type")).getValue();
                }
                OpAbstract *newOpA = createMemOp(
                        &op, parentLoop, lastHead, std::vector<Value>{mem}, operands,
                        OpAbstract::OpType::STORE_OP, signatures, distances, storageType);
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto writeRfOp = llvm::dyn_cast<aps::CpuRfWrite>(op)) {
                // Handle APS CPU register file write as a regular operation
                SmallVector<Value, 4> operands;
                operands.push_back(writeRfOp.getRd());
                operands.push_back(writeRfOp.getValue());
                
                // Create as a regular operation, not a memory operation
                OpAbstract *newOpA = createOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{},  // No result for write
                    operands,
                    OpAbstract::OpType::DEFINED_OP
                );
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto streamReadOp = llvm::dyn_cast<tor::StreamReadOp>(op)) {
                auto result = streamReadOp.getResult();
                OpAbstract *newOpA =
                        createStreamOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                    {},
                                    OpAbstract::OpType::STREAM_READ_OP);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto streamWriteOp = llvm::dyn_cast<tor::StreamWriteOp>(op)) {
                auto operand = streamWriteOp.getOperand(0);
                OpAbstract *newOpA =
                        createStreamOp(&op, parentLoop, lastHead, std::vector<Value>{},
                                    std::vector<Value>{operand},
                                    OpAbstract::OpType::STREAM_WRITE_OP); 
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);

            } else if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {

                bool pipelineFlag = false;
                int targetII = -1;

                if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("pipeline"))
                    pipelineFlag = attr.getInt();
                if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("II"))
                    targetII = attr.getInt();

                op.removeAttr("pipeline");
                op.removeAttr("II");

                Loops.push_back(std::make_unique<Loop>(
                        Loop(parentLoop, &op, pipelineFlag, targetII)));

                Loop *curLoop = Loops.back().get();
                if (parentLoop != nullptr)
                    parentLoop->addChildLoop(curLoop);

                LoopMap[&op] = curLoop;

                auto bodyBB = buildCFG(*forOp.getBody(), curLoop);

                BasicBlocks.push_back(
                    createBasicBlock(curLoop));
                auto forHead = BasicBlocks.back().get();

                BasicBlock::addControlDependency(
                    {bodyBB.second, forHead, ControlEdge::LOOPBACK});
                BasicBlock::addControlDependency(
                    {forHead, bodyBB.first, ControlEdge::FORWARD});
                BasicBlock::addControlDependency(
                    {forHead, lastHead, ControlEdge::COND});

                auto yieldOp = forOp.getBody()->getTerminator();

                Value induction = forOp.getInductionVar();
                OpAbstract *newOpA = createOp(
                    &op, curLoop, forHead, std::vector<Value>{induction},
                    std::vector<Value>{forOp.getLowerBound(), induction},
                    OpAbstract::OpType::PHI_OP);
                forHead->addOperation(newOpA);
                ValueMap.insert(std::make_pair(induction, newOpA));

                // forOp iteration variables
                for (auto iter: llvm::enumerate(forOp.getRegionIterArgs())) {
                    Value x = iter.value();
                    OpAbstract *newOpA = createOp(
                        &op, curLoop, forHead, std::vector<Value>{x},
                        std::vector<Value>{forOp.getInitArgs()[iter.index()],
                                           yieldOp->getOperand(iter.index())},
                        OpAbstract::OpType::PHI_OP);
                    ValueMap.insert(std::make_pair(x, newOpA));
                    forHead->addOperation(newOpA);
                }

                // forOp results
                for (auto result: llvm::enumerate(forOp.getResults())) {
                    Value x = result.value();
                    OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, {x},
                                                  {yieldOp->getOperand(result.index())},
                                                  OpAbstract::OpType::ASSIGN_OP);
                    ValueMap.insert(std::make_pair(x, newOpA));
                    lastHead->addOperation(newOpA);
                }

                BasicBlocks.push_back(
                        createBasicBlock(parentLoop));
                lastHead = BasicBlocks.back().get();
                BasicBlock::addControlDependency(
                    ControlEdge(lastHead, forHead, ControlEdge::FORWARD));

            } else if (auto condOp = llvm::dyn_cast<tor::ConditionOp>(op)) {
                /// omited
                continue;
            } else if (auto yieldOp = llvm::dyn_cast<tor::YieldOp>(op)) {
                /// omited
                continue;
            } else if (auto callOp = llvm::dyn_cast<tor::CallOp>(op)) {
                if (isDataflowRegion) {
                    /// Do not schedule call op in dataflow region
                    continue;
                }
                OpAbstract *newOpA = createOp(
                        &op, parentLoop, lastHead,
                        std::vector<Value>{op.getResults().begin(), op.getResults().end()},
                        std::vector<Value>{op.getOperands().begin(), op.getOperands().end()},
                        OpAbstract::OpType::CALL_OP);
                lastHead->addOperation(newOpA);
                OperationMap[&op] = newOpA;
                for (auto result: op.getResults()) {
                    Value x = result;
                    ValueMap.insert(std::make_pair(x, newOpA));
                }
            } else if (auto mAxiCreateOp = llvm::dyn_cast<tor::AXICreateOp>(op)) {
                /// omited
            } else if (auto mAxiReadOp = llvm::dyn_cast<tor::AXIReadOp>(op)) {
                std::vector<Value> operands{mAxiReadOp.getIndices().begin(),
                                            mAxiReadOp.getIndices().end()};
                auto result = mAxiReadOp.getResult();
                OpAbstract *newOpA =
                        createMAxiOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                     operands, OpAbstract::OpType::M_AXI_READ_OP);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiWriteOp = llvm::dyn_cast<tor::AXIWriteOp>(op)) {
                std::vector<Value> operands{mAxiWriteOp.getIndices().begin(),
                                            mAxiWriteOp.getIndices().end()};
                operands.push_back(mAxiWriteOp.getValue());
                // todo memref 放进result? ---- storeOp有这么放
                OpAbstract *newOpA =
                        createMAxiOp(&op, parentLoop, lastHead, std::vector<Value>{},
                                     operands, OpAbstract::OpType::M_AXI_WRITE_OP);
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiReadRequestOp = llvm::dyn_cast<tor::AXIReadRequestOp>(op)) {
                std::vector<Value> operands{mAxiReadRequestOp.getOffset(),
                                            mAxiReadRequestOp.getLength()};
                auto result = mAxiReadRequestOp.getResult();
                OpAbstract *newOpA =
                        createMAxiBurstOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                          operands, OpAbstract::OpType::M_AXI_READ_REQUEST_OP);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiWriteRequestOp = llvm::dyn_cast<tor::AXIWriteRequestOp>(op)) {
                std::vector<Value> operands{mAxiWriteRequestOp.getOffset(),
                                            mAxiWriteRequestOp.getLength()};
                auto result = mAxiWriteRequestOp.getResult();
                OpAbstract *newOpA =
                        createMAxiBurstOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                          operands, OpAbstract::OpType::M_AXI_WRITE_REQUEST_OP);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiBurstReadOp = llvm::dyn_cast<tor::AXIBurstReadOp>(op)) {
                auto result = mAxiBurstReadOp.getResult();
                OpAbstract *newOpA =
                        createMAxiBurstOp(&op, parentLoop, lastHead, std::vector<Value>{result},
                                          {mAxiBurstReadOp.getRequest()}, OpAbstract::OpType::M_AXI_BURST_READ_OP);
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiBurstWriteOp = llvm::dyn_cast<tor::AXIBurstWriteOp>(op)) {
                std::vector<Value> operands{mAxiBurstWriteOp.getRequest(),
                                            mAxiBurstWriteOp.getValue()};
                OpAbstract *newOpA =
                        createMAxiBurstOp(&op, parentLoop, lastHead, std::vector<Value>{},
                                          operands, OpAbstract::OpType::M_AXI_BURST_WRITE_OP);
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto mAxiWriteResponseOp = llvm::dyn_cast<tor::AXIWriteResponseOp>(op)) {
                OpAbstract *newOpA =
                        createMAxiBurstOp(&op, parentLoop, lastHead, std::vector<Value>{},
                                          {mAxiWriteResponseOp.getRequest()},
                                          OpAbstract::OpType::M_AXI_RESPONSE_OP);
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(op)) {
                OpAbstract *newOpA = createOp(constOp, parentLoop, lastHead,
                                               {constOp.getResult()}, SmallVector<Value>{});
                lastHead->addOperation(newOpA);
                ValueMap[constOp.getResult()] = newOpA;
            } else if (auto getGlobalOp = llvm::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
                // Handle memref.get_global - get reference to global memory
                Value result = getGlobalOp.getResult();
                OpAbstract *newOpA = createOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{result},
                    std::vector<Value>{},  // No operands
                    OpAbstract::OpType::DEFINED_OP
                );
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto memLoadOp = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
                // Handle memref.load
                Value result = memLoadOp.getResult();
                OpAbstract *newOpA = createMemOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{result},
                    std::vector<Value>{memLoadOp.getIndices().begin(), memLoadOp.getIndices().end()},
                    OpAbstract::OpType::LOAD_OP,
                    SmallVector<int, 2>(),  // No dependence signatures
                    SmallVector<int, 2>(),  // No dependence distances
                    ""  // No storage type
                );
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto memStoreOp = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
                // Handle memref.store
                Value mem = memStoreOp.getMemref();
                std::vector<Value> operands{memStoreOp.getIndices().begin(), memStoreOp.getIndices().end()};
                operands.push_back(memStoreOp.getValueToStore());

                OpAbstract *newOpA = createMemOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{mem},
                    operands,
                    OpAbstract::OpType::STORE_OP,
                    SmallVector<int, 2>(),  // No dependence signatures
                    SmallVector<int, 2>(),  // No dependence distances
                    ""  // No storage type
                );
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto apsMemLoadOp = llvm::dyn_cast<aps::MemLoad>(op)) {
                // Handle aps.memload
                Value result = apsMemLoadOp.getResult();
                OpAbstract *newOpA = createMemOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{result},
                    std::vector<Value>{apsMemLoadOp.getIndices().begin(), apsMemLoadOp.getIndices().end()},
                    OpAbstract::OpType::LOAD_OP,
                    SmallVector<int, 2>(),
                    SmallVector<int, 2>(),
                    ""
                );
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto apsMemStoreOp = llvm::dyn_cast<aps::MemStore>(op)) {
                // Handle aps.memstore
                Value mem = apsMemStoreOp.getMemref();
                std::vector<Value> operands{apsMemStoreOp.getIndices().begin(), apsMemStoreOp.getIndices().end()};
                operands.push_back(apsMemStoreOp.getValue());

                OpAbstract *newOpA = createMemOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{mem},
                    operands,
                    OpAbstract::OpType::STORE_OP,
                    SmallVector<int, 2>(),
                    SmallVector<int, 2>(),
                    ""
                );
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto memBurstLoadOp = llvm::dyn_cast<aps::MemBurstLoad>(op)) {
                // Handle aps.memburstload - TileLink burst read
                std::vector<Value> operands{
                    memBurstLoadOp.getCpuAddr(),
                    memBurstLoadOp.getStart(),
                    memBurstLoadOp.getLength()
                };
                OpAbstract *newOpA = createTLOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{},  // No result
                    operands,
                    OpAbstract::OpType::TL_READ_OP
                );
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto memBurstStoreOp = llvm::dyn_cast<aps::MemBurstStore>(op)) {
                // Handle aps.memburststore - TileLink burst write
                std::vector<Value> operands{
                    memBurstStoreOp.getStart(),
                    memBurstStoreOp.getCpuAddr(),
                    memBurstStoreOp.getLength()
                };
                OpAbstract *newOpA = createTLOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{},  // No result
                    operands,
                    OpAbstract::OpType::TL_WRITE_OP
                );
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (auto memDeclareOp = llvm::dyn_cast<aps::MemDeclare>(op)) {
                // Handle aps.memdeclare - similar to alloc, can be omitted from scheduling
                // Just track the result value if needed
                Value result = memDeclareOp.getResult();
                OpAbstract *newOpA = createOp(
                    &op, parentLoop, lastHead,
                    std::vector<Value>{result},
                    std::vector<Value>{},
                    OpAbstract::OpType::DEFINED_OP
                );
                ValueMap[result] = newOpA;
                OperationMap[&op] = newOpA;
                lastHead->addOperation(newOpA);
            } else if (llvm::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp, mlir::memref::CastOp>(op)) {
                // Handle memref.alloc, memref.alloca, memref.cast - omit from scheduling
                // These are typically handled at compile time or don't need scheduling
                if (op.getNumResults() > 0) {
                    Value result = op.getResult(0);
                    OpAbstract *newOpA = createOp(
                        &op, parentLoop, lastHead,
                        std::vector<Value>{result},
                        std::vector<Value>{op.getOperands().begin(), op.getOperands().end()},
                        OpAbstract::OpType::DEFINED_OP
                    );
                    ValueMap[result] = newOpA;
                    OperationMap[&op] = newOpA;
                    lastHead->addOperation(newOpA);
                }
            } else {
                OpAbstract *newOpA = createOp(
                        &op, parentLoop, lastHead,
                        std::vector<Value>{op.getResults().begin(), op.getResults().end()},
                        std::vector<Value>{op.getOperands().begin(), op.getOperands().end()});
                lastHead->addOperation(newOpA);
                OperationMap[&op] = newOpA;
                for (auto result: op.getResults()) {
                    Value x = result;
                    ValueMap.insert(std::make_pair(x, newOpA));
                }
            }
        }

        return std::make_pair(lastHead, exitBlock);
    }

    void ScheduleBase::buildScalarDFG() {
        // dependence of scalars
        for (auto &&BB: BasicBlocks) {
            for (auto op: BB->getOperations()) {
                if (op->getType() == OpAbstract::OpType::PHI_OP) {
                    if (llvm::isa<tor::WhileOp>(op->getOp()) ||
                        llvm::isa<tor::ForOp>(op->getOp())) {
                        // second operand is loop recursion dependence
                        addDependency(Dependence(ValueMap[op->getOperands()[0]], op, 0,
                                                 Dependence::D_RAW));
                        addDependency(Dependence(ValueMap[op->getOperands()[1]], op, 1,
                                                 Dependence::D_RAW));
                        continue;
                    } // otherwise is phi in an ifop
                }
                for (auto v: op->getOperands()) {
                    if (ValueMap.find(v) != ValueMap.end()) {
                        llvm::errs() << "Adding RAW dependency: " 
                                     << ValueMap[v]->getOp()->getName() << " -> " 
                                     << op->getOp()->getName() << "\n";
                        addDependency(Dependence(ValueMap[v], op, 0, Dependence::D_RAW));
                    } else {
                        llvm::errs() << "WARNING: Value not found in ValueMap for op " 
                                     << op->getOp()->getName() << "\n";
                    }
                }
            }
        }
    }

    void ScheduleBase::buildStreamDFG() {
        // dependence of streams
        for (auto &&op1 : Operations) {
            if (op1->getType() != OpAbstract::OpType::STREAM_READ_OP &&
                op1->getType() != OpAbstract::OpType::STREAM_WRITE_OP)
                continue;
            for (auto &&op2 : Operations) {
                if (op2->getType() != OpAbstract::OpType::STREAM_READ_OP &&
                    op2->getType() != OpAbstract::OpType::STREAM_WRITE_OP)
                    continue;
                if (op1.get() == op2.get())
                    continue;

                auto streamop1 = op1->getStreamOp();
                auto streamop2 = op2->getStreamOp();

                if (streamop1->getStream() != streamop2->getStream())
                    continue;

                int Distance = -1;
                if (canReach(streamop1, streamop2, false))
                    Distance = 0;
                
                if (Distance == -1 && canReach(streamop2, streamop2, true))
                    Distance = 1;
                    
                if (Distance != -1) {
                    LLVM_DEBUG( 
                        llvm::outs() << "Stream dependence: " << Distance << "\n";
                        llvm::outs() << streamop1->getOpDumpId() << " " << streamop2->getOpDumpId() << "\n";
                    );
                    addDependency(Dependence(streamop1, streamop2, Distance, Dependence::D_RAW));
                }
            }
        }
    }

    void ScheduleBase::buildBurstMAxiDFG(std::vector<mlir::Operation*>& mAxiOps,
        std::unordered_map<mlir::Operation*, mlir::Operation*>& requestOpToLastOp)
    {
        for (auto &&op: Operations) {
            auto mAxiOp = op->getMAxiOp();
            if (mAxiOp == nullptr) {
                continue;
            }

            if (mAxiOp->isRequest()) {
                mlir::Operation* op = mAxiOp->getOp();
                mAxiOps.push_back(op);
                std::vector<mlir::Operation*> users;
                for (auto *user: op->getUsers()) {
                    if (llvm::isa<tor::AXIBurstReadOp, tor::AXIBurstWriteOp>(user)) {
                        users.push_back(user);
                    } else if (llvm::isa<tor::AXIWriteResponseOp>(user)) {
                        assert(requestOpToLastOp.find(op) == requestOpToLastOp.end()
                            && "request op should have at most one responseOp");
                        requestOpToLastOp[op] = user;
                    } else {
                        assert(false && "request Op should only have burst type user");
                    }
                }
                // sort burst op
                std::sort(users.begin(), users.end(), [&](mlir::Operation* lhs, mlir::Operation* rhs){
                    return canReach(OperationMap[lhs], OperationMap[rhs], false);
                });

                // dependency between burstOps from the same request
                for (size_t i = 1; i < users.size(); ++i) {
                    addDependency(Dependence(OperationMap[users[i - 1]], OperationMap[users[i]], 0, Dependence::D_RAW));
                }

                // if is write request, add dependency between response op and the last burstOp
                auto iter = requestOpToLastOp.find(op);
                if (iter != requestOpToLastOp.end()) {
                    addDependency(Dependence(OperationMap[users.back()], OperationMap[iter->second], 0, Dependence::D_RAW));
                }

                // loop back dependency in between burstOps
                for (size_t i = 0; i < users.size(); ++i) {
                    OpAbstract * opA1 = OperationMap[users[i]];
                    auto mAxiOp1 = opA1->getMAxiOp();
                    for (size_t j = i + 1; j < users.size(); ++j) {
                        OpAbstract * opA2 = OperationMap[users[j]];
                        auto mAxiOp2 = opA2->getMAxiOp();
                        auto loop1 = mAxiOp1->getParentLoop();
                        auto loop2 = mAxiOp2->getParentLoop();
                        if (loop1 != nullptr && loop1 == loop2) {
                            if (canReach(mAxiOp2, mAxiOp1, true)) {
                                addDependency(Dependence(opA2, opA1, 1, Dependence::D_RAW));
                            }
                        }
                    }
                }
            } else if (!mAxiOp->isBurst()) {
                // non-burst op
                mAxiOps.push_back(mAxiOp->getOp());
            }
        }
    }

    void ScheduleBase::buildMAxiDFG() {
        // todo add burst
        // 1. request and burstop dependency is added in cfg (done)
        // 2. cfg between burstop
        // 3. cfg between burstop and response op
        // 4. dfg between request and request, 但每个request
        // 5. resource constraint bus的资源限制，一个bus一个方向只能一次
        // 能burst都是区间内没有依赖关系才能burst的

        // operations的顺序是从后往前，callop对m axi的依赖暂时无分析
        // all request op or maxiReadOp or maxiWriteOp
        std::vector<mlir::Operation*> mAxiOps;
        // request op to last op that uses this request, this is considered as a whole
        std::unordered_map<mlir::Operation*, mlir::Operation*> requestOpToLastOp;
        buildBurstMAxiDFG(mAxiOps, requestOpToLastOp);

        auto depAnalysis = tor::DependenceAnalysis();
        for (auto op1: mAxiOps) {
            auto opA1 = OperationMap[op1];
            auto mAxiOp1 = opA1->getMAxiOp();

            for (auto op2 : mAxiOps) {
                auto opA2 = OperationMap[op2];
                auto mAxiOp2 = opA2->getMAxiOp();

                if (op1 == op2) {
                    continue;
                }

                // no data dependency if memref not the same memref
                if (mAxiOp1->getMemRef() != mAxiOp2->getMemRef()) {
                    continue;
                }

                if (mAxiOp1->isRead() && mAxiOp2->isRead()) {
                    // No data dependency if both read
                    continue;
                }

                // check if the effect of mAxiOp1 while reach mAxiOp2
                int distance = -1;

                // Check dependence analysis result
                auto loop1 = mAxiOp1->getParentLoop();
                auto loop2 = mAxiOp2->getParentLoop();
                if (loop1 != nullptr && loop1 == loop2) {
                    auto loopOp = loop1->getDefiningOp();
                    auto dep = depAnalysis.get_distance(mAxiOp1->getLength(), mAxiOp1->getAddr(),
                        mAxiOp2->getLength(), mAxiOp2->getAddr(), loopOp);
                    if (dep.type == tor::DependenceResult::NotDependent)
                        continue;

                    if (dep.type == tor::DependenceResult::Always) {
                        if (canReach(mAxiOp1, mAxiOp2, false))
                            distance = 0;
                        else if (canReach(mAxiOp1, mAxiOp2, true))
                            distance = 1;
                    }

                    if (dep.type == tor::DependenceResult::Dependent) {
                        // When mAxiOp1 cannot reach mAxiOp2 without loop-back edge,
                        // mAxiOp1 doesn't depends on mAxiOp2 even when their distance is
                        // zero
                        if (dep.dist == 0 && !canReach(mAxiOp1, mAxiOp2, false))
                            continue;
                        distance = dep.dist;
                    }
                }

                // mAxiOp1 can reach mAxiOp2 without loop-back edge
                if (distance == -1 && canReach(mAxiOp1, mAxiOp2, false))
                    distance = 0;

                // mAxiOp1 can reach mAxiOp2 using some loop-back edge,
                // pessimiticaly assume distance to be 1
                if (distance == -1 && canReach(mAxiOp1, mAxiOp2, true))
                    distance = 1;

                // When mAxiOp1 and mAxiOp2 are not in the same BB and LOOP, then there
                // is no dependence between them even when their distance=0 (canReach without loop-back edge)
                if (distance == 0 && mAxiOp1->getParentBB() != mAxiOp2->getParentBB()
                    && mAxiOp1->getParentLoop() != mAxiOp2->getParentLoop())
                  distance = -1;
                // mAxiOp1 can't reach mAxiOp2 (e.g. mutually exclusive branch of an
                // if statement)
                if (distance == -1)
                    continue;

                if (mAxiOp1->isRead() && mAxiOp2->isWrite()) {
                    // WAR
                    if (requestOpToLastOp.find(op1) != requestOpToLastOp.end()) {
                        mAxiOp1 = OperationMap[op1]->getMAxiOp();
                    }
                    addDependency(Dependence(mAxiOp1, mAxiOp2, distance, Dependence::D_WAR));
                } else if (mAxiOp1->isWrite() && mAxiOp2->isRead()) {
                    // RAW
                    if (requestOpToLastOp.find(op1) != requestOpToLastOp.end()) {
                        mAxiOp1 = OperationMap[op1]->getMAxiOp();
                    }
                    addDependency(Dependence(mAxiOp1, mAxiOp2, distance, Dependence::D_RAW));
                } else if (mAxiOp1->isWrite() && mAxiOp2->isWrite()) {
                    // WAW - For TileLink operations, check if they access overlapping addresses
                    // If they don't overlap, skip WAW dependency as they can run in parallel
                    // based on the "amount" resource parameter
                    if (opA1->getType() == OpAbstract::OpType::TL_WRITE_OP &&
                        opA2->getType() == OpAbstract::OpType::TL_WRITE_OP) {
                        // Use dependence analysis to check if address ranges overlap
                        auto loop1 = mAxiOp1->getParentLoop();
                        auto loop2 = mAxiOp2->getParentLoop();
                        if (loop1 != nullptr && loop1 == loop2) {
                            // In the same loop, use dependence analysis
                            auto loopOp = loop1->getDefiningOp();
                            auto dep = depAnalysis.get_distance(mAxiOp1->getLength(), mAxiOp1->getAddr(),
                                mAxiOp2->getLength(), mAxiOp2->getAddr(), loopOp);
                            if (dep.type == tor::DependenceResult::NotDependent) {
                                // No overlap, can skip WAW dependency
                                continue;
                            }
                        } else {
                            // Not in the same loop - check if addresses are statically different
                            // If both have constant addresses that don't overlap, skip dependency
                            auto addr1 = mAxiOp1->getAddr();
                            auto addr2 = mAxiOp2->getAddr();
                            if (addr1 != addr2) {
                                // Different SSA values for addresses - likely different ranges
                                // Skip WAW dependency, resource constraints will handle parallelism
                                continue;
                            }
                        }
                    }
                    if (requestOpToLastOp.find(op1) != requestOpToLastOp.end()) {
                        mAxiOp1 = OperationMap[op1]->getMAxiOp();
                    }
                    addDependency(Dependence(mAxiOp1, mAxiOp2, distance, Dependence::D_WAW));
                }
            }
        }
    }

    void ScheduleBase::buildTensorDFG() {
        auto dep_analysis = tor::DependenceAnalysis();

        // Helper function to check if two memrefs refer to the same global memory
        // This is needed because memrefs in different blocks may have different SSA values
        // (from different memref.get_global operations) but refer to the same global symbol
        auto isSameGlobalMemref = [](Value memref1, Value memref2) -> bool {
            // Direct SSA value comparison
            if (memref1 == memref2)
                return true;

            // Check if both are results of memref.get_global operations
            auto getGlobal1 = memref1.getDefiningOp<mlir::memref::GetGlobalOp>();
            auto getGlobal2 = memref2.getDefiningOp<mlir::memref::GetGlobalOp>();

            if (getGlobal1 && getGlobal2) {
                // Compare the global symbol names
                return getGlobal1.getNameAttr() == getGlobal2.getNameAttr();
            }

            return false;
        };

        // dependence of tensors
        for (auto &&op1: Operations) {
            auto memop1 = op1->getMemOp();
            if (memop1 == nullptr)
                continue;

          for (auto &&op2 : Operations) {
            auto memop2 = op2->getMemOp();
            if (memop2 == nullptr || memop1 == memop2)
                continue;

            // no dependency if not the same memref
            if (!isSameGlobalMemref(memop1->getMemRef(), memop2->getMemRef()))
                continue;

            if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
                op2->getType() == OpAbstract::OpType::LOAD_OP) {
                // No data dependency
                continue;
            }

            // // When memop1 and memop2 are not in the same BB, then there
            // // is no dependence between them
            // if (memop1->getParentBB() != memop2->getParentBB())
            //     continue;

            // check if memop1 and memop2 use different memory bank
            if (memop1->hasFixedMemoryBank() && memop2->hasFixedMemoryBank()) {
                if (memop1->getPartitionIndicies() != memop2->getPartitionIndicies())
                    continue;
            }

            // check if the effect of memop1 while reach memop2
            int Distance = -1;
            bool annotated = false;

            // Annotated
            for (auto d1 : memop1->Dependences) {
                if (memop2->Dependences.find(d1.first) != memop2->Dependences.end()) {
                    Distance = d1.second;
                    annotated = true;
                    break;
                }
            }

            // Check dependence analysis result
            auto loop1 = memop1->getParentLoop();
            auto loop2 = memop2->getParentLoop();

            // This is a fast fix for determine the dependence of two memops
            // First we check if they appears in the sameloop, if so, we compute their distance
            // by analyzing their indices. Otherwise, we simply check if a memop can reach another 
            // disregarding their indices, and set dependence if so.
            if (loop1 != nullptr && loop2 != nullptr && loop1 == loop2) {
                auto loopOp = loop1->getDefiningOp();
                auto dep = dep_analysis.get_distance(1, memop1->getAddr(), 1, memop2->getAddr(), loopOp);
                if (dep.type == tor::DependenceResult::Dependent && dep.dist == 0) {
                  if (op1->getOp()->getAttr("distance") && op2->getOp()->getAttr("distance")) {
                    auto distance1 = dyn_cast<IntegerAttr>(op1->getOp()->getAttr("distance")).getInt();
                    auto distance2 = dyn_cast<IntegerAttr>(op2->getOp()->getAttr("distance")).getInt();
                    if (distance1 == distance2) {
                      dep.dist = distance1;
                    }
                  }
                }
                if (dep.type == tor::DependenceResult::NotDependent)
                    continue;

                if (dep.type == tor::DependenceResult::Always) {
                    if (canReach(memop1, memop2, false))
                        Distance = 0;
                    else if (canReach(memop1, memop2, true))
                        Distance = 1;
                }

                if (dep.type == tor::DependenceResult::Dependent) {
                    // When memop1 cannot reach memop2 without loop-back edge,
                    // memop1 doesn't depends on memop2 even when their distance is
                    // zero
                    if (dep.dist == 0 && !canReach(memop1, memop2, false))
                        continue;
                    Distance = dep.dist;
                }
            }

            // memop1 can reach memop2 without loop-back edge
            if (Distance == -1 && canReach(memop1, memop2, false))
                Distance = 0;

            // memop1 can reach memop2 using some loop-back edge,
            // pessimiticaly assume distance to be 1
            if (!annotated && Distance == -1 && canReach(memop1, memop2, true))
                Distance = 1;

            // When memop1 and memop2 are not in the same BB and not in the same parent loop,
            // then there is no dependence between them even when their distance=0
            if (Distance == 0 && memop1->getParentBB() != memop2->getParentBB()
                && memop1->getParentLoop() != memop2->getParentLoop()) {
                Distance = -1;
            }

            // memop1 can't reach memop2 (e.g. mutually exclusive branch of an
            // if statement)
            if (Distance == -1)
                continue;

            if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
                op2->getType() == OpAbstract::OpType::STORE_OP) {
                // WAR
                addDependency(Dependence(memop1, memop2, Distance, Dependence::D_WAR));
            } else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
                       op2->getType() == OpAbstract::OpType::LOAD_OP) {
                // RAW
                addDependency(Dependence(memop1, memop2, Distance, Dependence::D_RAW));
            } else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
                       op2->getType() == OpAbstract::OpType::STORE_OP) {
                // WAW
                addDependency(Dependence(memop1, memop2, Distance, Dependence::D_WAW));
            } else {
                // RAR
                addDependency(Dependence(memop1, memop2, Distance, Dependence::D_RAR));
            }
          }
        }

        // Dependencies between regular memory operations and burst operations
        // Burst operations should wait for all regular operations on the same memref to complete
        for (auto &&op1: Operations) {
            auto memop1 = op1->getMemOp();
            if (memop1 == nullptr)
                continue;

            for (auto &&op2 : Operations) {
                auto maxiop2 = op2->getMAxiOp();
                if (maxiop2 == nullptr || op1.get() == op2.get())
                    continue;

                // Only process burst operations (M_AXI burst and TileLink)
                if (op2->getType() != OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                    op2->getType() != OpAbstract::OpType::M_AXI_BURST_WRITE_OP &&
                    op2->getType() != OpAbstract::OpType::TL_READ_OP &&
                    op2->getType() != OpAbstract::OpType::TL_WRITE_OP)
                    continue;

                // Check if memrefs match
                // For aps.memburstload/aps.memburststore, we need to check all variadic memrefs
                bool memrefMatches = false;
                if (auto memBurstLoadOp = llvm::dyn_cast<aps::MemBurstLoad>(op2->getOp())) {
                    // Check if any of the burst load's memrefs match
                    for (auto memref : memBurstLoadOp.getMemrefs()) {
                        if (isSameGlobalMemref(memop1->getMemRef(), memref)) {
                            memrefMatches = true;
                            break;
                        }
                    }
                } else if (auto memBurstStoreOp = llvm::dyn_cast<aps::MemBurstStore>(op2->getOp())) {
                    // Check if any of the burst store's memrefs match
                    for (auto memref : memBurstStoreOp.getMemrefs()) {
                        if (isSameGlobalMemref(memop1->getMemRef(), memref)) {
                            memrefMatches = true;
                            break;
                        }
                    }
                } else {
                    // For regular burst operations (tor::AXI*), use the single memref
                    memrefMatches = isSameGlobalMemref(memop1->getMemRef(), maxiop2->getMemRef());
                }

                if (!memrefMatches) {
                    continue;
                }

                // Debug: log when memrefs match
                llvm::errs() << "DEBUG: Memref match found between "
                             << op1->getOp()->getName() << " and "
                             << op2->getOp()->getName() << "\n";
                llvm::errs() << "  memop1 ParentBB: " << memop1->getParentBB()
                             << ", ParentLoop: " << memop1->getParentLoop() << "\n";
                llvm::errs() << "  maxiop2 ParentBB: " << maxiop2->getParentBB()
                             << ", ParentLoop: " << maxiop2->getParentLoop() << "\n";

                // Check if memop1 can reach maxiop2
                int Distance = -1;
                bool canReachDirect = canReach(memop1, maxiop2, false);
                bool canReachLoop = canReach(memop1, maxiop2, true);

                llvm::errs() << "  canReach(false): " << canReachDirect
                             << ", canReach(true): " << canReachLoop << "\n";

                // Special case: if memop1 is inside a loop and maxiop2 is outside the loop,
                // this is a loop-exit dependency and should have Distance = 0
                if (memop1->getParentLoop() != nullptr && maxiop2->getParentLoop() == nullptr) {
                    // Check if maxiop2 is in a block that follows the loop
                    if (canReachDirect || canReachLoop) {
                        Distance = 0;
                        llvm::errs() << "  Special case: loop-exit dependency, setting Distance = 0\n";
                    }
                } else if (canReachDirect) {
                    Distance = 0;
                } else if (canReachLoop) {
                    Distance = 1;
                }

                if (Distance == -1) {
                    llvm::errs() << "  Distance == -1, skipping dependency\n";
                    continue;
                }

                llvm::errs() << "  Distance: " << Distance << ", adding dependency\n";

                // Add appropriate dependencies
                if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
                    (op2->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP ||
                     op2->getType() == OpAbstract::OpType::TL_WRITE_OP)) {
                    // LOAD -> BURST_WRITE/TL_WRITE: WAR
                    addDependency(Dependence(memop1, maxiop2, Distance, Dependence::D_WAR));
                } else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
                           (op2->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP ||
                            op2->getType() == OpAbstract::OpType::TL_READ_OP)) {
                    // STORE -> BURST_READ/TL_READ: WAR
                    addDependency(Dependence(memop1, maxiop2, Distance, Dependence::D_WAR));
                } else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
                           (op2->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP ||
                            op2->getType() == OpAbstract::OpType::TL_WRITE_OP)) {
                    // STORE -> BURST_WRITE/TL_WRITE: RAW (burst write reads from memory)
                    addDependency(Dependence(memop1, maxiop2, Distance, Dependence::D_RAW));
                } else if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
                           (op2->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP ||
                            op2->getType() == OpAbstract::OpType::TL_READ_OP)) {
                    // LOAD -> BURST_READ/TL_READ: RAR (no true dependency, but for ordering)
                    addDependency(Dependence(memop1, maxiop2, Distance, Dependence::D_RAR));
                }
            }
        }

        // Dependencies from burst operations to regular memory operations
        for (auto &&op1: Operations) {
            auto maxiop1 = op1->getMAxiOp();
            if (maxiop1 == nullptr)
                continue;

            // Only process burst operations (M_AXI burst and TileLink)
            if (op1->getType() != OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                op1->getType() != OpAbstract::OpType::M_AXI_BURST_WRITE_OP &&
                op1->getType() != OpAbstract::OpType::TL_READ_OP &&
                op1->getType() != OpAbstract::OpType::TL_WRITE_OP)
                continue;

            for (auto &&op2 : Operations) {
                auto memop2 = op2->getMemOp();
                if (memop2 == nullptr || op1.get() == op2.get())
                    continue;

                // Check if memrefs match
                // For aps.memburstload/aps.memburststore, we need to check all variadic memrefs
                bool memrefMatches = false;
                if (auto memBurstLoadOp = llvm::dyn_cast<aps::MemBurstLoad>(op1->getOp())) {
                    // Check if any of the burst load's memrefs match
                    for (auto memref : memBurstLoadOp.getMemrefs()) {
                        if (isSameGlobalMemref(memref, memop2->getMemRef())) {
                            memrefMatches = true;
                            break;
                        }
                    }
                } else if (auto memBurstStoreOp = llvm::dyn_cast<aps::MemBurstStore>(op1->getOp())) {
                    // Check if any of the burst store's memrefs match
                    for (auto memref : memBurstStoreOp.getMemrefs()) {
                        if (isSameGlobalMemref(memref, memop2->getMemRef())) {
                            memrefMatches = true;
                            break;
                        }
                    }
                } else {
                    // For regular burst operations (tor::AXI*), use the single memref
                    memrefMatches = isSameGlobalMemref(maxiop1->getMemRef(), memop2->getMemRef());
                }

                if (!memrefMatches)
                    continue;

                // Debug: log when memrefs match
                llvm::errs() << "DEBUG: Memref match found (reverse) between "
                             << op1->getOp()->getName() << " and "
                             << op2->getOp()->getName() << "\n";
                llvm::errs() << "  maxiop1 ParentBB: " << maxiop1->getParentBB()
                             << ", ParentLoop: " << maxiop1->getParentLoop() << "\n";
                llvm::errs() << "  memop2 ParentBB: " << memop2->getParentBB()
                             << ", ParentLoop: " << memop2->getParentLoop() << "\n";

                // Check if maxiop1 can reach memop2
                int Distance = -1;
                bool canReachDirect = canReach(maxiop1, memop2, false);
                bool canReachLoop = canReach(maxiop1, memop2, true);

                llvm::errs() << "  canReach(burst->mem, false): " << canReachDirect
                             << ", canReach(burst->mem, true): " << canReachLoop << "\n";

                if (canReachDirect)
                    Distance = 0;
                else if (canReachLoop)
                    Distance = 1;

                if (Distance == -1) {
                    llvm::errs() << "  Distance == -1, skipping dependency\n";
                    continue;
                }

                llvm::errs() << "  Distance: " << Distance << ", adding dependency from burst to mem\n";

                // Add appropriate dependencies
                if ((op1->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP ||
                     op1->getType() == OpAbstract::OpType::TL_READ_OP) &&
                    op2->getType() == OpAbstract::OpType::LOAD_OP) {
                    // BURST_READ/TL_READ -> LOAD: RAW (burst read writes to memory)
                    addDependency(Dependence(maxiop1, memop2, Distance, Dependence::D_RAW));
                } else if ((op1->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP ||
                            op1->getType() == OpAbstract::OpType::TL_READ_OP) &&
                           op2->getType() == OpAbstract::OpType::STORE_OP) {
                    // BURST_READ/TL_READ -> STORE: WAW (burst read writes to memory)
                    addDependency(Dependence(maxiop1, memop2, Distance, Dependence::D_WAW));
                } else if ((op1->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP ||
                            op1->getType() == OpAbstract::OpType::TL_WRITE_OP) &&
                           op2->getType() == OpAbstract::OpType::LOAD_OP) {
                    // BURST_WRITE/TL_WRITE -> LOAD: WAR
                    addDependency(Dependence(maxiop1, memop2, Distance, Dependence::D_WAR));
                } else if ((op1->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP ||
                            op1->getType() == OpAbstract::OpType::TL_WRITE_OP) &&
                           op2->getType() == OpAbstract::OpType::STORE_OP) {
                    // BURST_WRITE/TL_WRITE -> STORE: RAR (no true dependency, but for ordering)
                    addDependency(Dependence(maxiop1, memop2, Distance, Dependence::D_RAR));
                }
            }
        }

        // Dependencies between burst operations themselves
        // Burst operations must be serialized to respect resource constraints and memory ordering
        // Note: TileLink operations are excluded here as they are handled by addResourceConstr with amount limit
        for (auto &&op1: Operations) {
            auto maxiop1 = op1->getMAxiOp();
            if (maxiop1 == nullptr)
                continue;

            // Only process burst operations
            if (op1->getType() != OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                op1->getType() != OpAbstract::OpType::M_AXI_BURST_WRITE_OP)
                continue;

            for (auto &&op2 : Operations) {
                auto maxiop2 = op2->getMAxiOp();
                if (maxiop2 == nullptr || op1.get() == op2.get())
                    continue;

                // Only process burst operations
                if (op2->getType() != OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                    op2->getType() != OpAbstract::OpType::M_AXI_BURST_WRITE_OP)
                    continue;

                // Check if maxiop1 can reach maxiop2
                int Distance = -1;
                if (canReach(maxiop1, maxiop2, false))
                    Distance = 0;
                else if (canReach(maxiop1, maxiop2, true))
                    Distance = 1;

                if (Distance == -1)
                    continue;

                // Add appropriate dependencies based on operation types
                if (op1->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                    op2->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP) {
                    // BURST_READ -> BURST_READ: enforce ordering for resource sharing
                    // If they access the same memref, enforce WAW semantics
                    if (isSameGlobalMemref(maxiop1->getMemRef(), maxiop2->getMemRef())) {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_WAW));
                    } else {
                        // Different memrefs, but still need to serialize for resource constraint
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAR));
                    }
                } else if (op1->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP &&
                           op2->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP) {
                    // BURST_READ -> BURST_WRITE
                    if (isSameGlobalMemref(maxiop1->getMemRef(), maxiop2->getMemRef())) {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_WAR));
                    } else {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAR));
                    }
                } else if (op1->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP &&
                           op2->getType() == OpAbstract::OpType::M_AXI_BURST_READ_OP) {
                    // BURST_WRITE -> BURST_READ
                    if (isSameGlobalMemref(maxiop1->getMemRef(), maxiop2->getMemRef())) {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAW));
                    } else {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAR));
                    }
                } else if (op1->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP &&
                           op2->getType() == OpAbstract::OpType::M_AXI_BURST_WRITE_OP) {
                    // BURST_WRITE -> BURST_WRITE: enforce ordering for resource sharing
                    if (isSameGlobalMemref(maxiop1->getMemRef(), maxiop2->getMemRef())) {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAR));
                    } else {
                        addDependency(Dependence(maxiop1, maxiop2, Distance, Dependence::D_RAR));
                    }
                }
            }
        }
    }

    void getDescendants(
        mlir::Operation* currOp, mlir::Operation* ancestorOp,
        const llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>>& callOpChildrenMap,
        llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>>& callOpDescendantsMap)
    {
        auto iter = callOpChildrenMap.find(currOp);
        if (iter == callOpChildrenMap.end()) {
            return;
        }
        for (auto& op : iter->second) {
            if (callOpDescendantsMap[ancestorOp].find(op) != callOpDescendantsMap[ancestorOp].end())
                continue;
            getDescendants(op, ancestorOp, callOpChildrenMap, callOpDescendantsMap);
        }
    }

    llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>>
        getCallOpUsesMap(tor::DesignOp designOp,
            const llvm::DenseMap<StringRef, mlir::Operation*>& nameToFuncOp)
    {
        llvm::DenseMap<StringRef, mlir::Operation*> nameToOp;
        designOp.walk([&](tor::FuncOp funcOp) {
            nameToOp[funcOp.getName()] = funcOp;
        });

        // key: funcOp, value: its children, direct use by funcOp
        llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>> callOpChildrenMap;
        for (auto &[name, op] : nameToOp) {
            auto funcOp = llvm::dyn_cast<mlir::tor::FuncOp>(op);
            funcOp.walk([&](tor::CallOp callOp) {
                auto callOpName = callOp.getCallee();
                auto funcOpIt = nameToOp.find(callOpName);
                assert(funcOpIt != nameToOp.end());
                callOpChildrenMap[funcOp].insert(funcOpIt->second);
            });
        }

        // key: funcOp, value: its descendant funcOp, direct or indirect use by key (funcOp)
        llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>> callOpDescendantsMap;
        for (auto &[name, funcOp] : nameToFuncOp) {
            getDescendants(funcOp, funcOp, callOpChildrenMap, callOpDescendantsMap);
        }

        // key: funcOp, value: its ancestors funcOps, direct or indirect use them by callOp
        llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>> callOpAncestorsMap;
        // llvm::outs() << "print ancestor map:\n";
        for (auto &[ancestorOp, descendantsOp] : callOpDescendantsMap) {
            for (auto descedantOp : descendantsOp) {
                callOpAncestorsMap[descedantOp].insert(ancestorOp);
                // llvm::outs() << "\tdescedantOp " << descedantOp << ", andcestorOp " << ancestorOp << "\n";
            }
        }
        return callOpAncestorsMap;
    }

    llvm::DenseMap<StringRef, mlir::Operation*>
        getStringRefToFuncOp(tor::DesignOp designOp,
            const llvm::DenseMap<StringRef, OpAbstract*>& callOpToOpAbstract)
    {
        llvm::DenseMap<StringRef, mlir::Operation*> callOpToFuncOp;
        designOp.walk([&](tor::FuncOp funcOp) {
            if (callOpToOpAbstract.find(funcOp.getName()) != callOpToOpAbstract.end())
                callOpToFuncOp[funcOp.getName()] = funcOp;
        });
        return callOpToFuncOp;
    }

    mlir::Operation* opAbstractToOperation(OpAbstract *opA,
        const llvm::DenseMap<StringRef, mlir::Operation*>& nameToFuncOp)
    {
        if (opA->getType() == OpAbstract::OpType::CALL_OP) {
            auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(opA->getOp());
            return nameToFuncOp.at(callOp.getCallee());
        }
        return opA->getOp();
    }

    bool isDataDependent(
        const llvm::DenseMap<mlir::Value, std::unordered_set<mlir::Operation*>>& memrefUseOp,
        const llvm::DenseMap<StringRef, mlir::Operation*>& nameToFuncOp,
        OpAbstract *otherOp, OpAbstract * callOp)
    {
        mlir::Operation *op1 = opAbstractToOperation(otherOp, nameToFuncOp);
        mlir::Operation *op2 = opAbstractToOperation(callOp, nameToFuncOp);
        for (auto&[val, useSet] : memrefUseOp) {
            if (useSet.find(op1) != useSet.end() && useSet.find(op2) != useSet.end()) {
                // llvm::outs() << "op1 (" << *op1 << ") and op2 (" << *op2 << ") is data dependent\n";
                return true;
            }
        }
        return false;
    }

    bool ScheduleBase::isLocalOp(mlir::Operation* op) const {
        auto currOp = op;
        while (currOp) {
            if (currOp == containingOp)
                return true;
            currOp = currOp->getParentOp();
        }
        return false;
    }

    mlir::Operation* getParentFuncOp(mlir::Operation* op) {
        auto currOp = op;
        while (currOp) {
            if (llvm::isa<tor::FuncOp>(currOp)) {
                break;
            }
            currOp = currOp->getParentOp();
        }
        return currOp;
    }

    void ScheduleBase::buildCallOpTensorDFG() {
        llvm::DenseMap<StringRef, OpAbstract*> callOpToOpAbstract; // call op name to OpAbstract
        for (auto &&BB: BasicBlocks) {
            for (auto op: llvm::reverse(BB->getOperations())) {
                if (op->getType() == OpAbstract::OpType::CALL_OP) {
                    auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op->getOp());
                    callOpToOpAbstract[callOp.getCallee()] = op;
                }
            }
        }
        if (callOpToOpAbstract.empty()) {
            return;
        }

        // dependence of tensors in callop in the same basicblocks
        // [TODO] maybe need update if support basicblocks run in parallel
        auto funcOp = llvm::dyn_cast<tor::FuncOp>(containingOp);
        auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
        auto nameToFuncOp = getStringRefToFuncOp(designOp, callOpToOpAbstract);
        auto callOpUsesMap = getCallOpUsesMap(designOp, nameToFuncOp);
        llvm::DenseMap<mlir::Value, std::unordered_set<mlir::Operation*>> memrefUseOp;
        designOp.walk([&](tor::AllocOp alloc) {
            auto value = alloc.getResult();
            // llvm::outs() << "memref value: " << value << "\n";
            for (auto op : value.getUsers()) {
                // 假如是containingOp直接调用的call funcOp里面直接或间接使用到的
                if (isLocalOp(op)) {
                    memrefUseOp[value].insert(op);
                    // llvm::outs() << "\t" << *op << "\n";
                } else {
                    auto parentOp = getParentFuncOp(op);
                    if (parentOp) {
                        auto parentFuncOp = llvm::dyn_cast<tor::FuncOp>(parentOp);
                        auto usesIt = callOpUsesMap.find(parentOp);
                        if (usesIt != callOpUsesMap.end()) {
                            for (auto& ancestorOp : usesIt->second) {
                                memrefUseOp[value].insert(ancestorOp);
                                // llvm::outs() << "\t" << *ancestorOp << "\n";
                            }
                        } else if (callOpToOpAbstract.find(parentFuncOp.getName()) != callOpToOpAbstract.end()) {
                            memrefUseOp[value].insert(parentOp);
                            // llvm::outs() << "\t" << *parentOp << "\n";
                        }
                    }
                }
            }
        });


        for (auto &&BB: BasicBlocks) {
            std::vector<OpAbstract *> previousOps;
            for (auto op: llvm::reverse(BB->getOperations())) {
                if (op->getType() == OpAbstract::OpType::CALL_OP) {
                    for (auto prev: previousOps) {
                        if (isDataDependent(memrefUseOp, nameToFuncOp, prev, op)) {
                            addDependency(Dependence(prev, op, 0, Dependence::D_WAW));
                        }
                    }
                }
                previousOps.push_back(op);
            }

            std::vector<OpAbstract *> succOps;
            for (auto op: BB->getOperations()) {
                if (op->getType() == OpAbstract::OpType::CALL_OP) {
                    for (auto succ: succOps)
                        if (isDataDependent(memrefUseOp, nameToFuncOp, succ, op)) {
                            addDependency(Dependence(op, succ, 0, Dependence::D_WAW));
                        }
                } else {
                    succOps.push_back(op);
                }
            }
        }
    }

    void ScheduleBase::buildDFG() {
        buildScalarDFG();
        buildTensorDFG();
        buildCallOpTensorDFG();
        buildStreamDFG();
        buildMAxiDFG();
    }

    void ScheduleBase::buildFromContainingOp() {
        if (auto funcOp = llvm::dyn_cast<tor::FuncOp>(containingOp)) {
            /// instantiate a funOp
            Region &region = funcOp.getBody();
            auto bbs = buildCFG(region.front(), nullptr);

            /// add function op at beginning
            OpAbstract *funcArgOp = createOp(
                containingOp, nullptr, bbs.first,
                {funcOp.getArguments().begin(), funcOp.getArguments().end()},
                {}, OpAbstract::OpType::ASSIGN_OP);

            for (Value v : funcOp.getArguments())
              ValueMap[v] = funcArgOp;

            for (auto &&BB: BasicBlocks)
                if (BB->getBranchValue().getImpl() != nullptr) {
                  if (isa<BlockArgument>(BB->getBranchValue()) ||
                      !llvm::isa<arith::ConstantOp>(
                          BB->getBranchValue().getDefiningOp())) {
                    // Special handling for constant branch value
                    BB->setBranchOp(ValueMap[BB->getBranchValue()]);
                    assert(ValueMap[BB->getBranchValue()]);
                  }
                }

            // We need to manually add the constantOp which is not in the current
            // funcOp.
            auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
            for (auto &op: designOp.getBody().front().getOperations())
                if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
                    OpAbstract *opA = createOp(constOp, nullptr, bbs.first,
                                               {constOp.getResult()}, SmallVector<Value>{});

                    ValueMap[constOp.getResult()] = opA;
                    bbs.first->addOperation(opA);
                }

            bbs.first->addOperation(funcArgOp);
            EntryBB = bbs.first, ExitBB = bbs.second;

            buildDFG();
            // printCDFG();
        }
    }

    void ScheduleBase::printCDFG() {
        for (unsigned i = 0, n = BasicBlocks.size(); i < n; ++i) {
            llvm::outs() << "BasicBlock " << i << ": " << BasicBlocks[i].get() << "\n";
            llvm::outs() << "===================================\n";

            for (auto op: BasicBlocks[i]->getOperations()) {
                if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
                    llvm::outs() << op->getOp()->getName().getStringRef().str() << ": \n";
                } else if (op->getType() == OpAbstract::OpType::PHI_OP) {
                    llvm::outs() << "PHI: \n";
                } else if (op->getType() == OpAbstract::OpType::ASSIGN_OP) {
                    llvm::outs() << "ASSIGN: \n";
                } else if (op->getType() == OpAbstract::OpType::LOAD_OP) {
                    llvm::outs() << "LOAD: \n";
                } else if (op->getType() == OpAbstract::OpType::STORE_OP) {
                    llvm::outs() << "STORE: \n";
                } else if (op->getType() == OpAbstract::OpType::CALL_OP) {
                    llvm::outs() << "CALL: \n";
                }

                llvm::outs() << "operands(";
                for (auto opr: op->getOperands())
                    llvm::outs() << mlir::hash_value(opr) << ", ";
                llvm::outs() << ")\n";

                llvm::outs() << "results(";
                for (auto res: op->getResults())
                    llvm::outs() << mlir::hash_value(res) << ", ";
                llvm::outs() << ")\n";

                llvm::outs() << "ParentLoop: " << op->getParentLoop() << "\n";
            }

            llvm::outs() << "Successor BB: \n -------------------------\n";
            for (auto &succ: BasicBlocks[i]->getSucc()) {
                if (succ.type == ControlEdge::FORWARD)
                    llvm::outs() << "Forward Edge: ";
                else if (succ.type == ControlEdge::LOOPBACK)
                    llvm::outs() << "Loop Back Edge: ";
                else if (succ.type == ControlEdge::COND)
                    llvm::outs() << "Condition Edge: ";
                llvm::outs() << succ.toBB << "\n";
            }
            llvm::outs() << "----------------------\n";

            llvm::outs() << "END of printCDFG===================================\n";
        }

        llvm::outs() << "\n";
        llvm::outs() << "Data Dependencies: \n ===========================\n";
        for (auto &D: Dependencies) {
            llvm::outs() << D->SourceOp->getOp()->getName().getStringRef() << " -> "
                         << D->DestinationOp->getOp()->getName().getStringRef()
                         << "  Distance: " << D->Distance << "\n";
        }
        llvm::outs() << "=============================\n";
    }

    void ScheduleBase::printSchedule() {
        for (unsigned i = 0, n = BasicBlocks.size(); i < n; ++i) {
            llvm::outs() << "BasicBlock " << i << ": " << BasicBlocks[i].get() << "\n";
            llvm::outs() << "=====================================\n";

            for (auto op: BasicBlocks[i]->getOperations()) {
                if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
                    llvm::outs() << op->getOp()->getName().getStringRef().str()
                                 << ": operands(";
                    for (auto opr: op->getOperands())
                        llvm::outs() << mlir::hash_value(opr) << ", ";
                    llvm::outs() << ")";
                } else if (op->getType() == OpAbstract::OpType::PHI_OP) {
                    llvm::outs() << "PHI: operands(";
                    for (auto opr: op->getOperands())
                        llvm::outs() << mlir::hash_value(opr) << ", ";
                    llvm::outs() << ")";
                } else if (op->getType() == OpAbstract::OpType::ASSIGN_OP) {
                    llvm::outs() << "ASSIGN: operands(";
                    for (auto opr: op->getOperands())
                        llvm::outs() << mlir::hash_value(opr) << ", ";
                    llvm::outs() << ")";
                } else if (op->getType() == OpAbstract::OpType::CALL_OP) {
                    llvm::outs() << "CALL: \n";
                    llvm::outs() << "operands(";
                    for (auto opr: op->getOperands())
                        llvm::outs() << mlir::hash_value(opr) << ", ";
                    llvm::outs() << ")\n";

                    llvm::outs() << "results(";
                    for (auto res: op->getResults())
                        llvm::outs() << mlir::hash_value(res) << ", ";
                    llvm::outs() << ")\n";
                }

                llvm::outs() << " at cycle " << op->getStartTime() << "\n";
            }

            llvm::outs() << "=====================================\n";
        }
    }

    LogicalResult ScheduleBase::verify() { return success(); }

} // namespace scheduling
