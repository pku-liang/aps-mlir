#ifndef SCHEDULE_RESOURCEDB_H
#define SCHEDULE_RESOURCEDB_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "TOR/TOR.h"
#include "TOR/Utils.h"
#include "APS/APSOps.h"
// #include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <functional>

const int BIT_WIDTH_TYPE = 7;

namespace scheduling {

/// @brief A hardware component. Can either be a builtin component
///        or retrieved from a function without side effect.
struct Component {
  std::string name;         /// Name of the component.
  std::vector<float> delay; /// combinatorial delay of the component. Different
                            /// bitwidth may have different delay
  std::vector<int> latency; /// latency of the component of different bitwidth
  int II; /// Initial interval of a pipelined component. Equals to latency when
          /// not a pipeline function
  bool constr; /// should this components has constrained usage
  int amount;  /// Amount of resources of this kind. -1 when the resource can't
               /// be shared

  Component(const std::string &name, const std::vector<float> &delay,
            const std::vector<int> &latency, int II, bool constr = -1,
            int amount = -1)
      : name(name), delay(delay), latency(latency), II(II), constr(constr),
        amount(amount) {}

  Component(const std::string &name, float d, int l, int II, bool constr,
            int amount)
      : name(name), II(II), constr(constr), amount(amount) {
    delay.resize(BIT_WIDTH_TYPE, d);
    latency.resize(BIT_WIDTH_TYPE, l);
  }
};

/// @brief This class contains the allocation information for the scheduler
class ResourceDB {
public:
  int getResourceID(mlir::Operation *op) {
    auto name = op->getName().stripDialect().str();
    if (op->hasAttr("latency") || op->hasAttr("bind_op_latency") || op->hasAttr("impl")) {
      assert(NameToID.find(name) != NameToID.end() && "user defined bind op name not found in RDB");
      if (Components[NameToID[name]].latency.back() == 0) {
        setPragmaStructureAttrStatusByOp(op, "bind_op", false);
        llvm::errs() << "warning: op_bind does not support combinational op " << name
                     << "\n\tDowngrade to default setting.\n";
        if (op->hasAttr("bind_op_latency"))
          op->removeAttr("bind_op_latency");
        if (op->hasAttr("bind_op_impl"))
          op->removeAttr("bind_op_impl");
      } else {
        setPragmaStructureAttrStatusByOp(op, "bind_op");
        return addOrGetResourceID(name, op);
      }
    }

    if (NameToID.find(name) != NameToID.end())
      return NameToID[name];

    if (llvm::isa<mlir::func::CallOp>(op) || llvm::isa<mlir::tor::CallOp>(op)) {
      return NameToID["call"];
      // FIXME: Here we assume callop's latency is 1. The following code
      // should be used after we compute the latency for each function.
      /*
      auto func = callOp.getCallee().str();
      if (NameToID.find(func) != NameToID.end())
        return NameToID[func];
      */
    }

    return NameToID["nop"];
    // llvm::errs() << name << "\n";
    // assert(0 && "Error: Can't find corresponding resource");
  }

  int getResourceID(const std::string &str) {
    if (NameToID.find(str) != NameToID.end())
      return NameToID[str];
    llvm::outs() << str << "\n";
    assert(0 && "Error: Can't find corresponding resource");
  }

  bool hasResource(const std::string &str) {
    return NameToID.find(str) != NameToID.end();
  }

  int getII(int id) { return Components[id].II; }

  int bitwidthIdx(int bitwidth) {
    if (!(bitwidth == 0 || __builtin_popcount(bitwidth) == 1)) {
      llvm::errs() << "ERROR: Invalid bitwidth " << bitwidth << " (not a power of 2)\n";
    }
    assert((bitwidth == 0 || __builtin_popcount(bitwidth) == 1) &&
           "bitwidth should be the power of 2");
    return (__CHAR_BIT__ * sizeof(uint32_t) - __builtin_clz(bitwidth | 1) - 1);
  }

  float getDelay(int id, int bitwidth) {
    // Normalize bitwidth to nearest power of 2 (round up)
    if (bitwidth > 0 && __builtin_popcount(bitwidth) != 1) {
      bitwidth = 1 << (32 - __builtin_clz(bitwidth - 1));
    }
    int index = bitwidthIdx(bitwidth);
    return Components[id].delay[index];
  }

  std::string getName(int id) { return Components[id].name; }

  int getLatency(int id, int bitwidth) {
    // Normalize bitwidth to nearest power of 2 (round up)
    // This handles constants with arbitrary bitwidths (e.g., 5-bit constants)
    if (bitwidth > 0 && __builtin_popcount(bitwidth) != 1) {
      // Round up to next power of 2
      int normalized = 1 << (32 - __builtin_clz(bitwidth - 1));
      // llvm::errs() << "getLatency: normalizing bitwidth " << bitwidth
      //              << " -> " << normalized << " for resource " << Components[id].name << "\n";
      bitwidth = normalized;
    }
    int index = bitwidthIdx(bitwidth);
    if (Components[id].name == "addf" && index != 6) {
//      llvm::outs() << "error\n";
    }
    return Components[id].latency[index];
  }

  int getNumResource() { return Components.size(); }

  int getAmount(int id) { return Components[id].amount; }

  bool isCombLogic(int id, int bitwidth) {
    // Normalize bitwidth to nearest power of 2 (round up)
    if (bitwidth > 0 && __builtin_popcount(bitwidth) != 1) {
      bitwidth = 1 << (32 - __builtin_clz(bitwidth - 1));
    }
    int index = bitwidthIdx(bitwidth);
    return Components[id].latency[index] == 0;
  }

  // bool hasHardLimit(int id) { return Components[id].constr == true; }

  bool hasHardLimit(int id) { return Components[id].amount != -1; }

  ResourceDB() {
    NameToID.clear();
    Components.clear();
  }

  ResourceDB(nlohmann::json &config) {
    for (auto &res : config.items()) {
      auto info = res.value();

      std::string name = res.key();
      std::vector<float> delay;
      std::vector<int> latency;
      int amount = -1;
      int II = 0;
      bool constr = false;

      for (auto &item : info.items()) {
        if (item.key() == "delay") {
          for (auto &f : item.value().items())
            delay.push_back(f.value().get<float>());
        } else if (item.key() == "latency") {
          for (auto &f : item.value().items())
            latency.push_back(f.value().get<int>());
        } else if (item.key() == "amount") {
          amount = item.value().get<int>();
        } else if (item.key() == "II") {
          II = item.value().get<int>();
        } else if (item.key() == "constr") {
          constr = item.value().get<int>();
        }
      }

      addComponent(Component(name, delay, latency, II, constr, amount));
      if (name == "memport") {
        addComponent(Component("memport_RAM_1P", delay, latency, II, true, 1));
        addComponent(Component("memport_RAM_T2P", delay, latency, II, true, 2));
      }
    }
    addComponent(Component("call", 0, 1, 1, false, -1));
    addComponent(Component("m_axi_read", 0, 2, 2, true, 1)); // II不可参考, 操作本身不支持pipeline
    addComponent(Component("m_axi_write", 0, 3, 3, true, 1)); // II不可参考, 操作本身不支持pipeline
    addComponent(Component("m_axi_burst", 0, 16, 15, true, 1)); // latency = 15, II不可参考, 操作本身不支持pipeline
  }

  void addUsage(mlir::tor::FuncOp funcOp, std::vector<int> &usage) {
    usages[funcOp] = usage;
  }

  const std::vector<int>& getUsage(mlir::tor::FuncOp funcOp) const {
    return usages.at(funcOp);
  }

  bool isUsageSatisfied(const std::vector<int>& usage) {
    for (size_t i = 0; i < usage.size(); ++i) {
      int amount = getAmount(i);
      if (amount == -1 || amount >= usage[i]) {
        continue;
      }
      return false;
    }
    return true;
  }

  /// Get or create a per-memref resource ID for APS memory operations
  /// This allows different memrefs to be scheduled independently
  ///
  /// All memories are now treated as 1RW (one-read-one-write) with amount=1,
  /// meaning only one read or write operation can occur per cycle.
  ///
  /// Special handling for _cpu_memory: latency is 1 cycle, others have 0 latency
  int getOrCreateMemrefResource(mlir::Value memref) {
    // Generate a unique name for this memref resource
    std::string memrefName = "memport_unknown";
    bool isCpuMemory = false;

    llvm::errs() << "getOrCreateMemrefResource called for memref\n";
    if (auto defOp = memref.getDefiningOp()) {
      llvm::errs() << "  defOp: " << defOp->getName() << "\n";
      if (auto getGlobalOp = llvm::dyn_cast<mlir::memref::GetGlobalOp>(defOp)) {
        std::string globalName = getGlobalOp.getName().str();
        llvm::errs() << "  Found GetGlobalOp, globalName=" << globalName << "\n";
        // Check if this is _cpu_memory
        isCpuMemory = (globalName.find("_cpu_memory") != std::string::npos);
        // All memories are 1RW regardless of attributes
        memrefName = "memport_" + globalName + "_1rw";
      } else if (auto allocOp = llvm::dyn_cast<tor::AllocOp>(defOp)) {
        // For tor.alloc, always treat as 1RW
        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        os << defOp;
        size_t hash = std::hash<std::string>{}(os.str());
        memrefName = "memport_alloc_1rw_" + std::to_string(hash);
        isCpuMemory = false; // tor.alloc is not _cpu_memory
      } else if (defOp->getName().getStringRef().str() == "aps.memdeclare") {
        // For aps.memdeclare, always treat as 1RW
        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        os << defOp;
        size_t hash = std::hash<std::string>{}(os.str());
        memrefName = "memport_aps_1rw_" + std::to_string(hash);
        isCpuMemory = false; // aps.memdeclare is not _cpu_memory
      }
    }

    // Check if resource already exists
    if (NameToID.find(memrefName) != NameToID.end()) {
      llvm::errs() << "  Returning existing resource: " << memrefName
                   << " (ID=" << NameToID[memrefName] << ")\n";
      return NameToID[memrefName];
    }

    // Create a new resource with the same properties as base memport
    int baseMemportId = NameToID["memport"];
    const Component& baseMemport = Components[baseMemportId];

    // All memories are now 1RW: single port with resource constraint
    int amount = 1;         // 1 means single port (one read or write per cycle)
    bool hasConstraint = true;  // All memories have resource constraints

    // Keep base memport latency for all APS memories
    // Special handling for load/store will be done at scheduling constraint generation level
    std::vector<int> latency = baseMemport.latency;
    llvm::errs() << "  Using base latency=" << baseMemport.latency[0] << " (isCpuMemory=" << isCpuMemory << ")\n";

    llvm::errs() << "  Creating new resource: " << memrefName
                 << " amount=" << amount << " hasConstraint=" << hasConstraint << "\n";

    // Create a new component with the appropriate latency
    Component newComp(memrefName, baseMemport.delay, latency,
                      baseMemport.II, hasConstraint, amount);
    addComponent(newComp);

    llvm::errs() << "  New resource ID=" << NameToID[memrefName] << "\n";
    return NameToID[memrefName];
  }

private:
  std::vector<int> getOrFindLatency(const std::string& name, mlir::Operation *op, bool& isValid) {
    if (op->hasAttr("latency")) {
      return std::vector<int>(BIT_WIDTH_TYPE, dyn_cast<mlir::IntegerAttr>(op->getAttr("latency")).getInt());
    }

    if (op->hasAttr("bind_op_latency")) {
      auto latency = dyn_cast<mlir::IntegerAttr>(op->getAttr("bind_op_latency")).getInt();
      if (latency == 0) {
        llvm::errs() << "warning: Cannot set op_bind latency to 0 " << name
                     << "\n\tDowngrade to default latency.\n";
      } else if (latency > 0) {
        return std::vector<int>(BIT_WIDTH_TYPE, latency);
      }
      isValid = false;
    }
    return Components[NameToID[name]].latency;
  }

  std::string getImpl(mlir::Operation *op) {
    if (op->hasAttr("impl")) {
      return dyn_cast<mlir::StringAttr>(op->getAttr("impl")).getValue().str();
    }

    if (op->hasAttr("bind_op_impl")) {
      return dyn_cast<mlir::StringAttr>(op->getAttr("bind_op_impl")).getValue().str();
    }
    return "";
  }

  int addOrGetResourceID(const std::string& name, mlir::Operation *op) {
    bool isValid = true;
    auto impl = getImpl(op);
    auto latency = getOrFindLatency(name, op, isValid);
    if (!isValid && impl.empty()) {
      llvm::errs() << "warning: Invalid op_bind setting for " << name
                   << "\n\tDowngrade to default setting.\n";
      return NameToID[name];
    }
    std::stringstream latStr;
    std::copy(latency.begin(), latency.end(), std::ostream_iterator<int>(latStr, "_"));
    auto fullname = name + "_" + latStr.str() + impl;

    if (NameToID.find(fullname) != NameToID.end()) {
      return NameToID[fullname];
    }

    int rcs_id = NameToID[name];
    if (Components[rcs_id].amount == -1) {
      addComponent(
        Component(fullname, Components[rcs_id].delay, latency, Components[rcs_id].II, false, -1));
    } else if (Components[rcs_id].amount > 1) {
      addComponent(
        Component(fullname, Components[rcs_id].delay, latency, Components[rcs_id].II, true, 1));
      --Components[rcs_id].amount;
    } else if (Components[rcs_id].amount == 1) {
      llvm::errs() << "warning: Cannot derive component " << fullname << " from " << name
                   << ", amount is not satisfied.\n\tDowngrade to the default\n\t"
                   << "Note: this should not happend in firrtl stage";
      fullname = name;
      if (op->hasAttr("bind_op_latency"))
        op->removeAttr("bind_op_latency");
      if (op->hasAttr("bind_op_impl"))
        op->removeAttr("bind_op_impl");
    } else {
      assert(false && "Cannot create new op from component, amount is zero, should not happen");
    }
    return NameToID[fullname];
  }

  void addComponent(const Component &c) {
    NameToID[c.name] = Components.size();
    Components.push_back(c);
  }

private:
  std::map<std::string, int> NameToID;
  std::vector<Component> Components;
  llvm::DenseMap<mlir::tor::FuncOp, std::vector<int>> usages;
};

struct ScheduleTime {
public:
  int cycle, time;
  ScheduleTime(int c = 0, int t = 0) : cycle(c), time(t) {}

  ScheduleTime(const std::pair<int, int> &x) {
    cycle = x.first, time = x.second;
  }

  bool operator==(const ScheduleTime &x) const {
    return cycle == x.cycle && time == x.time;
  }
  bool operator<(const ScheduleTime &x) const {
    return cycle < x.cycle || (cycle == x.cycle && time < x.time);
  }
  bool operator>(const ScheduleTime &x) const { return x < *this; }
  bool operator<=(const ScheduleTime &x) const {
    return *this < x || *this == x;
  }
  bool operator>=(const ScheduleTime &x) const {
    return *this > x || *this == x;
  }
};

} // namespace scheduling

#endif
