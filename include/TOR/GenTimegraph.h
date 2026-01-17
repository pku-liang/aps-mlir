#ifndef GEN_TIMEGRAPH_H
#define GEN_TIMEGRAPH_H

#include "Timegraph.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include <regex>
#include <string>

struct GenTimgraph {
  std::string dumpStr;
  std::string outputDirName;

  GenTimgraph(mlir::ModuleOp moduleOp, std::string outputDirName)
      : outputDirName(outputDirName) {
    llvm::raw_string_ostream output(dumpStr);
    moduleOp->print(output);
  };

  std::string replaceStr(std::string str, const std::string &pattern,
                         const std::string &to) {
    std::regex r(pattern);
    return std::regex_replace(str, r, to);
  }

  void writeTimegraphTxtFile() {
    std::istringstream iss(dumpStr);
    std::string str, line;
    while (std::getline(iss, line)) {
      if (line.find("arith.constant") == std::string::npos &&
          line.find("tor.alloc") == std::string::npos &&
          line.find("tor.yield") == std::string::npos &&
          line.find("tor.condition") == std::string::npos &&
          line.find("^bb") == std::string::npos &&
          line.find("math") == std::string::npos &&
          line.find("arith.select") == std::string::npos) {
        if (!(*(line.rbegin()) == '{' &&
              line.find("on ") != std::string::npos &&
              line.find("if") == std::string::npos) ||
            line.find("tor.design") != std::string::npos) {
          if (str.length() != 0) {
            str += "\n";
          }
        }
        str += line;
      }
    }

    // format func
    str = replaceStr(str, "func @(.*?)\\((.*?)cycle = (.*?) : (.*?)\n",
                     "func @$1 {cycle = $3} {\n");
    // old format func
    str = replaceStr(str, "func @(.*?)\\((.*?)\n", "func @$1 {\n");

    // clear behind on unused info
    str = replaceStr(str, "on \\((.*?)\\)(.*?)\n", "on ($1)\n");
    // if op beautify
    str = replaceStr(str, "if(.*?)on \\((.*?)\\)\n", "if on ($2) then {\n");
    // while op beautify
    str = replaceStr(str, "while(.*?)on \\((.*?)\\)\n", "while on ($2) {\n");
    str = replaceStr(str, "\\} do\\{", "} do {");
    // for op beautify
    str = replaceStr(str, "for(.*?)on \\((.*?)\\)\n", "for on ($2) {\n");
    // store op beautify
    str = replaceStr(str, "store(.*?)on \\((.*?)\\)\n", "store on ($2)\n");

    // not tor op handle
    str = replaceStr(str,
                     "\\{dump = (.*?), endtime = (\\d+) : i32, starttime = "
                     "(\\d+) : i32\\} : (.*?)\n",
                     "on ($3 to $2)\n");

    // clear unused info
    str = replaceStr(str, "attributes \\{(.*?)\\}", "");
    str = replaceStr(str, " \\{dump = \"(.*?)\\}", "");
    // HoistConstCondIfOp
    str = replaceStr(str, "\\{Hoist(.*?)\\}", "");

    // clear %*
    str = replaceStr(str, "\\%(.*?)\\s", "");
    // clear remain =
    str = replaceStr(str, "  = ", "  ");

    // clear special mark
    str = replaceStr(str, "tor\\.", "");
    str = replaceStr(str, "arith\\.", "");

    // time graph handle
    str = replaceStr(str, "succ (.*?) : \\[", "Back edge ($1 to ");
    str = replaceStr(str, " : i\\d+", "");
    str = replaceStr(str, " to i\\d+", "");
    str = replaceStr(str, "\\] \\[(.*?)\n", ")\n");

    // format ") {"
    str = replaceStr(str, "\\)(\\s*?)\\{\n", ") {\n");

    // call op
    str = replaceStr(str, "call @(.*?)\\(", "call @$1 ");

    // II
    str = replaceStr(str, "II = (.*?), (.*?)\\}", "II = $1}");

    // cycle
    str = replaceStr(str, "cycle = (.*?), (.*?)\\}", "cycle = $1}");

    std::error_code EC;
    auto llvmFileOutput = std::make_unique<llvm::raw_fd_ostream>(
        outputDirName + "/timegraph.txt", EC, llvm::sys::fs::OF_None);
    *llvmFileOutput << str;
  }

  void writeTimegraphPngFile() {
    make_graph gs(dumpStr);
    gs.simplify_graph();
    gs.out(outputDirName);
  }

  void writeTimegraph() {
    writeTimegraphPngFile();
    writeTimegraphTxtFile();
  }
};
#endif