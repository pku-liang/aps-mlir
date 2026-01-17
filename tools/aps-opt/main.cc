#include "aps_opt.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <climits>
#include <filesystem>

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aps-opt", "0.1");

    program.add_argument("-i", "--input")
        .required()
        .help("Input MLIR file path");
    program.add_argument("-o", "--output")
        .required()
        .help("Output Cement2 MLIR file path");
    program.add_argument("--clock")
        .required()
        .default_value(std::string("6.0"))
        .help("Clock period in ns (default: 6.0)");
    program.add_argument("--resource")
        .required()
        .default_value(std::string("examples/resource_ihp130.json"))
        .help("Path to resource.json for scheduling information");
    program.add_argument("--print-ir-after-all")
        .default_value(false)
        .implicit_value(true)
        .help("Print MLIR after each pass");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program << std::endl;
        return EXIT_FAILURE;
    }

    const auto inputFile = program.get<std::string>("--input");
    const auto clockPeriod = program.get<std::string>("--clock");
    const auto resourceFile = program.get<std::string>("--resource");
    const auto outputPath = program.get<std::string>("--output");
    const auto printIrAfterAll = program.get<bool>("--print-ir-after-all");

    // Validate input file exists
    if (access(inputFile.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] Input file " << inputFile << " does not exist"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Validate resource file exists
    if (access(resourceFile.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] Resource file " << resourceFile
                  << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // tmpfile
    auto tempDir = std::filesystem::temp_directory_path();

    // Build argument vector for hector_driver
    std::vector<std::string> args;
    args.push_back(argv[0]);
    args.push_back(inputFile);
    args.push_back("-o");
    args.push_back(outputPath);

    args.push_back("--allow-unregistered-dialect");
    if (printIrAfterAll) {
        args.push_back("--mlir-print-ir-after-all");
    }

    // Add passes
    args.push_back("--aps-hoist-readrf");
    args.push_back("--memory-map");
    args.push_back("--scf-for-index-cast");
    args.push_back("--aps-mem-to-memref");
    args.push_back("--canonicalize");
    args.push_back("--raise-scf-to-affine");
    args.push_back("--canonicalize");
    args.push_back("--affine-raise-from-memref");
    args.push_back("--infer-affine-mem-access");
    args.push_back("--canonicalize");
    args.push_back("--hls-unroll");
    args.push_back("--cse");
    args.push_back("--canonicalize");
    args.push_back("--affine-loop-normalize");
    args.push_back("--canonicalize");
    args.push_back("--new-array-partition");
    args.push_back("--canonicalize");
    args.push_back("--affine-mem-to-aps-mem");
    args.push_back("--memref-to-aps-mem");
    args.push_back("--aps-scalar-mem-to-global");
    args.push_back("--arith-muldiv-to-shift");
    args.push_back("--canonicalize");
    // Lower affine to SCF
    args.push_back("--lower-affine-for");
    args.push_back("--canonicalize");
    args.push_back("--expression-balance");

    // Convert input with top function, clock, and resource parameters
    args.push_back("--convert-input=clock=" + clockPeriod +
                               " resource=" + resourceFile +
                               " output-path=" + tempDir.string());

    args.push_back("--canonicalize");
    args.push_back("--scf-to-tor");
    args.push_back("--canonicalize");
    args.push_back("--schedule-tor");
    args.push_back("--aps-split-memory-ops");
    args.push_back("--tor-time-graph");
    args.push_back("--aps-memload-duplication"); //
    args.push_back("--canonicalize");

    // Cmt2!
    args.push_back("--aps-to-cmt2-gen");

    // Convert std::vector<std::string> to char** for C-style API
    std::vector<char *> args_c;
    args_c.reserve(args.size());
    for (auto &arg : args) {
        args_c.push_back(const_cast<char *>(arg.c_str()));
    }

    // Call hector_driver with the constructed arguments
    return(aps_opt_driver(static_cast<int>(args_c.size()), args_c.data()));
}
