#include "ptobc/ptobc_format.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <PTO/IR/PTO.h>

#include <iostream>

namespace ptobc {
mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path);
PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module);
void decodeFileToPTO(const std::string& inPath, const std::string& outPath);
}

static void usage() {
  std::cerr << "ptobc (v0)\n\n"
            << "Usage:\n"
            << "  ptobc encode <input.pto> -o <out.ptobc>\n"
            << "  ptobc decode <input.ptobc> -o <out.pto>\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }

  std::string cmd = argv[1];
  std::string in;
  std::string out;

  if (cmd == "encode" || cmd == "decode") {
    if (argc < 5) {
      usage();
      return 2;
    }
    in = argv[2];
    for (int i = 3; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "-o" && i + 1 < argc) {
        out = argv[++i];
      }
    }
    if (out.empty()) {
      std::cerr << "Missing -o\n";
      return 2;
    }
  }

  try {
    if (cmd == "encode") {
      mlir::DialectRegistry registry;
      registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::pto::PTODialect>();
      mlir::MLIRContext ctx(registry);
      ctx.allowUnregisteredDialects(true);

      // Preload dialects so custom op/type parsing is available.
      (void)ctx.getOrLoadDialect<mlir::func::FuncDialect>();
      (void)ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
      (void)ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
      (void)ctx.getOrLoadDialect<mlir::pto::PTODialect>();

      auto module = ptobc::parsePTOFile(ctx, in);
      auto file = ptobc::encodeFromMLIRModule(*module);
      auto bytes = file.serialize();
      ptobc::writeFile(out, bytes);
      return 0;
    }

    if (cmd == "decode") {
      ptobc::decodeFileToPTO(in, out);
      return 0;
    }

    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
