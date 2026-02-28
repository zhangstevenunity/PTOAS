//===- InsertSyncDebug.cpp - Debug printing for PTO InsertSync ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"

#include "mlir/IR/AsmState.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

llvm::cl::opt<unsigned> insertSyncDebugLevelOpt(
    "pto-insert-sync-debug",
    llvm::cl::desc("Debug verbosity for PTOInsertSync: "
                   "0=off, 1=phase, 2=syncir, 3=trace"),
    llvm::cl::init(0));

} // namespace

unsigned mlir::pto::getInsertSyncDebugLevel() { return insertSyncDebugLevelOpt; }

bool mlir::pto::isInsertSyncDebugEnabled(InsertSyncDebugLevel minLevel) {
  return getInsertSyncDebugLevel() >= static_cast<unsigned>(minLevel);
}

static llvm::StringRef getPipelineName(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return "PIPE_S";
  case PipelineType::PIPE_V:
    return "PIPE_V";
  case PipelineType::PIPE_M:
    return "PIPE_M";
  case PipelineType::PIPE_MTE1:
    return "PIPE_MTE1";
  case PipelineType::PIPE_MTE2:
    return "PIPE_MTE2";
  case PipelineType::PIPE_MTE3:
    return "PIPE_MTE3";
  case PipelineType::PIPE_ALL:
    return "PIPE_ALL";
  case PipelineType::PIPE_MTE4:
    return "PIPE_MTE4";
  case PipelineType::PIPE_MTE5:
    return "PIPE_MTE5";
  case PipelineType::PIPE_V2:
    return "PIPE_V2";
  case PipelineType::PIPE_FIX:
    return "PIPE_FIX";
  case PipelineType::VIRTUAL_PIPE_MTE2_L1A:
    return "VIRTUAL_PIPE_MTE2_L1A";
  case PipelineType::VIRTUAL_PIPE_MTE2_L1B:
    return "VIRTUAL_PIPE_MTE2_L1B";
  case PipelineType::PIPE_NUM:
    return "PIPE_NUM";
  case PipelineType::PIPE_UNASSIGNED:
    return "PIPE_UNASSIGNED";
  }
  return "PIPE_UNKNOWN";
}

static llvm::StringRef getBranchKindName(KindOfBranch kind) {
  switch (kind) {
  case KindOfBranch::IF_BEGIN:
    return "IF_BEGIN";
  case KindOfBranch::ELSE_BEGIN:
    return "ELSE_BEGIN";
  case KindOfBranch::IF_END:
    return "IF_END";
  }
  return "BRANCH_UNKNOWN";
}

static llvm::StringRef getLoopKindName(KindOfLoop kind) {
  switch (kind) {
  case KindOfLoop::LOOP_BEGIN:
    return "LOOP_BEGIN";
  case KindOfLoop::LOOP_END:
    return "LOOP_END";
  }
  return "LOOP_UNKNOWN";
}

static llvm::StringRef getMemScopeName(pto::AddressSpace scope) {
  switch (scope) {
  case pto::AddressSpace::Zero:
    return "Zero";
  case pto::AddressSpace::GM:
    return "GM";
  case pto::AddressSpace::VEC:
    return "VEC";
  case pto::AddressSpace::MAT:
    return "MAT";
  case pto::AddressSpace::ACC:
    return "ACC";
  case pto::AddressSpace::LEFT:
    return "LEFT";
  case pto::AddressSpace::RIGHT:
    return "RIGHT";
  case pto::AddressSpace::BIAS:
    return "BIAS";
  case pto::AddressSpace::SCALING:
    return "SCALING";
  }
  return "SCOPE_UNKNOWN";
}

static void dumpEventIds(llvm::raw_ostream &os,
                         const SmallVector<int> &eventIds) {
  os << "[";
  for (size_t i = 0; i < eventIds.size(); ++i) {
    os << eventIds[i];
    if (i + 1 != eventIds.size())
      os << ",";
  }
  os << "]";
}

static void dumpSyncOp(llvm::raw_ostream &os, const SyncOperation *op,
                       bool showUselessSync) {
  if (!op)
    return;
  if (op->uselessSync && !showUselessSync)
    return;

  os << SyncOperation::TypeName(op->GetType());
  os << " <" << getPipelineName(op->GetSrcPipe()) << " -> "
     << getPipelineName(op->GetDstPipe()) << ">";
  os << " idx=" << op->GetSyncIndex();

  if (op->GetForEndIndex().has_value())
    os << " forEnd=" << op->GetForEndIndex().value();

  if (op->eventIdNum != 1)
    os << " eventIdNum=" << op->eventIdNum;

  if (op->isCompensation)
    os << " compensation";
  if (op->uselessSync)
    os << " useless";

  if (!op->eventIds.empty()) {
    os << " eventIds=";
    dumpEventIds(os, op->eventIds);
  }
}

static void dumpMemInfoList(llvm::raw_ostream &os, llvm::StringRef tag,
                            const SmallVector<const BaseMemInfo *> &list,
                            mlir::AsmState *state) {
  os << tag << "=[";
  for (size_t i = 0; i < list.size(); ++i) {
    const BaseMemInfo *info = list[i];
    if (!info) {
      os << "<null>";
    } else if (info->rootBuffer) {
      if (state) {
        info->rootBuffer.printAsOperand(os, *state);
      } else {
        os << "<value>";
      }
      os << "(" << getMemScopeName(info->scope) << ")";
    } else {
      os << "<null-root>";
    }
    if (i + 1 != list.size())
      os << ", ";
  }
  os << "]";
}

static void dumpSyncIR(llvm::raw_ostream &os, const SyncIRs &syncIR,
                       Operation *opForPrinting, InsertSyncDumpOptions options,
                       bool showMemInfo) {
  std::optional<mlir::AsmState> state;
  if (showMemInfo && opForPrinting)
    state.emplace(opForPrinting);

  int indent = 0;
  auto indentBy = [&](int extra = 0) {
    return static_cast<unsigned>(std::max(0, indent) * 2 + extra);
  };

  for (const auto &e : syncIR) {
    if (!e)
      continue;

    if (auto *loop = dyn_cast<LoopInstanceElement>(e.get())) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_END)
        indent = std::max(0, indent - 1);
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(e.get())) {
      if (branch->getBranchKind() == KindOfBranch::IF_END ||
          branch->getBranchKind() == KindOfBranch::ELSE_BEGIN)
        indent = std::max(0, indent - 1);
    }

    os.indent(indentBy());
    os << llvm::formatv("[{0,4}] ", e->GetIndex());

    switch (e->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = cast<CompoundInstanceElement>(e.get());
      os << "COMPOUND " << comp->opName.getStringRef() << " ["
         << getPipelineName(comp->kPipeValue) << "]";
      os << "\n";
      if (showMemInfo) {
        os.indent(indentBy(2));
        dumpMemInfoList(os, "def", comp->defVec, state ? &*state : nullptr);
        os << "\n";
        os.indent(indentBy(2));
        dumpMemInfoList(os, "use", comp->useVec, state ? &*state : nullptr);
        os << "\n";
      }
      break;
    }
    case InstanceElement::KindTy::LOOP: {
      auto *loop = cast<LoopInstanceElement>(e.get());
      os << "LOOP " << getLoopKindName(loop->getLoopKind())
         << " (begin=" << loop->beginId << ", end=" << loop->endId << ")\n";
      break;
    }
    case InstanceElement::KindTy::BRANCH: {
      auto *branch = cast<BranchInstanceElement>(e.get());
      os << "BRANCH " << getBranchKindName(branch->getBranchKind())
         << " (begin=" << branch->beginId << ", branch=" << branch->branchId
         << ", end=" << branch->endId << ")\n";
      break;
    }
    case InstanceElement::KindTy::PLACE_HOLDER: {
      auto *ph = cast<PlaceHolderInstanceElement>(e.get());
      os << "PLACE_HOLDER (parentScopeId=" << ph->parentScopeId;
      if (ph->isVirtualElse)
        os << ", virtualElse";
      os << ")\n";
      break;
    }
    }

    auto dumpOps = [&](llvm::StringRef prefix, const SyncOps &ops) {
      for (const auto *op : ops) {
        if (!op)
          continue;
        if (op->uselessSync && !options.showUselessSync)
          continue;
        os.indent(indentBy(2));
        os << prefix << ": ";
        dumpSyncOp(os, op, options.showUselessSync);
        os << "\n";
      }
    };

    dumpOps("PRE ", e->pipeBefore);
    dumpOps("POST", e->pipeAfter);

    if (auto *loop = dyn_cast<LoopInstanceElement>(e.get())) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_BEGIN)
        indent += 1;
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(e.get())) {
      if (branch->getBranchKind() == KindOfBranch::IF_BEGIN ||
          branch->getBranchKind() == KindOfBranch::ELSE_BEGIN)
        indent += 1;
    }
  }
}

void mlir::pto::dumpInsertSyncPhase(llvm::StringRef phase, const SyncIRs &syncIR,
                                   const SyncOperations &syncOperations,
                                   Operation *opForPrinting,
                                   llvm::raw_ostream &os) {
  const unsigned level = getInsertSyncDebugLevel();
  if (level < static_cast<unsigned>(InsertSyncDebugLevel::Phase))
    return;

  unsigned activeOps = 0;
  unsigned setCnt = 0, waitCnt = 0, barrierCnt = 0;
  unsigned blockSetCnt = 0, blockWaitCnt = 0, blockAllCnt = 0;
  for (const auto &group : syncOperations) {
    for (const auto &op : group) {
      if (!op)
        continue;
      if (op->uselessSync)
        continue;
      activeOps++;
      switch (op->GetType()) {
      case SyncOperation::TYPE::SET_EVENT:
        setCnt++;
        break;
      case SyncOperation::TYPE::WAIT_EVENT:
        waitCnt++;
        break;
      case SyncOperation::TYPE::PIPE_BARRIER:
      case SyncOperation::TYPE::PIPE_BARRIER_CUBE:
      case SyncOperation::TYPE::PIPE_BARRIER_VECTOR:
        barrierCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_SET:
        blockSetCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_WAIT:
        blockWaitCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_ALL:
        blockAllCnt++;
        break;
      }
    }
  }

  os << "\n// === [PTOInsertSync Debug] " << phase << " === //\n";
  os << llvm::formatv("// nodes={0}, syncGroups={1}, activeOps={2} "
                      "(set={3}, wait={4}, barrier={5}, blockSet={6}, "
                      "blockWait={7}, blockAll={8})\n",
                      syncIR.size(), syncOperations.size(), activeOps, setCnt,
                      waitCnt, barrierCnt, blockSetCnt, blockWaitCnt,
                      blockAllCnt);

  if (level < static_cast<unsigned>(InsertSyncDebugLevel::SyncIR)) {
    os << "// ========================================= //\n";
    return;
  }

  InsertSyncDumpOptions options;
  const bool showMemInfo =
      level >= static_cast<unsigned>(InsertSyncDebugLevel::Trace);
  options.showMemInfo = showMemInfo;
  options.showUselessSync = showMemInfo;

  dumpSyncIR(os, syncIR, opForPrinting, options, showMemInfo);
  os << "// ========================================= //\n";
}
