#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ptobc {

// PTOBC v0 constants.
static constexpr uint8_t kSectionStrings = 0x01;
static constexpr uint8_t kSectionTypes = 0x02;
static constexpr uint8_t kSectionAttrs = 0x03;
static constexpr uint8_t kSectionConstPool = 0x04;
static constexpr uint8_t kSectionOpcodeSchemaExt = 0x05;
static constexpr uint8_t kSectionModule = 0x06;
static constexpr uint8_t kSectionDebugInfo = 0x07;
static constexpr uint8_t kSectionExtra = 0x7F;

static constexpr uint16_t kVersionV0 = 0x0000;
static constexpr uint16_t kFlagsV0 = 0x0000;
static constexpr uint16_t kOpcodeGeneric = 0xFFFF;

struct Buffer {
  std::vector<uint8_t> bytes;
  void append(const void* p, size_t n);
  void appendU8(uint8_t v);
  void appendU16LE(uint16_t v);
  void appendU32LE(uint32_t v);
};

struct StringTable {
  std::unordered_map<std::string, uint64_t> toId;
  std::vector<std::string> fromId;

  uint64_t intern(const std::string& s);
};

struct DebugFileEntry {
  uint64_t pathSid;
  uint8_t hashKind;                 // 0=none, 1=sha256
  std::vector<uint8_t> hashBytes;   // only if hashKind!=0
};

struct DebugValueNameEntry {
  uint64_t funcId;
  uint64_t valueId;
  uint64_t nameSid;
};

struct DebugLocationEntry {
  uint64_t funcId;
  uint64_t opId;
  uint64_t fileId;
  uint64_t sl;
  uint64_t sc;
  uint64_t el;
  uint64_t ec;
};

struct DebugSnippetEntry {
  uint64_t funcId;
  uint64_t opId;
  uint64_t snippetSid;
};

struct ConstEntry {
  uint8_t tag;
  std::vector<uint8_t> payload; // tag-specific payload bytes
};

struct PTOBCFile {
  // Tables
  StringTable strings;
  std::vector<std::string> typeAsm; // 1-based IDs; 0 means none
  std::vector<std::string> attrAsm; // 1-based IDs; 0 means none

  // Const pool
  std::vector<ConstEntry> consts;

  // DebugInfo tables (optional section)
  std::vector<DebugFileEntry> dbgFiles;
  std::vector<DebugValueNameEntry> dbgValueNames;
  std::vector<DebugLocationEntry> dbgLocations;
  std::vector<DebugSnippetEntry> dbgSnippets;

  // Sections payloads
  std::vector<uint8_t> moduleBytes;

  std::vector<uint8_t> buildStringsSection() const;
  std::vector<uint8_t> buildTypesSection() const;
  std::vector<uint8_t> buildAttrsSection() const;
  std::vector<uint8_t> buildConstPoolSection() const;
  std::vector<uint8_t> buildDebugInfoSection() const;

  std::vector<uint8_t> serialize() const;
};

// Helpers to read a PTOBC file from disk.
std::vector<uint8_t> readFile(const std::string& path);
void writeFile(const std::string& path, const std::vector<uint8_t>& data);

} // namespace ptobc
