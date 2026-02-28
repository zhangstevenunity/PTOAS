#include "ptobc/ptobc_format.h"

#include "ptobc/leb128.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ptobc {

void Buffer::append(const void* p, size_t n) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
  bytes.insert(bytes.end(), b, b + n);
}

void Buffer::appendU8(uint8_t v) { bytes.push_back(v); }

void Buffer::appendU16LE(uint16_t v) {
  bytes.push_back(uint8_t(v & 0xff));
  bytes.push_back(uint8_t((v >> 8) & 0xff));
}

void Buffer::appendU32LE(uint32_t v) {
  bytes.push_back(uint8_t(v & 0xff));
  bytes.push_back(uint8_t((v >> 8) & 0xff));
  bytes.push_back(uint8_t((v >> 16) & 0xff));
  bytes.push_back(uint8_t((v >> 24) & 0xff));
}

uint64_t StringTable::intern(const std::string& s) {
  auto it = toId.find(s);
  if (it != toId.end()) return it->second;
  uint64_t id = fromId.size();
  fromId.push_back(s);
  toId.emplace(s, id);
  return id;
}

static std::vector<uint8_t> buildSection(uint8_t id, const std::vector<uint8_t>& data) {
  Buffer b;
  b.appendU8(id);
  b.appendU32LE(uint32_t(data.size()));
  if (!data.empty()) b.append(data.data(), data.size());
  return std::move(b.bytes);
}

std::vector<uint8_t> PTOBCFile::buildStringsSection() const {
  Buffer b;
  writeULEB128(strings.fromId.size(), b.bytes);
  for (const auto& s : strings.fromId) {
    writeULEB128(s.size(), b.bytes);
    b.append(s.data(), s.size());
  }
  return b.bytes;
}

std::vector<uint8_t> PTOBCFile::buildTypesSection() const {
  Buffer b;
  writeULEB128(typeAsm.size(), b.bytes);
  for (const auto& asmStr : typeAsm) {
    // tag=0x00 opaque
    b.appendU8(0x00);
    // flags: bit0=has_asm
    b.appendU8(0x01);
    auto sid = strings.toId.at(asmStr);
    writeULEB128(sid, b.bytes);
  }
  return b.bytes;
}

std::vector<uint8_t> PTOBCFile::buildAttrsSection() const {
  Buffer b;
  writeULEB128(attrAsm.size(), b.bytes);
  for (const auto& asmStr : attrAsm) {
    b.appendU8(0x00);
    b.appendU8(0x01);
    auto sid = strings.toId.at(asmStr);
    writeULEB128(sid, b.bytes);
  }
  return b.bytes;
}

std::vector<uint8_t> PTOBCFile::buildConstPoolSection() const {
  Buffer b;
  writeULEB128(consts.size(), b.bytes);
  for (const auto &c : consts) {
    b.appendU8(c.tag);
    if (!c.payload.empty()) b.append(c.payload.data(), c.payload.size());
  }
  return b.bytes;
}

std::vector<uint8_t> PTOBCFile::buildDebugInfoSection() const {
  Buffer b;

  // FileTable
  writeULEB128(dbgFiles.size(), b.bytes);
  for (const auto &f : dbgFiles) {
    writeULEB128(f.pathSid, b.bytes);
    b.appendU8(f.hashKind);
    if (f.hashKind != 0) {
      writeULEB128(f.hashBytes.size(), b.bytes);
      if (!f.hashBytes.empty()) b.append(f.hashBytes.data(), f.hashBytes.size());
    }
  }

  // ValueNames
  writeULEB128(dbgValueNames.size(), b.bytes);
  for (const auto &vn : dbgValueNames) {
    writeULEB128(vn.funcId, b.bytes);
    writeULEB128(vn.valueId, b.bytes);
    writeULEB128(vn.nameSid, b.bytes);
  }

  // OpLocations
  writeULEB128(dbgLocations.size(), b.bytes);
  for (const auto &l : dbgLocations) {
    writeULEB128(l.funcId, b.bytes);
    writeULEB128(l.opId, b.bytes);
    writeULEB128(l.fileId, b.bytes);
    writeULEB128(l.sl, b.bytes);
    writeULEB128(l.sc, b.bytes);
    writeULEB128(l.el, b.bytes);
    writeULEB128(l.ec, b.bytes);
  }

  // OpSnippets
  writeULEB128(dbgSnippets.size(), b.bytes);
  for (const auto &s : dbgSnippets) {
    writeULEB128(s.funcId, b.bytes);
    writeULEB128(s.opId, b.bytes);
    writeULEB128(s.snippetSid, b.bytes);
  }

  return b.bytes;
}

std::vector<uint8_t> PTOBCFile::serialize() const {
  // Build payload sections in fixed order.
  auto sStrings = buildStringsSection();
  auto sTypes = buildTypesSection();
  auto sAttrs = buildAttrsSection();
  auto sConst = buildConstPoolSection();

  auto secStrings = buildSection(kSectionStrings, sStrings);
  auto secTypes = buildSection(kSectionTypes, sTypes);
  auto secAttrs = buildSection(kSectionAttrs, sAttrs);
  auto secConst = buildSection(kSectionConstPool, sConst);
  auto secModule = buildSection(kSectionModule, moduleBytes);

  std::vector<uint8_t> payload;
  payload.insert(payload.end(), secStrings.begin(), secStrings.end());
  payload.insert(payload.end(), secTypes.begin(), secTypes.end());
  payload.insert(payload.end(), secAttrs.begin(), secAttrs.end());
  payload.insert(payload.end(), secConst.begin(), secConst.end());
  payload.insert(payload.end(), secModule.begin(), secModule.end());

  // Optional DEBUGINFO section.
  const bool hasDebug = !dbgFiles.empty() || !dbgValueNames.empty() || !dbgLocations.empty() || !dbgSnippets.empty();
  if (hasDebug) {
    auto secDbg = buildSection(kSectionDebugInfo, buildDebugInfoSection());
    payload.insert(payload.end(), secDbg.begin(), secDbg.end());
  }

  Buffer out;
  const char magic[6] = {'P','T','O','B','C','\0'};
  out.append(magic, 6);
  out.appendU16LE(kVersionV0);
  out.appendU16LE(kFlagsV0);
  out.appendU32LE(uint32_t(payload.size()));
  out.append(payload.data(), payload.size());

  return out.bytes;
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("Failed to open: " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return buf;
}

void writeFile(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) throw std::runtime_error("Failed to write: " + path);
  ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
}

} // namespace ptobc
