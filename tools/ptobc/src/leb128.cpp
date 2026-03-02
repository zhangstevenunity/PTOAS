#include "ptobc/leb128.h"

#include <stdexcept>

namespace ptobc {

void writeULEB128(uint64_t value, std::vector<uint8_t>& out) {
  do {
    uint8_t byte = static_cast<uint8_t>(value & 0x7fu);
    value >>= 7;
    if (value != 0) byte |= 0x80u;
    out.push_back(byte);
  } while (value != 0);
}

void writeSLEB128(int64_t value, std::vector<uint8_t>& out) {
  bool more = true;
  while (more) {
    uint8_t byte = static_cast<uint8_t>(value & 0x7f);
    int64_t sign = byte & 0x40;
    value >>= 7;
    if ((value == 0 && sign == 0) || (value == -1 && sign != 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push_back(byte);
  }
}

size_t readULEB128(const uint8_t* data, size_t size, uint64_t& value) {
  value = 0;
  unsigned shift = 0;
  for (size_t i = 0; i < size; ++i) {
    uint8_t byte = data[i];
    value |= (uint64_t(byte & 0x7fu) << shift);
    if ((byte & 0x80u) == 0) return i + 1;
    shift += 7;
    if (shift > 63) throw std::runtime_error("ULEB128 too large");
  }
  throw std::runtime_error("Unexpected EOF in ULEB128");
}

size_t readSLEB128(const uint8_t* data, size_t size, int64_t& value) {
  value = 0;
  unsigned shift = 0;
  uint8_t byte = 0;
  size_t i = 0;
  for (; i < size; ++i) {
    byte = data[i];
    value |= (int64_t(byte & 0x7f) << shift);
    shift += 7;
    if ((byte & 0x80u) == 0) break;
    if (shift > 63) throw std::runtime_error("SLEB128 too large");
  }
  if (i == size) throw std::runtime_error("Unexpected EOF in SLEB128");

  // sign extend
  if ((shift < 64) && (byte & 0x40u)) {
    value |= (-1ll) << shift;
  }
  return i + 1;
}

} // namespace ptobc
