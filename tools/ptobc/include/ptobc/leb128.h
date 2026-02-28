#pragma once

#include <cstdint>
#include <vector>

namespace ptobc {

void writeULEB128(uint64_t value, std::vector<uint8_t>& out);
void writeSLEB128(int64_t value, std::vector<uint8_t>& out);

// Returns bytes consumed.
size_t readULEB128(const uint8_t* data, size_t size, uint64_t& value);
size_t readSLEB128(const uint8_t* data, size_t size, int64_t& value);

} // namespace ptobc
