// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "grittibanzli.h"

#include <assert.h>
#include <limits.h>

#include <iostream>
#include <algorithm>

namespace grittibanzli {

namespace {

// returns index of last element that is <= value, or 0 if none
int BinarySearch(int value, const std::vector<int>& values) {
  if (value > values.back()) return values.size() - 1;
  int result = std::lower_bound(values.begin(),
      values.end(), value) - values.begin();
  if (result > 0 && values[result] > value) result--;
  return result;
}

int PeekBit(const std::vector<uint8_t>& data, size_t bitpos) {
  return (data[bitpos >> 3] >> (bitpos & 7)) & 1;
}

int ReadBit(const std::vector<uint8_t>& data, size_t* bitpos) {
  int result = PeekBit(data, *bitpos);
  (*bitpos)++;
  return result;
}

int ReadBits(int num, const std::vector<uint8_t>& data, size_t* bitpos) {
  int result = 0;
  for (int i = 0; i < num; i++) {
    result |= ReadBit(data, bitpos) << i;
  }
  return result;
}

int ReadBitsInv(int num, const std::vector<uint8_t>& data, size_t* bitpos) {
  int result = 0;
  for (int i = 0; i < num; i++) {
    result = (result << 1) | ReadBit(data, bitpos);
  }
  return result;
}

int PeekBitsInv(int num, const std::vector<uint8_t>& data, size_t bitpos) {
  return ReadBitsInv(num, data, &bitpos);
}

void AppendBit(int bit, std::vector<uint8_t>* data, size_t* bitpos) {
  int m = (*bitpos) & 7;
  if (m == 0) {
    data->push_back(0);
  }
  data->back() |= bit << m;
  (*bitpos)++;
}

void AppendBits(int bits, int num, std::vector<uint8_t>* data, size_t* bitpos) {
  for (int i = 0; i < num; i++) {
    AppendBit((bits >> i) & 1, data, bitpos);
  }
}

void AppendBitsInv(int bits, int num, std::vector<uint8_t>* data,
    size_t* bitpos) {
  for (int i = 0; i < num; i++) {
    AppendBit((bits >> (num - 1 - i)) & 1, data, bitpos);
  }
}

////////////////////////////////////////////////////////////////////////////////

struct BlockChoices {
  int type;

  // uncompressed bytes range of this block
  int start_byte;
  int end_byte;
  // range in all LZ77 length symbols of this block
  int start_lz77;
  int end_lz77;

  std::vector<int> lengths;  // 1 means literal
  std::vector<int> dists;  // 0 means literal

  // huffman trees
  int hlit;
  int hdist;
  int hclen;
  std::vector<int> ht_lengths;  // code length code lengths
  std::vector<int> all_rle;
  std::vector<int> all_extra;
};

// choices of a single deflate stream
struct DeflateChoices {
  // The deflate stream itself, as blocks
  std::vector<BlockChoices> blocks;
  // Position in uncompressed data, which can be different from 0 when
  // concatenated from multiple deflate and other sources
  size_t abs_pos = 0;
};

////////////////////////////////////////////////////////////////////////////////

// Huffman coding

// Given huffman code lengths, gives the huffman symbol bits, as in the deflate
// specification
std::vector<int> CodeLengthsToSymbols(const std::vector<int>& lengths,
    int maxbits = 15) {
  std::vector<int> bl_count(maxbits + 1, 0);
  std::vector<int> next_code(maxbits + 1);
  std::vector<int> result(lengths.size(), 0);
  int code;

  // 1) Count the number of codes for each code length. Let bl_count[N] be the
  // number of codes of length N, N >= 1.
  for (size_t i = 0; i < lengths.size(); i++) {
    assert(lengths[i] <= maxbits);
    bl_count[lengths[i]]++;
  }
  // 2) Find the numerical value of the smallest code for each code length.
  code = 0;
  bl_count[0] = 0;
  for (int bits = 1; bits <= maxbits; bits++) {
    code = (code + bl_count[bits - 1]) << 1;
    next_code[bits] = code;
  }
  // 3) Assign numerical values to all codes, using consecutive values for all
  // codes of the same length with the base values determined at step 2.
  for (size_t i = 0; i < lengths.size(); i++) {
    int len = lengths[i];
    if (len != 0) {
      result[i] = next_code[len];
      next_code[len]++;
    }
  }

  return result;
}

// Makes huffman table for decoding, based on "root" bits
std::vector<std::pair<int, int>> DecodableHuffmanTree(
    const std::vector<int>& lengths, int maxbits = 15, int root_bits = 8) {
  const std::vector<int> symbols = CodeLengthsToSymbols(lengths, maxbits);
  int rootnum = (1 << root_bits);
  int rootmask = rootnum - 1;
  std::vector<std::pair<int, int>> result(rootnum, {-1, -1});

  // Longest symbol length for symbols with size > root_bits
  std::vector<int> maxlengths(rootnum, -1);
  for (size_t i = 0; i < symbols.size(); i++) {
    if (lengths[i] > root_bits) {
      int prefix = (symbols[i] >> (lengths[i] - root_bits)) & rootmask;
      maxlengths[prefix] = std::max(maxlengths[prefix], lengths[i]);
    }
  }

  for (size_t i = 0; i < symbols.size(); i++) {
    if (lengths[i] == 0) {
      continue;
    } else if (lengths[i] == root_bits) {
      // Symbol bits exactly matches table bits
      result[symbols[i]] = {lengths[i], i};
    } else if (lengths[i] < root_bits) {
      // Multiple root table entries with the same prefix for this symbol.
      int shift = (root_bits - lengths[i]);
      int num = 1 << shift;
      for (int j = 0; j < num; j++) {
        int b = (symbols[i] << shift) + j;
        assert(b <= rootmask);
        result[b] = {lengths[i], i};
      }
    } else {
      // root_bits is now just a prefix.
      int prefix = (symbols[i] >> (lengths[i] - root_bits)) & rootmask;
      int maxlen = maxlengths[prefix];
      int num = 1 << (maxlen - root_bits);
      int mask = num - 1;
      int pointer = result[prefix].second;
      if (pointer == -1) {
        pointer = result.size() - prefix;
        result.resize(result.size() + num, {-1, -1});
        result[prefix] = {maxlen, pointer};
      }
      int postfix = (symbols[i] << (maxlen - lengths[i])) & mask;
      int index = prefix + pointer + postfix;
      if (lengths[i] == maxlen) {
        // Symbol bits exactly match subtable bits
        result[index] = {lengths[i], i};
      } else {
        assert(lengths[i] < maxlen);
        int shift = (maxlen - lengths[i]);
        int num2 = 1 << shift;
        for (int j = 0; j < num2; j++) {
          result[index + j] = {lengths[i], i};
        }
      }
    }
  }
  return result;
}


// Decodes a single huffman symbol. Returns -1 on error.
int HuffmanDecodeSymbol(const std::vector<uint8_t>& compressed,
    const std::vector<std::pair<int, int>>& tree, size_t* bitpos,
    int root_bits = 8) {
  int index = PeekBitsInv(root_bits, compressed, *bitpos);

  if (tree[index].first <= root_bits) {
    *bitpos += tree[index].first;
    if (*bitpos > compressed.size() * 8) return -1;
    return tree[index].second;
  }

  *bitpos += root_bits;
  if (*bitpos > compressed.size() * 8) return -1;
  int numbits = tree[index].first - root_bits;

  int index2 = index + tree[index].second +
      PeekBitsInv(numbits, compressed, *bitpos);
  // tree[index2].first is length, tree[index2].second is the value.
  *bitpos += (tree[index2].first - root_bits);
  if (*bitpos > compressed.size() * 8) return -1;
  return tree[index2].second;
}


////////////////////////////////////////////////////////////////////////////////

// base lengths for codes 257-285
static const std::vector<int> kLengthBase = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258};

// num extra bits for length codes 257-285
static const std::vector<int> kLengthExtra = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 5, 0};

// base distances for distance codes
static const std::vector<int> kDistanceBase = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

// num extra bits for distance codes
static const std::vector<int> kDistanceExtra = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13};

// Order of code length alphabet code lengths
static const std::vector<int> kClClOrder = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

void GetSymbol(int length, int distance, int* lengthsymbol, int* lengthextra,
    int* lengthextranum, int* distsymbol, int* distextra, int* distextranum) {
  *lengthsymbol = BinarySearch(length, kLengthBase);
  *lengthextra = length - kLengthBase[*lengthsymbol];
  *lengthextranum = kLengthExtra[*lengthsymbol];
  *distsymbol = BinarySearch(distance, kDistanceBase);
  *distextra = distance - kDistanceBase[*distsymbol];
  *distextranum = kDistanceExtra[*distsymbol];
}

////////////////////////////////////////////////////////////////////////////////

bool DeflateEncodeHuffmanHeader(
    const BlockChoices& block, size_t* bitpos, std::vector<uint8_t>* result,
    std::vector<int>* ll_lengths, std::vector<int>* dist_lengths) {
  std::vector<int> ht_lengths = block.ht_lengths;
  std::vector<int> ht_symbols = CodeLengthsToSymbols(ht_lengths, 7);
  int hlit = block.hlit;
  int hdist = block.hdist;
  int hclen = block.hclen;

  std::vector<int> all_rle = block.all_rle;
  std::vector<int> all_extra = block.all_extra;

  AppendBits(hlit, 5, result, bitpos);
  AppendBits(hdist, 5, result, bitpos);
  AppendBits(hclen, 4, result, bitpos);
  for (int i = 0; i < hclen + 4; i++) {
    AppendBits(ht_lengths[kClClOrder[i]], 3, result, bitpos);
  }

  for (size_t i = 0; i < all_rle.size(); i++) {
    int s = all_rle[i];
    AppendBitsInv(ht_symbols[s], ht_lengths[s], result, bitpos);
    if (s == 16) AppendBits(all_extra[i], 2, result, bitpos);
    if (s == 17) AppendBits(all_extra[i], 3, result, bitpos);
    if (s == 18) AppendBits(all_extra[i], 7, result, bitpos);
  }

  std::vector<int> all_lengths;
  for (size_t i = 0; i < all_rle.size(); i++) {
    int s = all_rle[i];
    int e = all_extra[i];
    if (s < 16) {
      all_lengths.push_back(s);
    } else {
      int rep = 0;
      if (s == 16) {
        rep = e + 3;
        s = all_lengths.empty() ? 0 : all_lengths.back();
      } else if (s == 17) {
        rep = e + 3;
        s = 0;
      } else if (s == 18) {
        rep = e + 11;
        s = 0;
      }
      for (int i = 0; i < rep; i++) all_lengths.push_back(s);
    }
    if (all_lengths.size() >= static_cast<size_t>(hlit + 257 + hdist + 1)) {
      break;
    }
  }
  ll_lengths->assign(
      all_lengths.begin(), all_lengths.begin() + block.hlit + 257);
  dist_lengths->assign(
      all_lengths.begin() + block.hlit + 257, all_lengths.end());
  return true;
}

bool DeflateEncodeHuffmanHeader(const BlockChoices& block, size_t* bitpos,
    std::vector<uint8_t>* result) {
  std::vector<int> dummy_ll, dummy_dist;
  return DeflateEncodeHuffmanHeader(block, bitpos, result,
                                    &dummy_ll, &dummy_dist);
}

void MakeFixedCodes(
    std::vector<int>* ll_lengths, std::vector<int>* dist_lengths) {
  for (int i = 0; i <= 143; i++) ll_lengths->push_back(8);
  for (int i = 144; i <= 255; i++) ll_lengths->push_back(9);
  for (int i = 256; i <= 279; i++) ll_lengths->push_back(7);
  for (int i = 280; i <= 287; i++) ll_lengths->push_back(8);
  for (int i = 0; i <= 31; i++) dist_lengths->push_back(5);
}


bool DeflateDecodeHuffmanHeader(
    const std::vector<uint8_t>& data, size_t* bitpos,
    BlockChoices* blockchoices, std::vector<int>* ll_lengths,
    std::vector<int>* dist_lengths) {
  size_t data_bits = data.size() * 8;
  if (*bitpos + 14 > data_bits) return false;
  int hlit = ReadBits(5, data, bitpos);
  int hdist = ReadBits(5, data, bitpos);
  int hclen = ReadBits(4, data, bitpos);
  std::vector<int> ht_lengths(19, 0);

  for (int i = 0; i < hclen + 4; i++) {
    if (*bitpos + 3 > data_bits) return false;
    ht_lengths[kClClOrder[i]] = ReadBits(3, data, bitpos);
  }
  std::vector<int> all_lengths;
  std::vector<std::pair<int, int>> ht_tree = DecodableHuffmanTree(ht_lengths);
  for (;;) {
    int s = HuffmanDecodeSymbol(data, ht_tree, bitpos);
    if (s < 0) return false;
    blockchoices->all_rle.push_back(s);
    blockchoices->all_extra.push_back(0);
    if (s < 16) {
      all_lengths.push_back(s);
    } else {
      int rep = 0;
      if (s == 16) {
        if (*bitpos + 2 > data_bits) return false;
        rep = ReadBits(2, data, bitpos) + 3;
        blockchoices->all_extra.back() = rep - 3;
        s = all_lengths.empty() ? 0 : all_lengths.back();
      } else if (s == 17) {
        if (*bitpos + 3 > data_bits) return false;
        rep = ReadBits(3, data, bitpos) + 3;
        blockchoices->all_extra.back() = rep - 3;
        s = 0;
      } else if (s == 18) {
        if (*bitpos + 7 > data_bits) return false;
        rep = ReadBits(7, data, bitpos) + 11;
        blockchoices->all_extra.back() = rep - 11;
        s = 0;
      }
      for (int i = 0; i < rep; i++) all_lengths.push_back(s);
    }
    if (all_lengths.size() >= static_cast<size_t>(hlit + 257 + hdist + 1)) {
      break;
    }
  }

  blockchoices->hlit = hlit;
  blockchoices->hdist = hdist;
  blockchoices->hclen = hclen;
  blockchoices->ht_lengths = ht_lengths;
  ll_lengths->assign(all_lengths.begin(), all_lengths.begin() + hlit + 257);
  dist_lengths->assign(all_lengths.begin() + hlit + 257, all_lengths.end());
  return true;
}

bool DeflateDecodeHuffmanHeader(const std::vector<uint8_t>& data,
    size_t* bitpos, BlockChoices* blockchoices) {
  std::vector<int> dummy_ll, dummy_dist;
  return DeflateDecodeHuffmanHeader(data, bitpos, blockchoices,
                                    &dummy_ll, &dummy_dist);
}

bool DeflateDecode(const std::vector<uint8_t>& deflated,
                   std::vector<uint8_t>* result, DeflateChoices* choices) {
  size_t bitpos = 0;
  int lz77_counter = 0;
  size_t origsize = result->size();

  bool bfinal = false;
  while (!bfinal) {
    if (bitpos >= deflated.size() * 8) return false;
    bfinal = ReadBits(1, deflated, &bitpos);
    int btype = ReadBits(2, deflated, &bitpos);

    choices->blocks.resize(choices->blocks.size() + 1);
    BlockChoices* blockchoices = &choices->blocks.back();
    blockchoices->type = btype;
    blockchoices->start_byte = result->size() - origsize;
    blockchoices->start_lz77 = lz77_counter;

    if (btype == 0) {
      size_t pos = (bitpos & 7) ? (bitpos >> 3) + 1 : (bitpos >> 3);
      if (pos + 4 > deflated.size()) return false;

      int len = deflated[pos] + 256 * deflated[pos + 1];
      pos += 2;
      int nlen = deflated[pos] + 256 * deflated[pos + 1];
      pos += 2;
      if (pos + len > deflated.size()) return false;
      if (len != 65535 - nlen) return false;

      for (int i = 0; i < len; i++) result->push_back(deflated[pos++]);

      bitpos = pos * 8;
    } else {
      // huffman trees
      std::vector<int> ll_lengths;
      std::vector<int> dist_lengths;

      if (btype == 1) {
        MakeFixedCodes(&ll_lengths, &dist_lengths);
      } else {
        if (!DeflateDecodeHuffmanHeader(
            deflated, &bitpos, blockchoices, &ll_lengths, &dist_lengths)) {
          return false;
        }
      }
      std::vector<std::pair<int, int>> ll_tree =
          DecodableHuffmanTree(ll_lengths);
      std::vector<std::pair<int, int>> dist_tree =
          DecodableHuffmanTree(dist_lengths);

      for (;;) {
        int s = HuffmanDecodeSymbol(deflated, ll_tree, &bitpos);
        if (s < 0) return false;
        if (s == 256) {
          break;
        } else if (s > 256) {
          lz77_counter++;
          int ll_extra = ReadBits(kLengthExtra[s - 257], deflated, &bitpos);
          int d = HuffmanDecodeSymbol(deflated, dist_tree, &bitpos);
          if (d < 0) return false;
          int dist_extra = ReadBits(kDistanceExtra[d], deflated, &bitpos);
          int length = kLengthBase[s - 257] + ll_extra;
          int dist = kDistanceBase[d] + dist_extra;
          if (static_cast<size_t>(dist) > result->size()) return false;
          for (int i = 0; i < length; i++) {
            result->push_back((*result)[result->size() - dist]);
          }
          blockchoices->lengths.push_back(length);
          blockchoices->dists.push_back(dist);
        } else {
          lz77_counter++;
          result->push_back(s);
          blockchoices->lengths.push_back(1);
          blockchoices->dists.push_back(0);
        }
      }
    }
    blockchoices->end_byte = result->size() - origsize;
    blockchoices->end_lz77 = lz77_counter;
  }
  // Larger than 32 bit sizes not yet supported
  if (result->size() > 0xffffffff) return false;
  return true;
}

bool DeflateEncode(const uint8_t* data, size_t size,
    const DeflateChoices& choices, std::vector<uint8_t>* result) {
  size_t bitpos = 0;
  size_t pos = 0;
  for (size_t b = 0; b < choices.blocks.size(); b++) {
    const BlockChoices& block = choices.blocks[b];
    int blocksize = block.end_byte - block.start_byte;
    int btype = block.type;

    AppendBits(pos + blocksize >= size, 1, result, &bitpos);
    AppendBits(btype, 2, result, &bitpos);

    if (btype == 0) {
      if (blocksize >= 65536) return false;
      result->push_back(blocksize & 255);
      result->push_back((blocksize >> 8) & 255);
      result->push_back((65535 - blocksize) & 255);
      result->push_back(((65535 - blocksize) >> 8) & 255);
      for (int i = 0; i < blocksize; i++) result->push_back(data[pos + i]);
      bitpos = result->size() * 8;
    } else {
      std::vector<int> literals;
      std::vector<int> lengths;
      std::vector<int> distances;
      size_t pos2 = pos;
      for (size_t i = 0; i < block.lengths.size(); i++) {
        if (block.lengths[i] > 1) {
          lengths.push_back(block.lengths[i]);
          distances.push_back(block.dists[i]);
          literals.push_back(INT_MAX);
        } else {
          lengths.push_back(0);
          distances.push_back(0);
          literals.push_back(data[pos2]);
        }
        pos2 += block.lengths[i];
      }

      // huffman trees
      std::vector<int> ll_lengths, ll_symbols;
      std::vector<int> dist_lengths, dist_symbols;

      if (btype == 1) {
        MakeFixedCodes(&ll_lengths, &dist_lengths);
      } else {
        // dynamic huffman codes
        if (!DeflateEncodeHuffmanHeader(block, &bitpos, result,
                                        &ll_lengths, &dist_lengths)) {
          return false;
        }
      }
      ll_symbols = CodeLengthsToSymbols(ll_lengths, 15);
      dist_symbols = CodeLengthsToSymbols(dist_lengths, 15);
      for (size_t i = 0; i < distances.size(); i++) {
        if (distances[i]) {
          int lengthsymbol, lengthextra, lengthextranum;
          int distsymbol, distextra, distextranum;
          GetSymbol(lengths[i], distances[i], &lengthsymbol, &lengthextra,
                   &lengthextranum, &distsymbol, &distextra, &distextranum);
          AppendBitsInv(ll_symbols[lengthsymbol + 257],
                        ll_lengths[lengthsymbol + 257], result, &bitpos);
          AppendBits(lengthextra, lengthextranum, result, &bitpos);
          AppendBitsInv(dist_symbols[distsymbol], dist_lengths[distsymbol],
                        result, &bitpos);
          AppendBits(distextra, distextranum, result, &bitpos);
        } else {
          AppendBitsInv(ll_symbols[literals[i]], ll_lengths[literals[i]],
                        result, &bitpos);
        }
      }
      // end symbol
      AppendBitsInv(ll_symbols[256], ll_lengths[256], result, &bitpos);
    }

    pos += blocksize;
    if (pos >= size) break;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////


// Appends a at the end of result
void AppendTo(const std::vector<uint8_t>& a, std::vector<uint8_t>* result) {
  result->insert(result->end(), a.begin(), a.end());
}

// Written with varint to be future-proof to support 64-bit values later
void Write32Bit(uint32_t value, std::vector<uint8_t>* data) {
  for (;;) {
    uint8_t byte = (value & 127);
    if (value > 127) byte |= 128;
    data->push_back(byte);
    value >>= 7u;
    if (!value) return;
  }
}

bool Read32Bit(const std::vector<uint8_t>& data,
    size_t* pos, uint32_t* value) {
  int num = 0;
  *value = 0;
  for (;;) {
    if (*pos >= data.size()) return false;
    uint8_t byte = data[(*pos)++];
    if (num > 4 || (num == 4 && byte > 15)) return false;  // 32-bit overflow
    *value |= (uint32_t)(byte & 127) << (num * 7);
    if (byte < 128) return true;  // success
    num++;
  }
}

bool Read32Bit(const std::vector<uint8_t>& data,
    size_t* pos, int* value) {
  uint32_t value32;
  Read32Bit(data, pos, &value32);
  if (value32 > 0x7fffffff) return false;
  *value = static_cast<int>(value32);
  return true;
}

static const int kMinDeflateLength = 3;
static const int kMaxDeflateLength = 258;
// For speed: in FindLongestMatch, we only need to know if there is one or not.
static const int kMaxLongestLength = 3;
static const int kWindowSize = 32768;
static const int kWindowMask = (kWindowSize - 1);


// hash chain
static const unsigned kHashBits = 15;
static const unsigned kHashNumValues = 1 << kHashBits;
static const unsigned kHashBitMask = kHashNumValues - 1;
static const unsigned kHashShift = 5;

struct HashChain {
  int* head;
  uint16_t* chain;
  int* val;

  int undo_head = 0;
  uint16_t undo_chain = 0;
  int undo_val = 0;

  // Speed up repetitions of zero
  int* headz;
  uint16_t* chainz;
  uint16_t* zeros;
  uint32_t numzeros = 0;

  int undo_headz = 0;
  uint16_t undo_chainz = 0;
  int undo_zeros = 0;
  uint32_t undo_numzeros = 0;

  HashChain() {
    this->head = (int*)malloc(sizeof(int) * kHashNumValues);
    this->val = (int*)malloc(sizeof(int) * kWindowSize);
    this->chain = (uint16_t*)malloc(sizeof(uint16_t) * kWindowSize);

    for (uint32_t i = 0; i < kHashNumValues; ++i) {
      this->head[i] = -1;
    }
    for (uint32_t i = 0; i < kWindowSize; ++i) {
      this->val[i] = -1;
      this->chain[i] = i;  // same value as index indicates uninitialized
    }

    this->zeros = (uint16_t*)malloc(sizeof(uint16_t) * kWindowSize);
    this->headz = (int*)malloc(sizeof(int) * (kWindowSize + 1));
    this->chainz =
        (uint16_t*)malloc(sizeof(uint16_t) * kWindowSize);

    for (uint32_t i = 0; i < kHashNumValues; ++i) {
      this->headz[i] = -1;
    }
    for (uint32_t i = 0; i < kWindowSize; ++i) {
      this->chainz[i] = i;
    }
  }

  ~HashChain() {
    free(this->head);
    free(this->val);
    free(this->chain);

    free(this->headz);
    free(this->zeros);
    free(this->chainz);
  }
};

uint32_t getHash(const uint8_t* data, size_t size, size_t pos) {
  uint32_t result = 0;
  if (pos + 2 < size) {
    result ^= (uint32_t)(data[pos + 0] << 0u);
    result ^= (uint32_t)(data[pos + 1] << kHashShift);
    result ^= (uint32_t)(data[pos + 2] << (kHashShift * 2));
  } else {
    size_t amount, i;
    if (pos >= size) return 0;
    amount = size - pos;
    for (i = 0; i < amount; ++i) {
      result ^= (uint32_t)(data[pos + i] << (i * kHashShift));
    }
  }
  return result & kHashBitMask;
}

uint32_t countZeros(const uint8_t* data, size_t size, size_t pos,
                    uint32_t prevzeros) {
  size_t end = pos + kWindowSize;
  if (end > size) end = size;
  if (prevzeros > 0) {
    if (prevzeros >= kWindowMask && data[end - 1] == 0)  {
      return prevzeros;
    } else {
      return prevzeros - 1;
    }
  }
  uint32_t num = 0;
  while (pos + num < end && data[pos + num] == 0) num++;
  return num;
}

// wpos = pos & kWindowMask
void updateHashChain(const uint8_t* data, size_t size, size_t pos,
                     HashChain* hash) {
  uint32_t hashval = getHash(data, size, pos);
  uint32_t wpos = pos & kWindowMask;

  hash->undo_val = hash->val[wpos];
  hash->undo_chain = hash->chain[wpos];
  hash->undo_head = hash->head[hashval];

  hash->val[wpos] = (int)hashval;
  if (hash->head[hashval] != -1) hash->chain[wpos] = hash->head[hashval];
  hash->head[hashval] = wpos;

  uint32_t numzeros = countZeros(data, size, pos, hash->numzeros);
  hash->undo_zeros = hash->zeros[wpos];
  hash->undo_chainz = hash->chainz[wpos];
  hash->undo_headz = hash->headz[numzeros];
  hash->undo_numzeros = hash->numzeros;

  hash->zeros[wpos] = numzeros;
  if (hash->headz[numzeros] != -1) hash->chainz[wpos] = hash->headz[numzeros];
  hash->headz[numzeros] = wpos;
  hash->numzeros = numzeros;
}


void undoUpdateHashChain(const uint8_t* data, size_t size, size_t pos,
                         HashChain* hash) {
  uint32_t hashval = getHash(data, size, pos);
  uint32_t wpos = pos & kWindowMask;

  hash->val[wpos] = hash->undo_val;
  hash->chain[wpos] = hash->undo_chain;
  hash->head[hashval] = hash->undo_head;

  uint32_t numzeros = hash->numzeros;
  hash->zeros[wpos] = hash->undo_zeros;
  hash->chainz[wpos] = hash->undo_chainz;
  hash->headz[numzeros] = hash->undo_headz;
  hash->numzeros = hash->undo_numzeros;
}

////////////////////////////////////////////////////////////////////////////////

// The prediction tries to predict the LZ77 lengths and distances. For the
// length, it predicts the longest lazy match and dynamically adjusts. For the
// distance, it predicts the shortest possible distance for that length.
struct Predictor {
  Predictor(const uint8_t* data, size_t size) : data(data), size(size) {
  }

  bool PredictLength(int* pred_len, int* pred_dist, int* longest_possible) {
    if (byte >= size) return false;
    updateHashChain(data, size, byte, &chain);
    updateHashChain(data, size, byte, &chain_longest);
    int dummy_dist = 0;

    if (static_cast<size_t>(lazy_stored) == byte) {
      *pred_len = lazy_len;
      *pred_dist = lazy_dist;
      *longest_possible = lazy_longest_possible;
    } else {
      FindMatch(data, size, byte, kWindowSize,
                kMinDeflateLength, kMaxDeflateLength,
                maxchainlength, &chain, pred_dist, pred_len);
      FindLongestMatch(data, size, byte, kWindowSize,
                       kMinDeflateLength, kMaxLongestLength,
                       &chain_longest, &dummy_dist, longest_possible);
    }

    lazy_stored = 0;
    lazy_prev = false;
    if (*pred_len >= kMinDeflateLength &&
        byte + 1 < size && *pred_len < maxlazymatch) {
      updateHashChain(data, size, byte + 1, &chain);
      updateHashChain(data, size, byte + 1, &chain_longest);
      int maxchainlen = maxchainlength;
      if (*pred_len > good_length) maxchainlen >>= 2;
      FindMatch(data, size, byte + 1, kWindowSize,
                kMinDeflateLength, kMaxDeflateLength,
                maxchainlen, &chain,
                &lazy_dist, &lazy_len);
      FindLongestMatch(data, size, byte + 1, kWindowSize,
                       kMinDeflateLength, kMaxLongestLength,
                       &chain_longest,
                       &dummy_dist, &lazy_longest_possible);
      undoUpdateHashChain(data, size, byte + 1, &chain);
      undoUpdateHashChain(data, size, byte + 1, &chain_longest);
      if (lazy_len > *pred_len) {
        *pred_len = 1;
        *pred_dist = 0;
        lazy_stored = byte + 1;
        lazy_prev = true;
      }
    }

    if (*pred_len == kMinDeflateLength && *pred_dist > toofar) {
      *pred_len = 1;
      *pred_dist = 0;
    }

    if (*pred_len < 0 || *pred_len > kMaxDeflateLength || *pred_dist < 0
        || *pred_dist > kWindowSize || static_cast<size_t>(*pred_dist) > byte) {
      std::cout << "prediction error: " << *pred_len << " " << *pred_dist
                << " " << byte << std::endl;
      std::exit(1);
    }

    return true;
  }

  // use only if actual_len >= kMinDeflateLength. prev_dist is a previously
  // predicted dist, for predicting a next one
  void PredictDist(int actual_len, int* pred_dist, int prev_dist = 0) {
    FindShortestDistForLength(data, size, byte, kWindowSize,
                              kMinDeflateLength, kMaxDeflateLength, &chain,
                              actual_len, pred_dist, prev_dist);
  }

  void Update(int actual_len, int /*pred_len*/) {
    if (actual_len <= max_insert_length) {
      for (int i = 1; i < actual_len; i++) {
        updateHashChain(data, size, byte + i, &chain);
      }
    }
    for (int i = 1; i < actual_len; i++) {
      updateHashChain(data, size, byte + i, &chain_longest);
    }

    byte += actual_len;
  }

  // skip uncompresed bytes, but still update the hash chain
  void SkipBytes(int num) {
    if (num < kWindowSize) {
      for (int i = 0; i < num; i++) {
        updateHashChain(data, size, byte + i, &chain);
      }
      byte += num;
    } else {
      byte = byte + num - kWindowSize;
      for (int i = 0; i < kWindowSize; i++) {
        updateHashChain(data, size, byte + i, &chain);
      }
      byte += kWindowSize;
    }
  }

  void SetZlibLevel(int level) {
    int data[40] = {
      0, 0, 0, 0,
      4, 4, 8, 4,  // 1
      4, 5, 16, 8,  // 2
      4, 6, 32, 32,  // 3
      4, 4, 16, 16,  // 4
      8, 16, 32, 32,  // 5
      8, 16, 128, 128,  // 6
      8, 32, 128, 256,  // 7
      32, 128, 258, 1024,  // 8
      32, 258, 258, 4096  // 9
    };
    if (level >= 1 && level <= 9) {
      good_length = data[level * 4 + 0];
      maxlazymatch = data[level * 4 + 1];
      nice_length = data[level * 4 + 2];
      maxchainlength = data[level * 4 + 3];
      // emulate fast also
      if (level < 4) {
        max_insert_length = maxlazymatch;
        maxlazymatch = 0;
        toofar = 32768;
      }
    }
    // everything at the maximum
    if (level == 10) {
      good_length = kMaxDeflateLength;
      maxlazymatch = kMaxDeflateLength;
      nice_length = kMaxDeflateLength;
      maxchainlength = kWindowSize;
      max_insert_length = kMaxDeflateLength;
      toofar = 4096;
    }
  }

  void FindShortestDistForLength(const uint8_t* data, size_t size, size_t pos,
                               int max_dist, int /*min_len*/, int max_len,
                               HashChain* chain, int actual_len,
                               int* result_dist, int prev_result_dist) {
    size_t pos2 = pos - prev_result_dist;
    uint32_t wpos = pos & kWindowMask;
    uint32_t wpos2 = pos2 & kWindowMask;
    uint32_t hashval = getHash(data, size, pos);
    uint32_t hashpos = chain->chain[wpos2];

    int prev_dist = prev_result_dist;
    int end = std::min<int>(pos + max_len, size);
    max_dist = std::min<int>(max_dist, pos);
    *result_dist = 0;

    for (;;) {
      int dist = (hashpos <= wpos) ?
          (wpos - hashpos) : (wpos - hashpos + kWindowMask + 1);
      // went completely around the circular buffer
      if (dist < prev_dist) break;
      prev_dist = dist;

      int len = 0;
      if (dist > 0) {
        int i = pos;
        int j = pos - dist;
        if (chain->numzeros > 3) {
          int r = std::min<int>(chain->numzeros, chain->zeros[hashpos]);
          i += r;
          j += r;
          len += r;
        }
        while (i < end && data[i] == data[j]) {
          i++;
          j++;
          len++;
        }
        if (len >= actual_len) {
          *result_dist = dist;
          return;
        }
      }

      if (chain->numzeros >= 3 && len > static_cast<int>(chain->numzeros)) {
        if (hashpos == chain->chainz[hashpos]) break;
        hashpos = chain->chainz[hashpos];
        if (chain->zeros[hashpos] != chain->numzeros) break;
      } else {
        if (hashpos == chain->chain[hashpos]) break;
        hashpos = chain->chain[hashpos];
        if (chain->val[hashpos] != (int)hashval) break;  // outdated hash value
      }
    }
  }

  // Finds longest LZ77 match as length, distance pair. Emulates the concept of
  // max chain length for the particular hash used, but not with speedup in mind
  // but to emulate zlib for better prediction. It will continue searching
  // without max chain length and store the longest theoretically possible
  // length in longest_possible.
  void FindMatch(const uint8_t* data, size_t size, size_t pos, int max_dist,
      int min_len, int max_len, uint32_t maxchainlength, HashChain* chain,
      int* result_dist, int* result_len) {
    uint32_t wpos = pos & kWindowMask;
    uint32_t hashval = getHash(data, size, pos);
    uint32_t hashpos = chain->chain[wpos];

    int prev_dist = 0;
    int end = std::min<int>(pos + max_len, size);
    max_dist = std::min<int>(max_dist, pos);
    *result_len = 1;
    *result_dist = 0;

    uint32_t chainlength = 0;

    for (;;) {
      int dist = (hashpos <= wpos) ?
          (wpos - hashpos) : (wpos - hashpos + kWindowMask + 1);
      // went completely around the circular buffer
      if (dist < prev_dist) break;
      prev_dist = dist;
      int len = 0;
      if (dist > 0) {
        int i = pos;
        int j = pos - dist;
        if (chain->numzeros > 3) {
          int r = std::min<int>(chain->numzeros, chain->zeros[hashpos]);
          i += r;
          j += r;
          len += r;
        }
        if (len > max_len) len = max_len;
        while (i < end && data[i] == data[j]) {
          i++;
          j++;
          len++;
        }
        if (len >= min_len && len > *result_len) {
          *result_len = len;
          *result_dist = dist;
          if (len >= nice_length) break;
        }
      }

      chainlength++;
      if (chainlength >= maxchainlength) break;

      if (chain->numzeros >= 3 && len > static_cast<int>(chain->numzeros)) {
        if (hashpos == chain->chainz[hashpos]) break;
        hashpos = chain->chainz[hashpos];
        if (chain->zeros[hashpos] != chain->numzeros) break;
      } else {
        if (hashpos == chain->chain[hashpos]) break;
        hashpos = chain->chain[hashpos];
        if (chain->val[hashpos] != (int)hashval) break;  // outdated hash value
      }
    }
  }


  void FindLongestMatch(const uint8_t* data, size_t size, size_t pos,
      int max_dist, int min_len, int max_len, HashChain* chain,
      int* result_dist, int* result_len) {
    uint32_t wpos = pos & kWindowMask;
    uint32_t hashval = getHash(data, size, pos);
    uint32_t hashpos = chain->chain[wpos];

    int prev_dist = 0;
    int end = std::min<int>(pos + max_len, size);
    max_dist = std::min<int>(max_dist, pos);
    *result_len = 1;
    *result_dist = 0;

    for (;;) {
      int dist = (hashpos <= wpos) ?
          (wpos - hashpos) : (wpos - hashpos + kWindowMask + 1);
      // went completely around the circular buffer
      if (dist < prev_dist) break;
      prev_dist = dist;
      int len = 0;
      if (dist > 0) {
        int i = pos;
        int j = pos - dist;
        if (chain->numzeros > 3) {
          int r = std::min<int>(chain->numzeros, chain->zeros[hashpos]);
          i += r;
          j += r;
          len += r;
        }
        if (len > max_len) len = max_len;
        while (i < end && data[i] == data[j]) {
          i++;
          j++;
          len++;
        }
        if (len >= min_len && len > *result_len) {
          *result_len = len;
          *result_dist = dist;
          if (len >= max_len) break;
        }
      }

      if (chain->numzeros >= 3 && len > static_cast<int>(chain->numzeros)) {
        if (hashpos == chain->chainz[hashpos]) break;
        hashpos = chain->chainz[hashpos];
        if (chain->zeros[hashpos] != chain->numzeros) break;
      } else {
        if (hashpos == chain->chain[hashpos]) break;
        hashpos = chain->chain[hashpos];
        if (chain->val[hashpos] != (int)hashval) break;  // outdated hash value
      }
    }
  }

  size_t byte = 0;  // byte position

  const uint8_t* data;
  size_t size;
  HashChain chain;
  HashChain chain_longest;

  // set to 0 to disable lazy matching, max value is kMaxDeflateLength.
  int maxlazymatch = kMaxDeflateLength;
  // set to kMaxDeflateLength to do regular encoding that finds shortest dist,
  // or less (4, 5 or 6) to do like the fast modes of zlib do (= don't update
  // the hash chain if length longer than this).
  int max_insert_length  = kMaxDeflateLength;
  uint32_t maxchainlength = kWindowSize;  // kWindowSize to allow all
  int good_length = kMaxDeflateLength;
  int nice_length = kMaxDeflateLength;

  int lazy_len = 0;
  int lazy_dist = 0;
  int lazy_longest_possible = 0;
  int lazy_stored = -1;
  bool lazy_prev = false;
  int toofar = 4096;  // see official deflate.c "TOO_FAR"
};

// The encoded choices as separate stream, each with different entropy
struct ChoicesEncoded {
  std::vector<uint8_t> other;
  std::vector<uint8_t> blockheaders;
  std::vector<uint8_t> lencodes;
  std::vector<uint8_t> distcodes;
  std::vector<uint8_t> lenextra;
  std::vector<uint8_t> distextra;
};


std::vector<uint8_t> Combine(const ChoicesEncoded& encoded) {
  std::vector<uint8_t> result;
  Write32Bit(encoded.other.size(), &result);
  AppendTo(encoded.other, &result);
  Write32Bit(encoded.blockheaders.size(), &result);
  AppendTo(encoded.blockheaders, &result);
  Write32Bit(encoded.lencodes.size(), &result);
  AppendTo(encoded.lencodes, &result);
  Write32Bit(encoded.distcodes.size(), &result);
  AppendTo(encoded.distcodes, &result);
  Write32Bit(encoded.lenextra.size(), &result);
  AppendTo(encoded.lenextra, &result);
  Write32Bit(encoded.distextra.size(), &result);
  AppendTo(encoded.distextra, &result);
  return result;
}

bool Split(const std::vector<uint8_t>& data, ChoicesEncoded* result) {
  size_t pos = 0;
  uint32_t size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->other.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->blockheaders.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->lencodes.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->distcodes.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->lenextra.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  if (!Read32Bit(data, &pos, &size)) return false;
  if (pos + size > data.size()) return false;
  result->distextra.assign(data.begin() + pos, data.begin() + pos + size);
  pos += size;

  return true;
}

static const int kMaxDistTries = 16;

int GuessZlibLevel(const std::vector<uint8_t>& data,
                   const DeflateChoices& stream) {
  int bestlevel = 1;
  int bestcorrect = 0;
  int maxpredictions = 65536;

  for (int level = 1; level <= 10; level++) {
    int numcorrect = 0;
    int numdone = 0;
    size_t datapos = stream.abs_pos;
    size_t datasize =
        stream.blocks.back().end_byte - stream.blocks[0].start_byte;
    Predictor predictor(data.data() + datapos, datasize);
    predictor.SetZlibLevel(level);
    for (size_t i = 0; i < stream.blocks.size(); i++) {
      const BlockChoices& block = stream.blocks[i];
      if (block.type == 0) continue;
      for (size_t j = 0; j < block.lengths.size(); j++) {
        int actual_dist = block.dists[j];
        int actual_len = block.lengths[j];

        int pred_len, pred_dist, longest_possible;
        if (!predictor.PredictLength(
            &pred_len, &pred_dist, &longest_possible)) {
          return 0;
        }
        predictor.Update(actual_len, pred_len);
        if (pred_len == actual_len && pred_dist == actual_dist) numcorrect++;
        numdone++;
        if (numdone > maxpredictions) break;
      }
      if (numdone > maxpredictions) break;
    }

    if (numcorrect > bestcorrect) {
      bestlevel = level;
      bestcorrect = numcorrect;
    }
  }
  return bestlevel;
}

bool EncodeChoices(const std::vector<uint8_t>& data,
                   const DeflateChoices& choices,
                   ChoicesEncoded* encoded) {
  int level = GuessZlibLevel(data, choices);
  if (level == 0) return false;
  encoded->other.push_back(level);
  Write32Bit(choices.blocks.size(), &encoded->other);
  if (choices.blocks.empty()) return true;
  size_t datapos = choices.abs_pos;
  size_t datasize =
      choices.blocks.back().end_byte - choices.blocks[0].start_byte;
  Predictor predictor(data.data() + datapos, datasize);
  predictor.SetZlibLevel(level);
  Write32Bit(datapos, &encoded->other);
  Write32Bit(datasize, &encoded->other);
  for (size_t i = 0; i < choices.blocks.size(); i++) {
    const BlockChoices& block = choices.blocks[i];

    encoded->blockheaders.push_back(block.type);
    Write32Bit(block.start_byte, &encoded->blockheaders);
    Write32Bit(block.end_byte, &encoded->blockheaders);
    Write32Bit(block.start_lz77, &encoded->blockheaders);
    Write32Bit(block.end_lz77, &encoded->blockheaders);
    size_t bitpos = encoded->blockheaders.size() * 8;
    if (block.type == 2) {
      if (!DeflateEncodeHuffmanHeader(
          block, &bitpos, &encoded->blockheaders)) {
        return false;
      }
    }
    if (block.type == 0) {
      predictor.SkipBytes(block.end_byte - block.start_byte);
      continue;
    }
    for (size_t j = 0; j < block.lengths.size(); j++) {
      int actual_dist = block.dists[j];
      int actual_len = block.lengths[j];

      int pred_len, pred_dist, longest_possible;
      if (!predictor.PredictLength(
          &pred_len, &pred_dist, &longest_possible)) {
        return false;
      }
      int encoded_len = 0;

      if (longest_possible >= kMinDeflateLength) {
        // encoded_len meaning:
        // 0: prediction correct
        // 1: actual len is 1
        // [2..pred_len-1]: actual len is encoded_len
        // [pred_len..254]: actual len is pred_len - encoded_len + 1
        // 255: actual len encoded exactly in lenextra

        if (pred_len == actual_len) {
          encoded_len = 0;
        } else if (actual_len == 1) {
          encoded_len = 1;
        } else {
          encoded_len = (actual_len > pred_len)
            ? actual_len : (pred_len - actual_len + 1);
          if (encoded_len < 2 || encoded_len > 254) encoded_len = 255;
        }

        encoded->lencodes.push_back(encoded_len);
        if (encoded_len == 255) {
          encoded->lenextra.push_back(actual_len - kMinDeflateLength);
        }
      }

      if (actual_len > 1) {
        // only relevant for predicting low qualities, which use unoptimal
        // dists. The higher, the slower. In high qualities, dist always
        // predicted correctly so tries never increments and it's fast.
        if (pred_len != actual_len) {
          predictor.PredictDist(actual_len, &pred_dist, 0);
        }
        int tries = 0;
        while (tries < kMaxDistTries && pred_dist != actual_dist) {
          predictor.PredictDist(actual_len, &pred_dist, pred_dist);
          tries++;
        }
        // encoded_dist meaning:
        // 0: prediction correct
        // [1..kMaxDistTries-1]: hop this many times to next possible dists
        // kMaxDistTries: actual dist encoded in 2 bytes of distextra
        int encoded_dist = (tries < kMaxDistTries) ? tries : kMaxDistTries;
        encoded->distcodes.push_back(encoded_dist);
        if (encoded_dist == kMaxDistTries) {
          encoded->distextra.push_back((actual_dist - 1) & 255);
          encoded->distextra.push_back(((actual_dist - 1) >> 8) & 255);
        }
      }

      predictor.Update(actual_len, pred_len);
    }
  }

  return true;
}

bool DecodeChoices(const std::vector<uint8_t>& data,
                   const ChoicesEncoded& encoded,
                   DeflateChoices* result) {
  size_t opos = 0;
  size_t headerpos = 0;
  size_t lenpos = 0;
  size_t distpos = 0;
  size_t lenextrapos = 0;
  size_t distextrapos = 0;

  if (opos >= encoded.other.size()) return false;
  int level = encoded.other[opos++];
  if (level < 1 || level > 10) return false;
  uint32_t numblocks;
  if (!Read32Bit(encoded.other, &opos, &numblocks)) return false;
  if (numblocks == 0) return true;
  uint32_t datapos;
  if (!Read32Bit(encoded.other, &opos, &datapos)) return false;
  result->abs_pos = datapos;
  uint32_t datasize;
  if (!Read32Bit(encoded.other, &opos, &datasize)) return false;
  Predictor predictor(data.data() + datapos, datasize);
  predictor.SetZlibLevel(level);

  for (uint32_t ib = 0; ib < numblocks; ib++) {
    result->blocks.resize(result->blocks.size() + 1);
    BlockChoices* block = &result->blocks.back();
    if (headerpos >= encoded.blockheaders.size()) return false;
    block->type = encoded.blockheaders[headerpos++];
    if (!Read32Bit(encoded.blockheaders, &headerpos, &block->start_byte)) {
      return false;
    }
    if (!Read32Bit(encoded.blockheaders, &headerpos, &block->end_byte)) {
      return false;
    }
    if (!Read32Bit(encoded.blockheaders, &headerpos, &block->start_lz77)) {
      return false;
    }
    if (!Read32Bit(encoded.blockheaders, &headerpos, &block->end_lz77)) {
      return false;
    }
    size_t bitpos = headerpos * 8;
    if (block->type == 2) {
      if (!DeflateDecodeHuffmanHeader(encoded.blockheaders, &bitpos, block)) {
        return false;
      }
    }
    headerpos = (bitpos + 7) / 8;
    if (block->type == 0) {
      predictor.SkipBytes(block->end_byte - block->start_byte);
      continue;
    }

    size_t lz77size = block->end_lz77 - block->start_lz77;
    for (size_t ie = 0; ie < lz77size; ie++) {
      int pred_len, pred_dist, longest_possible;
      if (!predictor.PredictLength(
          &pred_len, &pred_dist, &longest_possible)) {
        return false;
      }

      int actual_len = 1;
      int encoded_len = 1;
      if (longest_possible >= kMinDeflateLength) {
        if (lenpos >= encoded.lencodes.size()) return false;

        encoded_len = encoded.lencodes[lenpos++];

        if (encoded_len == 255) {
          if (lenextrapos >= encoded.lenextra.size()) return false;
          actual_len = (encoded.lenextra[lenextrapos++] + kMinDeflateLength);
        } else if (encoded_len == 0) {
          actual_len = pred_len;
        } else if (encoded_len == 1) {
          actual_len = 1;
        } else if (encoded_len >= pred_len) {
          actual_len = encoded_len;
        } else {
          actual_len = pred_len - encoded_len + 1;
        }

        if (actual_len < 0 || actual_len > kMaxDeflateLength) {
          std::cout << "error: invalid length: " << actual_len << " "
                    << pred_len << " " << encoded_len << " "
                    << longest_possible << " " << predictor.byte
                    << std::endl;
          return false;
        }
      }

      int actual_dist = 0;
      if (actual_len > 1) {
        if (distpos >= encoded.distcodes.size()) return false;
        int encoded_dist = encoded.distcodes[distpos++];

        if (encoded_dist < 0 || encoded_dist > 255) {
          std::cout << "error: invalid dist: " << encoded_dist << " "
                    << pred_dist << " " << actual_len << " " << pred_len
                    << " " << encoded_len << " " << predictor.byte
                    << std::endl;
          return false;
        }
        if (pred_len != actual_len) {
          predictor.PredictDist(actual_len, &pred_dist, 0);
        }

        if (encoded_dist == kMaxDistTries) {
          if (distextrapos + 2 > encoded.distextra.size()) return false;
          actual_dist = encoded.distextra[distextrapos] +
              (encoded.distextra[distextrapos + 1] << 8) + 1;
          distextrapos += 2;
        } else {
          for (int tries = 0; tries < encoded_dist; tries++) {
            predictor.PredictDist(actual_len, &pred_dist, pred_dist);
          }
          actual_dist = pred_dist;
        }
      }

      predictor.Update(actual_len, pred_len);

      block->lengths.push_back(actual_len);
      block->dists.push_back(actual_dist);
    }
  }

  return true;
}

bool EncodeChoices(const DeflateChoices& choices,
                   const std::vector<uint8_t>& data,
                   std::vector<uint8_t>* result) {
  ChoicesEncoded encoded;
  if (!EncodeChoices(data, choices, &encoded)) return false;
  *result = Combine(encoded);
  return true;
}

bool DecodeChoices(const std::vector<uint8_t>& data,
                   const std::vector<uint8_t>& encoded,
                   DeflateChoices* result) {
  if (encoded.empty()) return false;
  ChoicesEncoded choices_encoded;
  if (!Split(encoded, &choices_encoded)) return false;
  return DecodeChoices(data, choices_encoded, result);
}

}  // namespace

bool Grittibanzli(const std::vector<uint8_t>& deflated,
                  std::vector<uint8_t>* uncompressed,
                  std::vector<uint8_t>* choices_encoded) {
  if (deflated.size() > 0xffffffff) {
    // Larger than 32 bit sizes not yet supported
    return false;
  }

  DeflateChoices choices;
  // deflate *de*code is for *en*coding for us
  if (!DeflateDecode(deflated, uncompressed, &choices)) {
    return false;
  }

  // quick verify (not full verification)
  std::vector<uint8_t> test;
  if (!DeflateEncode(uncompressed->data(), uncompressed->size(), choices, &test)
      || test != deflated) {
    std::cout << "internal verification failed" << std::endl;
    return false;
  }

  if (!EncodeChoices(choices, *uncompressed, choices_encoded)) {
    return false;
  }

  return true;
}

bool Ungrittibanzli(const std::vector<uint8_t>& uncompressed,
                    const std::vector<uint8_t>& choices_encoded,
                    std::vector<uint8_t>* deflated) {
  DeflateChoices choices;
  if (!DecodeChoices(uncompressed, choices_encoded, &choices)) {
    return false;
  }
  // deflate *en*code is for *de*coding for us
  if (!DeflateEncode(uncompressed.data(), uncompressed.size(),
      choices, deflated)) {
    return false;
  }

  return true;
}

}  // namespace grittibanzli
