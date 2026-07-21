/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

// Single-class (person-only) YOLOv8-Pose parser.
// Model output per detection: [cx, cy, w, h, person_score, 17*3 keypoints] = 56 channels

#include <algorithm>
#include <cmath>

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

#define NMS_THRESH 0.45;

#define NUM_KEYPOINTS    17
#define BBOX_CHANNELS    4
#define CLASS_CHANNELS   1
#define KPT_CHANNELS     (NUM_KEYPOINTS * 3)   // 51
#define KPT_START_IDX    (BBOX_CHANNELS + CLASS_CHANNELS)  // 5
#define EXPECTED_CHANNELS (BBOX_CHANNELS + CLASS_CHANNELS + KPT_CHANNELS)  // 56

// Overhead factory camera: a real person never occupies more than 20% of the frame.
#define MAX_BBOX_RATIO   0.20f

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

extern "C" bool
NvDsInferParseYoloPoseE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static std::vector<NvDsInferInstanceMaskInfo>
nonMaximumSuppression(std::vector<NvDsInferInstanceMaskInfo> binfo)
{
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };

  auto computeIoU = [&overlap1D](NvDsInferInstanceMaskInfo& bbox1, NvDsInferInstanceMaskInfo& bbox2) -> float {
    float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
    float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
    float area1 = (bbox1.width) * (bbox1.height);
    float area2 = (bbox2.width) * (bbox2.height);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };

  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferInstanceMaskInfo& b1, const NvDsInferInstanceMaskInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferInstanceMaskInfo> out;
  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
  }
  return out;
}

static void
addPoseProposal(const float* output, const uint& channelsSize, const uint& netW, const uint& netH, const uint& b,
    NvDsInferInstanceMaskInfo& bbi)
{
  bbi.mask = new float[KPT_CHANNELS];
  for (uint p = 0; p < NUM_KEYPOINTS; ++p) {
    bbi.mask[p * 3 + 0] = clamp(output[b * channelsSize + KPT_START_IDX + p * 3 + 0], 0, netW);
    bbi.mask[p * 3 + 1] = clamp(output[b * channelsSize + KPT_START_IDX + p * 3 + 1], 0, netH);
    bbi.mask[p * 3 + 2] = output[b * channelsSize + KPT_START_IDX + p * 3 + 2];
  }
  bbi.mask_width = netW;
  bbi.mask_height = netH;
  bbi.mask_size = sizeof(float) * KPT_CHANNELS;
}

static NvDsInferInstanceMaskInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferInstanceMaskInfo b;

  float x1 = clamp(bx1, 0, netW);
  float y1 = clamp(by1, 0, netH);
  float x2 = clamp(bx2, 0, netW);
  float y2 = clamp(by2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

// YOLOv8-Pose format: output[b] = [cx, cy, w, h, person_score, kpt0_x, kpt0_y, kpt0_conf, ...]
static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloPose(const float* output, const uint& outputSize, const uint& channelsSize, const uint& netW,
    const uint& netH, const float threshold)
{
  std::vector<NvDsInferInstanceMaskInfo> binfo;

  static bool first_run = true;
  if (first_run) {
    std::cout << "Parser (single-class) - outputSize: " << outputSize
              << ", channelsSize: " << channelsSize
              << ", expected: " << EXPECTED_CHANNELS
              << ", threshold: " << threshold << std::endl;
    if (channelsSize != EXPECTED_CHANNELS) {
      std::cerr << "WARNING: channelsSize(" << channelsSize << ") != expected("
                << EXPECTED_CHANNELS << "). Model may not be single-class!" << std::endl;
    }
    first_run = false;
  }

  for (uint b = 0; b < outputSize; ++b) {
    float score = output[b * channelsSize + 4];

    if (score < threshold) {
      continue;
    }

    float bxc = output[b * channelsSize + 0];
    float byc = output[b * channelsSize + 1];
    float bw  = output[b * channelsSize + 2];
    float bh  = output[b * channelsSize + 3];

    if (bw > MAX_BBOX_RATIO * netW || bh > MAX_BBOX_RATIO * netH) {
      continue;
    }

    float bx1 = bxc - bw / 2;
    float by1 = byc - bh / 2;
    float bx2 = bx1 + bw;
    float by2 = by1 + bh;

    NvDsInferInstanceMaskInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

    if (bbi.width < 1 || bbi.height < 1) {
      continue;
    }

    bbi.detectionConfidence = score;
    bbi.classId = 0;

    addPoseProposal(output, channelsSize, netW, netH, b, bbi);
    binfo.push_back(bbi);
  }

  return binfo;
}

// YOLOv8-PoseE (end-to-end) variant: bbox already in [x1, y1, x2, y2] format
static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloPoseE(const float* output, const uint& outputSize, const uint& channelsSize, const uint& netW,
    const uint& netH, const float threshold)
{
  std::vector<NvDsInferInstanceMaskInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float score = output[b * channelsSize + 4];

    if (score < threshold) {
      continue;
    }

    float bx1 = output[b * channelsSize + 0];
    float by1 = output[b * channelsSize + 1];
    float bx2 = output[b * channelsSize + 2];
    float by2 = output[b * channelsSize + 3];

    if ((bx2 - bx1) > MAX_BBOX_RATIO * netW || (by2 - by1) > MAX_BBOX_RATIO * netH) {
      continue;
    }

    NvDsInferInstanceMaskInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

    if (bbi.width < 1 || bbi.height < 1) {
      continue;
    }

    bbi.detectionConfidence = score;
    bbi.classId = 0;

    addPoseProposal(output, channelsSize, netW, netH, b, bbi);
    binfo.push_back(bbi);
  }

  return binfo;
}

static bool
NvDsInferParseCustomYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0];
  const uint channelsSize = output.inferDims.d[1];
  const float threshold = detectionParams.perClassPreclusterThreshold[0];

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorYoloPose(
      (const float*) (output.buffer), outputSize, channelsSize,
      networkInfo.width, networkInfo.height, threshold);

  objectList.clear();
  objectList = nonMaximumSuppression(objects);

  return true;
}

static bool
NvDsInferParseCustomYoloPoseE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0];
  const uint channelsSize = output.inferDims.d[1];
  const float threshold = detectionParams.perClassPreclusterThreshold[0];

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorYoloPoseE(
      (const float*) (output.buffer), outputSize, channelsSize,
      networkInfo.width, networkInfo.height, threshold);

  objectList.clear();
  objectList = nonMaximumSuppression(objects);

  return true;
}

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloPose(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloPoseE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloPoseE(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPose);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPoseE);
