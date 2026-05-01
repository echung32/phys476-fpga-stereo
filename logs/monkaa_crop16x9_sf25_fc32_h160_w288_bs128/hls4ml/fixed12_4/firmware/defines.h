#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<12,4>, 1*1> input_t;
typedef nnet::array<ap_fixed<12,4>, 1*1> layer8_t;
typedef ap_fixed<29,13> feat_conv1_accum_t;
typedef nnet::array<ap_fixed<29,13>, 32*1> feat_conv1_result_t;
typedef ap_fixed<12,4> feat_conv1_weight_t;
typedef ap_fixed<12,4> feat_conv1_bias_t;
typedef nnet::array<ap_fixed<12,4>, 32*1> layer3_t;
typedef ap_fixed<18,8> feat_conv1_relu_table_t;
typedef nnet::array<ap_fixed<12,4>, 32*1> layer9_t;
typedef ap_fixed<34,18> feat_conv2_accum_t;
typedef nnet::array<ap_fixed<34,18>, 32*1> feat_conv2_result_t;
typedef ap_fixed<12,4> feat_conv2_weight_t;
typedef ap_fixed<12,4> feat_conv2_bias_t;
typedef nnet::array<ap_fixed<12,4>, 32*1> layer5_t;
typedef ap_fixed<18,8> feat_conv2_relu_table_t;
typedef nnet::array<ap_fixed<12,4>, 32*1> layer10_t;
typedef ap_fixed<34,18> feat_conv3_accum_t;
typedef nnet::array<ap_fixed<34,18>, 16*1> feat_conv3_result_t;
typedef ap_fixed<12,4> feat_conv3_weight_t;
typedef ap_fixed<12,4> feat_conv3_bias_t;
typedef nnet::array<ap_fixed<12,4>, 16*1> result_t;
typedef ap_fixed<18,8> feat_conv3_relu_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
