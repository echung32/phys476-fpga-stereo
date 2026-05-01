#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &image,
    hls::stream<result_t> &layer7_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=image,layer7_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<feat_conv1_weight_t, 288>(w2, "w2.txt");
        nnet::load_weights_from_txt<feat_conv1_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<feat_conv2_weight_t, 9216>(w4, "w4.txt");
        nnet::load_weights_from_txt<feat_conv2_bias_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<feat_conv3_weight_t, 4608>(w6, "w6.txt");
        nnet::load_weights_from_txt<feat_conv3_bias_t, 16>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=46980

    hls::stream<feat_conv1_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=46080

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=46080

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=46980

    hls::stream<feat_conv2_result_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=46080

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=46080

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=46980

    hls::stream<feat_conv3_result_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=46080

    nnet::zeropad2d_cl<input_t, layer8_t, config8>(image, layer8_out); // zp2d_feat_conv1

    nnet::conv_2d_cl<layer8_t, feat_conv1_result_t, config2>(layer8_out, layer2_out, w2, b2); // feat_conv1

    nnet::relu<feat_conv1_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // feat_conv1_relu

    nnet::zeropad2d_cl<layer3_t, layer9_t, config9>(layer3_out, layer9_out); // zp2d_feat_conv2

    nnet::conv_2d_cl<layer9_t, feat_conv2_result_t, config4>(layer9_out, layer4_out, w4, b4); // feat_conv2

    nnet::relu<feat_conv2_result_t, layer5_t, relu_config5>(layer4_out, layer5_out); // feat_conv2_relu

    nnet::zeropad2d_cl<layer5_t, layer10_t, config10>(layer5_out, layer10_out); // zp2d_feat_conv3

    nnet::conv_2d_cl<layer10_t, feat_conv3_result_t, config6>(layer10_out, layer6_out, w6, b6); // feat_conv3

    nnet::relu<feat_conv3_result_t, result_t, relu_config7>(layer6_out, layer7_out); // feat_conv3_relu

}

