#include "cwru_dnn_scaler.h"

const float cwru_dnn_scaler_mean[CWRU_DNN_NUM_FEATURES] = {
    4.824706614e-01f,
    -5.008059740e-02f,
    5.405549049e+00f,
    2.873501750e+06f,
    2.646982910e+03f,
    4.038933411e+02f,
    6.357210279e-01f,
    2.345085740e-01f,
    -4.962733015e-03f,
    2.528882504e+00f,
    6.202101875e+05f,
    2.557715088e+03f,
    1.854801941e+02f,
    6.703225970e-01f,
};

const float cwru_dnn_scaler_scale[CWRU_DNN_NUM_FEATURES] = {
    3.438723683e-01f,
    1.495509148e-01f,
    6.646598816e+00f,
    3.628526750e+06f,
    8.722544556e+02f,
    3.827723389e+02f,
    6.036820263e-02f,
    1.663670242e-01f,
    1.524247527e-01f,
    5.098065376e+00f,
    9.677476875e+05f,
    1.370485596e+03f,
    1.386295166e+02f,
    8.481730521e-02f,
};

const float cwru_dnn_input_quant_scale = 3.369183093e-02f;
const int cwru_dnn_input_quant_zero_point = -9;
const float cwru_dnn_output_quant_scale = 3.906250000e-03f;
const int cwru_dnn_output_quant_zero_point = -128;
