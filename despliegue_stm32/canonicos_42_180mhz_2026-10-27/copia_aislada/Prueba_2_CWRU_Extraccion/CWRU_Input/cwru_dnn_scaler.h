#ifndef CWRU_DNN_SCALER_H_
#define CWRU_DNN_SCALER_H_

#define CWRU_DNN_NUM_FEATURES 14

extern const float cwru_dnn_scaler_mean[CWRU_DNN_NUM_FEATURES];
extern const float cwru_dnn_scaler_scale[CWRU_DNN_NUM_FEATURES];
extern const float cwru_dnn_input_quant_scale;
extern const int cwru_dnn_input_quant_zero_point;
extern const float cwru_dnn_output_quant_scale;
extern const int cwru_dnn_output_quant_zero_point;

#endif  // CWRU_DNN_SCALER_H_
