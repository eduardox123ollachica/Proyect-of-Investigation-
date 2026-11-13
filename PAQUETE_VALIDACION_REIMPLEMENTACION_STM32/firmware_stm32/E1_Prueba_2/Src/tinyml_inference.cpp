#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <stdint.h>
#include <stddef.h>
#include <math.h>

namespace
{
    const tflite::Model* model = nullptr;

    tflite::MicroMutableOpResolver<2> resolver;

    constexpr size_t kTensorArenaSize = 16 * 1024;

    alignas(16) uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter* interpreter = nullptr;

    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
}


/*
 * Inicialización de TensorFlow Lite Micro.
 */
extern "C" int TinyML_Init(void)
{
    model = tflite::GetModel(g_model);

    if (model == nullptr)
    {
        return -1;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        return -2;
    }

    if (resolver.AddFullyConnected() != kTfLiteOk)
    {
        return -3;
    }

    if (resolver.AddSoftmax() != kTfLiteOk)
    {
        return -4;
    }

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        return -5;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input == nullptr || output == nullptr)
    {
        return -6;
    }

    if (input->type != kTfLiteInt8)
    {
        return -7;
    }

    if (output->type != kTfLiteInt8)
    {
        return -8;
    }

    if (input->bytes != 21)
    {
        return -9;
    }

    if (output->bytes != 3)
    {
        return -10;
    }

    return 0;
}


/*
 * Funciones de diagnóstico.
 */

extern "C" int TinyML_InputType(void)
{
    if (input == nullptr)
        return -1;

    return (int)input->type;
}

extern "C" int TinyML_OutputType(void)
{
    if (output == nullptr)
        return -1;

    return (int)output->type;
}

extern "C" int TinyML_InputBytes(void)
{
    if (input == nullptr)
        return -1;

    return (int)input->bytes;
}

extern "C" int TinyML_OutputBytes(void)
{
    if (output == nullptr)
        return -1;

    return (int)output->bytes;
}

extern "C" float TinyML_InputScale(void)
{
    if (input == nullptr)
        return 0.0f;

    return input->params.scale;
}

extern "C" int TinyML_InputZeroPoint(void)
{
    if (input == nullptr)
        return 0;

    return input->params.zero_point;
}

extern "C" float TinyML_OutputScale(void)
{
    if (output == nullptr)
        return 0.0f;

    return output->params.scale;
}

extern "C" int TinyML_OutputZeroPoint(void)
{
    if (output == nullptr)
        return 0;

    return output->params.zero_point;
}


/*
 * NUEVO:
 * Ejecuta una inferencia completa.
 */
extern "C" int TinyML_Predict(
    const float features[21],
    int *clase,
    float probs[3]
)
{
    if (interpreter == nullptr ||
        input == nullptr ||
        output == nullptr)
    {
        return -1;
    }

    if (features == nullptr ||
        clase == nullptr ||
        probs == nullptr)
    {
        return -2;
    }

    /*
     * Parámetros de cuantización del tensor de entrada.
     */
    const float input_scale =
        input->params.scale;

    const int input_zero_point =
        input->params.zero_point;

    /*
     * Cuantización:
     *
     * q = round(x / scale) + zero_point
     */
    for (int i = 0; i < 21; i++)
    {
        int32_t q = (int32_t)roundf(
            features[i] / input_scale
        ) + input_zero_point;

        /*
         * Saturación al rango INT8.
         */
        if (q > 127)
            q = 127;

        if (q < -128)
            q = -128;

        input->data.int8[i] =
            (int8_t)q;
    }

    /*
     * Ejecutar red neuronal.
     */
    if (interpreter->Invoke() != kTfLiteOk)
    {
        return -3;
    }

    /*
     * Parámetros de cuantización de la salida.
     */
    const float output_scale =
        output->params.scale;

    const int output_zero_point =
        output->params.zero_point;

    int mejor_indice = 0;
    float mejor_valor = -1.0f;

    /*
     * Desquantizar las 3 salidas.
     *
     * valor_float =
     * (q - zero_point) * scale
     */
    for (int i = 0; i < 3; i++)
    {
        int8_t q =
            output->data.int8[i];

        probs[i] =
            ((int)q - output_zero_point)
            * output_scale;

        if (probs[i] > mejor_valor)
        {
            mejor_valor = probs[i];
            mejor_indice = i;
        }
    }

    *clase = mejor_indice;

    return 0;
}
