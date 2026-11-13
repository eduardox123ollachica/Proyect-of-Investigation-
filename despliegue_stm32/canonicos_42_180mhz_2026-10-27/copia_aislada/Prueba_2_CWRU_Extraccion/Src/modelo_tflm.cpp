#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" int verificar_modelo_tflm(void)
{
    const tflite::Model* model =
        tflite::GetModel(g_model);

    if (model == nullptr)
    {
        return -1;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        return -2;
    }

    return 0;
}
