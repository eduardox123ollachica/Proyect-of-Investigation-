################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Src/feature_extraction.cpp \
../Src/kmeans_anomaly.cpp \
../Src/kmeans_params.cpp \
../Src/modelo_tflm.cpp \
../Src/test_tflm.cpp \
../Src/tinyml_inference.cpp 

CC_SRCS += \
../Src/debug_log.cc \
../Src/micro_log.cc \
../Src/model_data.cc 

C_SRCS += \
../Src/cmd_benchall.c \
../Src/cmd_benchinfo.c \
../Src/cmd_benchraw.c \
../Src/cmd_info.c \
../Src/cmd_kmeansraw.c \
../Src/cmd_modelo.c \
../Src/cmd_pred.c \
../Src/cmd_quant.c \
../Src/cmd_rawall.c \
../Src/cmd_rawfeat.c \
../Src/cmd_rawpred.c \
../Src/cmd_stackbench.c \
../Src/cmd_tinyml.c \
../Src/main.c \
../Src/stack_monitor.c \
../Src/stm32f4xx_hal.c \
../Src/stm32f4xx_hal_cortex.c \
../Src/stm32f4xx_hal_gpio.c \
../Src/stm32f4xx_hal_rcc.c \
../Src/stm32f4xx_hal_rcc_ex.c \
../Src/stm32f4xx_hal_uart.c \
../Src/syscalls.c \
../Src/sysmem.c \
../Src/system_stm32f4xx.c \
../Src/uart_commands.c 

C_DEPS += \
./Src/cmd_benchall.d \
./Src/cmd_benchinfo.d \
./Src/cmd_benchraw.d \
./Src/cmd_info.d \
./Src/cmd_kmeansraw.d \
./Src/cmd_modelo.d \
./Src/cmd_pred.d \
./Src/cmd_quant.d \
./Src/cmd_rawall.d \
./Src/cmd_rawfeat.d \
./Src/cmd_rawpred.d \
./Src/cmd_stackbench.d \
./Src/cmd_tinyml.d \
./Src/main.d \
./Src/stack_monitor.d \
./Src/stm32f4xx_hal.d \
./Src/stm32f4xx_hal_cortex.d \
./Src/stm32f4xx_hal_gpio.d \
./Src/stm32f4xx_hal_rcc.d \
./Src/stm32f4xx_hal_rcc_ex.d \
./Src/stm32f4xx_hal_uart.d \
./Src/syscalls.d \
./Src/sysmem.d \
./Src/system_stm32f4xx.d \
./Src/uart_commands.d 

CC_DEPS += \
./Src/debug_log.d \
./Src/micro_log.d \
./Src/model_data.d 

OBJS += \
./Src/cmd_benchall.o \
./Src/cmd_benchinfo.o \
./Src/cmd_benchraw.o \
./Src/cmd_info.o \
./Src/cmd_kmeansraw.o \
./Src/cmd_modelo.o \
./Src/cmd_pred.o \
./Src/cmd_quant.o \
./Src/cmd_rawall.o \
./Src/cmd_rawfeat.o \
./Src/cmd_rawpred.o \
./Src/cmd_stackbench.o \
./Src/cmd_tinyml.o \
./Src/debug_log.o \
./Src/feature_extraction.o \
./Src/kmeans_anomaly.o \
./Src/kmeans_params.o \
./Src/main.o \
./Src/micro_log.o \
./Src/model_data.o \
./Src/modelo_tflm.o \
./Src/stack_monitor.o \
./Src/stm32f4xx_hal.o \
./Src/stm32f4xx_hal_cortex.o \
./Src/stm32f4xx_hal_gpio.o \
./Src/stm32f4xx_hal_rcc.o \
./Src/stm32f4xx_hal_rcc_ex.o \
./Src/stm32f4xx_hal_uart.o \
./Src/syscalls.o \
./Src/sysmem.o \
./Src/system_stm32f4xx.o \
./Src/test_tflm.o \
./Src/tinyml_inference.o \
./Src/uart_commands.o 

CPP_DEPS += \
./Src/feature_extraction.d \
./Src/kmeans_anomaly.d \
./Src/kmeans_params.d \
./Src/modelo_tflm.d \
./Src/test_tflm.d \
./Src/tinyml_inference.d 


# Each subdirectory must supply rules for building sources it contributes
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.cc Src/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../Middlewares/CMSIS-DSP/Include -I../TFLM/third_party/cmsis_nn -I../TFLM/third_party/ruy -I../TFLM/third_party/gemmlowp -I../TFLM/third_party/flatbuffers/include -I../TFLM -I../TFLM/third_party/cmsis/CMSIS/Core/Include -I../TFLM/third_party/cmsis_nn/Include -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.cpp Src/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../Middlewares/CMSIS-DSP/Include -I../TFLM/third_party/cmsis_nn -I../TFLM/third_party/ruy -I../TFLM/third_party/gemmlowp -I../TFLM/third_party/flatbuffers/include -I../TFLM -I../TFLM/third_party/cmsis/CMSIS/Core/Include -I../TFLM/third_party/cmsis_nn/Include -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Src

clean-Src:
	-$(RM) ./Src/cmd_benchall.cyclo ./Src/cmd_benchall.d ./Src/cmd_benchall.o ./Src/cmd_benchall.su ./Src/cmd_benchinfo.cyclo ./Src/cmd_benchinfo.d ./Src/cmd_benchinfo.o ./Src/cmd_benchinfo.su ./Src/cmd_benchraw.cyclo ./Src/cmd_benchraw.d ./Src/cmd_benchraw.o ./Src/cmd_benchraw.su ./Src/cmd_info.cyclo ./Src/cmd_info.d ./Src/cmd_info.o ./Src/cmd_info.su ./Src/cmd_kmeansraw.cyclo ./Src/cmd_kmeansraw.d ./Src/cmd_kmeansraw.o ./Src/cmd_kmeansraw.su ./Src/cmd_modelo.cyclo ./Src/cmd_modelo.d ./Src/cmd_modelo.o ./Src/cmd_modelo.su ./Src/cmd_pred.cyclo ./Src/cmd_pred.d ./Src/cmd_pred.o ./Src/cmd_pred.su ./Src/cmd_quant.cyclo ./Src/cmd_quant.d ./Src/cmd_quant.o ./Src/cmd_quant.su ./Src/cmd_rawall.cyclo ./Src/cmd_rawall.d ./Src/cmd_rawall.o ./Src/cmd_rawall.su ./Src/cmd_rawfeat.cyclo ./Src/cmd_rawfeat.d ./Src/cmd_rawfeat.o ./Src/cmd_rawfeat.su ./Src/cmd_rawpred.cyclo ./Src/cmd_rawpred.d ./Src/cmd_rawpred.o ./Src/cmd_rawpred.su ./Src/cmd_stackbench.cyclo ./Src/cmd_stackbench.d ./Src/cmd_stackbench.o ./Src/cmd_stackbench.su ./Src/cmd_tinyml.cyclo ./Src/cmd_tinyml.d ./Src/cmd_tinyml.o ./Src/cmd_tinyml.su ./Src/debug_log.cyclo ./Src/debug_log.d ./Src/debug_log.o ./Src/debug_log.su ./Src/feature_extraction.cyclo ./Src/feature_extraction.d ./Src/feature_extraction.o ./Src/feature_extraction.su ./Src/kmeans_anomaly.cyclo ./Src/kmeans_anomaly.d ./Src/kmeans_anomaly.o ./Src/kmeans_anomaly.su ./Src/kmeans_params.cyclo ./Src/kmeans_params.d ./Src/kmeans_params.o ./Src/kmeans_params.su ./Src/main.cyclo ./Src/main.d ./Src/main.o ./Src/main.su ./Src/micro_log.cyclo ./Src/micro_log.d ./Src/micro_log.o ./Src/micro_log.su ./Src/model_data.cyclo ./Src/model_data.d ./Src/model_data.o ./Src/model_data.su ./Src/modelo_tflm.cyclo ./Src/modelo_tflm.d ./Src/modelo_tflm.o ./Src/modelo_tflm.su ./Src/stack_monitor.cyclo ./Src/stack_monitor.d ./Src/stack_monitor.o ./Src/stack_monitor.su ./Src/stm32f4xx_hal.cyclo ./Src/stm32f4xx_hal.d ./Src/stm32f4xx_hal.o ./Src/stm32f4xx_hal.su ./Src/stm32f4xx_hal_cortex.cyclo ./Src/stm32f4xx_hal_cortex.d ./Src/stm32f4xx_hal_cortex.o ./Src/stm32f4xx_hal_cortex.su ./Src/stm32f4xx_hal_gpio.cyclo ./Src/stm32f4xx_hal_gpio.d ./Src/stm32f4xx_hal_gpio.o ./Src/stm32f4xx_hal_gpio.su ./Src/stm32f4xx_hal_rcc.cyclo ./Src/stm32f4xx_hal_rcc.d ./Src/stm32f4xx_hal_rcc.o ./Src/stm32f4xx_hal_rcc.su ./Src/stm32f4xx_hal_rcc_ex.cyclo ./Src/stm32f4xx_hal_rcc_ex.d ./Src/stm32f4xx_hal_rcc_ex.o ./Src/stm32f4xx_hal_rcc_ex.su ./Src/stm32f4xx_hal_uart.cyclo ./Src/stm32f4xx_hal_uart.d ./Src/stm32f4xx_hal_uart.o ./Src/stm32f4xx_hal_uart.su ./Src/syscalls.cyclo ./Src/syscalls.d ./Src/syscalls.o ./Src/syscalls.su ./Src/sysmem.cyclo ./Src/sysmem.d ./Src/sysmem.o ./Src/sysmem.su ./Src/system_stm32f4xx.cyclo ./Src/system_stm32f4xx.d ./Src/system_stm32f4xx.o ./Src/system_stm32f4xx.su ./Src/test_tflm.cyclo ./Src/test_tflm.d ./Src/test_tflm.o ./Src/test_tflm.su ./Src/tinyml_inference.cyclo ./Src/tinyml_inference.d ./Src/tinyml_inference.o ./Src/tinyml_inference.su ./Src/uart_commands.cyclo ./Src/uart_commands.d ./Src/uart_commands.o ./Src/uart_commands.su

.PHONY: clean-Src

