################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../CWRU_Canon8/cwru_total_catalog.cpp \
../CWRU_Canon8/cwru_total_nn.cpp 

CC_SRCS += \
../CWRU_Canon8/cwru_total_debug_log.cc 

C_SRCS += \
../CWRU_Canon8/cwru_total.c 

C_DEPS += \
./CWRU_Canon8/cwru_total.d 

CC_DEPS += \
./CWRU_Canon8/cwru_total_debug_log.d 

OBJS += \
./CWRU_Canon8/cwru_total.o \
./CWRU_Canon8/cwru_total_catalog.o \
./CWRU_Canon8/cwru_total_debug_log.o \
./CWRU_Canon8/cwru_total_nn.o 

CPP_DEPS += \
./CWRU_Canon8/cwru_total_catalog.d \
./CWRU_Canon8/cwru_total_nn.d 


# Each subdirectory must supply rules for building sources it contributes
CWRU_Canon8/%.o CWRU_Canon8/%.su CWRU_Canon8/%.cyclo: ../CWRU_Canon8/%.c CWRU_Canon8/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_Canon8 -I../CWRU_TotalSR -I../TFLM/third_party/cmsis_nn/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
CWRU_Canon8/%.o CWRU_Canon8/%.su CWRU_Canon8/%.cyclo: ../CWRU_Canon8/%.cpp CWRU_Canon8/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../Middlewares/CMSIS-DSP/Include -I../TFLM/third_party/cmsis_nn -I../TFLM/third_party/ruy -I../TFLM/third_party/gemmlowp -I../TFLM/third_party/flatbuffers/include -I../TFLM -I../TFLM/third_party/cmsis/CMSIS/Core/Include -I../TFLM/third_party/cmsis_nn/Include -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_Canon8 -I../CWRU_TotalSR -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -std=gnu++17 -fno-threadsafe-statics -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
CWRU_Canon8/%.o CWRU_Canon8/%.su CWRU_Canon8/%.cyclo: ../CWRU_Canon8/%.cc CWRU_Canon8/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../Middlewares/CMSIS-DSP/Include -I../TFLM/third_party/cmsis_nn -I../TFLM/third_party/ruy -I../TFLM/third_party/gemmlowp -I../TFLM/third_party/flatbuffers/include -I../TFLM -I../TFLM/third_party/cmsis/CMSIS/Core/Include -I../TFLM/third_party/cmsis_nn/Include -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_Canon8 -I../CWRU_TotalSR -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -std=gnu++17 -fno-threadsafe-statics -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-CWRU_Canon8

clean-CWRU_Canon8:
	-$(RM) ./CWRU_Canon8/cwru_total.cyclo ./CWRU_Canon8/cwru_total.d ./CWRU_Canon8/cwru_total.o ./CWRU_Canon8/cwru_total.su ./CWRU_Canon8/cwru_total_catalog.cyclo ./CWRU_Canon8/cwru_total_catalog.d ./CWRU_Canon8/cwru_total_catalog.o ./CWRU_Canon8/cwru_total_catalog.su ./CWRU_Canon8/cwru_total_debug_log.cyclo ./CWRU_Canon8/cwru_total_debug_log.d ./CWRU_Canon8/cwru_total_debug_log.o ./CWRU_Canon8/cwru_total_debug_log.su ./CWRU_Canon8/cwru_total_nn.cyclo ./CWRU_Canon8/cwru_total_nn.d ./CWRU_Canon8/cwru_total_nn.o ./CWRU_Canon8/cwru_total_nn.su

.PHONY: clean-CWRU_Canon8

