################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.c 

C_DEPS += \
./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.d 

OBJS += \
./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.o 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/%.o Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/%.su Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/%.cyclo: ../Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/%.c Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_Canon8 -I../CWRU_TotalSR -I../TFLM/third_party/cmsis_nn/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-ComplexMathFunctions

clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-ComplexMathFunctions:
	-$(RM) ./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.cyclo ./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.d ./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.o ./Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-ComplexMathFunctions

