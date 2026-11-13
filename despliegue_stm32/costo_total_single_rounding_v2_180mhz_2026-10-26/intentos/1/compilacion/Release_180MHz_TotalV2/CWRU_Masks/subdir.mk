################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../CWRU_Masks/cwru_masks.c 

C_DEPS += \
./CWRU_Masks/cwru_masks.d 

OBJS += \
./CWRU_Masks/cwru_masks.o 


# Each subdirectory must supply rules for building sources it contributes
CWRU_Masks/%.o CWRU_Masks/%.su CWRU_Masks/%.cyclo: ../CWRU_Masks/%.c CWRU_Masks/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_TotalV2 -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-CWRU_Masks

clean-CWRU_Masks:
	-$(RM) ./CWRU_Masks/cwru_masks.cyclo ./CWRU_Masks/cwru_masks.d ./CWRU_Masks/cwru_masks.o ./CWRU_Masks/cwru_masks.su

.PHONY: clean-CWRU_Masks

