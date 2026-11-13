################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.c 

C_DEPS += \
./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.d 

OBJS += \
./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.o 


# Each subdirectory must supply rules for building sources it contributes
CWRU_TotalSR/%.o CWRU_TotalSR/%.su CWRU_TotalSR/%.cyclo: ../CWRU_TotalSR/%.c CWRU_TotalSR/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_MASK_BENCH=1 -DCWRU_TOTALV2_TEST=1 -DCMSIS_NN_USE_SINGLE_ROUNDING=1 -DTFLITE_SINGLE_ROUNDING=1 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../CWRU_Masks -I../CWRU_Canon8 -I../CWRU_TotalSR -I../TFLM/third_party/cmsis_nn/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-CWRU_TotalSR

clean-CWRU_TotalSR:
	-$(RM) ./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.cyclo ./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.d ./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.o ./CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.su

.PHONY: clean-CWRU_TotalSR

