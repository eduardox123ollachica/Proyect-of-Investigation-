################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.c 

C_DEPS += \
./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.d 

OBJS += \
./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.o 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-DSP/Source/TransformFunctions/%.o Middlewares/CMSIS-DSP/Source/TransformFunctions/%.su Middlewares/CMSIS-DSP/Source/TransformFunctions/%.cyclo: ../Middlewares/CMSIS-DSP/Source/TransformFunctions/%.c Middlewares/CMSIS-DSP/Source/TransformFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-TransformFunctions

clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-TransformFunctions:
	-$(RM) ./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.cyclo ./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.d ./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.o ./Middlewares/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-TransformFunctions

