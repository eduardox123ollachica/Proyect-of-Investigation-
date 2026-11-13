################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.c 

C_DEPS += \
./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.d 

OBJS += \
./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.o 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-DSP/Source/CommonTables/%.o Middlewares/CMSIS-DSP/Source/CommonTables/%.su Middlewares/CMSIS-DSP/Source/CommonTables/%.cyclo: ../Middlewares/CMSIS-DSP/Source/CommonTables/%.c Middlewares/CMSIS-DSP/Source/CommonTables/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-CommonTables

clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-CommonTables:
	-$(RM) ./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.cyclo ./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.d ./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.o ./Middlewares/CMSIS-DSP/Source/CommonTables/CommonTables.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-DSP-2f-Source-2f-CommonTables

