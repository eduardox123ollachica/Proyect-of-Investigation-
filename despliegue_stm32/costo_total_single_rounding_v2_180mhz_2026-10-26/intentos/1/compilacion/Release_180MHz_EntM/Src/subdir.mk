################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Src/cmd_cwru_feat.c \
../Src/cwru_feature_extraction.c \
../Src/main.c \
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
./Src/cmd_cwru_feat.d \
./Src/cwru_feature_extraction.d \
./Src/main.d \
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

OBJS += \
./Src/cmd_cwru_feat.o \
./Src/cwru_feature_extraction.o \
./Src/main.o \
./Src/stm32f4xx_hal.o \
./Src/stm32f4xx_hal_cortex.o \
./Src/stm32f4xx_hal_gpio.o \
./Src/stm32f4xx_hal_rcc.o \
./Src/stm32f4xx_hal_rcc_ex.o \
./Src/stm32f4xx_hal_uart.o \
./Src/syscalls.o \
./Src/sysmem.o \
./Src/system_stm32f4xx.o \
./Src/uart_commands.o 


# Each subdirectory must supply rules for building sources it contributes
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F446xx -DSTM32 -DSTM32F4 -DSTM32F446RETx -DNUCLEO_F446RE -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -c -I../Drivers/CMSIS/Include -I../Middlewares/CMSIS-DSP/Include -I../Middlewares/CMSIS-DSP/PrivateInclude -I../TFLM -I../Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/STM32F4xx_HAL_Driver/Inc -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Src

clean-Src:
	-$(RM) ./Src/cmd_cwru_feat.cyclo ./Src/cmd_cwru_feat.d ./Src/cmd_cwru_feat.o ./Src/cmd_cwru_feat.su ./Src/cwru_feature_extraction.cyclo ./Src/cwru_feature_extraction.d ./Src/cwru_feature_extraction.o ./Src/cwru_feature_extraction.su ./Src/main.cyclo ./Src/main.d ./Src/main.o ./Src/main.su ./Src/stm32f4xx_hal.cyclo ./Src/stm32f4xx_hal.d ./Src/stm32f4xx_hal.o ./Src/stm32f4xx_hal.su ./Src/stm32f4xx_hal_cortex.cyclo ./Src/stm32f4xx_hal_cortex.d ./Src/stm32f4xx_hal_cortex.o ./Src/stm32f4xx_hal_cortex.su ./Src/stm32f4xx_hal_gpio.cyclo ./Src/stm32f4xx_hal_gpio.d ./Src/stm32f4xx_hal_gpio.o ./Src/stm32f4xx_hal_gpio.su ./Src/stm32f4xx_hal_rcc.cyclo ./Src/stm32f4xx_hal_rcc.d ./Src/stm32f4xx_hal_rcc.o ./Src/stm32f4xx_hal_rcc.su ./Src/stm32f4xx_hal_rcc_ex.cyclo ./Src/stm32f4xx_hal_rcc_ex.d ./Src/stm32f4xx_hal_rcc_ex.o ./Src/stm32f4xx_hal_rcc_ex.su ./Src/stm32f4xx_hal_uart.cyclo ./Src/stm32f4xx_hal_uart.d ./Src/stm32f4xx_hal_uart.o ./Src/stm32f4xx_hal_uart.su ./Src/syscalls.cyclo ./Src/syscalls.d ./Src/syscalls.o ./Src/syscalls.su ./Src/sysmem.cyclo ./Src/sysmem.d ./Src/sysmem.o ./Src/sysmem.su ./Src/system_stm32f4xx.cyclo ./Src/system_stm32f4xx.d ./Src/system_stm32f4xx.o ./Src/system_stm32f4xx.su ./Src/uart_commands.cyclo ./Src/uart_commands.d ./Src/uart_commands.o ./Src/uart_commands.su

.PHONY: clean-Src

