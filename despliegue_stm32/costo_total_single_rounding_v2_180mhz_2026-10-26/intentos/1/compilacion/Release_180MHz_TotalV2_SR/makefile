################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Startup/subdir.mk
-include Src/subdir.mk
-include Middlewares/CMSIS-DSP/Source/TransformFunctions/subdir.mk
-include Middlewares/CMSIS-DSP/Source/ComplexMathFunctions/subdir.mk
-include Middlewares/CMSIS-DSP/Source/CommonTables/subdir.mk
-include Drivers/STM32F4xx_HAL_Driver/Src/subdir.mk
-include CWRU_TotalV2/subdir.mk
-include CWRU_TotalSR/subdir.mk
-include CWRU_Masks/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(CCM_DEPS)),)
-include $(CCM_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(CXXM_DEPS)),)
-include $(CXXM_DEPS)
endif
ifneq ($(strip $(C++M_DEPS)),)
-include $(C++M_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := Prueba_2_CWRU_Extraccion
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
Prueba_2_CWRU_Extraccion.elf \

MAP_FILES += \
Prueba_2_CWRU_Extraccion.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
Prueba_2_CWRU_Extraccion.list \


# All Target
all: main-build

# Main-build Target
main-build: Prueba_2_CWRU_Extraccion.elf secondary-outputs

# Tool invocations
Prueba_2_CWRU_Extraccion.elf Prueba_2_CWRU_Extraccion.map: $(OBJS) $(USER_OBJS) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-g++ -o "Prueba_2_CWRU_Extraccion.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m4 -T"C:\Users\LENOVO\Desktop\Proyecto Investigación\despliegue_stm32\costo_total_single_rounding_v2_180mhz_2026-10-26\copia_aislada\Prueba_2_CWRU_Extraccion\STM32F446RETX_FLASH.ld" -Wl,-Map="Prueba_2_CWRU_Extraccion.map" -Wl,--gc-sections -static -L../Lib -u _printf_float --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

Prueba_2_CWRU_Extraccion.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "Prueba_2_CWRU_Extraccion.list"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) Prueba_2_CWRU_Extraccion.elf Prueba_2_CWRU_Extraccion.list Prueba_2_CWRU_Extraccion.map default.size.stdout
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified

-include ../makefile.targets
