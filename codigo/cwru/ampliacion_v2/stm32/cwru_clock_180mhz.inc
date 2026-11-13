/* Arranque opcional para STM32F446RE, VDD 2.7--3.6 V.
 * Solo se incluye con CWRU_TARGET_HCLK_HZ=180000000U.
 * HSI 16 MHz / 16 * 360 / 2 = 180 MHz nominales.
 * RM0390, sec. 3.4.1, y DS10693, condiciones de operacion.
 * No modifica extraccion, ventanas ni frecuencia de muestreo del dataset.
 */
static volatile uint32_t g_cwru_clock_stage;

static void CWRU_SystemClock180_Config(void)
{
    RCC_OscInitTypeDef oscillator = {0};
    RCC_ClkInitTypeDef clocks = {0};

    g_cwru_clock_stage = 1U;
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    oscillator.HSIState = RCC_HSI_ON;
    oscillator.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    oscillator.PLL.PLLState = RCC_PLL_ON;
    oscillator.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    oscillator.PLL.PLLM = 16U;
    oscillator.PLL.PLLN = 360U;
    oscillator.PLL.PLLP = RCC_PLLP_DIV2;
    oscillator.PLL.PLLQ = 8U; /* 45 MHz: no se habilitan USB/SDIO/RNG. */
    oscillator.PLL.PLLR = 6U; /* No se selecciona PLLR como SYSCLK. */

    g_cwru_clock_stage = 2U;
    if (HAL_RCC_OscConfig(&oscillator) != HAL_OK)
    {
        Error_Handler();
    }

    /* Todavia se ejecuta desde HSI; GPIO y USART no se habilitaron. */
    g_cwru_clock_stage = 3U;
    if (HAL_PWREx_EnableOverDrive() != HAL_OK)
    {
        Error_Handler();
    }

    clocks.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                       RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clocks.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clocks.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clocks.APB1CLKDivider = RCC_HCLK_DIV4;
    clocks.APB2CLKDivider = RCC_HCLK_DIV2;

    /* HAL ajusta Flash antes del cambio y recalcula SystemCoreClock/SysTick. */
    g_cwru_clock_stage = 4U;
    if (HAL_RCC_ClockConfig(&clocks, FLASH_LATENCY_5) != HAL_OK)
    {
        Error_Handler();
    }

    g_cwru_clock_stage = 5U;
    if (HAL_RCC_GetSysClockFreq() != 180000000U ||
        HAL_RCC_GetHCLKFreq() != 180000000U ||
        HAL_RCC_GetPCLK1Freq() != 45000000U ||
        HAL_RCC_GetPCLK2Freq() != 90000000U ||
        SystemCoreClock != 180000000U ||
        __HAL_FLASH_GET_LATENCY() != FLASH_LATENCY_5 ||
        !__HAL_PWR_GET_FLAG(PWR_FLAG_ODRDY) ||
        !__HAL_PWR_GET_FLAG(PWR_FLAG_ODSWRDY))
    {
        Error_Handler();
    }
    g_cwru_clock_stage = 100U;
}
