Las 30 pruebas finalizaron con retorno 0 y se guardaron en
`PRUEBAS_CLIENTE_SR.json`. Al imprimir después ese informe en la consola
CP1252, el wrapper encontró un carácter de sustitución procedente de la
decodificación del log y terminó con `UnicodeEncodeError`. La lectura
posterior con salida ASCII confirmó `PASS`, retorno 0 y `Ran 30 tests ... OK`.
No se repitieron ni se modificaron las pruebas por ese error de presentación.
