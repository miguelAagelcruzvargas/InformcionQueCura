Para Windows:

Instalación como Servicio: MongoDB puede ejecutarse como un servicio en Windows, lo que significa que se
iniciará automáticamente cuando se encienda la computadora.

Pasos:

Abre el Símbolo del sistema como administrador.

Ejecuta el siguiente comando para instalar MongoDB como servicio:

"C:\Program Files\MongoDB\Server\<version>\bin\mongod.exe" --config "C:\Program Files\MongoDB\Server\<version>\bin\mongod.cfg" --install
Asegúrate de cambiar <version> por la versión de MongoDB que tengas instalada.

Para iniciar el servicio inmediatamente:
net start MongoDB
Para asegurarte de que MongoDB se inicie automáticamente al encender la PC:

1. Abre el Administrador de servicios (pulsa Win + R, escribe services.msc y presiona Enter).
2.Busca el servicio de MongoDB.
3.Haz clic derecho sobre él, selecciona "Propiedades" y en "Tipo de inicio", elige "Automático".

OPCION 2 :

Método 1: Usando el Instalador de MongoDB (Automático)
Si instalaste MongoDB usando el instalador oficial de MongoDB para Windows, existe una opción para 
instalarlo automáticamente como un servicio durante el proceso de instalación.

Descarga e instala MongoDB:

Si no lo tienes instalado aún, puedes descargar el instalador desde el sitio oficial de MongoDB.
Durante la instalación, selecciona la opción Install MongoDB as a Service. Esto asegurará que MongoDB 
se configure como un servicio de Windows, que se iniciará automáticamente cada vez que enciendas la computadora.
Verificar el servicio:

Después de la instalación, abre el Administrador de servicios (services.msc) y busca "MongoDB".
Asegúrate de que el tipo de inicio esté configurado como "Automático".

Pasos para configurar MongoDB como un servicio:
Instalación de MongoDB: Durante el proceso de instalación con el instalador MSI de MongoDB, asegúrate de seleccionar la opción "Install MongoDB as a Service".

Ruta del instalador: Descargar el instalador MSI de MongoDB desde el sitio oficial.
En la opción de configuración del servicio, selecciona "Run the service as Network Service user" 
(configuración predeterminada).
Inicio automático del servicio: Después de la instalación, MongoDB se configurará para 
ejecutarse automáticamente cada vez que Windows arranque.
No necesitarás iniciar MongoDB manualmente.
