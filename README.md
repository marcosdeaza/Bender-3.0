# Bender 3.0
<p align="center">
  <img src="https://img.icons8.com/doodle/512/futurama-bender.png" width="150" alt="Bender"><br>
  <sub align="center">Este texto estará centrado y más chiquito debajo de Bender</sub>
</p>

Bot de Discord con **conversación por IA** y un **asistente de voz en tiempo real** pensado para un canal de voz: escucha la llamada, detecta cuándo le hablan a él y responde hablando, en español y de forma natural.

Más allá del chat, integra un sistema completo de comunidad: canales de voz temporales por usuario, control de acceso por *keys*, reputación/XP, música, clips de voz y un puente opcional con WhatsApp.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white">
  <img alt="discord.py" src="https://img.shields.io/badge/discord.py-2.7-5865F2?logo=discord&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## ✨ Características

- **🎙️ Asistente de voz en tiempo real.** Se une a un canal de voz, transcribe lo que se dice y responde por voz. Pensado para funcionar con **varias personas hablando a la vez** sin saturarse.
- **🧠 Detección de *wake word* por contexto.** En español *«Bender»* y *«vender»* suenan idénticos. En lugar de fiarse del sonido, un **clasificador vocativo** decide si te diriges al bot (*«Bender, pon música»*) o solo estás usando el verbo (*«voy a vender la casa»*) según la posición, la gramática y los pronombres. (Ver [Pipeline de voz](#-pipeline-de-voz).)
- **💬 Conversación con IA.** Respuestas con personalidad propia (configurable), memoria de conversación y **búsqueda web** automática cuando la pregunta lo requiere.
- **🔊 Canales de voz temporales.** Cada usuario genera su propio canal al entrar en el «creador»; se autodestruye al quedar vacío. Con panel de control (modos público / fantasma / cristal, kick, renombrar, transferir propiedad).
- **🔑 Acceso por *keys* + reputación.** Sistema de invitaciones con clave, roles, anti-spam y XP/rangos por actividad.
- **🎵 Música y clips.** Reproducción de audio en el canal de voz (con *ducking* música+voz) y clips de los últimos segundos de la llamada.
- **📱 Puente con WhatsApp (opcional).** El mismo bot puede atender un grupo de WhatsApp e identificar a cada persona de forma coherente entre plataformas.

---

## 🎛️ Pipeline de voz

El reto principal es la latencia: el servidor de producción es una CPU compartida donde *Whisper* tarda entre 4 y 20 s por frase. La solución es un pipeline en cascada que prioriza reaccionar **siempre** y rápido:

```
 Audio del canal (48 kHz)
        │
        ▼
 Buffer por usuario  ──►  detección de fin de frase (~0.4 s de silencio)
        │
        ▼
 ETAPA 1 · Vosk sobre los primeros 3 s        ← barato; decide si es candidato
        │   ├─ Clasificador vocativo (¿me llaman o es el verbo "vender"?)
        │   └─ Segunda pasada: gramática restringida que "caza" el nombre
        ▼
 ETAPA 2 · Vosk sobre la frase completa        ← solo para candidatos
        │   └─ Respaldo Whisper para el contenido si Vosk se queda corto
        ▼
 LLM (vía OpenRouter)  ──►  Piper TTS  ──►  reproducción en la llamada
```

Decisiones de diseño destacadas:

- **[Vosk](https://alphacephei.com/vosk/) como motor principal** (modelo grande español), con *faster-whisper* solo como respaldo de contenido. Vosk transcribe en ~0.4 s lo que Whisper tarda segundos.
- **Decisión en dos etapas:** el 95 % del audio es charla ambiente, así que se decide con los primeros 3 s antes de gastar en transcribir la frase entera. Evita que la cola se atasque con varias personas hablando.
- **Descartado de audio rancio:** si una frase espera demasiado en la cola, se tira en vez de responder a algo de hace 20 s.
- **Anti-repetición** y ventana de conversación para no contestar varias veces a lo mismo ni reaccionar a interjecciones sueltas.
- **TTS con [Piper](https://github.com/rhasspy/piper)** (voz local en español) mezclado con la música mediante *ducking*.

---

## 🧰 Stack

| Área | Tecnología |
|------|-----------|
| Bot / gateway | [discord.py](https://github.com/Rapptz/discord.py) + [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv) |
| STT | [Vosk](https://alphacephei.com/vosk/) (principal) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (respaldo) |
| TTS | [Piper](https://github.com/rhasspy/piper) |
| LLM | [OpenRouter](https://openrouter.ai/) (API compatible con OpenAI) |
| Web / WhatsApp bridge | aiohttp |

---

## 🚀 Puesta en marcha

### 1. Requisitos
- Python 3.11+
- `ffmpeg` instalado y en el `PATH` (necesario para el audio de Discord)

### 2. Instalación
```bash
git clone https://github.com/MARKITOS-E/Bender-3.0.git
cd Bender-3.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuración
```bash
cp .env.example .env
```
Edita el `.env` con tu token de Discord, tu clave de OpenRouter y los IDs de tu servidor (clic derecho → *Copiar ID* con el Modo Desarrollador de Discord activado). Todos los valores están documentados en [`.env.example`](.env.example).

### 4. Modelos de voz (para el asistente de voz)
No se incluyen en el repo por su tamaño. Descárgalos aparte:

- **Vosk (español):** descarga un modelo de https://alphacephei.com/vosk/models y descomprímelo. El código busca `vosk-model-es-0.42` y, como alternativa, `vosk-model-es`. *(Para el modelo grande puedes borrar las carpetas `rescore/` y `rnnlm/` para reducir el uso de RAM a la mitad sin apenas perder precisión.)*
- **Piper (voz española):** coloca un modelo `.onnx` en `voice_models/` y ajusta `PIPER_PATH` si cambias la ruta.

> El bot arranca y funciona en modo solo-texto aunque no estén los modelos de voz.

### 5. Ejecutar
```bash
python bot.py
```

---

## 🗂️ Estructura

```
Bender-3.0/
├── bot.py              # Todo el bot (gateway, IA, voz, comunidad, WhatsApp)
├── requirements.txt
├── .env.example        # Plantilla de configuración
├── .gitignore
└── README.md
```

El estado en runtime (usuarios, canales, reputación, vault…) se guarda en `bender_data.json`, que se crea solo en el primer arranque y **no** se versiona.

---

## 📝 Notas

- **Personalidad configurable.** Bender tiene una personalidad deliberadamente satírica e irreverente, definida en el *system prompt* (`SERVER_CONTEXT` dentro de `bot.py`). Es solo texto: ajústalo al tono que quieras para tu comunidad.
- **IDs específicos de servidor.** Los roles de color y emojis personalizados (`COLOR_ROLES` / `ROLE_NAMES`) están atados a un servidor concreto; cámbialos por los tuyos o ignóralos (degradan con elegancia si no existen).
- Proyecto personal para un servidor privado de amigos; se publica como muestra de la arquitectura.
