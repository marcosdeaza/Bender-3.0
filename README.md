# Bender 3.0
<p align="center">
  <img src="https://img.icons8.com/doodle/512/futurama-bender.png" width="150" alt="Bender"><br>
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

- **🎙️ Asistente de voz en tiempo real.** Se une a un canal de voz, transcribe lo que se dice con **Groq Whisper** y responde por voz con **Piper TTS**. Pensado para funcionar con **varias personas hablando a la vez** sin saturarse.
- **🧠 Detección de *wake word* por contexto.** En español *«Bender»* y *«vender»* suenan idénticos. Un **clasificador vocativo** decide si te diriges al bot (*«Bender, pon música»*) o solo estás usando el verbo (*«voy a vender la casa»*) según la posición, la gramática y los pronombres.
- **💬 Conversación con IA.** Respuestas con personalidad propia (configurable), memoria de conversación y **búsqueda web** automática cuando la pregunta lo requiere.
- **🔊 Canales de voz temporales.** Cada usuario genera su propio canal al entrar en el «creador»; se autodestruye al quedar vacío. Con panel de control (modos público / fantasma / cristal, kick, renombrar, transferir propiedad).
- **🔑 Acceso por *keys* + reputación.** Sistema de invitaciones con clave, roles, anti-spam y XP/rangos por actividad.
- **🎵 Música y clips.** Reproducción de audio en el canal de voz (con *ducking* música+voz) y clips de los últimos segundos de la llamada.
- **📱 Puente con WhatsApp (opcional).** El mismo bot puede atender un grupo de WhatsApp, identificar a cada persona, **ver fotos** y responder de forma coherente entre plataformas.
- **🔒 Parches E2E.** Incluye parches para el descifrado DAVE de Discord (sin ellos, ~40% de los paquetes de audio no se descifran) y manejo de paquetes Opus corruptos.

---

## 🎛️ Pipeline de voz

```
 Audio del canal (48 kHz)
        │
        ▼
 Buffer por usuario  ──►  VAD (filtro de energía) + detección de fin de frase (~0.4 s de silencio)
        │
        ▼
 Groq Whisper API (whisper-large-v3)     ← transcripción en la nube, ~0.5-2s
        │   ├─ Filtro de alucinaciones ("¡Gracias!", "¡Suscríbete!" → descartadas)
        │   └─ Clasificador vocativo (¿me llaman o es el verbo "vender"?)
        ▼
 LLM (vía OpenRouter, Gemini 2.5 Flash)  ──►  Piper TTS  ──►  reproducción en la llamada
```

Decisiones de diseño destacadas:

- **Groq Whisper** sustituye a Vosk + faster-whisper. Transcripción de alta calidad con baja latencia vía API.
- **VAD por energía RMS** filtra el silencio antes de enviar a Groq, evitando alucinaciones y ahorrando cuota.
- **Límite diario configurable** (7200s por defecto) con reset automático a medianoche. Cuando se alcanza, el bot avisa que está "cansado" sin mencionar APIs.
- **Anti-repetición** y ventana de conversación para no contestar varias veces a lo mismo.
- **TTS con [Piper](https://github.com/rhasspy/piper)** (voz local en español) mezclado con la música mediante *ducking*.
- **Parche DAVE-decrypt**: corrige la firma de descifrado E2E de discord-ext-voice-recv para que los paquetes se descifren correctamente.

---

## 🧰 Stack

| Área | Tecnología |
|------|-----------|
| Bot / gateway | [discord.py](https://github.com/Rapptz/discord.py) + [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv) |
| STT | [Groq Whisper](https://console.groq.com/) (whisper-large-v3) |
| TTS | [Piper](https://github.com/rhasspy/piper) |
| LLM | [OpenRouter](https://openrouter.ai/) (Gemini 2.5 Flash) |
| E2E | [davey](https://pypi.org/project/davey/) (descifrado DAVE) |
| Web / WhatsApp bridge | aiohttp |

---

## 🚀 Puesta en marcha

### 1. Requisitos
- Python 3.11+
- `ffmpeg` instalado y en el `PATH`

### 2. Instalación
```bash
git clone https://github.com/marcosdeaza/Bender-3.0.git
cd Bender-3.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuración
```bash
cp .env.example .env
```
Edita el `.env` con tus credenciales (Discord, OpenRouter, Groq) y los IDs de tu servidor. Ver [`.env.example`](.env.example) para todos los valores.

### 4. Modelo de voz Piper (TTS)
Descarga un modelo `.onnx` de Piper en español desde [HuggingFace](https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES) y colócalo en `voice_models/bender.onnx` (o ajusta `PIPER_PATH` en el código).

> El bot arranca y funciona en modo solo-texto aunque no haya modelo de voz.

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

---

## 📝 Notas

- **Personalidad configurable.** La personalidad se define en `SERVER_CONTEXT` dentro de `bot.py`. Ajústala al tono de tu comunidad.
- **IDs específicos de servidor.** Los roles de color y emojis (`COLOR_ROLES` / `ROLE_NAMES`) están atados a un servidor concreto; cámbialos por los tuyos.
- **Límite de Groq.** El plan gratuito de Groq tiene un límite diario de segundos de audio. El bot lo gestiona automáticamente y avisa cuando está "cansado" sin revelar que es un límite de API.
- Proyecto personal para un servidor privado de amigos; se publica como muestra de la arquitectura.
