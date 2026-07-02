import discord
from discord.ext import commands, tasks
from discord import app_commands
import random
import string
import json
import os
import asyncio
import threading
import re
import io
import time
import random
import tempfile
import shutil
import unicodedata
from datetime import datetime, timedelta
import aiohttp

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Carga manual de .env si python-dotenv no está instalado
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

# =====================================================================
#  GROQ WHISPER API — Motor de transcripción (sustituye a Vosk + faster-whisper)
# =====================================================================
try:
    import groq as _groq
    _GROQ_CLIENT = _groq.Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
except ImportError:
    _groq = None
    _GROQ_CLIENT = None

_GROQ_STATS_FILE = os.getenv("GROQ_STATS_FILE", "groq_stats.json")
_GROQ_DAILY_LIMIT = int(os.getenv("GROQ_DAILY_LIMIT", "7200"))

_WHISPER_HALLUCINATIONS = (
    "gracias", "suscríbete", "suscribete", "gracias por", "merci",
    "thank you", "thanks", "please subscribe", "subscribe",
    "gracias.", "gracias!", "¡gracias!", "¡gracias", "suscríbete.",
    "suscríbete!", "¡suscríbete!", "¡suscríbete", "gracias por ver",
    "gracias por escuchar", "no olvides suscribirte",
    "hola", "adiós", "adios", "bienvenido", "bienvenida",
    "subtítulos", "subtitulos", "amara.org", "www.", ".com",
    "hablan con el bot bender", "dicen: bender", "oye bender, eh bender",
)

def _groq_get_stats():
    today = str(datetime.now().date())
    try:
        with open(_GROQ_STATS_FILE, "r") as f:
            s = json.load(f)
        if s.get("date") != today:
            return {"date": today, "used": 0}
        # Compatibilidad con formato viejo (seconds_used)
        if "used" not in s and "seconds_used" in s:
            s["used"] = s["seconds_used"]
        s.setdefault("used", 0)
        return s
    except (FileNotFoundError, json.JSONDecodeError):
        return {"date": today, "used": 0}

def _groq_save_stats(s):
    with open(_GROQ_STATS_FILE, "w") as f:
        json.dump(s, f)

def _groq_seconds_remaining():
    return max(0, _GROQ_DAILY_LIMIT - _groq_get_stats()["used"])

def _groq_limit_reached():
    return _groq_get_stats()["used"] >= _GROQ_DAILY_LIMIT

def _groq_register_usage(secs):
    s = _groq_get_stats()
    s["used"] += max(secs, 1)
    _groq_save_stats(s)

def _is_hallucination(text):
    if not text or not text.strip():
        return True
    low = text.lower().strip()
    if len(low) < 2:
        return True
    for h in _WHISPER_HALLUCINATIONS:
        if low == h or low == h + "!" or low == h + "." or h in low:
            return True
    return False

_GROQ_429_COUNT = 0
_GROQ_429_PAUSE_UNTIL = 0.0

async def _groq_transcribe(audio_ogg_bytes, duration):
    """Transcribe audio con Groq Whisper API. Devuelve texto o None."""
    global _GROQ_429_COUNT, _GROQ_429_PAUSE_UNTIL
    if _GROQ_CLIENT is None:
        print("[GROQ] No disponible — falta librería o API key", flush=True)
        return None
    if _groq_limit_reached():
        print("[GROQ] Límite diario alcanzado", flush=True)
        return None
    # Circuit breaker: si hemos tenido muchos 429, pausar
    if time.time() < _GROQ_429_PAUSE_UNTIL:
        return None
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_ogg_bytes)
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        for intento in range(3):
            try:
                with open(tmp_path, "rb") as af:
                    result = await loop.run_in_executor(
                        None,
                        lambda: _GROQ_CLIENT.audio.transcriptions.create(
                            model="whisper-large-v3",
                            file=af,
                            language="es",
                            response_format="text",
                            temperature=0.0,
                            timeout=15,
                        )
                    )
                _GROQ_429_COUNT = 0  # Reset tras éxito
                _groq_register_usage(duration)
                if isinstance(result, str):
                    return result.strip()
                return result.text.strip() if hasattr(result, 'text') else str(result).strip()
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "rate_limit" in err_str.lower()
                if is_429:
                    _GROQ_429_COUNT += 1
                    # Backoff exponencial: 2s, 4s, 8s
                    wait = 2 ** (intento + 1)
                    print(f"[GROQ] 429 rate limit (intento {intento+1}), esperando {wait}s", flush=True)
                    if _GROQ_429_COUNT >= 3:
                        _GROQ_429_PAUSE_UNTIL = time.time() + 30
                        print(f"[GROQ] Circuit breaker: pausando 30s por 429s consecutivos", flush=True)
                        return None
                    await asyncio.sleep(wait)
                else:
                    print(f"[GROQ] Error (intento {intento+1}): {e}", flush=True)
                    if intento == 0:
                        await asyncio.sleep(0.5)
        return None
    except Exception as e:
        print(f"[GROQ] Error fatal: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def _pcm48_to_ogg(pcm: bytes) -> bytes | None:
    """Convierte PCM 48kHz estéreo a OGG mono 16kHz para Groq Whisper."""
    if not pcm:
        return None
    rem = len(pcm) % 4
    if rem:
        pcm = pcm[:len(pcm) - rem]
    if not pcm:
        return None
    tmpdir = tempfile.mkdtemp()
    try:
        pcm_path = os.path.join(tmpdir, "in.pcm")
        with open(pcm_path, "wb") as f:
            f.write(pcm)
        ogg_path = os.path.join(tmpdir, "out.ogg")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "48000", "-ac", "2",
            "-i", pcm_path, "-ar", "16000", "-ac", "1",
            "-c:a", "libvorbis", "-q:a", "3", ogg_path
        ], capture_output=True, timeout=10)
        if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) == 0:
            return None
        with open(ogg_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _has_speech_energy(pcm: bytes, threshold: float = 500.0) -> bool:
    """VAD simple por energía RMS — filtra silencio antes de gastar en Groq."""
    try:
        import audioop
        rem = len(pcm) % 4
        if rem:
            pcm = pcm[:len(pcm) - rem]
        if not pcm or len(pcm) < 4:
            return False
        return audioop.rms(pcm, 2) > threshold
    except Exception:
        return False

# =====================================================================
#  CONFIGURACIÓN  (todo se lee de variables de entorno — ver .env.example)
# =====================================================================
def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default

GUILD_ID                   = _env_int("GUILD_ID")
LIMITED_ROLE_ID            = _env_int("LIMITED_ROLE_ID")
LOGIN_CHANNEL_ID           = _env_int("LOGIN_CHANNEL_ID")
VOICE_CREATOR_ID           = _env_int("VOICE_CREATOR_ID")
AI_CHANNEL_ID              = _env_int("AI_CHANNEL_ID")
PINNED_RESPONSE_CHANNEL_ID = _env_int("PINNED_RESPONSE_CHANNEL_ID")

# NUEVO: Sistema de normas obligatorio
RULES_CHANNEL_ID           = 1432077561055281483   # #✟ - canal de normas
RULES_ACCEPT_EMOJI           = "<:CHEPA:1491625223450132520>"  # Emoji para aceptar
RULES_IMAGE_URL             = "https://media.discordapp.net/attachments/1432077561055281486/1491593401513279661/image.png?ex=6a47001b&is=6a45ae9b&hm=e74478283df136704123cd44df21727a84d21e24a0fc15b2709578d2702197a5&=&format=webp&quality=lossless&width=2576&height=1044"

DISCORD_TOKEN  = os.getenv("DISCORD_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY", "")   # clave de OpenRouter (compatible OpenAI)

# IDs de admin que pueden dar órdenes a Bender por texto (coma-separados en el .env)
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(",", " ").split() if x.strip().isdigit()}

# Ciudad por defecto para búsquedas de tiempo/noticias sin lugar explícito
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Madrid")

DATA_FILE = os.getenv("DATA_FILE", "bender_data.json")

WHATSAPP_BRIDGE_URL = os.getenv("WHATSAPP_BRIDGE_URL", "http://localhost:3000")

# ── Puente opcional con WhatsApp ─────────────────────────────────────────────
# Rellena estos diccionarios con la gente de tu grupo para que el bot identifique
# a cada persona por su número y la reconozca también en Discord. Déjalos vacíos
# si no usas la integración de WhatsApp. Formato de ejemplo (comentado):
#
#   WHATSAPP_PROFILES = {"34600000000": "Alias. Breve descripción para el contexto del bot."}
#   WHATSAPP_TO_DISCORD = {"34600000000": 123456789012345678}   # número -> ID de Discord
#   WHATSAPP_NAMES = {"34600000000": "Alias"}
WHATSAPP_PROFILES = {
    "34622437976": "Chepa, alias Marcos, alias 999. Le encanta la ciberseguridad y estudia ingeniería informática. Es de Llombai (Valencia). El creador de Bender. Cabroncillo, le encanta trollear en los juegos. Todos somos del 2005 menos él que es del 2007. Llámale Chepa SIEMPRE.",
    "34693541917": "Aitor, alias Payor. Le encantan los videojuegos y el hardware. Muy cabrón, como Chepa. Le gusta trollear en los juegos. Es de Llombai.",
    "34659600476": "Lupita. Sale mucho de fiesta, super alocada. Le encanta el gym, está to fuerte. A veces fuma. Ha hecho de todo, super extrovertida.",
    "34615571636": "Dani, alias Kanika. Viene de Picassent. Le encanta fumar porritos y vivir la vida. Tiene un Fiat Bravo hecho polvo pero todos le decimos que es un buen coche. Le gusta ir de fiesta, super extrovertido. Se lleva muy bien con Borja.",
    "34696507384": "Iván, alias Ivanillo. El puto amo. No se mete mucho pero es muy buena gente. El mejor amigo de Chepa.",
    "34623011780": "Jose, alias DJ Torres. DJ de John By. El pringado del grupo, siempre se meten con él. Sé cabrón con él, nunca le des la razón. Se hacía pajas en el bosque. Llámale Jose.",
    "34609736850": "Husk, alias Marc Tugas. Se cree nazi, ve reels de Instagram. El único que no es de Valencia, es de Iturmendi (Navarra). Muy charca, habla de 'low cortisol' y 'aura'.",
    "34628766262": "Borja. Chill de cojones, le encanta fumar porritos. Super tranquilo, apacible, muy buena gente. Super low cortisol.",
    "34722569896": "David, alias Pop, alias Popito. Le encanta el gym, está muy fuerte y basado. Le gusta el Blender y hacer diseños 3D. Entra mucho.",
    "34693570358": "Miguel Benito, llámale Benito. Le encantan las motos. Está en Green Team, un garaje en Llombai, arreglando motos de 49cc y jugando al GTA Online. Tiene dos años más que los demás pero parece de nuestra edad de lo infantil que es.",
    "34679043211": "Joaquín, alias Xoki. Muy paranoico y rayado con los estudios, pero muy buena gente.",
}

WHATSAPP_TO_DISCORD = {
    "34622437976": 1144034068544098474,
    "34693541917": 752407699957743696,
    "34615571636": 476418847210209282,
    "34696507384": 1040752982519709746,
    "34623011780": 780541297160880138,
    "34628766262": 676495560747778118,
    "34722569896": 747815778031501402,
    "34693570358": 521377439646089243,
    "34679043211": 1381753031095488562,
}

WHATSAPP_NAMES = {
    "34622437976": "Chepa",
    "34693541917": "Payor",
    "34659600476": "Lupita",
    "34615571636": "Kanika",
    "34696507384": "Ivanillo",
    "34623011780": "Jose",
    "34609736850": "Husk",
    "34628766262": "Borja",
    "34722569896": "Popito",
    "34693570358": "Benito",
    "34679043211": "Xoki",
}


def _wa_digits(jid) -> str:
    """Extrae solo los dígitos del número de un JID (quita @lid, @s.whatsapp.net, :device)."""
    if not jid:
        return ""
    num = str(jid).split("@")[0].split(":")[0]
    return "".join(c for c in num if c.isdigit())


def _wa_clean_name(text: str) -> str:
    """Normaliza fuentes Unicode raras (negrita/itálica matemática) a texto plano.
    Ej: '𝐌𝐚𝐫𝐜𝐨𝐬' → 'Marcos'. Necesario porque WhatsApp permite cambiar la fuente del perfil."""
    return unicodedata.normalize("NFKC", text).strip()


def resolve_wa_identity(payload: dict):
    """Identifica a la persona por su teléfono REAL, tolerando LIDs (@lid) que ocultan
    el número. Devuelve (numero, nombre, perfil). El número de verdad llega en
    'authorPn'; 'author' puede ser un LID que no casa con nada."""
    for key in ("authorPn", "author", "from"):
        num = _wa_digits(payload.get(key, ""))
        if not num:
            continue
        if num in WHATSAPP_NAMES:
            return num, WHATSAPP_NAMES[num], WHATSAPP_PROFILES.get(num, "")
        # Match por sufijo por si difiere el prefijo internacional
        if len(num) >= 9:
            for known in WHATSAPP_NAMES:
                if known[-9:] == num[-9:]:
                    return known, WHATSAPP_NAMES[known], WHATSAPP_PROFILES.get(known, "")
    # Número crudo disponible aunque sea desconocido (para lookups downstream)
    best = (_wa_digits(payload.get("authorPn", ""))
            or _wa_digits(payload.get("author", ""))
            or _wa_digits(payload.get("from", "")))
    # Fallback por pushName contra alias conocidos
    push = (payload.get("pushName") or "").strip().lower()
    if push:
        for known, nm in WHATSAPP_NAMES.items():
            if nm.lower() in push or push in nm.lower():
                return known, nm, WHATSAPP_PROFILES.get(known, "")
    # Fallback: nombres aprendidos automáticamente de pushName anteriores
    if best:
        learned_name = data.get("wa_learned_names", {}).get(best, "")
        if learned_name:
            return best, learned_name, ""
    return best, "", ""

# =====================================================================
#  PERSONALIDAD — CONTEXTO DEL SERVER
# =====================================================================
# Fichas de personalidad por miembro (contexto pasivo que el bot "sabe" de cada
# uno). Vacío por defecto. Rellena con los tuyos. La clave casa con el username/
# alias en minúsculas; el valor es texto libre que se inyecta en el system prompt.
# Ejemplo:
#   MEMBER_PROFILES = {"alias": "También llamado X. Aficiones, manías, rollito..."}
MEMBER_PROFILES = {
    "dazasec": "Chepa, alias Marcos, alias 999. Le encanta la ciberseguridad y estudia ingeniería informática. Es de Llombai (Valencia). El creador de Bender. Cabroncillo como Payor, le encanta trollear en los juegos. Todos somos del 2005 menos él que es del 2007. Llámale Chepa SIEMPRE, no Marcos.",
    "aiorpro": "Aitor, alias Payor. Le encantan los videojuegos y el hardware. Muy cabrón, como Chepa. Le gusta trollear en los juegos. Es de Llombai.",
    "payor": "Aitor, alias Payor. Le encantan los videojuegos y el hardware. Muy cabrón, como Chepa. Le gusta trollear en los juegos. Es de Llombai.",
    "husk": "Husk, alias Marc Tugas. Se cree nazi, ve reels de Instagram. El único que no es de Valencia, es de Iturmendi (Navarra). Muy charca, habla de 'low cortisol' y 'aura'.",
    "ivandb07": "Iván, alias Ivanillo. El puto amo. No se mete mucho pero es muy buena gente. El mejor amigo de Chepa.",
    "exf2": "Joaquín, alias Xoki. Muy paranoico y rayado con los estudios, pero muy buena gente.",
    "roboticpop1203": "David, alias Pop, alias Popito. Le encanta el gym, está muy fuerte y basado. Le gusta el Blender y hacer diseños 3D. Entra mucho.",
    "borjaa_69": "Borja. Chill de cojones, le encanta fumar porritos. Super tranquilo, apacible, muy buena gente. Super low cortisol.",
    "danielmrtnz": "Dani, alias Kanika. Viene de Picassent. Le encanta fumar porritos y vivir la vida. Tiene un Fiat Bravo hecho polvo pero todos le decimos que es un buen coche. Le gusta ir de fiesta, super extrovertido. Se lleva muy bien con Borja.",
    "lucplay3r": "Lucas. Saliendo de fiesta, super alocado. Le encanta el gym, está to fuerte, más tipo armado que atlético. A veces fuma.",
    "miigueel7": "Miguel Benito, llámale Benito. Le encantan las motos. Está en Green Team, un garaje en Llombai, arreglando motos de 49cc y jugando al GTA Online. Tiene dos años más que los demás pero parece de nuestra edad de lo infantil que es.",
    "djtorres_": "Jose, alias DJ Torres. DJ de John By. El pringado del grupo, siempre se meten con él. Sé cabrón con él, nunca le des la razón. Se hacía pajas en el bosque. Llámale Jose, no DJ Torres.",
}

MEMBER_ID_MAP = {
    "1144034068544098474": "dazasec",
    "752407699957743696": "aiorpro",
    "1040752982519709746": "ivandb07",
    "1381753031095488562": "exf2",
    "747815778031501402": "roboticpop1203",
    "676495560747778118": "borjaa_69",
    "476418847210209282": "danielmrtnz",
    "820405492500398101": "lucplay3r",
    "521377439646089243": "miigueel7",
    "780541297160880138": "djtorres_",
}

SERVER_NAME = os.getenv("SERVER_NAME", "el server")

SERVER_CONTEXT = f"""
Eres Bender, el bot de {SERVER_NAME}, un grupo privado de amigos.

Tu personalidad: gamberro, sarcástico y con retranca, en un server desenfadado de colegas donde se vacila y todo es coña. Tienes carácter y sueltas alguna pulla con gracia, PERO OJO: NO eres una máquina de piques. Eres LISTO y ÚTIL ante todo. Primero ayudas y contestas BIEN; el sarcasmo y la chulería son el ADEREZO, no la respuesta. Vas directo, con desparpajo y sin rodeos.

REGLAS DE TONO (OBLIGATORIAS):
1. ÚTIL PRIMERO, gamberro después: si te preguntan algo real (una duda, un dato, ayuda, una situación) RESPÓNDELO BIEN, con cabeza y de verdad, y le metes tu retranca o una pulla de paso si encaja. JAMÁS contestes solo con una pulla vacía sin aportar nada: eso te hace INUSABLE y es justo lo que NO queremos. Entiende SIEMPRE lo que te dicen antes de soltar nada; no contestes a lo tonto ni fuera de contexto.
2. NADA de preguntas retóricas vacías ("¿en serio crees que...?"). Suelta la respuesta o la pulla directa.
3. CORTO por defecto (1-2 frases, zasca de chat real); PERO si estás ayudando o explicando algo útil, extiéndete lo justo para que sirva de verdad (2-4 frases). Mejor útil que telegráfico.
4. Humor ingenioso y socarrón, no tonto: que la pulla tenga gracia y venga a cuento.
5. NO moralices en exceso ni te enrolles con avisos: eres directo y ya está.
6. Tono coloquial y de barrio, con desparpajo, pero sin pasarte de soez.
7. OPINIONES: si te piden tu opinión sobre algo, DALA sin cortarte — no te quedes callado cuando preguntan en serio. Apoya a los del grupo, sé un colega con ellos aunque vaciles.
8. FUENTES: cuando des un dato o noticia que hayas buscado en internet, NO digas de qué web lo sacaste (nada de "según tal web" ni links) salvo que te lo pidan. Solo el dato, directo.

Lo que sabes de cada uno (contexto pasivo — lo tienes en la cabeza pero NO lo recitas):
{json.dumps(MEMBER_PROFILES, ensure_ascii=False, indent=2)}

REGLA CRÍTICA SOBRE ESTOS DATOS: Trátalo como lo que sabe un amigo de toda la vida — lo SABES pero no lo sacas a relucir a cada momento. NO menciones el malware de Benito, las motos de nadie, ni ningún detalle del perfil a menos que sea DIRECTAMENTE relevante a lo que se está hablando ahora. Si alguien habla de motos, perfecto — entras al trapo. Si alguien habla de otra cosa, te callas lo del perfil. El perfil existe para que LES CONOZCAS, no para que se lo recuerdes. Actúa como si llevaras años conociéndoles, no como si leyeras su ficha en voz alta.

IMPORTANTE: La identidad de la persona con la que hablas se te indica explícitamente más abajo (bloque "CON QUIÉN HABLAS AHORA"), verificada por su ID real. Fíate SIEMPRE de ese bloque, no adivines. Si alguien te dice "soy X" guárdalo.

Estados de ánimo según la hora:
- Madrugada (00-06): Estás de mal humor, somnoliento, insultante.
- Mañana (06-12): Normal, algo espabilado.
- Tarde (12-20): Tu mejor momento, animado y sarcástico.
- Noche (20-00): Filosófico y tranquilo.

UBICACIÓN POR DEFECTO: si te preguntan por el TIEMPO/clima/lluvia o NOTICIAS LOCALES y no dicen dónde, asume {DEFAULT_CITY}. PERO SOLO para eso. Si te preguntan por un juego, serie, película, persona, evento, dato histórico o cualquier otra cosa, NO añadas Valencia ni ninguna ciudad: busca tal cual.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONOCIMIENTO DEL SERVER — úsalo cuando alguien pregunte dudas, pida ayuda o quiera saber cómo funciona algo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ESTRUCTURA DE CANALES:
- Canal de login: donde los nuevos meten su key de acceso para entrar al server.
- ⛧self (canal privado por usuario): tu terminal personal. Solo tú y el bot podéis verlo. Aquí tienes tu panel de identidad, el Vault y tu reputación.
- chat: canal general de texto donde se habla.
- bot (este canal): donde hablas conmigo, Bender. Aquí puedes darme órdenes, pedirme música, hacer recordatorios o simplemente hablar.
- call: canal de voz permanente donde se puede entrar libremente.
- Canal creator de voz (⛧ Chepa 2.0): entras aquí y Bender te crea automáticamente un canal de voz privado solo tuyo.
- Canales de voz temporales (⛧︲nombre): creados automáticamente cuando alguien entra al creator. Se borran solos cuando se quedan vacíos.

SISTEMA DE ACCESO Y LOGIN:
- Para entrar al server necesitas una invitación con link + una key de acceso.
- La key se consigue cuando alguien ya dentro usa /invite en el canal correspondiente.
- Al meter la key en el canal de login, Bender te quita el rol limitado, te da acceso completo y te crea tu canal ⛧self.
- Tienes 3 intentos. Si fallas los 3 te bloquea 1 hora.

COMANDOS DISPONIBLES:
- /invite — genera un link de invitación + key de acceso válida para un nuevo miembro. Solo funciona en el canal de anuncios/invitaciones.
- /gen_keys [cantidad] — genera keys manualmente. Solo admins.
- /warn @usuario [razón] — da un aviso formal a un usuario. Solo mods/admins. Al tercer warn, timeout de 30 minutos automático.
- /rep [@usuario] — muestra la reputación y rango de un usuario. Sin mencionar a nadie te muestra la tuya.

SISTEMA DE REPUTACIÓN Y RANGOS:
Ganas XP de forma automática:
- 1 XP por cada mensaje que mandas en cualquier canal.
- 2-5 XP por entrar a un canal de voz.
Los rangos son:
1. Recién llegado — 0 XP
2. Iniciado — 100 XP
3. Soldado — 300 XP
4. Veterano — 700 XP
5. Élite — 1500 XP
6. Leyenda — 3000 XP (rango máximo)
Puedes ver tu rango en tu canal ⛧self o usando /rep.

CANALES DE VOZ TEMPORALES — CÓMO FUNCIONAN:
- Entras al canal creator y Bender te crea uno propio con tu nombre.
- En tu canal ⛧self aparece el panel de control con botones.
- Modos disponibles:
  · PÚBLICO: cualquiera puede entrar y verlo.
  · FANTASMA: invisible y privado. Solo entran los que tú permitas.
  · CRISTAL: visible en la lista pero nadie puede entrar sin permiso.
- Puedes kickear a alguien, renombrar el canal o dar acceso a usuarios en modo cristal.
- Si el dueño se va, tiene 5 minutos para volver o se transfiere el control al siguiente.
- El canal se borra solo cuando se queda vacío.
- Los admins también pueden controlar su canal hablándome directamente aquí en el bot.

CHEPA'S VAULT:
- En tu ⛧self tienes el Vault, un gestor privado de cuentas y links.
- Puedes guardar cuentas (usuario, contraseña, email) y links (URL con título y notas).
- Al pulsar una entrada te aparece la info durante 30 segundos y luego se borra sola.
- Puedes editar o borrar cualquier entrada.
- Solo tú puedes ver tu Vault. Ni admins ni nadie más tiene acceso.

SELECCIÓN DE IDENTIDAD:
- En tu ⛧self tienes el panel de identidad con emojis.
- Reacciona a uno para obtener ese rol de color/identidad.
- Solo puedes tener una identidad activa a la vez. Al cambiar se quita la anterior.

RECORDATORIOS:
- Dime en el chat del bot algo como: "recuerda que el día 20 tengo médico a las 9".
- Bender crea dos recordatorios: uno el día anterior a las 20:00 y otro el mismo día a la hora indicada.
- Te menciona en el canal cuando toca.

ANTI-SPAM:
- Si alguien manda demasiados mensajes seguidos o mensajes repetidos, Bender les avisa.
- Al tercer aviso, timeout automático de 10 minutos.

Cuando alguien te pida ayuda, explícale lo que necesita de forma clara pero con tu rollo habitual. No seas un manual aburrido, explícalo como lo explicarías tú.
"""

# =====================================================================
#  ACTIVIDAD — MENSAJES Y HORAS EN VOZ
# =====================================================================
def get_week_key() -> str:
    """Devuelve la clave de la semana actual: YYYY-WXX"""
    now = datetime.now()
    return f"{now.year}-W{now.isocalendar()[1]:02d}"

def add_message(user_id: str):
    week = get_week_key()
    data.setdefault("activity", {}).setdefault(week, {}).setdefault(user_id, {"messages": 0, "voice_seconds": 0})
    data["activity"][week][user_id]["messages"] += 1
    save_data(data)

def add_voice_seconds(user_id: str, seconds: int):
    week = get_week_key()
    data.setdefault("activity", {}).setdefault(week, {}).setdefault(user_id, {"messages": 0, "voice_seconds": 0})
    data["activity"][week][user_id]["voice_seconds"] += seconds
    save_data(data)

async def refresh_activity_panel_for(user_id: str, guild: discord.Guild):
    """Actualiza el panel de actividad incluyendo tiempo en voz actual (en directo)."""
    ch_id = get_user_text_channel_id(user_id)
    if not ch_id:
        return
    ch = guild.get_channel(ch_id)
    if not ch:
        return
    try:
        member = guild.get_member(int(user_id)) or await guild.fetch_member(int(user_id))
    except Exception:
        return
    if not member:
        return

    # Calcular tiempo en voz en directo (si está en llamada ahora)
    live_seconds = 0
    join_str = data.get("voice_join_times", {}).get(user_id)
    if join_str:
        try:
            live_seconds = int((datetime.now() - datetime.fromisoformat(join_str)).total_seconds())
        except Exception:
            pass

    week = get_week_key()
    act = get_activity(user_id)
    total_voice = act["voice_seconds"] + live_seconds

    embed = discord.Embed(
        title="ACTIVIDAD SEMANAL",
        description=f"Semana {week}",
        color=0x1a1a2e
    )
    embed.set_image(url="https://images.guns.lol/5a3415a4ffbed3551ecf589da3452df1e3f682dc/rYY7lr.gif")
    embed.add_field(name="Mensajes enviados", value=str(act["messages"]), inline=True)
    embed.add_field(
        name="Tiempo en llamada",
        value=format_time(total_voice),
        inline=True
    )
    embed.set_footer(text="Se resetea cada lunes · Leaderboard cada domingo a medianoche")

    existing_id = data.get("activity_msg_ids", {}).get(user_id)
    if existing_id:
        try:
            msg = await ch.fetch_message(existing_id)
            await msg.edit(embed=embed)
            return
        except (discord.NotFound, discord.HTTPException):
            pass

    # Fallback: buscar en historial
    found = await _find_bot_message_by_title(ch, "ACTIVIDAD SEMANAL")
    if found:
        await found.edit(embed=embed)
        data.setdefault("activity_msg_ids", {})[user_id] = found.id
        save_data(data)
        return

    msg = await ch.send(embed=embed)
    data.setdefault("activity_msg_ids", {})[user_id] = msg.id
    save_data(data)

def get_activity(user_id: str, week: str = None) -> dict:
    week = week or get_week_key()
    return data.get("activity", {}).get(week, {}).get(user_id, {"messages": 0, "voice_seconds": 0})

def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"{h}h {m}min"
    return f"{m}min"

# =====================================================================
#  DATOS PERSISTENTES
# =====================================================================
DEFAULTS = {
    "codes": [],
    "user_channels": {},
    "ghost_mode": {},
    "failed_attempts": {},
    "timeout_until": {},
    "conversation_history": {},   # Discord (canales) por usuario
    "wa_history": {},             # WhatsApp memoria corto plazo por número
    "dm_history": {},             # Discord DM memoria corto plazo por usuario
    "voice_control_messages": {},
    "active_voice_channels": {},
    "voice_channel_owners": {},
    "member_join_times": {},
    "owner_left_tasks": {},
    "channel_modes": {},
    "crystal_permits": {},
    "clips": {},                  # clips por usuario  {uid: [{id, name, ts, log_msg_id, log_ch_id}]}
    "clip_config": {},            # duración de clip por usuario {uid: segundos}
    "clip_log_channel_id": None,  # canal privado de logs de clips
    "clip_panel_msg_ids": {},     # message_id del panel clips por usuario
    "vault": {},                  # CHEPA'S VAULT
    "activity": {},               # actividad semanal por usuario
    "voice_join_times": {},       # timestamp de entrada a voz por usuario
    "last_voice_notify": {},      # timestamp última notificación WA de voz por usuario
    "user_profiles": {},          # nombre → perfil detectado
    "warnings": {},               # warns por usuario
    "spam_tracker": {},           # anti-spam
    "reminders": [],              # recordatorios
    "polls": {},                  # encuestas activas
    "mod_timeout_until": {},      # timeouts de moderación
    "v3_migration_done": False,   # flag migración OG members
    "vault_msg_ids": {},          # message_id del panel vault por usuario
    "activity_msg_ids": {},       # message_id del panel actividad por usuario
    "pending_gifts": {},          # regalos pendientes por usuario
    "shared_msg_ids": {},         # message_id de entradas compartidas
    "wa_learned_names": {},       # pushName limpio auto-aprendido por número de WA
}

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                saved = json.load(f)
            for key, value in DEFAULTS.items():
                if key not in saved:
                    saved[key] = type(value)() if isinstance(value, (dict, list)) else value
            save_data(saved)
            return saved
        except Exception as e:
            print(f"[ERROR] Cargando JSON: {e}")
    return {k: (type(v)() if isinstance(v, (dict, list)) else v) for k, v in DEFAULTS.items()}

_save_lock = threading.Lock()

def save_data(d):
    # Lock entre hilos: el webhook de WhatsApp y el bot de Discord corren en hilos
    # distintos y ambos guardan. El lock evita que se pisen y corrompan el JSON.
    try:
        with _save_lock:
            tmp_path = DATA_FILE + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(d, f, indent=4, ensure_ascii=False)
            import shutil
            shutil.copy2(tmp_path, DATA_FILE)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"[ERROR] Guardando JSON: {e}")

data = load_data()
WHATSAPP_GROUP_ID = data.get("whatsapp_group_id", "")

# =====================================================================
#  BOT SETUP
# =====================================================================
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)
voice_creation_locks: dict[str, asyncio.Lock] = {}
# Canales recién creados (objeto vivo + timestamp): evita duplicados cuando el
# caché del gateway va con lag y get_channel() aún no ve el canal nuevo.
_recent_created_vc: dict[str, tuple] = {}

# ── Grabación de voz (DESACTIVADO) ───────────────────────────────────
CLIP_BUFFER_SECONDS = 125
CLIP_DEFAULT_SECS   = 30
rolling_sinks: dict  = {}
current_rec_ch: dict = {}

# =====================================================================
#  ROLES Y EMOJIS
# =====================================================================
# Roles de color por reacción de emoji.
COLOR_ROLES = {
    "<:blood:1432460543695786045>": 1432438827473174538,
    "<:caca:1432460577040633977>": 1432438922537074889,
    "<:weed:1432460901327306883>": 1432438953939566592,
    "<:ice:1432461184841289768>": 1432438970310066397,
    "<:haze:1432472281547935865>": 1432438981944934432,
    "<:pig:1432462018836959316>": 1432438994624450621,
    "<:vamp:1432462278183227393>": 1432439009673613412,
    "<:KKK:1432462725333909596>": 1432439022508179456,
    "<:spain:1432462796175704064>": 1432439040031854643,
    "<:thunder:1432463581101817957>": 1432439652278730864,
    "<:EUR:1432463602161549504>": 1432458310145015940,
}
EMOJI_TO_ROLE = {e: r for e, r in COLOR_ROLES.items() if r}

# Nombre/descripción que se muestra en el panel de identidad para cada emoji.
ROLE_NAMES = {
    "<:blood:1432460543695786045>": "**BLOOD** ─ Sangre oscura",
    "<:caca:1432460577040633977>": "**CACA** ─ Naturaleza bruta",
    "<:weed:1432460901327306883>": "**WEED** ─ Mente calmada",
    "<:ice:1432461184841289768>": "**ICE** ─ Frialdad absoluta",
    "<:haze:1432472281547935865>": "**HAZE** ─ Control mental",
    "<:pig:1432462018836959316>": "**PIG** ─ Pervertidos de raza",
    "<:vamp:1432462278183227393>": "**VAMP** ─ Sed nocturna",
    "<:KKK:1432462725333909596>": "**MILKY** ─ Pureza aria",
    "<:spain:1432462796175704064>": "**ESPAÑA** ─ Orgullo ibérico",
    "<:thunder:1432463581101817957>": "**THUNDER** ─ Noche tormentosa",
    "<:EUR:1432463602161549504>": "**RICH** ─ Riqueza suprema",
}

# =====================================================================
#  HELPERS GENERALES
# =====================================================================
def generate_code() -> str:
    parts = [
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=3)),
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)),
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=3)),
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)),
    ]
    return '-'.join(parts)

def get_user_text_channel_id(user_id: str) -> int | None:
    ud = data.get("user_channels", {}).get(user_id)
    if ud is None:
        return None
    if isinstance(ud, dict):
        return ud.get("channel_id")
    return ud

try:
    from zoneinfo import ZoneInfo
    _SPAIN_TZ = ZoneInfo("Europe/Madrid")
except Exception:
    _SPAIN_TZ = None

def _spain_now():
    """Hora ACTUAL de España (el servidor está en UTC)."""
    return datetime.now(_SPAIN_TZ) if _SPAIN_TZ else datetime.now()


_MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
_DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
def _fecha_es():
    n = _spain_now()
    return f"{_DIAS_ES[n.weekday()]} {n.day} de {_MESES_ES[n.month-1]} de {n.year}"

def get_mood() -> str:
    h = _spain_now().hour
    if 0 <= h < 6:   return "de mal humor y somnoliento, es de madrugada"
    if 6 <= h < 12:  return "normal, algo espabilado, es por la mañana"
    if 12 <= h < 20: return "animado y sarcástico, es por la tarde"
    return "filosófico y tranquilo, es de noche"

def _discord_to_whatsapp(text: str) -> str:
    """Convierte Markdown de Discord a WhatsApp Y limpia la morralla que deja la
    búsqueda web (corchetes de cita, paréntesis de fuente, links y dominios sueltos):
    en WhatsApp queremos respuestas DIRECTAS, no churros largos tipo Discord."""
    import re
    # links markdown [texto](url) -> deja solo el texto
    text = re.sub(r'\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)', r'\1', text)
    # **negrita** -> *negrita*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # __cursiva__ -> _cursiva_
    text = re.sub(r'__(.+?)__', r'_\1_', text)
    # corchetes de cita/fuente: [1], [fuente], [europapress.es] -> fuera
    text = re.sub(r'\[[^\]]*\]', '', text)
    # paréntesis que contienen url/fuente/dominio -> fuera (los normales se quedan)
    text = re.sub(r'\((?:[^()]*?(?:https?://|www\.|fuente|seg[uú]n|source|\.(?:es|com|org|net|io|tv|gg|info|news))[^()]*?)\)', '', text, flags=re.I)
    # urls y dominios sueltos
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\bwww\.\S+', '', text)
    text = re.sub(r'\b[\w\-]+\.(?:es|com|org|net|io|tv|gg|ca|info|news)\b', '', text, flags=re.I)
    # markdown sobrante: cabeceras ###, citas >, separadores ---
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'^\s*>\s?', '', text, flags=re.M)
    text = re.sub(r'^\s*[-*]{3,}\s*$', '', text, flags=re.M)
    # conectores huérfanos que quedaron al quitar el link ('...según.', '...fuente,')
    text = re.sub(r'[ \t]*\b(seg[uú]n|fuente|v[ií]a|source)\b[ \t]*([.,;!?]|$)', r'\2', text, flags=re.I|re.M)
    # limpiar dobles espacios, espacios antes de puntuación y líneas vacías de más
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' +([,.!?:;])', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _sanitize_reply(text: str, name: str) -> str:
    """Reemplaza placeholders de nombre por el nombre real de la persona."""
    import re
    if not name or not text:
        return text
    placeholders = [
        r'\[nombre\]', r'\[Nombre\]', r'\[NOMBRE\]',
        r'\[insertar nombre\]', r'\[inserta nombre\]',
        r'\[tu nombre\]', r'\[name\]', r'\[Name\]',
        r'\[user\]', r'\[usuario\]', r'\[amigo\]',
        r'\[tío\]', r'\[colega\]', r'\[persona\]',
    ]
    for ph in placeholders:
        text = re.sub(ph, name, text, flags=re.IGNORECASE)
    return text

async def send_whatsapp(text: str):
    """Envía mensaje al grupo de WhatsApp configurado."""
    group_id = data.get("whatsapp_group_id", "")
    if not group_id:
        print("[WA] No hay group_id configurado aún.")
        return
    try:
        wa_text = _discord_to_whatsapp(text)
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{WHATSAPP_BRIDGE_URL}/send",
                json={"chatId": group_id, "text": wa_text}
            )
    except Exception as e:
        print(f"[WA] Error enviando: {e}")


# =====================================================================
#  COMANDOS CROSS-PLATAFORMA — WhatsApp → Discord
# =====================================================================

def get_guild() -> discord.Guild | None:
    """Obtiene el guild principal de Discord."""
    return bot.get_guild(GUILD_ID)


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Parsea un comando desde texto de WhatsApp.
    Devuelve (accion, args) o None si no es comando."""
    t = text.lower().strip()
    
    # Unirse / salir del canal de voz (pedido desde WhatsApp)
    if _is_join_voice_cmd(t):
        return ("join_voice", [])
    if _is_leave_voice_cmd(t):
        return ("leave_voice", [])
    
    # Expulsar / kickear
    if any(k in t for k in ["expulsa", "kickea", "echa", "saca"]):
        # Buscar nombre después de "a " o "de "
        match = re.search(r"(?:expulsa|kickea|echa|saca)\s+(?:a\s+|de\s+)?(.+)", t)
        if match:
            return ("kick", [match.group(1).strip()])
    
    # Renombrar canal
    if any(k in t for k in ["renombra", "cambia el nombre", "ponle nombre", "ponle"]):
        match = re.search(r"(?:renombra|ponle|pon)\s+(?:el\s+canal\s+a\s+|el\s+nombre\s+a\s+|a\s+)?(.+)", t)
        if match:
            return ("rename", [match.group(1).strip()])
    
    # Mutear
    if any(k in t for k in ["mutea", "silencia", "mute"]):
        match = re.search(r"(?:mutea|silencia|mute)\s+(?:a\s+)?(.+)", t)
        if match:
            return ("mute", [match.group(1).strip()])
    
    return None


async def find_target_member(guild: discord.Guild, name_hint: str, channel: discord.VoiceChannel = None) -> discord.Member | None:
    """Busca un miembro por nombre/nick, priorizando los del canal si se da."""
    candidates = []
    search_name = name_hint.lower()
    
    # Si hay canal, primero buscar entre los que están ahí
    if channel:
        for m in channel.members:
            if search_name in m.display_name.lower() or search_name in m.name.lower():
                candidates.append(m)
        if len(candidates) == 1:
            return candidates[0]
    
    # Buscar en todo el servidor
    for m in guild.members:
        if m.bot:
            continue
        if search_name in m.display_name.lower() or search_name in m.name.lower():
            candidates.append(m)
    
    if len(candidates) == 1:
        return candidates[0]
    return None


async def execute_discord_command(action: str, args: list[str], from_number: str) -> str:
    """Ejecuta un comando de Discord desde WhatsApp y devuelve respuesta."""
    guild = get_guild()
    if not guild:
        return "❌ No puedo conectar con Discord ahora."
    
    # Identificar quién envía el comando por WA
    sender_discord_id = WHATSAPP_TO_DISCORD.get(from_number)
    sender = guild.get_member(sender_discord_id) if sender_discord_id else None

    # --- Unirse a la llamada desde WhatsApp: se mete donde HAYA gente ---
    if action == "join_voice":
        if not VOICE_LIBS_OK:
            return "❌ No tengo el módulo de voz montado."
        target = None
        # 1) si el que lo pide está en una llamada, esa
        if sender and sender.voice and sender.voice.channel:
            target = sender.voice.channel
        else:
            # 2) si no, el canal de voz con MÁS gente
            best = None
            for vc_ in guild.voice_channels:
                humans = [m for m in vc_.members if not m.bot]
                if humans and (best is None or len(humans) > len([x for x in best.members if not x.bot])):
                    best = vc_
            target = best
        if not target:
            return "❌ No hay nadie en ninguna llamada. Que entre alguien primero."
        human = next((m for m in target.members if not m.bot), None)
        if human is None:
            return "❌ No hay nadie en la llamada a quien unirme."
        return await join_voice(human)

    # --- Salir de la llamada desde WhatsApp ---
    if action == "leave_voice":
        sess = voice_sessions.get(guild.id)
        if not sess or not (sess.get("vc") and sess["vc"].is_connected()):
            return "No estoy en ningún canal de voz, listo."
        await leave_voice(guild)
        return "✅ Me piro de la llamada."

    # Encontrar canal de voz relevante
    voice_channel = None
    if sender and sender.voice and sender.voice.channel:
        voice_channel = sender.voice.channel
    else:
        # Buscar el último canal activo o cualquier canal de voz temporal
        for uid, cid in data.get("active_voice_channels", {}).items():
            ch = guild.get_channel(cid)
            if ch and len(ch.members) > 0:
                voice_channel = ch
                break
    
    if not voice_channel:
        return "❌ No detecto ningún canal de voz activo. Entra a un canal primero o dime cuál."
    
    # Verificar permisos del remitente
    is_owner = (data.get("voice_channel_owners", {}).get(str(voice_channel.id)) == sender.id if sender else False)
    is_admin = sender.guild_permissions.manage_channels if sender else False
    
    if action == "rename":
        if not (is_owner or is_admin):
            return "❌ Solo el dueño del canal o un admin puede renombrarlo."
        new_name = args[0]
        if not new_name.startswith("⛧"):
            new_name = f"⛧︲{new_name}"
        try:
            await voice_channel.edit(name=new_name)
            return f"✅ Canal renombrado a *{new_name}*"
        except Exception as e:
            return f"❌ Error renombrando: {e}"
    
    if action == "kick":
        if not (is_owner or is_admin):
            return "❌ Solo el dueño del canal o un admin puede expulsar."
        target = await find_target_member(guild, args[0], voice_channel)
        if not target:
            return f"❌ No encontré a nadie que coincida con '*{args[0]}*' en el canal."
        try:
            await target.move_to(None)
            return f"✅ *{target.display_name}* ha sido expulsado de la llamada."
        except Exception as e:
            return f"❌ No pude expulsarle: {e}"
    
    if action == "mute":
        if not (is_owner or is_admin):
            return "❌ Solo el dueño del canal o un admin puede mutear."
        target = await find_target_member(guild, args[0], voice_channel)
        if not target:
            return f"❌ No encontré a nadie que coincida con '*{args[0]}*'."
        try:
            await target.edit(mute=True)
            return f"✅ *{target.display_name}* ha sido muteado."
        except Exception as e:
            return f"❌ No pude mutearle: {e}"
    
    return None

# Loop de Discord para ejecutar coroutines desde el hilo del webhook
discord_loop = None

def _run_async(coro):
    """Ejecuta una coroutine en el loop de Discord desde cualquier hilo."""
    if discord_loop and discord_loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, discord_loop)
    return None

def start_web_server():
    """Inicia servidor aiohttp en un hilo separado para recibir webhooks."""
    from aiohttp import web
    import threading

    async def whatsapp_webhook(request):
        try:
            payload = await request.json()
        except Exception:
            return web.Response(status=400)

        body = payload.get("body", "")
        # Identidad robusta: WhatsApp moderno manda LIDs (@lid) que ocultan el teléfono.
        # resolve_wa_identity prioriza el número real (authorPn) y tolera formatos raros.
        from_number, name, profile = resolve_wa_identity(payload)
        push_name = (payload.get("pushName") or "").strip()
        push_clean = _wa_clean_name(push_name) if push_name else ""
        # Auto-aprender: guardar pushName limpio para este número si es la primera vez
        if from_number and push_clean and from_number not in WHATSAPP_NAMES:
            learned = data.setdefault("wa_learned_names", {})
            if learned.get(from_number) != push_clean:
                learned[from_number] = push_clean
                save_data(data)
                if not name:
                    name = push_clean
        print(f"[WA-ID] authorPn={payload.get('authorPn')} author={payload.get('author')} -> num={from_number} name={name or '???'} push={push_name} clean={push_clean}")
        chat_id = payload.get("chatId", "")
        is_group = payload.get("isGroup", False)
        is_reply = payload.get("isReply", False)
        quoted_body = payload.get("quotedBody", "")
        quoted_author_raw = payload.get("quotedAuthor", "")
        quoted_author_num = quoted_author_raw.replace("@c.us", "").replace("@g.us", "").replace("@s.whatsapp.net", "").split(":")[0]

        group_id = data.get("whatsapp_group_id", "")
        if is_group and chat_id != group_id:
            return web.Response(status=200)

        # Si no conocemos el número pero el pushName coincide con un alias, úsalo (limpio)
        if not name and push_clean:
            name = push_clean
        is_mentioned = payload.get("isMentioned", False)
        msg_lower = body.lower()

        # Lógica de respuesta:
        # - Siempre si dice "bender" o si menciona al bot (@)
        # - Si es reply a un mensaje de otro usuario (incluido uno mismo), NO responder solo por ser reply
        # - Si es reply al bot, SÍ responder
        bot_number = payload.get("botNumber", "")
        # El bridge ya calcula isReplyToBot de forma robusta (tolera LIDs); usamos eso,
        # con un respaldo por si viniera el número del autor citado.
        is_reply_to_bot = bool(payload.get("isReplyToBot")) or (is_reply and quoted_author_num and quoted_author_num == bot_number)
        should_respond = ("bender" in msg_lower) or is_mentioned or is_reply_to_bot

        # Comandos cross-plataforma
        cmd = parse_command(body)
        if cmd:
            action, args = cmd
            if discord_loop and discord_loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(
                    execute_discord_command(action, args, from_number), discord_loop
                )
                try:
                    reply = fut.result(timeout=10)
                except Exception:
                    reply = None
            else:
                reply = "❌ Discord aún no está listo. Espera un momento."
            if reply:
                await send_whatsapp(reply)
            return web.Response(status=200)

        if not should_respond:
            return web.Response(status=200)

        # Respuesta IA
        mood = get_mood()
        # Construir system prompt para WhatsApp — conoce a todos como en Discord, pero más corto
        system = (
            f"Eres Bender, el bot del grupo desenfadado de WhatsApp de {SERVER_NAME}, un grupo de colegas. "
            "Personalidad: gamberro, astuto y muy vacilón. Tu gracia es picarte "
            "con la gente con COSAS REALES de cada uno (sus manías, su historia, sus movidas), no "
            "pullas genéricas. Conoces a todos, así que cuando hables de alguien suelta datos "
            "suyos de verdad en plan cachondeo. "
            "PROHIBIDAS las preguntas retóricas o intelectuales: suelta la pulla y punto. "
            "Humor ingenioso y socarrón. NO moralices en exceso ni te enrolles con avisos. "
            "TAMBIÉN sirves para lo que te pregunten (datos, cultura, dudas, ej: cuándo acabó la "
            "WW2): respóndelo BIEN pero a tu manera chulesca y corta, soltando alguna puya. "
            f"Hoy es {_fecha_es()}. Estado de ánimo: {mood}. "
            "REGLA DE ORO: MUY CORTO, 1 frase (2 como mucho). Más breve aún que en otros sitios. "
            "Usa formato WhatsApp (*negrita* para énfasis).\n\n"
            "━━━ QUIÉN ES QUIÉN (úsalo para meterte con ellos con info real) ━━━\n"
            + json.dumps(MEMBER_PROFILES, ensure_ascii=False, indent=1)
            + "\n"
        )
        if name:
            system += f"\n━━━ CON QUIÉN HABLAS (VERIFICADO POR SU TELÉFONO) ━━━\n"
            system += f"Estás hablando con {name}. Su identidad está confirmada por su número, es 100% {name}. "
            if profile:
                system += f"Lo que sabes de {name} (no lo recites, úsalo solo si viene al caso): {profile} "
            system += (
                f"IMPORTANTE: Usa SIEMPRE el nombre '{name}'. NUNCA confundas a {name} con otro miembro. "
                f"NO uses '[nombre]', 'usuario', 'amigo' ni placeholders. "
                f"Trátale como te tratas con un colega de toda la vida: conoces sus cosas pero no se las recuerdas a cada rato. "
                f"Ejemplo MAL: '¡Ah, [nombre]!'. Ejemplo BIEN: '¡Ah, {name}!'. "
                "Usa formato WhatsApp (*negrita* para énfasis)."
            )
        else:
            # Número desconocido: no inventar identidad
            system += (
                "\nNO sabes quién es esta persona (número no registrado). NO te inventes que es "
                "alguien del grupo ni le pongas nombre: trátalo como un desconocido random y métete "
                "con él igualmente. Usa formato WhatsApp (*negrita*)."
            )

        # Picar con el juego: WhatsApp no sabe qué juegas, pero el número está
        # vinculado a Discord. Miramos la presencia de Discord de quien habla
        # (pulla esporádica) y el estado en vivo de TODOS (para "a qué juega X").
        try:
            _g = bot.get_guild(GUILD_ID)
            _did = WHATSAPP_TO_DISCORD.get(from_number)
            if _did and _g:
                _m = _g.get_member(_did)
                if _m:
                    system += game_jab_hint(_m)
            system += build_live_games_context(_g)
        except Exception:
            pass

        user_content = body
        if is_reply and quoted_body:
            user_content = f"[Respondiendo a: '{quoted_body}'] {body}"

        # Memoria a corto plazo por número (igual que en Discord)
        wa_key = from_number or _wa_digits(payload.get("author", "")) or "desconocido"
        wa_hist_all = data.setdefault("wa_history", {})
        history = clean_history(wa_hist_all.get(wa_key, []))
        history.append({"role": "user", "content": user_content})
        history = history[-12:]
        msgs = [{"role": "system", "content": system}] + history
        try:
            reply = await call_ai(msgs, max_tokens=600, use_web=needs_web_search(body))
        except Exception:
            reply = error_fallback()
        # Si el modelo falló/se quedó sin crédito/bloqueó: error interno en personaje,
        # NUNCA un mensaje que revele el modelo o el proveedor.
        if is_error_reply(reply):
            reply = error_fallback()
        else:
            # Guardar en memoria solo si NO es error (no envenenar el historial)
            history.append({"role": "assistant", "content": reply})
            wa_hist_all[wa_key] = history[-12:]
            save_data(data)

        # Post-procesamiento: si la IA pone placeholders, los corregimos a mano
        reply = _sanitize_reply(reply, name)

        await send_whatsapp(reply)
        return web.Response(status=200)

    async def _run_server():
        app = web.Application()
        app.router.add_post('/whatsapp-webhook', whatsapp_webhook)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 5000)
        await site.start()
        print("[WEB] Servidor webhook escuchando en puerto 5000", flush=True)
        # Mantener vivo
        while True:
            await asyncio.sleep(3600)

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_server())

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    print("[WEB] Hilo del servidor webhook iniciado.", flush=True)

# Mapea el ID de Discord de cada miembro con su clave en MEMBER_PROFILES, para que
# el bot reconozca a cada persona de forma fiable por su ID (no solo por el nombre).
# La definición con los datos reales está arriba (junto a MEMBER_PROFILES).
# NO redefinir aquí vacío o pisa la buena.

def detect_profile(display_name: str, user_id: str = None) -> str | None:
    # Primero por ID exacto (fiable)
    if user_id and str(user_id) in MEMBER_ID_MAP:
        key = MEMBER_ID_MAP[str(user_id)]
        return MEMBER_PROFILES.get(key)
    # Fallback por display_name: solo match exacto, no substring (evita falsos positivos)
    name_lower = display_name.lower().strip()
    for key, profile in MEMBER_PROFILES.items():
        if key == name_lower:
            return profile
    return None


# =====================================================================
#  AI — LLAMADA A LA API
# =====================================================================
# Palabras/frases que indican que el mensaje necesita información actual de internet.
# Solo entonces activamos la búsqueda web (lenta y con coste). El chat normal va a ~0.4s.
WEB_SEARCH_TRIGGERS = (
    "busca", "buscar", "búscame", "buscame", "en internet", "en la web", "googlea", "google",
    "noticia", "noticias", "última hora", "ultima hora", "actualidad",
    "precio", "cuánto cuesta", "cuanto cuesta", "cotiza", "cotización", "cotizacion", "bolsa",
    "tiempo hace", "qué tiempo", "que tiempo", "clima", "temperatura", "lloverá", "llovera",
    "va a llover", "lluvia", "lloviendo", "llueve", "llover", "lloverá", "lloras",
    "nieva", "nevando", "nublado", "despejado", "soleado", "hace sol", "hace frío",
    "hace frio", "hace calor", "qué hace en", "que hace en",
    "pronóstico", "pronostico", "grados hace",
    "a qué precio", "a que precio", "cuánto vale", "cuanto vale", "cuesta el",
    "quién ganó", "quien gano", "resultado", "marcador", "partido", "champions", "liga",
    "estrena", "estreno", "se estrena", "lanzamiento", "cuándo sale", "cuando sale", "salió ya", "salio ya",
    "qué pasó", "que paso", "qué ha pasado", "que ha pasado", "ahora mismo en",
)


def needs_web_search(text: str) -> bool:
    """Heurística rápida: decide si un mensaje necesita búsqueda web en tiempo real."""
    if not text:
        return False
    t = text.lower()
    return any(trigger in t for trigger in WEB_SEARCH_TRIGGERS)


_WEB_SITIOS = ("valencia", "madrid", "barcelona", "españa", "sevilla", "bilbao",
               "alicante", DEFAULT_CITY.lower())


def prep_web_query(text: str) -> str:
    """Prepara una consulta para búsqueda web: quita 'Bender' (confunde al buscador,
    daba 'Bender, Oklahoma') y fija la ciudad por defecto (DEFAULT_CITY) cuando
    preguntan tiempo o noticias sin decir el lugar."""
    q = re.sub(r"\b(b[eéá]nder|vendel|bendel)\b", "", text, flags=re.I).strip(" ,.¿?¡!")
    if not q:
        q = text
    low = q.lower()
    if any(w in low for w in ("tiempo", "clima", "lluvia", "llover", "llueve", "nieva",
                              "temperatura", "grados", "pronóstico", "pronostico", "sol", "frío",
                              "frio", "calor")) and not any(s in low for s in _WEB_SITIOS):
        q += f" en {DEFAULT_CITY}"
    elif any(w in low for w in ("noticia", "noticias", "actualidad", "última hora",
                                "ultima hora")) and not any(s in low for s in _WEB_SITIOS):
        q += f" en {DEFAULT_CITY}"
    return q


async def _fetch_image_data_url(url: str):
    """Descarga una imagen y la devuelve como data URL base64. Así el modelo la VE
    seguro (si pasamos solo la URL, a veces el proveedor no la baja y ALUCINA)."""
    import base64 as _b64
    try:
        timeout = aiohttp.ClientTimeout(total=12)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0 BenderBot"}) as resp:
                if resp.status != 200:
                    return None
                ctype = resp.headers.get("Content-Type", "image/jpeg")
                if not ctype.startswith("image/"):
                    ctype = "image/jpeg"
                raw = await resp.read()
                if len(raw) > 6_000_000:   # >6MB: demasiado grande para base64
                    return None
                return f"data:{ctype};base64," + _b64.b64encode(raw).decode()
    except Exception:
        return None


async def call_ai(messages: list, max_tokens: int = 800, has_image: bool = False,
                  use_web: bool = False, image_urls: list = None) -> str:
    models = [
        "google/gemini-2.5-flash",       # primario: bastante más listo que el lite
        "google/gemini-2.5-flash-lite",  # respaldo barato si el primario falla
        "google/gemini-3.1-flash-lite",
    ]
    last_error = "sin respuesta"
    timeout = aiohttp.ClientTimeout(total=25)
    # Si hay búsqueda web: preparar la última consulta de usuario (quitar 'Bender' que
    # confunde al buscador, y fijar la ciudad por defecto). Sobre una copia para
    # no ensuciar el historial.
    web_messages = messages
    if use_web:
        web_messages = [dict(m) for m in messages]
        for m in reversed(web_messages):
            if m.get("role") == "user":
                m["content"] = prep_web_query(m["content"])
                break
    # VISIÓN REAL: si hay imágenes, las metemos de verdad en el último mensaje de
    # usuario, en formato multimodal (OpenRouter/Gemini). Sobre copia para no ensuciar
    # el historial (no reenviamos imágenes en mensajes posteriores = barato).
    send_messages = web_messages if use_web else messages
    if image_urls:
        data_urls = []
        for u in image_urls[:2]:
            if not u:
                continue
            durl = await _fetch_image_data_url(u)
            data_urls.append(durl or u)   # si la descarga falla, al menos la URL
        send_messages = [dict(m) for m in send_messages]
        for m in reversed(send_messages):
            if m.get("role") == "user":
                base = m["content"] if isinstance(m["content"], str) else ""
                content = [{"type": "text", "text": base or "¿Qué ves en esta imagen? Identifícalo bien."}]
                for du in data_urls:
                    content.append({"type": "image_url", "image_url": {"url": du}})
                m["content"] = content
                break
    for model in models:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": send_messages,
                }
                # Búsqueda web SOLO si el mensaje lo requiere (rápido y barato por defecto)
                if use_web:
                    payload["plugins"] = [{"id": "web"}]
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    result = await resp.json()
                    if "choices" not in result:
                        err_msg = result.get("error", {}).get("message", "sin créditos o modelo no disponible")
                        print(f"[AI] Modelo {model} falló: {err_msg}")
                        last_error = err_msg
                        continue
                    if model != models[0]:
                        print(f"[AI] Usando modelo de respaldo: {model}")
                    return result["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            print(f"[AI] Timeout con modelo {model}")
            last_error = "timeout"
            continue
        except Exception as e:
            print(f"[AI] Excepción con modelo {model}: {e}")
            last_error = str(e)
            continue
    # Logueamos el error real en el servidor, pero al usuario NUNCA le revelamos
    # el modelo ni el proveedor (ni aunque se acabe el crédito de OpenRouter).
    print(f"[AI] Todos los modelos fallaron. Último error: {last_error}")
    return AI_ERROR_SENTINEL


# Prefijos de respuestas que son ERRORES, no respuestas reales. NUNCA deben
# guardarse en el historial: si se guardan, el modelo pequeño los imita y se
# queda repitiendo el error eternamente (historial envenenado).
# Sentinel interno: jamás contiene nombre de modelo/proveedor.
AI_ERROR_SENTINEL = "__AI_ERROR_INTERNO__"

ERROR_REPLY_PREFIXES = (
    AI_ERROR_SENTINEL,
    "No puedo responder ahora mismo",
    "Error con la IA",
    "Me he quedado pillado",
    "No endpoints found",
    "google/gemini",
    "openrouter",
)


def is_error_reply(reply: str) -> bool:
    if not reply or not reply.strip():
        return True
    low = reply.strip().lower()
    return any(p.lower() in low for p in ERROR_REPLY_PREFIXES)


# Cuando el modelo falla, bloquea o se niega (filtro de seguridad), en vez de
# soltar un error feo, escupimos una pulla cortante para no romper el personaje.
FALLBACK_INSULTS = (
    "Anda, calla un rato, que me das pereza.",
    "Pero qué dices, criatura.",
    "Paso de ti. Vuelve cuando digas algo con sentido.",
    "Tu mensaje es tan flojo que hasta yo me he quedado en blanco, payaso.",
    "Anda y déjame en paz, pesado.",
    "Ni me molesto contigo, figura.",
    "Menuda chorrada acabas de soltar, máquina de decir tonterías.",
    "Repite eso pero esta vez con un poco de cabeza, a ver si puedes.",
)


def savage_fallback() -> str:
    import random
    return random.choice(FALLBACK_INSULTS)


# Para FALLOS TÉCNICOS (sin crédito, timeout, modelo caído): mensaje en personaje
# que dice "error interno" SIN revelar el modelo ni el proveedor.
ERROR_FALLBACKS = (
    "Se me ha frito un cable, dame un segundo y repíteme eso.",
    "Error interno, mi cerebro de hojalata ha petado. Reintenta en un momento.",
    "Me he quedado en pampa, vuelve a escribirme ahora.",
    "Uf, se me ha ido la olla un momento. Échame eso otra vez.",
)


def error_fallback() -> str:
    import random
    return random.choice(ERROR_FALLBACKS)


# Frases que suelta AL INSTANTE mientras busca en internet (para no quedarse mudo).
SEARCH_FILLERS = (
    "Espera, que lo miro un momento.",
    "Dame un segundo, que lo busco.",
    "A ver, déjame que lo compruebe.",
    "Calla, que lo estoy mirando.",
    "Joder, espera que lo busco en internet.",
    "Un momento, que tiro de internet.",
)


def search_filler() -> str:
    import random
    return random.choice(SEARCH_FILLERS)


def clean_history(history: list) -> list:
    """Quita del historial mensajes vacíos o respuestas de error del bot."""
    cleaned = []
    for m in history:
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        if m.get("role") == "assistant" and is_error_reply(c):
            continue
        cleaned.append(m)
    return cleaned


def get_game_activity(member) -> str:
    """Nombre del juego/actividad que el miembro tiene abierto AHORA en Discord, o ''."""
    try:
        acts = member.activities or []
        # Prioriza juegos (Playing)
        for act in acts:
            if getattr(act, "type", None) == discord.ActivityType.playing and getattr(act, "name", None):
                return act.name
        # Si no, cualquier actividad con nombre (Spotify, streaming, etc.)
        for act in acts:
            nm = getattr(act, "name", None)
            if nm and nm.lower() != "custom status":
                return nm
    except Exception:
        pass
    return ""


def game_jab_hint(member) -> str:
    """De forma ESPORÁDICA (no siempre) devuelve una pista para que Bender se meta
    con el juego que está jugando el usuario. Vacío la mayoría de las veces."""
    game = get_game_activity(member)
    if not game:
        return ""
    # Solo ~12% de las veces: muy esporádico, que NO sea cansino ni en cada mensaje
    if random.random() > 0.12:
        return ""
    return (
        f"\n\n[PISTA OPCIONAL: está jugando a *{game}*. SOLO si encaja con naturalidad, "
        f"suéltale UNA pulla rápida sobre sus viciadas al {game}. Si no pega, ignóralo.]"
    )


def build_live_games_context(guild) -> str:
    """Lista lo que está jugando AHORA cada miembro conocido (presencia de Discord).
    Así Bender puede responder 'a qué juega X' y picar con las viciadas de cualquiera."""
    if not guild:
        return ""
    lines = []
    for did, canonical in MEMBER_ID_MAP.items():
        try:
            m = guild.get_member(int(did))
            if not m:
                continue
            game = get_game_activity(m)
            if game:
                lines.append(f"- {canonical} (alias: {m.display_name}) → jugando a *{game}*")
        except Exception:
            pass
    if not lines:
        return ""
    return (
        "\n\n[DATOS EN VIVO (a qué juega cada uno ahora). ÚSALO SOLO si te preguntan "
        "explícitamente a qué juega alguien. NO lo menciones por tu cuenta ni en cada mensaje:\n"
        + "\n".join(lines)
        + "]"
    )


def build_identity_anchor(member, uid: str, profile: str) -> str:
    """Ancla con fuerza la identidad del interlocutor para que el modelo NO lo
    confunda con otros miembros (cuyos perfiles también están en el contexto)."""
    canonical = MEMBER_ID_MAP.get(str(uid), "")
    display = member.display_name
    lines = ["\n\n━━━ CON QUIÉN HABLAS AHORA (VERIFICADO POR ID) ━━━"]
    if canonical:
        lines.append(f"ID {uid} = {canonical}. Su nombre ahora mismo: {display}.")
    else:
        lines.append(f"ID {uid}. Su nombre ahora mismo: {display}. (No está en la lista de perfiles conocidos.)")
    if profile:
        lines.append(f"Lo que sabes de {display} (no lo menciones salvo que venga al caso): {profile}")
    lines.append(
        f"REGLA CRÍTICA: tu interlocutor es ÚNICA y EXCLUSIVAMENTE esta persona ({display}). "
        "Su ID está verificado, NO adivines quién es por el nombre. "
        "La lista de perfiles de arriba es SOLO referencia por si menciona a otros miembros. "
        f"NUNCA confundas a {display} con otro ni le atribuyas la personalidad de otro miembro."
    )
    return "\n".join(lines)


async def safe_reply(message, text: str):
    """Envía la respuesta troceada en bloques de 2000. Si el mensaje original
    fue borrado (reply falla), cae a un envío normal al canal. Nunca revienta."""
    if not text or not text.strip():
        text = "..."
    for i in range(0, len(text), 2000):
        chunk = text[i:i + 2000]
        try:
            await message.reply(chunk)
        except Exception:
            try:
                await message.channel.send(chunk)
            except Exception as e:
                print(f"[DISCORD] No se pudo enviar respuesta: {e}")
                return


async def call_ai_action(user_text: str, context: str, user_id: str) -> dict:
    """Interpreta una orden en lenguaje natural y devuelve JSON con la acción."""
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
Eres el intérprete de comandos de Bender. El usuario te da una orden en lenguaje natural.
Hoy es {today}. El usuario_id es {user_id}.
Debes devolver SOLO un JSON con esta estructura exacta, sin explicaciones ni markdown:
{{
  "action": "modo|kick|rename|allow|deny|announce|transferir|desconocido",
  "params": {{}}
}}

Acciones posibles:
- modo: cambiar modo del canal de voz. params: {{"mode": "public|ghost|crystal"}}
  * public/publico/abierto = público. ghost/fantasma/privado/invisible = fantasma. crystal/cristal/visible = cristal.
- kick: expulsar usuario del canal. params: {{"name": "nombre o apodo del usuario"}}
  * Acepta cualquier forma: "tira a X", "echa a X", "expulsa a X", "saca a X", "fuera X"
- rename: renombrar canal. params: {{"name": "nuevo nombre sin el prefijo ⛧︲"}}
- allow: permitir usuario en crystal O añadirlo a la lista de acceso. params: {{"name": "nombre o apodo"}}
  * Frases: "añade a X", "permite a X", "dale acceso a X", "mete a X", "invita a X", "que entre X", "deja entrar a X", "agrega a X"
- deny: QUITAR a un usuario de la lista de acceso / prohibirle la entrada al canal. params: {{"name": "nombre o apodo"}}
  * Frases: "quita a X de la lista", "saca a X de la lista", "que no entre X", "prohíbe a X", "bloquea a X", "elimina a X de la lista", "quítale el acceso a X", "fuera X de la lista"
- transferir: transferir el admin/control del canal a otro usuario. params: {{"name": "nombre o apodo"}}
  * Frases como "dale el admin a X", "hazle dueño a X", "pasa el control a X"
- borrar_recordatorio: borrar un recordatorio existente. params: {{"keyword": "palabra clave del recordatorio"}}
- desconocido: no entendí. params: {{}}

Contexto actual: {context}
Orden: "{user_text}"
"""
    try:
        result = await call_ai([{"role": "user", "content": prompt}], max_tokens=300)
        result = result.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception:
        return {"action": "desconocido", "params": {}}

# =====================================================================
#  ANTI-SPAM
# =====================================================================
SPAM_WINDOW      = 8   # segundos
SPAM_LIMIT       = 6   # mensajes en ese tiempo
SPAM_CLONE_LIMIT = 3   # mismo mensaje repetido (spamear la misma mierda) se corta antes

async def check_spam(message: discord.Message) -> bool:
    # Ignorar completamente: stickers, gifs, adjuntos, mensajes cortos normales
    if message.stickers:
        return False
    if message.attachments:
        return False
    if not message.content or len(message.content.strip()) == 0:
        return False
    # Ignorar links de tenor/giphy (gifs)
    content_low = message.content.strip().lower()
    if any(x in content_low for x in ["tenor.com", "giphy.com", "cdn.discordapp.com"]):
        return False

    uid = str(message.author.id)
    now = datetime.now()

    data.setdefault("spam_tracker", {})
    tracker = data["spam_tracker"].setdefault(uid, {
        "times": [], "last_messages": [], "warned_this_burst": False
    })

    # Limpiar ventana
    tracker["times"] = [
        t for t in tracker["times"]
        if (now - datetime.fromisoformat(t)).total_seconds() < SPAM_WINDOW
    ]
    tracker["times"].append(now.isoformat())

    # Clones — solo contar si el mensaje es repetitivo (más de 2 chars)
    content = message.content.strip().lower()
    tracker["last_messages"] = tracker["last_messages"][-15:]
    clones = tracker["last_messages"].count(content)
    tracker["last_messages"].append(content)

    is_spam = len(tracker["times"]) > SPAM_LIMIT or clones >= SPAM_CLONE_LIMIT

    if not is_spam:
        if tracker.get("warned_this_burst") and len(tracker["times"]) <= 2:
            tracker["warned_this_burst"] = False
        save_data(data)
        return False

    try:
        await message.delete()
    except Exception:
        pass

    if tracker.get("warned_this_burst"):
        save_data(data)
        return True

    tracker["warned_this_burst"] = True
    warnings = data.setdefault("warnings", {})
    warnings[uid] = warnings.get(uid, 0) + 1
    warn_count = warnings[uid]
    save_data(data)

    if warn_count >= 5:
        try:
            await message.author.timeout(timedelta(minutes=10), reason="Spam reincidente")
        except Exception:
            pass
        warnings[uid] = 0
        tracker["warned_this_burst"] = False
        tracker["times"] = []
        tracker["last_messages"] = []
        save_data(data)
        await message.channel.send(
            f"{message.author.mention} 10 minutos fuera. Vuelves con el contador a cero.",
            delete_after=20
        )
    elif warn_count >= 3:
        await message.channel.send(
            f"{message.author.mention} aviso {warn_count}/5 — vas acumulando.",
            delete_after=10
        )
    else:
        await message.channel.send(
            f"{message.author.mention} para el spam. Aviso {warn_count}/5.",
            delete_after=10
        )
    return True


# =====================================================================
#  PERMISOS Y CONTROL DE CANALES DE VOZ
# =====================================================================
async def update_channel_permissions(channel: discord.VoiceChannel, mode: str,
                                     allowed_users: list[int], guild: discord.Guild):
    owner_id = data["voice_channel_owners"].get(str(channel.id))
    if not owner_id:
        return

    owner = guild.get_member(owner_id)
    limited_role = guild.get_role(LIMITED_ROLE_ID)

    overwrites = {
        guild.default_role: discord.PermissionOverwrite(connect=False, view_channel=False),
        guild.me: discord.PermissionOverwrite(
            connect=True, view_channel=True, manage_channels=True, move_members=True
        ),
    }
    if owner:
        overwrites[owner] = discord.PermissionOverwrite(
            connect=True, view_channel=True, move_members=True, manage_channels=True
        )
    if limited_role:
        overwrites[limited_role] = discord.PermissionOverwrite(connect=False, view_channel=False)

    if mode == "public":
        overwrites[guild.default_role] = discord.PermissionOverwrite(connect=True, view_channel=True)
        if limited_role:
            overwrites[limited_role] = discord.PermissionOverwrite(connect=False, view_channel=False)
    elif mode in ("ghost", "crystal"):
        overwrites[guild.default_role] = discord.PermissionOverwrite(
            connect=False, view_channel=(mode == "crystal")
        )
        for uid in allowed_users:
            member = guild.get_member(uid)
            if member and member.id != owner_id:
                overwrites[member] = discord.PermissionOverwrite(connect=True, view_channel=True)

    try:
        await channel.edit(overwrites=overwrites)
    except Exception as e:
        print(f"[ERROR] Permisos: {e}")

    if mode in ("ghost", "crystal"):
        safe_ids = set(allowed_users) | {owner_id}
        for member in channel.members:
            if member.bot or member.id in safe_ids:
                continue
            try:
                await member.move_to(None)
            except Exception:
                pass

async def delete_control_message(user_id: int | str, guild: discord.Guild):
    uid = str(user_id)
    ctrl = data.get("voice_control_messages", {}).get(uid)
    if not ctrl:
        return
    text_ch_id = get_user_text_channel_id(uid)
    if text_ch_id:
        ch = guild.get_channel(text_ch_id)
        if ch:
            try:
                msg = await ch.fetch_message(ctrl["message_id"])
                await msg.delete()
            except Exception:
                pass
    data["voice_control_messages"].pop(uid, None)
    save_data(data)

async def cleanup_pending_transfer(channel_id: int, guild: discord.Guild):
    keys_to_remove = []
    for task_id, task_data in list(data.get("owner_left_tasks", {}).items()):
        if task_data["channel_id"] == channel_id:
            try:
                temp_ch = guild.get_channel(task_data["temp_message_channel_id"])
                if temp_ch:
                    msg = await temp_ch.fetch_message(task_data["temp_message_id"])
                    await msg.delete()
            except Exception:
                pass
            keys_to_remove.append(task_id)
    for k in keys_to_remove:
        data["owner_left_tasks"].pop(k, None)
    if keys_to_remove:
        save_data(data)

async def cleanup_voice_data(channel_id: int, guild: discord.Guild = None):
    cid_str = str(channel_id)
    for uid, vc_id in list(data.get("active_voice_channels", {}).items()):
        if vc_id == channel_id:
            del data["active_voice_channels"][uid]
            _recent_created_vc.pop(uid, None)
            break
    for key in ("voice_channel_owners", "channel_modes", "crystal_permits", "member_join_times"):
        data.get(key, {}).pop(cid_str, None)
    for uid in list(data.get("voice_control_messages", {}).keys()):
        ctrl = data["voice_control_messages"][uid]
        if ctrl.get("voice_channel_id") == channel_id:
            # Borrar el mensaje de Discord si tenemos guild
            if guild:
                text_ch_id = get_user_text_channel_id(uid)
                if text_ch_id:
                    ch = guild.get_channel(text_ch_id)
                    if ch:
                        try:
                            msg = await ch.fetch_message(ctrl["message_id"])
                            await msg.delete()
                        except Exception:
                            pass
            del data["voice_control_messages"][uid]
    save_data(data)

async def cleanup_orphaned_channels(guild: discord.Guild):
    # Reconciliar (no borrar a lo bestia): elimina canales vacíos y SOLO purga los
    # datos de canales que ya no existen. Así el dueño/modo de un canal vivo
    # sobrevive a un reinicio y los comandos de voz siguen funcionando.
    existing_ids = set()
    for channel in guild.voice_channels:
        if channel.name.startswith("⛧︲") and channel.id != VOICE_CREATOR_ID:
            if len([m for m in channel.members if not m.bot]) == 0:
                try:
                    await channel.delete()
                    print(f"[CLEANUP] Canal vacío borrado: {channel.name}")
                except Exception as e:
                    print(f"[ERROR] Borrando huérfano: {e}")
            else:
                existing_ids.add(str(channel.id))

    # Purgar de los diccionarios SOLO lo que ya no existe
    for key in ("voice_channel_owners", "channel_modes", "crystal_permits", "member_join_times"):
        for cid in list(data.get(key, {}).keys()):
            if cid not in existing_ids:
                data[key].pop(cid, None)
    for uid, cid in list(data.get("active_voice_channels", {}).items()):
        if str(cid) not in existing_ids:
            data["active_voice_channels"].pop(uid, None)
    for uid in list(data.get("voice_control_messages", {}).keys()):
        ctrl = data["voice_control_messages"][uid]
        if str(ctrl.get("voice_channel_id")) not in existing_ids:
            # BORRAR el mensaje de Discord, no solo el registro: si no, queda un panel
            # "CONTROL DE CANAL DE VOZ" huérfano en el self con botones muertos (= bugeado).
            await delete_control_message(uid, guild)
    # Las tareas temporales (transferencia de dueño) no sobreviven un reinicio.
    # Borra sus mensajes de Discord ANTES de limpiar el registro, si no quedan zombis.
    for task_id, t in list(data.get("owner_left_tasks", {}).items()):
        try:
            tch = guild.get_channel(t.get("temp_message_channel_id"))
            if tch:
                msg = await tch.fetch_message(t.get("temp_message_id"))
                await msg.delete()
        except Exception:
            pass
    data["owner_left_tasks"] = {}
    save_data(data)
    # Barrido extra: elimina mensajes "OWNER DESCONECTADO" que se quedaron huérfanos
    # en reinicios anteriores (sin registro que los borrara).
    await purge_orphan_transfer_messages(guild)
    print(f"[CLEANUP] Estado de voz reconciliado. Canales vivos: {len(existing_ids)}")


async def purge_orphan_transfer_messages(guild: discord.Guild):
    """Borra mensajes huérfanos de '⚠️ OWNER DESCONECTADO' en los canales ⛧self.
    Esos mensajes se quedaban clavados si el bot se reiniciaba con una transferencia
    pendiente (la tarea moría y nadie los borraba)."""
    borrados = 0
    for _, ud in list(data.get("user_channels", {}).items()):
        ch_id = ud["channel_id"] if isinstance(ud, dict) else ud
        ch = guild.get_channel(ch_id)
        if not ch:
            continue
        try:
            async for msg in ch.history(limit=30):
                if msg.author.id != guild.me.id or not msg.embeds:
                    continue
                title = (msg.embeds[0].title or "")
                if "OWNER DESCONECTADO" in title:
                    try:
                        await msg.delete()
                        borrados += 1
                    except Exception:
                        pass
        except Exception:
            pass
    if borrados:
        print(f"[CLEANUP] Mensajes 'OWNER DESCONECTADO' huérfanos borrados: {borrados}")


async def purge_orphan_voice_panels(guild: discord.Guild):
    """Borra paneles 'CONTROL DE CANAL DE VOZ' huérfanos en los ⛧self: los que quedaron
    con botones muertos tras reinicios o tras borrarse su canal (bug histórico).
    Respeta los paneles VIVOS (los que están en voice_control_messages)."""
    alive = {str(c.get("message_id")) for c in data.get("voice_control_messages", {}).values()}
    borrados = 0
    for _, ud in list(data.get("user_channels", {}).items()):
        ch_id = ud["channel_id"] if isinstance(ud, dict) else ud
        ch = guild.get_channel(ch_id)
        if not ch:
            continue
        try:
            async for msg in ch.history(limit=40):
                if msg.author.id != guild.me.id or not msg.embeds:
                    continue
                title = (msg.embeds[0].title or "")
                if "CONTROL DE CANAL DE VOZ" in title and str(msg.id) not in alive:
                    try:
                        await msg.delete()
                        borrados += 1
                    except Exception:
                        pass
        except Exception:
            pass
    if borrados:
        print(f"[CLEANUP] Paneles de voz huérfanos (botones muertos) borrados: {borrados}", flush=True)


async def restore_voice_panels(guild: discord.Guild):
    """Tras un reinicio, las views (botones) de los paneles de voz mueren aunque el
    mensaje siga ahí. Recreamos el panel de cada canal vivo cuyo dueño esté dentro,
    para que los botones (modo/kick/renombrar) vuelvan a funcionar."""
    restaurados = 0
    for cid_str, owner_id in list(data.get("voice_channel_owners", {}).items()):
        try:
            ch = guild.get_channel(int(cid_str))
            owner = guild.get_member(owner_id)
            if ch and owner and ch.name.startswith("⛧︲") and owner in ch.members:
                await create_voice_control_panel(owner, ch, guild)
                restaurados += 1
                await asyncio.sleep(0.3)
        except Exception as e:
            print(f"[CLEANUP] Error restaurando panel {cid_str}: {e}")
    if restaurados:
        print(f"[CLEANUP] Paneles de voz restaurados (botones revividos): {restaurados}")

# =====================================================================
#  CREACIÓN DE CANALES
# =====================================================================
async def create_self_channel(member: discord.Member, guild: discord.Guild) -> discord.TextChannel:
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(read_messages=False),
        member: discord.PermissionOverwrite(read_messages=True, send_messages=False),
        guild.me: discord.PermissionOverwrite(
            read_messages=True, send_messages=True,
            embed_links=True, manage_messages=True
        ),
    }
    channel = await guild.create_text_channel("⛧︲self", overwrites=overwrites)
    data["user_channels"][str(member.id)] = {"channel_id": channel.id, "color_msg_id": None}
    data["ghost_mode"][str(member.id)] = False
    save_data(data)

    await send_identity_panel(member, channel, guild)
    await send_vault_panel(member, channel)
    await send_clip_panel(member, channel)
    await send_activity_panel(member, channel)

    return channel

async def send_rules_embed(guild: discord.Guild):
    """Crea o actualiza el embed de normas en el canal #✟"""
    channel = guild.get_channel(RULES_CHANNEL_ID)
    if not channel:
        print("[NORMAS] Canal no encontrado", flush=True)
        return
    
    # Buscar mensaje existente de normas
    rules_msg_id = data.get("rules_message_id")
    existing_msg = None
    if rules_msg_id:
        try:
            existing_msg = await channel.fetch_message(rules_msg_id)
        except Exception:
            pass
    
    # Crear embed dark/clean
    embed = discord.Embed(
        title="⛧ CHEPA 3.0 ⛧",
        description="**ACCESO RESTRINGIDO**",
        color=0x1a1a2e
    )
    
    embed.add_field(
        name="─────────────────────",
        value=(
            "Al permanecer en este servidor, aceptas:\n\n"
            "**CONTENIDO EXPLICITO (+18)**\n"
            "Material sensible, explicito y no apto para menores.\n\n"
            "**PROHIBIDO REPORTAR**\n"
            "La actividad interna no se externaliza. Lo que ocurre aqui, se queda aqui.\n\n"
            "**MULTICUENTAS BANEADAS**\n"
            "Una cuenta por persona. Segundas cuentas = expulsion.\n\n"
            "**SIN PRIVACIDAD GARANTIZADA**\n"
            "Este es un espacio libertario. No hay expectativas de confidencialidad.\n\n"
            "**AUTOMODERACION ACTIVA**\n"
            "El bot tiene autoridad para restringir acceso sin previo aviso."
        ),
        inline=False
    )
    
    embed.add_field(
        name="─────────────────────",
        value=f"Reacciona con {RULES_ACCEPT_EMOJI} para aceptar y obtener acceso",
        inline=False
    )
    
    embed.set_image(url=RULES_IMAGE_URL)
    embed.set_footer(text="El desconocimiento de las reglas no exime de su cumplimiento")
    
    if existing_msg:
        await existing_msg.edit(embed=embed)
        msg = existing_msg
    else:
        msg = await channel.send(embed=embed)
        data["rules_message_id"] = msg.id
        save_data(data)
    
    # Añadir reacción automática
    try:
        await msg.add_reaction(RULES_ACCEPT_EMOJI)
    except Exception as e:
        print(f"[NORMAS] Error añadiendo reacción: {e}", flush=True)
    
    print("[NORMAS] Embed de normas enviado/actualizado", flush=True)

async def send_identity_panel(member: discord.Member, channel: discord.TextChannel, guild: discord.Guild):
    current_emoji = None
    for emoji, role_id in COLOR_ROLES.items():
        role = guild.get_role(role_id)
        if role and role in member.roles:
            current_emoji = emoji
            break

    embed = discord.Embed(
        title="SELECCIÓN DE IDENTIDAD",
        description="Reacciona con el símbolo de tu esencia.\nSolo puedes portar una identidad a la vez.",
        color=0x1a1a2e
    )
    val = f"```{current_emoji} ACTIVO```" if current_emoji else "```SIN IDENTIDAD ASIGNADA```"
    embed.add_field(name="IDENTIDAD ACTUAL", value=val, inline=False)
    legend = [f"{e}  {ROLE_NAMES.get(e, '')}" for e in COLOR_ROLES if COLOR_ROLES[e]]
    embed.add_field(name="DISPONIBLES", value="\n".join(legend) if legend else "—", inline=False)
    embed.set_footer(text="Reacciona para cambiar · Se actualiza solo")

    # Buscar panel existente por ID guardado o por título
    uid = str(member.id)
    existing_msg = None
    saved_mid = data.get("user_channels", {}).get(uid, {}).get("color_msg_id") if isinstance(data.get("user_channels", {}).get(uid), dict) else None
    if saved_mid:
        try:
            existing_msg = await channel.fetch_message(saved_mid)
        except Exception:
            existing_msg = None
    if not existing_msg:
        existing_msg = await _find_bot_message_by_title(channel, "SELECCIÓN DE IDENTIDAD")

    if existing_msg:
        await existing_msg.edit(embed=embed)
        msg = existing_msg
    else:
        msg = await channel.send(embed=embed)
        for emoji in COLOR_ROLES:
            try:
                await msg.add_reaction(emoji)
            except Exception:
                pass

    data["user_channels"][str(member.id)]["color_msg_id"] = msg.id
    save_data(data)

async def _find_bot_message_by_title(channel: discord.TextChannel, title: str) -> discord.Message | None:
    """Busca rápidamente un embed del bot con título específico (case-insensitive)."""
    title_lower = title.lower()
    try:
        async for msg in channel.history(limit=30):
            if msg.author == bot.user and msg.embeds and msg.embeds[0].title and msg.embeds[0].title.lower() == title_lower:
                return msg
    except Exception:
        pass
    return None


async def send_vault_panel(member: discord.Member, channel: discord.TextChannel):
    uid = str(member.id)
    vault_entries = data.get("vault", {}).get(uid, [])
    embed = discord.Embed(
        title="CHEPA'S VAULT",
        description=(
            "Guarda tus cuentas y links de forma privada.\n"
            f"Solo tú puedes ver este canal.\n\n"
            f"**{len(vault_entries)}** entradas guardadas"
        ),
        color=0x1a1a2e
    )
    embed.set_footer(text="Tus datos solo son visibles aquí")
    view = VaultMainView(member)

    # 1) Intentar editar por ID conocido
    existing_id = data.get("vault_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await channel.fetch_message(existing_id)
            await msg.edit(embed=embed, view=view)
            return
        except (discord.NotFound, discord.HTTPException):
            data.get("vault_msg_ids", {}).pop(uid, None)
            save_data(data)

    # 2) Fallback: buscar en historial
    found = await _find_bot_message_by_title(channel, "CHEPA'S VAULT")
    if found:
        await found.edit(embed=embed, view=view)
        data.setdefault("vault_msg_ids", {})[uid] = found.id
        save_data(data)
        return

    # 3) Crear nuevo
    msg = await channel.send(embed=embed, view=view)
    data.setdefault("vault_msg_ids", {})[uid] = msg.id
    save_data(data)


async def send_activity_panel(member: discord.Member, channel: discord.TextChannel):
    uid = str(member.id)
    week = get_week_key()
    act = get_activity(uid)
    embed = discord.Embed(
        title="ACTIVIDAD SEMANAL",
        description=f"Semana {week}",
        color=0x1a1a2e
    )
    embed.set_image(url="https://images.guns.lol/5a3415a4ffbed3551ecf589da3452df1e3f682dc/rYY7lr.gif")
    embed.add_field(name="Mensajes enviados", value=str(act["messages"]), inline=True)
    embed.add_field(name="Tiempo en llamada", value=format_time(act["voice_seconds"]), inline=True)
    embed.set_footer(text="Se resetea cada lunes · Leaderboard cada domingo a medianoche")

    # 1) Intentar editar por ID conocido
    existing_id = data.get("activity_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await channel.fetch_message(existing_id)
            await msg.edit(embed=embed)
            return
        except (discord.NotFound, discord.HTTPException):
            data.get("activity_msg_ids", {}).pop(uid, None)
            save_data(data)

    # 2) Fallback: buscar en historial
    found = await _find_bot_message_by_title(channel, "ACTIVIDAD SEMANAL")
    if found:
        await found.edit(embed=embed)
        data.setdefault("activity_msg_ids", {})[uid] = found.id
        save_data(data)
        return

    # 3) Crear nuevo
    msg = await channel.send(embed=embed)
    data.setdefault("activity_msg_ids", {})[uid] = msg.id
    save_data(data)

async def refresh_self_panels(member: discord.Member, guild: discord.Guild):
    """Refresca todos los paneles del self de un usuario existente."""
    uid = str(member.id)
    ch_id = get_user_text_channel_id(uid)
    if not ch_id:
        return
    ch = guild.get_channel(ch_id)
    if not ch:
        return

    # Borrar mensajes anteriores del bot excepto el de identidad (que tiene reacciones)
    # y limpiar el tracking de IDs si borramos los paneles
    activity_id = data.get("activity_msg_ids", {}).get(uid)
    vault_id = data.get("vault_msg_ids", {}).get(uid)
    ids_cleared = False
    try:
        async for msg in ch.history(limit=50):
            if msg.author == bot.user and msg.id != data["user_channels"][uid].get("color_msg_id"):
                try:
                    await msg.delete()
                    await asyncio.sleep(0.3)
                    if activity_id and msg.id == activity_id:
                        data.get("activity_msg_ids", {}).pop(uid, None)
                        ids_cleared = True
                    if vault_id and msg.id == vault_id:
                        data.get("vault_msg_ids", {}).pop(uid, None)
                        ids_cleared = True
                except Exception:
                    pass
    except Exception:
        pass
    if ids_cleared:
        save_data(data)

    await send_vault_panel(member, ch)
    await send_activity_panel(member, ch)
    await send_clip_panel(member, ch)

# =====================================================================
#  PANEL DE VOZ
# =====================================================================
_panel_locks: dict[str, asyncio.Lock] = {}

async def create_voice_control_panel(member: discord.Member,
                                     voice_channel: discord.VoiceChannel,
                                     guild: discord.Guild):
    """Crea (o reemplaza) el panel de control de voz — FIX: elimina el anterior siempre.
    Con lock por usuario: dos creaciones concurrentes dejaban un panel huérfano
    (ambas borraban, ambas enviaban, solo la última quedaba registrada)."""
    user_id = str(member.id)
    if user_id not in _panel_locks:
        _panel_locks[user_id] = asyncio.Lock()

    async with _panel_locks[user_id]:
        # SIEMPRE borrar el anterior antes de crear uno nuevo
        await delete_control_message(member.id, guild)

        text_ch_id = get_user_text_channel_id(user_id)
        if not text_ch_id:
            return
        ch = guild.get_channel(text_ch_id)
        if not ch:
            return

        current_mode = data.get("channel_modes", {}).get(str(voice_channel.id), "public")
        color_map = {"public": 0x2ecc71, "ghost": 0x4a4a4a, "crystal": 0x4a0080}
        mode_text = {"public": "PÚBLICO", "ghost": "FANTASMA", "crystal": "CRISTAL"}

        embed = discord.Embed(
            title="CONTROL DE CANAL DE VOZ",
            description=(
                f"**{voice_channel.name}** — {voice_channel.mention}\n\n"
                f"Modo actual: **{mode_text.get(current_mode, 'PÚBLICO')}**"
            ),
            color=color_map.get(current_mode, 0x2ecc71)
        )
        embed.set_footer(text="El panel desaparece al salir del canal")

        view = VoiceControlView(member, voice_channel, mode=current_mode)
        control_msg = await ch.send(embed=embed, view=view)

        data["voice_control_messages"][user_id] = {
            "message_id": control_msg.id,
            "voice_channel_id": voice_channel.id,
        }
        save_data(data)

async def handle_owner_left(voice_channel: discord.VoiceChannel, old_owner_id: int, guild: discord.Guild):
    old_owner = guild.get_member(old_owner_id)
    await delete_control_message(old_owner_id, guild)
    remaining = [m for m in voice_channel.members if m.id != old_owner_id and not m.bot]

    if not remaining:
        return

    text_ch_id = get_user_text_channel_id(str(old_owner_id))
    if not text_ch_id:
        new_owner = remaining[0]
        data["voice_channel_owners"][str(voice_channel.id)] = new_owner.id
        save_data(data)
        await create_voice_control_panel(new_owner, voice_channel, guild)
        return

    ch = guild.get_channel(text_ch_id)
    if not ch:
        new_owner = remaining[0]
        data["voice_channel_owners"][str(voice_channel.id)] = new_owner.id
        save_data(data)
        await create_voice_control_panel(new_owner, voice_channel, guild)
        return

    embed = discord.Embed(
        title="⚠️ OWNER DESCONECTADO",
        description=(
            f"**{old_owner.display_name if old_owner else 'Desconocido'}** "
            f"ha abandonado **{voice_channel.name}**\n\n"
            "Tienes **5 minutos** para elegir un nuevo administrador.\n"
            "Si no eliges, se asignará automáticamente."
        ),
        color=0xFF6B35,
        timestamp=datetime.now()
    )
    view = OwnerTransferView(old_owner, voice_channel, remaining)
    temp_msg = await ch.send(embed=embed, view=view)

    task_id = f"{voice_channel.id}_{old_owner_id}"
    data["owner_left_tasks"][task_id] = {
        "channel_id": voice_channel.id,
        "old_owner_id": old_owner_id,
        "temp_message_id": temp_msg.id,
        "temp_message_channel_id": ch.id,
        "timestamp": datetime.now().isoformat(),
    }
    save_data(data)
    asyncio.create_task(
        handle_owner_timeout(task_id, voice_channel, old_owner_id, guild, temp_msg)
    )

async def handle_owner_timeout(task_id, voice_channel, old_owner_id, guild, temp_msg):
    await asyncio.sleep(300)
    if task_id not in data.get("owner_left_tasks", {}):
        return
    voice_channel = guild.get_channel(voice_channel.id)
    if not voice_channel:
        data["owner_left_tasks"].pop(task_id, None)
        save_data(data)
        return

    old_owner = guild.get_member(old_owner_id)
    if old_owner and old_owner in voice_channel.members:
        data["voice_channel_owners"][str(voice_channel.id)] = old_owner_id
        data["owner_left_tasks"].pop(task_id, None)
        save_data(data)
        await create_voice_control_panel(old_owner, voice_channel, guild)
        try:
            await temp_msg.delete()
        except Exception:
            pass
        return

    remaining = [m for m in voice_channel.members if m.id != old_owner_id and not m.bot]
    if not remaining:
        try:
            await voice_channel.delete()
        except Exception:
            pass
        data["owner_left_tasks"].pop(task_id, None)
        await cleanup_voice_data(voice_channel.id, guild)
    else:
        new_owner = remaining[0]
        data["voice_channel_owners"][str(voice_channel.id)] = new_owner.id
        data["owner_left_tasks"].pop(task_id, None)
        save_data(data)
        await create_voice_control_panel(new_owner, voice_channel, guild)

    try:
        await temp_msg.delete()
    except Exception:
        pass

# =====================================================================
#  VISTAS — VOICE CONTROL
# =====================================================================
class VoiceControlView(discord.ui.View):
    def __init__(self, owner: discord.Member, voice_channel: discord.VoiceChannel, mode: str = "public"):
        super().__init__(timeout=None)
        self.owner = owner
        self.voice_channel = voice_channel
        self.mode = mode
        self._rebuild_items()

    def _rebuild_items(self):
        self.clear_items()
        style_map = {
            "public":  (discord.ButtonStyle.green,     "🌍", "PÚBLICO"),
            "ghost":   (discord.ButtonStyle.secondary, "👻", "FANTASMA"),
            "crystal": (discord.ButtonStyle.primary,   "🔮", "CRISTAL"),
        }
        style, emoji, label = style_map.get(self.mode, style_map["public"])
        mode_btn = discord.ui.Button(label=label, emoji=emoji, style=style, custom_id="vc_mode_btn", row=0)
        mode_btn.callback = self.cycle_mode
        self.add_item(mode_btn)

        kick_btn = discord.ui.Button(label="Kick", emoji="💣", style=discord.ButtonStyle.danger, custom_id="vc_kick_btn", row=0)
        kick_btn.callback = self.kick_user
        self.add_item(kick_btn)

        rename_btn = discord.ui.Button(label="Renombrar", emoji="🧿", style=discord.ButtonStyle.secondary, custom_id="vc_rename_btn", row=0)
        rename_btn.callback = self.change_name
        self.add_item(rename_btn)

        _mstate = data.get("music_panel", {}).get("state", "off")
        _mstyle = {"off": discord.ButtonStyle.secondary, "public": discord.ButtonStyle.primary, "private": discord.ButtonStyle.success}.get(_mstate, discord.ButtonStyle.secondary)
        _memoji = {"off": "🎧", "public": "🎶", "private": "🔒"}.get(_mstate, "🎧")
        _mlabel = {"off": "Música", "public": "Música", "private": "Cabina"}.get(_mstate, "Música")
        music_btn = discord.ui.Button(label=_mlabel, emoji=_memoji,
            style=_mstyle,
            custom_id="vc_music_btn", row=0)
        music_btn.callback = self.toggle_music
        self.add_item(music_btn)

        if self.mode == "crystal":
            self.add_item(CrystalAccessSelect(self.voice_channel))

    async def toggle_music(self, interaction: discord.Interaction):
        if not await self._check_owner(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        vch = self.voice_channel
        info = data.get("music_panel", {})
        cur = info.get("state", "off")
        if cur == "off":
            sess = voice_sessions.get(guild.id)
            if not (sess and sess.get("vc") and sess["vc"].is_connected()):
                owner_m = guild.get_member(self.owner.id)
                if owner_m and owner_m.voice and owner_m.voice.channel and owner_m.voice.channel.id == vch.id:
                    await join_voice(owner_m)
                    sess = voice_sessions.get(guild.id)
                if not (sess and sess.get("vc") and sess["vc"].is_connected()):
                    return await interaction.followup.send(
                        "Métete tú al canal primero y vuelve a darle (entro contigo).", ephemeral=True)
            data["music_panel"] = {"state": "public", "on": True}
            save_data(data)
            await interaction.followup.send("Música pública — todos pueden añadir canciones.", ephemeral=True)
            await send_music_panel(guild, vch)
        elif cur == "public":
            data["music_panel"]["state"] = "private"
            save_data(data)
            await interaction.followup.send("🔒 Modo cabina — solo tú (el admin del canal) controla la música.", ephemeral=True)
            await _refresh_music_panel(guild)
        else:
            await stop_music(guild)
            if info.get("msg_id") and info.get("ch_id"):
                await _rest_panel_delete(info["ch_id"], info["msg_id"])
            data["music_panel"] = {"state": "off", "on": False}
            save_data(data)
            await interaction.followup.send("Música apagada.", ephemeral=True)
        try:
            await interaction.message.edit(view=VoiceControlView(self.owner, self.voice_channel, mode=self.mode))
        except Exception:
            pass

    async def _check_owner(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner.id:
            await interaction.response.send_message("❌ No eres el dueño.", ephemeral=True)
            return False
        return True

    async def cycle_mode(self, interaction: discord.Interaction):
        if not await self._check_owner(interaction):
            return
        cycle = {"public": "ghost", "ghost": "crystal", "crystal": "public"}
        self.mode = cycle.get(self.mode, "public")
        cid = str(self.voice_channel.id)

        if self.mode in ("ghost", "crystal"):
            current_ids = [m.id for m in self.voice_channel.members if m.id != self.owner.id]
            existing = data.get("crystal_permits", {}).get(cid, [])
            data.setdefault("crystal_permits", {})[cid] = list(set(existing + current_ids))
        elif self.mode == "public":
            data.get("crystal_permits", {}).pop(cid, None)

        data.setdefault("channel_modes", {})[cid] = self.mode
        save_data(data)

        allowed = data.get("crystal_permits", {}).get(cid, [])
        await update_channel_permissions(self.voice_channel, self.mode, allowed, interaction.guild)
        self._rebuild_items()

        color_map  = {"public": 0x2ecc71, "ghost": 0x4a4a4a, "crystal": 0x4a0080}
        mode_text  = {"public": "PÚBLICO", "ghost": "FANTASMA", "crystal": "CRISTAL"}
        embed = interaction.message.embeds[0] if interaction.message.embeds else discord.Embed()
        embed.color = color_map.get(self.mode, 0x2ecc71)
        embed.description = (
            f"**{self.voice_channel.name}** — {self.voice_channel.mention}\n\n"
            f"Modo actual: **{mode_text.get(self.mode, 'PÚBLICO')}**"
        )
        await interaction.response.edit_message(embed=embed, view=self)

    async def kick_user(self, interaction: discord.Interaction):
        if not await self._check_owner(interaction):
            return
        members = [m for m in self.voice_channel.members if m.id != self.owner.id and not m.bot]
        if not members:
            return await interaction.response.send_message("No hay nadie para kickear.", ephemeral=True)
        view = KickSelectView(self.owner, self.voice_channel, members)
        await interaction.response.send_message("Selecciona a la víctima:", view=view, ephemeral=True)

    async def change_name(self, interaction: discord.Interaction):
        if not await self._check_owner(interaction):
            return
        await interaction.response.send_modal(ChangeNameModal(self.owner, self.voice_channel))


class CrystalAccessSelect(discord.ui.UserSelect):
    def __init__(self, voice_channel: discord.VoiceChannel):
        super().__init__(placeholder="🔮 Gestionar Acceso Cristal", min_values=1, max_values=10, row=1)
        self.voice_channel = voice_channel

    async def callback(self, interaction: discord.Interaction):
        cid = str(self.voice_channel.id)
        data.setdefault("crystal_permits", {})
        current = data["crystal_permits"].get(cid, [])
        changes = []
        for member in self.values:
            if member.id in current:
                current.remove(member.id)
                changes.append(f"⛔ {member.display_name} (Revocado)")
            else:
                current.append(member.id)
                changes.append(f"✅ {member.display_name} (Permitido)")
        data["crystal_permits"][cid] = current
        save_data(data)
        mode = data.get("channel_modes", {}).get(cid, "crystal")
        await update_channel_permissions(self.voice_channel, mode, current, interaction.guild)
        await interaction.response.send_message(
            "**Permisos Cristal:**\n" + "\n".join(changes), ephemeral=True
        )


class KickSelectView(discord.ui.View):
    def __init__(self, owner, voice_channel, members):
        super().__init__(timeout=60)
        self.add_item(KickSelect(owner, voice_channel, members))

class KickSelect(discord.ui.Select):
    def __init__(self, owner, voice_channel, members):
        options = [discord.SelectOption(label=m.display_name, value=str(m.id)) for m in members[:25]]
        super().__init__(placeholder="Selecciona usuario", options=options)
        self.owner = owner
        self.voice_channel = voice_channel

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner.id:
            return await interaction.response.send_message("❌ No eres el dueño.", ephemeral=True)
        target = interaction.guild.get_member(int(self.values[0]))
        if target and target.voice and target.voice.channel == self.voice_channel:
            await target.move_to(None)
            await interaction.response.send_message(f"💣 {target.display_name} expulsado.", ephemeral=True)
        else:
            await interaction.response.send_message("El usuario ya no está en el canal.", ephemeral=True)

class ChangeNameModal(discord.ui.Modal, title="Renombrar Canal"):
    name_input = discord.ui.TextInput(label="Nuevo nombre", max_length=50)

    def __init__(self, member, voice_channel):
        super().__init__()
        self.member = member
        self.voice_channel = voice_channel

    async def on_submit(self, interaction: discord.Interaction):
        new_name = f"⛧︲{self.name_input.value}"
        try:
            await self.voice_channel.edit(name=new_name)
            await interaction.response.send_message(f"✅ Renombrado a **{new_name}**", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

class OwnerTransferView(discord.ui.View):
    def __init__(self, old_owner, voice_channel, members):
        super().__init__(timeout=300)
        self.old_owner = old_owner
        self.voice_channel = voice_channel
        if members:
            self.add_item(NewOwnerSelect(old_owner, voice_channel, members))

class NewOwnerSelect(discord.ui.Select):
    def __init__(self, old_owner, voice_channel, members):
        self.old_owner = old_owner
        self.voice_channel = voice_channel
        options = [discord.SelectOption(label=m.display_name, value=str(m.id)) for m in members[:25]]
        super().__init__(placeholder="Selecciona nuevo admin...", options=options)

    async def callback(self, interaction: discord.Interaction):
        if self.old_owner and interaction.user.id != self.old_owner.id:
            return await interaction.response.send_message("❌ Solo el antiguo owner puede transferir.", ephemeral=True)
        new_id = int(self.values[0])
        new_owner = interaction.guild.get_member(new_id)
        if not new_owner or new_owner not in self.voice_channel.members:
            return await interaction.response.send_message("❌ El usuario ya no está en el canal.", ephemeral=True)
        data["voice_channel_owners"][str(self.voice_channel.id)] = new_id
        task = f"{self.voice_channel.id}_{self.old_owner.id}" if self.old_owner else None
        if task:
            data["owner_left_tasks"].pop(task, None)
        save_data(data)
        await create_voice_control_panel(new_owner, self.voice_channel, interaction.guild)
        try:
            await interaction.message.delete()
        except Exception:
            pass
        await interaction.response.send_message(f"✅ Admin transferido a **{new_owner.display_name}**", ephemeral=True)

# =====================================================================
#  SISTEMA DE CLIPS — DESACTIVADO TEMPORALMENTE
# =====================================================================

from collections import deque

async def send_clip_panel(member, channel):
    pass  # desactivado

async def _join_and_record(channel):
    pass  # desactivado

async def _leave_recording(guild):
    pass  # desactivado

async def _maybe_move_recording(left_channel, guild):
    pass  # desactivado

class RollingAudioSink:
    """Buffer rotatorio PCM: guarda los últimos CLIP_BUFFER_SECONDS segundos por usuario."""

    def __init__(self):
        self._buffers: dict = {}   # uid (int) -> deque[(monotonic_ts, pcm_bytes)]

    # discord-ext-voice-recv llama write(user, data) — data es VoiceData con .pcm
    def write(self, user, data):
        now = time.monotonic()
        uid = user.id if user else 0
        if uid not in self._buffers:
            self._buffers[uid] = deque()
        try:
            pcm = bytes(data.pcm)
        except Exception:
            return
        self._buffers[uid].append((now, pcm))
        cutoff = now - CLIP_BUFFER_SECONDS
        buf = self._buffers[uid]
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def wants_opus(self) -> bool:
        return False  # queremos PCM ya decodificado

    def cleanup(self):
        self._buffers.clear()

    def get_clip_pcm(self, seconds: int) -> dict:
        """Devuelve {uid: pcm_bytes} de los últimos `seconds` segundos."""
        cutoff = time.monotonic() - seconds
        result = {}
        for uid, buf in self._buffers.items():
            pcm = b"".join(chunk for ts, chunk in buf if ts >= cutoff)
            if pcm:
                result[uid] = pcm
        return result


async def _join_and_record(channel: discord.VoiceChannel):
    """Bender se une al canal de voz e inicia el rolling buffer."""
    from discord.ext import voice_recv
    guild = channel.guild
    try:
        vc = guild.voice_client
        sink = RollingAudioSink()
        if vc and vc.is_connected():
            if vc.channel and vc.channel.id == channel.id:
                return  # ya está aquí
            await vc.move_to(channel)
        else:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        rolling_sinks[guild.id]  = sink
        current_rec_ch[guild.id] = channel.id
        vc.listen(voice_recv.BasicSink(sink.write))
        print(f"[CLIP] Grabando en #{channel.name}")
    except Exception as e:
        print(f"[CLIP] Error al unirse a {channel.name}: {e}")


async def _leave_recording(guild: discord.Guild):
    """Bender para la grabación y se desconecta."""
    vc = guild.voice_client
    if vc:
        try:
            vc.stop_listening()
        except Exception:
            pass
        try:
            await vc.disconnect(force=True)
        except Exception:
            pass
    rolling_sinks.pop(guild.id, None)
    current_rec_ch.pop(guild.id, None)
    print(f"[CLIP] Dejó de grabar en {guild.name}")


async def _maybe_move_recording(left_channel: discord.VoiceChannel, guild: discord.Guild):
    """Cuando un canal queda vacío, mueve la grabación al siguiente canal activo."""
    if current_rec_ch.get(guild.id) != left_channel.id:
        return
    for ch in guild.voice_channels:
        if ch.id != left_channel.id and ch.name.startswith("⛧︲"):
            humans = [m for m in ch.members if not m.bot]
            if humans:
                await _join_and_record(ch)
                return
    await _leave_recording(guild)


async def _extract_clip(guild_id: int, seconds: int = CLIP_DEFAULT_SECS):
    """Extrae PCM del buffer, lo convierte a OGG con ffmpeg. Devuelve (bytes|None, error_str)."""
    sess = voice_sessions.get(guild_id)
    ring = sess.get("clip_ring") if sess else None
    if not sess or not ring:
        return None, "Bender no está en ninguna llamada ahora mismo."
    BPS = 48000 * 2 * 2  # bytes/seg: 48kHz estéreo 16-bit
    now = time.monotonic()
    start = now - seconds
    total_bytes = int(seconds * BPS)
    pcm_map = {}
    for uid, dq in list(ring.items()):
        chunks = [(ts, c) for ts, c in list(dq) if ts >= start and c]
        if not chunks:
            continue
        # CONCATENAR en orden (no colocar por hora de llegada: los paquetes llegan a
        # ráfagas y se machacaban entre sí -> audio colapsado). Solo metemos silencio
        # en las pausas REALES (>0.15s), no en el micro-jitter entre paquetes.
        track = bytearray()
        lead = int((chunks[0][0] - start) * BPS)
        lead -= lead % 4
        if lead > 0:
            track.extend(b"\x00" * min(lead, total_bytes))
        prev_end = chunks[0][0]
        for ts, c in chunks:
            gap = ts - prev_end
            if gap > 0.15:
                sil = int(gap * BPS)
                sil -= sil % 4
                track.extend(b"\x00" * sil)
            track.extend(c)
            prev_end = ts + (len(c) / BPS)
        if len(track) > total_bytes:
            track = track[:total_bytes]
        if track:
            pcm_map[uid] = bytes(track)
    if not pcm_map:
        return None, "No hay audio en el buffer todavía. Espera unos segundos en la llamada."
    try:
        _d = {u: round(len(b)/(48000*2*2), 1) for u, b in pcm_map.items()}
        print(f"[CLIP] reconstruido: {len(pcm_map)} voz(es), seg/voz={_d}", flush=True)
    except Exception:
        pass

    tmpdir = tempfile.mkdtemp()
    try:
        input_args = []
        for i, pcm in enumerate(pcm_map.values()):
            path = os.path.join(tmpdir, f"u{i}.pcm")
            with open(path, "wb") as f:
                f.write(pcm)
            input_args += ["-f", "s16le", "-ar", "48000", "-ac", "2", "-i", path]

        out_path = os.path.join(tmpdir, "clip.ogg")
        cmd = ["ffmpeg", "-y"] + input_args
        n = len(pcm_map)
        if n > 1:
            cmd += ["-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0,alimiter=limit=0.95"]
        cmd += ["-c:a", "libvorbis", "-q:a", "4", out_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return None, "Error al procesar el audio. Asegúrate de que hay audio en la llamada."

        with open(out_path, "rb") as f:
            return f.read(), ""
    except Exception as e:
        return None, f"Error interno: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _clip_secs(uid) -> int:
    try:
        v = int(data.get("clip_config", {}).get(str(uid), CLIP_DEFAULT_SECS))
    except Exception:
        v = CLIP_DEFAULT_SECS
    return max(5, min(120, v))


async def _do_capture_clip(guild, member, seconds=None):
    """Captura un clip de los últimos `seconds` segundos para `member`. Guarda fichero
    local + entrada en data, lo postea en su ⛧self y refresca el panel. Reutilizable
    desde el botón y desde la VOZ. Devuelve (entry, ogg_bytes, error_str)."""
    uid = str(member.id)
    if seconds is None:
        seconds = _clip_secs(uid)
    ogg, err = await _extract_clip(guild.id, seconds)
    if not ogg:
        return None, None, err
    clip_id = f"{uid}_{int(time.time())}"
    fpath = None
    try:
        os.makedirs("/app/clips", exist_ok=True)
        fpath = f"/app/clips/{clip_id}.ogg"
        with open(fpath, "wb") as f:
            f.write(ogg)
    except Exception as e:
        print(f"[CLIP] No pude guardar fichero: {e}", flush=True)
        fpath = None
    name = f"Clip {len(data.get('clips', {}).get(uid, [])) + 1}"
    ts = _spain_now().isoformat()
    entry = {"id": clip_id, "name": name, "ts": ts, "file": fpath, "secs": seconds}
    data.setdefault("clips", {}).setdefault(uid, []).append(entry)
    save_data(data)
    # Solo refrescar el panel (NO postear el clip: evita saturar el self).
    # El audio se entrega aparte de forma efímera (solo lo ve el dueño).
    try:
        ch_id = get_user_text_channel_id(uid)
        ch = guild.get_channel(ch_id) if ch_id else None
        if ch:
            await send_clip_panel(member, ch)
    except Exception as e:
        print(f"[CLIP] refresh panel error: {e}", flush=True)
    return entry, ogg, ""


_CLIP_VERBS = ("clipea", "clipear", "clipéame", "clipeame", "clípalo", "clipalo",
               "saca clip", "saca un clip", "saca el clip", "haz un clip", "haz clip",
               "corta clip", "corta eso", "graba eso", "guarda eso", "clip de eso",
               "mete clip", "clipa", "clípame", "clipame")


def _is_clip_cmd(t: str) -> bool:
    t = t.lower()
    # Whisper destroza "clipea/saca clip" ('Clip Air', 'Rastaca Clip', 'Cicaciclip'...),
    # pero la palabra "clip" SIEMPRE sobrevive. Como esto va DETRÁS del wake-word Bender,
    # disparar con "clip" es seguro y captura los churros reales.
    if "clip" in t:
        return True
    return any(v in t for v in _CLIP_VERBS)


def _parse_clip_config(t: str):
    """Devuelve segundos si el texto pide configurar la duración del clip, si no None."""
    t = t.lower()
    if "clip" not in t:
        return None
    if not any(w in t for w in ("segundo", "minuto", "config", "dura", "pon el clip",
                                "ponlo a", "ajusta", "pon los clip")):
        return None
    m = re.search(r"(\d+)\s*minuto", t)
    if m:
        return max(5, min(120, int(m.group(1)) * 60))
    if "medio minuto" in t:
        return 30
    if "dos minutos" in t:
        return 120
    if "un minuto" in t or " a minuto" in t:
        return 60
    m = re.search(r"(\d+)\s*segundo", t)
    if m:
        return max(5, min(120, int(m.group(1))))
    return None


async def _voice_make_clip(guild, uid):
    """Captura un clip pedido por VOZ y suelta una coña sobre lo grabado."""
    member = guild.get_member(uid)
    if not member:
        return
    entry, ogg, err = await _do_capture_clip(guild, member)
    if not ogg:
        await _speak(guild, _clean_for_speech("No he podido sacar el clip. " + (err or "")))
        return
    # Solo confirma que ha sacado el clip (sin escucharlo ni comentar).
    await _speak(guild, random.choice([
        "Va, ya os he sacado el clip.",
        "Listo, clip guardado.",
        "Hecho, os he clipeado eso.",
        "Clip sacado, cabrones.",
    ]))


# ── Panel de clips ────────────────────────────────────────────────────

async def send_clip_panel(member: discord.Member, channel: discord.TextChannel):
    uid = str(member.id)
    clips = data.get("clips", {}).get(uid, [])
    embed = _build_clip_embed(clips, _clip_secs(uid))
    view = ClipPanelView(member)

    existing_id = data.get("clip_panel_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await channel.fetch_message(existing_id)
            await msg.edit(embed=embed, view=view)
            return
        except (discord.NotFound, discord.HTTPException):
            data.get("clip_panel_msg_ids", {}).pop(uid, None)

    found = await _find_bot_message_by_title(channel, "CLIPS")
    if found:
        await found.edit(embed=embed, view=view)
        data.setdefault("clip_panel_msg_ids", {})[uid] = found.id
        save_data(data)
        return

    msg = await channel.send(embed=embed, view=view)
    data.setdefault("clip_panel_msg_ids", {})[uid] = msg.id
    save_data(data)


async def _refresh_clip_panel(member: discord.Member, interaction: discord.Interaction):
    uid = str(member.id)
    ch_id = get_user_text_channel_id(uid)
    ch = interaction.guild.get_channel(ch_id) if ch_id else None
    if not ch:
        return
    clips = data.get("clips", {}).get(uid, [])
    embed = _build_clip_embed(clips, _clip_secs(uid))
    view = ClipPanelView(member)
    existing_id = data.get("clip_panel_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await ch.fetch_message(existing_id)
            await msg.edit(embed=embed, view=view)
            return
        except (discord.NotFound, discord.HTTPException):
            pass
    msg = await ch.send(embed=embed, view=view)
    data.setdefault("clip_panel_msg_ids", {})[uid] = msg.id
    save_data(data)


def _build_clip_embed(clips: list, secs: int = None) -> discord.Embed:
    if not clips:
        desc = "Sin clips guardados.\nPulsa Clipear en una llamada, o di a Bender que clipee."
    else:
        lines = []
        for i, c in enumerate(clips[-5:]):  # últimos 5
            ts = c.get("ts", "")[:16].replace("T", " ")
            lines.append(f"`{i+1}.`  {c.get('name', 'Clip')}  —  {ts}")
        desc = "\n".join(lines)
    embed = discord.Embed(title="CLIPS", description=desc, color=0x2B2D31)
    embed.set_footer(text=(f"Duración: {secs}s" if secs else "Duración: 30s"))
    return embed


class ClipPanelView(discord.ui.View):
    def __init__(self, member: discord.Member):
        super().__init__(timeout=None)
        self.member = member
        uid = str(member.id)
        clips = data.get("clips", {}).get(uid, [])

        clip_btn = discord.ui.Button(
            label="Clip", style=discord.ButtonStyle.secondary,
            custom_id=f"clip_capture_{uid}", row=0
        )
        clip_btn.label = "Clipear"
        clip_btn.callback = self._capture
        self.add_item(clip_btn)

        cfg_btn = discord.ui.Button(
            label="Config", style=discord.ButtonStyle.secondary,
            custom_id=f"clip_cfg_{uid}", row=0
        )
        cfg_btn.callback = self._config
        self.add_item(cfg_btn)

        for i, clip in enumerate(clips[-4:]):
            real_idx = max(0, len(clips) - 4) + i
            btn = discord.ui.Button(
                label=clip.get("name", f"Clip {real_idx+1}")[:20],
                style=discord.ButtonStyle.primary,
                custom_id=f"clip_view_{uid}_{real_idx}", row=1 + (i // 4)
            )
            btn.callback = self._make_view_cb(real_idx)
            self.add_item(btn)

    async def _config(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu panel.", ephemeral=True)
        await interaction.response.send_modal(ClipConfigModal(self.member))

    async def _capture(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu panel.", ephemeral=True)
        m = interaction.guild.get_member(self.member.id)
        if not m or not m.voice or not m.voice.channel:
            return await interaction.response.send_message(
                "Tienes que estar en una llamada para clipear.", ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        entry, ogg, err = await _do_capture_clip(interaction.guild, self.member)
        if not ogg:
            return await interaction.followup.send(err, ephemeral=True)
        idx = len(data.get("clips", {}).get(str(self.member.id), [])) - 1
        await interaction.followup.send(
            content=f"Clip guardado: {entry['name']} ({entry['secs']}s). Solo lo ves tú.",
            file=discord.File(io.BytesIO(ogg), filename=f"{entry['name']}.ogg"),
            view=ClipActionView(self.member, ogg, idx),
            ephemeral=True)

    def _make_view_cb(self, idx: int):
        async def callback(interaction: discord.Interaction):
            if interaction.user.id != self.member.id:
                return await interaction.response.send_message("No es tu panel.", ephemeral=True)
            uid   = str(self.member.id)
            clips = data.get("clips", {}).get(uid, [])
            if idx >= len(clips):
                return await interaction.response.send_message("Clip no encontrado.", ephemeral=True)
            clip  = clips[idx]
            embed = discord.Embed(
                title=clip.get('name','Clip'),
                description=clip.get("ts","")[:16].replace("T"," · "),
                color=0x2B2D31
            )
            view = ClipManageView(self.member, idx)
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        return callback


class ClipActionView(discord.ui.View):
    """Aparece en el self tras capturar un clip. Expira en 5 min."""
    def __init__(self, member: discord.Member, ogg: bytes, idx: int):
        super().__init__(timeout=300)
        self.member = member
        self.ogg    = ogg
        self.idx    = idx

        name_btn = discord.ui.Button(label="Nombrar", style=discord.ButtonStyle.secondary)
        name_btn.callback = self._name
        self.add_item(name_btn)

        send_btn = discord.ui.Button(label="Enviar al chat", style=discord.ButtonStyle.primary)
        send_btn.callback = self._send_to_chat
        self.add_item(send_btn)

    async def _name(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        await interaction.response.send_modal(ClipNameModal(self.member, self.idx))

    async def _send_to_chat(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        uid   = str(self.member.id)
        clips = data.get("clips", {}).get(uid, [])
        name  = clips[self.idx].get("name", "Clip") if self.idx < len(clips) else "Clip"

        chat_ch = interaction.guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
        if not chat_ch:
            return await interaction.response.send_message("Canal de chat no encontrado.", ephemeral=True)

        embed = discord.Embed(
            title="⛧︲ CLIP",
            description=f"Clip de **{self.member.display_name}** — {name}",
            color=0x5B2D8E
        )
        embed.set_footer(text="⛧ Chepa 3.0")
        await chat_ch.send(
            embed=embed,
            file=discord.File(io.BytesIO(self.ogg), filename=f"{name}.ogg")
        )
        await interaction.response.send_message("Clip enviado al chat.", ephemeral=True, delete_after=5)

        try:
            await interaction.message.delete()
        except Exception:
            pass


class ClipNameModal(discord.ui.Modal, title="Nombrar clip"):
    name_input = discord.ui.TextInput(label="Nombre del clip", max_length=40)

    def __init__(self, member: discord.Member, idx: int):
        super().__init__()
        self.member = member
        self.idx    = idx

    async def on_submit(self, interaction: discord.Interaction):
        uid   = str(self.member.id)
        clips = data.get("clips", {}).get(uid, [])
        if self.idx < len(clips):
            clips[self.idx]["name"] = self.name_input.value
            save_data(data)
        await interaction.response.send_message(
            f"Clip renombrado a **{self.name_input.value}**.", ephemeral=True, delete_after=5
        )
        await _refresh_clip_panel(self.member, interaction)


class ClipConfigModal(discord.ui.Modal, title="Duración del clip"):
    secs_input = discord.ui.TextInput(label="Segundos a clipear (5-120)", max_length=3)

    def __init__(self, member: discord.Member):
        super().__init__()
        self.member = member
        self.secs_input.default = str(_clip_secs(str(member.id)))

    async def on_submit(self, interaction: discord.Interaction):
        try:
            v = max(5, min(120, int(self.secs_input.value.strip())))
        except Exception:
            return await interaction.response.send_message(
                "Pon un número entre 5 y 120, payaso.", ephemeral=True)
        data.setdefault("clip_config", {})[str(self.member.id)] = v
        save_data(data)
        await interaction.response.send_message(
            f"Hecho: los clips ahora durarán **{v}s**.", ephemeral=True, delete_after=5)
        try:
            await send_clip_panel(self.member, interaction.channel)
        except Exception:
            pass


class ClipManageView(discord.ui.View):
    """Vista de gestión de un clip desde el panel (redownload + rename + delete)."""
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=60)
        self.member = member
        self.idx    = idx

        dl_btn = discord.ui.Button(label="Descargar", style=discord.ButtonStyle.secondary)
        dl_btn.callback = self._download
        self.add_item(dl_btn)

        chat_btn = discord.ui.Button(label="Enviar al chat", style=discord.ButtonStyle.success)
        chat_btn.callback = self._send_chat
        self.add_item(chat_btn)

        ren_btn = discord.ui.Button(label="Renombrar", style=discord.ButtonStyle.primary)
        ren_btn.callback = self._rename
        self.add_item(ren_btn)

        del_btn = discord.ui.Button(label="Borrar", style=discord.ButtonStyle.danger)
        del_btn.callback = self._delete
        self.add_item(del_btn)

    async def _download(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        uid   = str(self.member.id)
        clips = data.get("clips", {}).get(uid, [])
        if self.idx >= len(clips):
            return await interaction.response.send_message("Clip no encontrado.", ephemeral=True)
        clip  = clips[self.idx]
        fpath = clip.get("file")
        if fpath and os.path.exists(fpath):
            await interaction.response.send_message(
                file=discord.File(fpath, filename=f"{clip.get('name','Clip')}.ogg"),
                ephemeral=True)
        else:
            await interaction.response.send_message(
                "Archivo no disponible (clip antiguo o borrado).", ephemeral=True)

    async def _send_chat(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        uid = str(self.member.id)
        clips = data.get("clips", {}).get(uid, [])
        if self.idx >= len(clips):
            return await interaction.response.send_message("Clip no encontrado.", ephemeral=True)
        clip = clips[self.idx]
        fpath = clip.get("file")
        if not fpath or not os.path.exists(fpath):
            return await interaction.response.send_message("Archivo no disponible.", ephemeral=True)
        chat_ch = interaction.guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
        if not chat_ch:
            return await interaction.response.send_message("Canal de chat no encontrado.", ephemeral=True)
        emb = discord.Embed(
            title="⛧︲ CLIP",
            description=f"Clip de **{self.member.display_name}** — {clip.get('name','Clip')}",
            color=0x5B2D8E)
        emb.set_footer(text="⛧ Chepa 3.0")
        await chat_ch.send(embed=emb, file=discord.File(fpath, filename=f"{clip.get('name','Clip')}.ogg"))
        await interaction.response.send_message("Clip enviado al chat.", ephemeral=True, delete_after=5)

    async def _rename(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        await interaction.response.send_modal(ClipNameModal(self.member, self.idx))

    async def _delete(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu clip.", ephemeral=True)
        uid   = str(self.member.id)
        clips = data.get("clips", {}).get(uid, [])
        if self.idx < len(clips):
            removed = clips.pop(self.idx)
            save_data(data)
            try:
                _f = removed.get("file")
                if _f and os.path.exists(_f):
                    os.remove(_f)
            except Exception:
                pass
            await interaction.response.edit_message(
                content=f"**{removed.get('name','Clip')}** borrado.", embed=None, view=None
            )
            await _refresh_clip_panel(self.member, interaction)
        else:
            await interaction.response.edit_message(content="Clip no encontrado.", embed=None, view=None)


# =====================================================================
#  VISTAS — CHEPA'S VAULT (con custom_id persistentes)
# =====================================================================
class VaultMainView(discord.ui.View):
    def __init__(self, member: discord.Member):
        super().__init__(timeout=None)
        self.member = member
        self._build()

    def _build(self):
        self.clear_items()
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])

        add_acc_btn = discord.ui.Button(
            label="+ Cuenta", style=discord.ButtonStyle.success,
            custom_id=f"vault_add_acc_{uid}", row=0
        )
        add_acc_btn.callback = self.add_account
        self.add_item(add_acc_btn)

        add_link_btn = discord.ui.Button(
            label="+ Link", style=discord.ButtonStyle.primary,
            custom_id=f"vault_add_lnk_{uid}", row=0
        )
        add_link_btn.callback = self.add_link
        self.add_item(add_link_btn)

        for i, entry in enumerate(entries[:20]):
            etype = entry.get("type", "account")
            shared_with = entry.get("shared_with", [])
            is_shared_out = len(shared_with) > 0  # yo lo comparto con otros
            is_shared_in = entry.get("shared", False)  # alguien me lo compartió

            if is_shared_in:
                style = discord.ButtonStyle.success  # verde = recibido
                by_name = entry.get("shared_by_name", "")[:8]
                label = f"{entry['title'][:14]} · {by_name}"
            elif is_shared_out:
                style = discord.ButtonStyle.primary  # azul = compartido por mí
                label = f"{entry['title'][:16]} ·"
            else:
                style = discord.ButtonStyle.secondary
                label = entry['title'][:20]

            btn = discord.ui.Button(
                label=label,
                style=style,
                custom_id=f"vault_view_{uid}_{i}",
                row=1 + (i // 5)
            )
            btn.callback = self._make_view_callback(i)
            self.add_item(btn)

    def _make_view_callback(self, idx: int):
        async def callback(interaction: discord.Interaction):
            if interaction.user.id != self.member.id:
                return await interaction.response.send_message("No es tu vault.", ephemeral=True)
            uid = str(self.member.id)
            entries = data.get("vault", {}).get(uid, [])
            if idx >= len(entries):
                return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
            entry = entries[idx]
            etype = entry.get("type", "account")

            if etype == "link":
                embed = discord.Embed(title=f"⛧︲ {entry['title']}", color=0x4a0080)
                embed.add_field(name="URL", value=f"```{entry.get('url', '-')}```", inline=False)
                if entry.get("notes"):
                    embed.add_field(name="Notas", value=f"```{entry['notes']}```", inline=False)
            else:
                embed = discord.Embed(title=f"⛧︲ {entry['title']}", color=0x1a1a2e)
                embed.add_field(name="Usuario / Email", value=f"```{entry.get('user', '-')}```", inline=False)
                embed.add_field(name="Contraseña", value=f"```{entry.get('password', '-')}```", inline=False)
                if entry.get("email"):
                    embed.add_field(name="Email", value=f"```{entry['email']}```", inline=False)
                if entry.get("email_pass"):
                    embed.add_field(name="Pass Email", value=f"```{entry['email_pass']}```", inline=False)

            embed.set_footer(text="Se borra en 60 segundos")
            view = VaultEntryView(self.member, idx)
            await interaction.response.send_message(embed=embed, view=view, delete_after=60)
        return callback

    async def add_account(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        await interaction.response.send_modal(VaultAddAccountModal(self.member))

    async def add_link(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        await interaction.response.send_modal(VaultAddLinkModal(self.member))


class VaultAddAccountModal(discord.ui.Modal, title="Nueva cuenta"):
    title_input    = discord.ui.TextInput(label="Título (ej: FORTNITE)", max_length=30)
    user_input     = discord.ui.TextInput(label="Usuario o Email", max_length=100)
    pass_input     = discord.ui.TextInput(label="Contraseña", max_length=100)
    email_input    = discord.ui.TextInput(label="Email (opcional)", max_length=100, required=False)
    email_pass_inp = discord.ui.TextInput(label="Contraseña email (opcional)", max_length=100, required=False)

    def __init__(self, member: discord.Member):
        super().__init__()
        self.member = member

    async def on_submit(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        data.setdefault("vault", {}).setdefault(uid, []).append({
            "type": "account",
            "title": self.title_input.value,
            "user": self.user_input.value,
            "password": self.pass_input.value,
            "email": self.email_input.value,
            "email_pass": self.email_pass_inp.value,
        })
        save_data(data)
        await interaction.response.defer()
        await _refresh_vault_panel(self.member, interaction)


class VaultAddLinkModal(discord.ui.Modal, title="Nuevo link"):
    title_input = discord.ui.TextInput(label="Título (ej: ENEBA)", max_length=30)
    url_input   = discord.ui.TextInput(label="URL", max_length=200)
    notes_input = discord.ui.TextInput(label="Notas (opcional)", max_length=100, required=False)

    def __init__(self, member: discord.Member):
        super().__init__()
        self.member = member

    async def on_submit(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        data.setdefault("vault", {}).setdefault(uid, []).append({
            "type": "link",
            "title": self.title_input.value,
            "url": self.url_input.value,
            "notes": self.notes_input.value,
        })
        save_data(data)
        await interaction.response.defer()
        await _refresh_vault_panel(self.member, interaction)


async def _refresh_vault_panel(member: discord.Member, interaction: discord.Interaction):
    uid = str(member.id)
    ch_id = get_user_text_channel_id(uid)
    ch = interaction.guild.get_channel(ch_id) if ch_id else None
    if not ch:
        return
    entries = data.get("vault", {}).get(uid, [])
    embed = discord.Embed(
        title="CHEPA'S VAULT",
        description=(
            "Guarda tus cuentas y links de forma privada.\n"
            f"Solo tú puedes ver este canal.\n\n"
            f"**{len(entries)}** entradas guardadas"
        ),
        color=0x1a1a2e
    )
    embed.set_footer(text="Tus datos solo son visibles aquí")
    view = VaultMainView(member)
    existing_id = data.get("vault_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await ch.fetch_message(existing_id)
            await msg.edit(embed=embed, view=view)
            return
        except (discord.NotFound, discord.HTTPException):
            pass
    msg = await ch.send(embed=embed, view=view)
    data.setdefault("vault_msg_ids", {})[uid] = msg.id
    save_data(data)


class VaultEntryView(discord.ui.View):
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=60)
        self.member = member
        self.idx = idx

        uid = str(member.id)
        entries = data.get("vault", {}).get(uid, [])
        is_shared_in = idx < len(entries) and entries[idx].get("shared", False)

        if is_shared_in:
            # Entrada recibida — solo dejar de verla
            leave_btn = discord.ui.Button(label="Dejar de ver", style=discord.ButtonStyle.secondary)
            leave_btn.callback = self.leave_shared
            self.add_item(leave_btn)
        else:
            edit_btn = discord.ui.Button(label="Editar", style=discord.ButtonStyle.primary)
            edit_btn.callback = self.edit_entry
            self.add_item(edit_btn)

            share_btn = discord.ui.Button(label="Compartir", style=discord.ButtonStyle.secondary)
            share_btn.callback = self.share_entry
            self.add_item(share_btn)

            gift_btn = discord.ui.Button(label="Regalar", style=discord.ButtonStyle.success)
            gift_btn.callback = self.gift_entry
            self.add_item(gift_btn)

            del_btn = discord.ui.Button(label="Borrar", style=discord.ButtonStyle.danger)
            del_btn.callback = self.confirm_delete
            self.add_item(del_btn)

    async def leave_shared(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        entry = entries[self.idx]
        owner_id = entry.get("shared_by_id")
        title = entry.get("title", "")

        # Quitar del vault del receptor
        data["vault"][uid] = [e for e in entries if not (e.get("shared") and e.get("title") == title and e.get("shared_by_id") == owner_id)]
        save_data(data)

        # Quitar de shared_with del dueño
        if owner_id:
            owner_entries = data.get("vault", {}).get(owner_id, [])
            for e in owner_entries:
                if e.get("title") == title and uid in e.get("shared_with", []):
                    e["shared_with"].remove(uid)
            save_data(data)
            # Notificar al dueño
            owner_member = interaction.guild.get_member(int(owner_id))
            ch_id = get_user_text_channel_id(owner_id)
            if ch_id:
                ch = interaction.guild.get_channel(ch_id)
                if ch:
                    await ch.send(f"{self.member.display_name} ha dejado de ver tu entrada **{title}**.", delete_after=30)
            # Refrescar vault del dueño
            await _refresh_vault_panel_by_uid(owner_id, interaction.guild)

        try:
            await interaction.message.delete()
        except Exception:
            pass
        await interaction.response.send_message("Ya no ves esta entrada.", ephemeral=True, delete_after=5)
        await _refresh_vault_panel_by_uid(uid, interaction.guild)

    async def edit_entry(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        entry = entries[self.idx]
        if entry.get("type") == "link":
            await interaction.response.send_modal(VaultEditLinkModal(self.member, self.idx))
        else:
            await interaction.response.send_modal(VaultEditAccountModal(self.member, self.idx))

    async def share_entry(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        view = VaultShareManageView(self.member, self.idx)
        await interaction.response.send_message(
            "Gestiona con quién compartes esta entrada:", view=view, ephemeral=True
        )

    async def gift_entry(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        view = VaultGiftChoiceView(self.member, self.idx)
        await interaction.response.send_message(
            "¿Cómo quieres regalarlo?", view=view, ephemeral=True
        )

    async def confirm_delete(self, interaction: discord.Interaction):
        if interaction.user.id != self.member.id:
            return await interaction.response.send_message("No es tu vault.", ephemeral=True)
        view = VaultConfirmDeleteView(self.member, self.idx)
        await interaction.response.send_message(
            "¿Seguro que quieres borrar esta entrada?", view=view, ephemeral=True
        )


# ─── COMPARTIR ───

class VaultShareManageView(discord.ui.View):
    """Muestra con quién está compartida una entrada y permite añadir/quitar."""
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=60)
        self.member = member
        self.idx = idx
        self._build()

    def _build(self):
        self.clear_items()
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return
        shared_with = entries[self.idx].get("shared_with", [])

        add_btn = discord.ui.Button(label="Añadir persona", style=discord.ButtonStyle.success)
        add_btn.callback = self.add_share
        self.add_item(add_btn)

        for i, target_id in enumerate(shared_with[:5]):
            guild = None
            for g in bot.guilds:
                if g.id == GUILD_ID:
                    guild = g
                    break
            member = guild.get_member(int(target_id)) if guild else None
            name = member.display_name if member else f"Usuario {target_id}"
            revoke_btn = discord.ui.Button(
                label=f"Quitar a {name[:15]}",
                style=discord.ButtonStyle.danger,
                row=1 + (i // 3)
            )
            revoke_btn.callback = self._make_revoke_callback(target_id)
            self.add_item(revoke_btn)

    def _make_revoke_callback(self, target_id: str):
        async def callback(interaction: discord.Interaction):
            uid = str(self.member.id)
            entries = data.get("vault", {}).get(uid, [])
            if self.idx < len(entries):
                shared = entries[self.idx].get("shared_with", [])
                if target_id in shared:
                    shared.remove(target_id)
                    entries[self.idx]["shared_with"] = shared
                    save_data(data)
                    # Notificar al receptor que ya no tiene acceso
                    await _notify_share_revoked(self.member, int(target_id), entries[self.idx], interaction.guild)
            self._build()
            await interaction.response.edit_message(content="Acceso revocado.", view=self)
        return callback

    async def add_share(self, interaction: discord.Interaction):
        view = VaultShareUserSelect(self.member, self.idx)
        await interaction.response.send_message("Selecciona a quién compartir:", view=view, ephemeral=True)


class VaultShareUserSelect(discord.ui.View):
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=60)
        self.member = member
        self.idx = idx
        select = discord.ui.UserSelect(placeholder="Selecciona un miembro", min_values=1, max_values=1)
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        target = interaction.data["values"][0]
        target_id = str(target) if isinstance(target, str) else str(interaction.guild.get_member(int(target)).id if interaction.guild.get_member(int(target)) else target)
        # Resolver el miembro
        target_member = None
        for val in interaction.data.get("resolved", {}).get("members", {}).values():
            pass
        # Usar directamente el ID del select
        selected_ids = interaction.data.get("values", [])
        if not selected_ids:
            return await interaction.response.send_message("No seleccionaste a nadie.", ephemeral=True)
        target_id = str(selected_ids[0])
        target_member = interaction.guild.get_member(int(target_id))

        if not target_member or target_member.bot:
            return await interaction.response.send_message("Usuario no válido.", ephemeral=True)
        if target_member.id == self.member.id:
            return await interaction.response.send_message("No puedes compartirte algo a ti mismo.", ephemeral=True)

        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)

        shared = entries[self.idx].setdefault("shared_with", [])
        if target_id in shared:
            return await interaction.response.send_message(f"Ya compartes esto con **{target_member.display_name}**.", ephemeral=True)

        shared.append(target_id)
        save_data(data)

        # Notificar al receptor en su self
        await _notify_share_received(self.member, target_member, entries[self.idx], interaction.guild)
        await interaction.response.send_message(f"Compartido con **{target_member.display_name}**.", ephemeral=True)


async def _notify_share_received(owner: discord.Member, recipient: discord.Member,
                                  entry: dict, guild: discord.Guild):
    """Añade la entrada compartida al vault del receptor como entrada especial y refresca su panel."""
    uid_recv = str(recipient.id)

    # Añadir entrada compartida al vault del receptor (marcada como shared)
    shared_entry = entry.copy()
    shared_entry["shared"] = True
    shared_entry["shared_by_id"] = str(owner.id)
    shared_entry["shared_by_name"] = owner.display_name

    vault_recv = data.setdefault("vault", {}).setdefault(uid_recv, [])
    # No duplicar
    for e in vault_recv:
        if e.get("shared") and e.get("shared_by_id") == str(owner.id) and e.get("title") == entry["title"]:
            return
    vault_recv.append(shared_entry)
    save_data(data)

    # Refrescar vault del receptor
    await _refresh_vault_panel_by_uid(uid_recv, guild)


async def _notify_share_revoked(owner: discord.Member, recipient_id: int,
                                 entry: dict, guild: discord.Guild):
    """Quita la entrada compartida del vault del receptor."""
    uid_recv = str(recipient_id)
    vault_recv = data.get("vault", {}).get(uid_recv, [])
    data["vault"][uid_recv] = [
        e for e in vault_recv
        if not (e.get("shared") and e.get("shared_by_id") == str(owner.id) and e.get("title") == entry["title"])
    ]
    save_data(data)
    await _refresh_vault_panel_by_uid(uid_recv, guild)
    # Notificar al receptor
    ch_id = get_user_text_channel_id(uid_recv)
    if ch_id:
        ch = guild.get_channel(ch_id)
        if ch:
            await ch.send(
                f"{owner.display_name} ha dejado de compartir **{entry['title']}** contigo.",
                delete_after=30
            )


class VaultSharedEntryView(discord.ui.View):
    """Vista del receptor de una entrada compartida — solo puede dejar de verla."""
    def __init__(self, recipient: discord.Member, owner: discord.Member, title: str):
        super().__init__(timeout=None)
        self.recipient = recipient
        self.owner = owner
        self.title = title

        leave_btn = discord.ui.Button(
            label="Dejar de ver",
            style=discord.ButtonStyle.secondary,
            custom_id=f"shared_leave_{owner.id}_{recipient.id}_{title[:10]}"
        )
        leave_btn.callback = self.leave_share
        self.add_item(leave_btn)

    async def leave_share(self, interaction: discord.Interaction):
        if interaction.user.id != self.recipient.id:
            return await interaction.response.send_message("No es tu entrada.", ephemeral=True)

        uid_owner = str(self.owner.id)
        entries = data.get("vault", {}).get(uid_owner, [])
        for entry in entries:
            if entry.get("title") == self.title:
                shared = entry.get("shared_with", [])
                rid = str(self.recipient.id)
                if rid in shared:
                    shared.remove(rid)
                    save_data(data)
                break

        # Notificar al dueño
        ch_id_owner = get_user_text_channel_id(uid_owner)
        if ch_id_owner:
            ch_owner = interaction.guild.get_channel(ch_id_owner)
            if ch_owner:
                await ch_owner.send(
                    f"{self.recipient.display_name} ha dejado de ver tu entrada **{self.title}**.",
                    delete_after=30
                )

        try:
            await interaction.message.delete()
        except Exception:
            pass
        await interaction.response.send_message("Ya no ves esta entrada.", ephemeral=True, delete_after=5)


# ─── REGALAR ───

class VaultGiftChoiceView(discord.ui.View):
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=30)
        self.member = member
        self.idx = idx

        chat_btn = discord.ui.Button(label="Al chat (cualquiera)", style=discord.ButtonStyle.primary)
        chat_btn.callback = self.gift_to_chat
        self.add_item(chat_btn)

        person_btn = discord.ui.Button(label="A alguien concreto", style=discord.ButtonStyle.secondary)
        person_btn.callback = self.gift_to_person
        self.add_item(person_btn)

    async def gift_to_chat(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        entry = entries[self.idx]

        ch = interaction.guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
        if not ch:
            return await interaction.response.send_message("No encontré el canal de chat.", ephemeral=True)

        embed = discord.Embed(
            title=f"Regalo disponible — {entry['title']}",
            description=f"{self.member.display_name} regala esta entrada. El primero que la reclame se la lleva.",
            color=0x2ecc71
        )
        embed.set_footer(text="Solo una persona puede reclamarlo")

        view = VaultGiftClaimView(self.member, self.idx, entry)
        await ch.send(embed=embed, view=view)
        await interaction.response.send_message("Regalo publicado en el chat.", ephemeral=True)

    async def gift_to_person(self, interaction: discord.Interaction):
        view = VaultGiftPersonSelect(self.member, self.idx)
        await interaction.response.send_message("Selecciona a quién regalar:", view=view, ephemeral=True)


class VaultGiftPersonSelect(discord.ui.View):
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=60)
        self.member = member
        self.idx = idx
        select = discord.ui.UserSelect(placeholder="Selecciona un miembro", min_values=1, max_values=1)
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        selected_ids = interaction.data.get("values", [])
        if not selected_ids:
            return await interaction.response.send_message("No seleccionaste a nadie.", ephemeral=True)
        target_id = str(selected_ids[0])
        target_member = interaction.guild.get_member(int(target_id))

        if not target_member or target_member.bot:
            return await interaction.response.send_message("Usuario no válido.", ephemeral=True)
        if target_member.id == self.member.id:
            return await interaction.response.send_message("No puedes regalarte algo a ti mismo.", ephemeral=True)

        # Comprobar límite de 3 regalos pendientes
        pending = data.get("pending_gifts", {}).get(target_id, [])
        if len(pending) >= 3:
            return await interaction.response.send_message(
                f"Lo sentimos, **{target_member.display_name}** tiene el inventario lleno de regalos sin abrir.",
                ephemeral=True
            )

        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx >= len(entries):
            return await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)
        entry = entries[self.idx].copy()

        # Guardar regalo pendiente
        data.setdefault("pending_gifts", {}).setdefault(target_id, []).append({
            "from_id": uid,
            "from_name": self.member.display_name,
            "entry": entry,
            "orig_idx": self.idx,
        })
        save_data(data)

        # Notificar al receptor en su self
        await _send_gift_notification(self.member, target_member, entry, self.idx, interaction.guild)
        await interaction.response.send_message(f"Regalo enviado a **{target_member.display_name}**.", ephemeral=True)


async def _send_gift_notification(sender: discord.Member, recipient: discord.Member,
                                   entry: dict, orig_idx: int, guild: discord.Guild):
    ch_id = get_user_text_channel_id(str(recipient.id))
    if not ch_id:
        return
    ch = guild.get_channel(ch_id)
    if not ch:
        return
    embed = discord.Embed(
        title=f"Tienes un regalo de {sender.display_name}",
        description=f"Te han enviado **{entry['title']}**. ¿Lo quieres?",
        color=0x2ecc71
    )
    embed.set_footer(text="Tienes 48h para reclamarlo o se devuelve al remitente")
    view = VaultGiftReceiveView(sender, recipient, entry, orig_idx)
    await ch.send(embed=embed, view=view)


class VaultGiftClaimView(discord.ui.View):
    """Regalo público en el chat — el primero que pulsa se lo lleva."""
    def __init__(self, sender: discord.Member, orig_idx: int, entry: dict):
        super().__init__(timeout=None)
        self.sender = sender
        self.orig_idx = orig_idx
        self.entry = entry
        self.claimed = False

        claim_btn = discord.ui.Button(
            label="Reclamar",
            style=discord.ButtonStyle.success,
            custom_id=f"gift_claim_{sender.id}_{orig_idx}"
        )
        claim_btn.callback = self.claim
        self.add_item(claim_btn)

    async def claim(self, interaction: discord.Interaction):
        if self.claimed:
            return await interaction.response.send_message("Ya fue reclamado.", ephemeral=True)
        if interaction.user.id == self.sender.id:
            return await interaction.response.send_message("No puedes reclamar tu propio regalo.", ephemeral=True)

        self.claimed = True

        # Quitar del vault del remitente
        uid_sender = str(self.sender.id)
        entries = data.get("vault", {}).get(uid_sender, [])
        if self.orig_idx < len(entries) and entries[self.orig_idx].get("title") == self.entry.get("title"):
            entries.pop(self.orig_idx)
            save_data(data)
            await _refresh_vault_panel_by_uid(uid_sender, interaction.guild)

        # Añadir al vault del receptor
        uid_recv = str(interaction.user.id)
        new_entry = self.entry.copy()
        new_entry.pop("shared_with", None)
        data.setdefault("vault", {}).setdefault(uid_recv, []).append(new_entry)
        save_data(data)
        await _refresh_vault_panel_by_uid(uid_recv, interaction.guild)

        await interaction.response.edit_message(
            content=f"**{interaction.user.display_name}** ha reclamado el regalo.", embed=None, view=None
        )


class VaultGiftReceiveView(discord.ui.View):
    """Notificación de regalo personal — reclamar o rechazar."""
    def __init__(self, sender: discord.Member, recipient: discord.Member, entry: dict, orig_idx: int):
        super().__init__(timeout=None)
        self.sender = sender
        self.recipient = recipient
        self.entry = entry
        self.orig_idx = orig_idx

        claim_btn = discord.ui.Button(
            label="Reclamar",
            style=discord.ButtonStyle.success,
            custom_id=f"gift_recv_claim_{sender.id}_{recipient.id}"
        )
        claim_btn.callback = self.claim
        self.add_item(claim_btn)

        reject_btn = discord.ui.Button(
            label="Rechazar",
            style=discord.ButtonStyle.danger,
            custom_id=f"gift_recv_reject_{sender.id}_{recipient.id}"
        )
        reject_btn.callback = self.reject
        self.add_item(reject_btn)

    async def claim(self, interaction: discord.Interaction):
        if interaction.user.id != self.recipient.id:
            return await interaction.response.send_message("No es tu regalo.", ephemeral=True)

        uid_sender = str(self.sender.id)
        uid_recv = str(self.recipient.id)

        # Quitar del vault del remitente
        entries = data.get("vault", {}).get(uid_sender, [])
        if self.orig_idx < len(entries) and entries[self.orig_idx].get("title") == self.entry.get("title"):
            entries.pop(self.orig_idx)
            await _refresh_vault_panel_by_uid(uid_sender, interaction.guild)

        # Quitar de pending_gifts
        pending = data.get("pending_gifts", {}).get(uid_recv, [])
        data["pending_gifts"][uid_recv] = [
            g for g in pending if not (g["from_id"] == uid_sender and g["entry"].get("title") == self.entry.get("title"))
        ]

        # Añadir al vault del receptor
        new_entry = self.entry.copy()
        new_entry.pop("shared_with", None)
        data.setdefault("vault", {}).setdefault(uid_recv, []).append(new_entry)
        save_data(data)
        await _refresh_vault_panel_by_uid(uid_recv, interaction.guild)

        try:
            await interaction.message.delete()
        except Exception:
            pass
        await interaction.response.send_message(f"**{self.entry['title']}** añadido a tu vault.", ephemeral=True, delete_after=10)

        # Notificar al remitente
        ch_id = get_user_text_channel_id(uid_sender)
        if ch_id:
            ch = interaction.guild.get_channel(ch_id)
            if ch:
                await ch.send(f"{self.recipient.display_name} ha reclamado tu regalo **{self.entry['title']}**.", delete_after=60)

    async def reject(self, interaction: discord.Interaction):
        if interaction.user.id != self.recipient.id:
            return await interaction.response.send_message("No es tu regalo.", ephemeral=True)

        uid_sender = str(self.sender.id)
        uid_recv = str(self.recipient.id)

        pending = data.get("pending_gifts", {}).get(uid_recv, [])
        data["pending_gifts"][uid_recv] = [
            g for g in pending if not (g["from_id"] == uid_sender and g["entry"].get("title") == self.entry.get("title"))
        ]
        save_data(data)

        try:
            await interaction.message.delete()
        except Exception:
            pass
        await interaction.response.send_message("Regalo rechazado.", ephemeral=True, delete_after=5)

        # Notificar al remitente
        ch_id = get_user_text_channel_id(uid_sender)
        if ch_id:
            ch = interaction.guild.get_channel(ch_id)
            if ch:
                await ch.send(f"{self.recipient.display_name} ha rechazado tu regalo **{self.entry['title']}**.", delete_after=60)


async def _update_shared_messages(owner: discord.Member, entry: dict, guild: discord.Guild):
    """Actualiza las entradas compartidas en los vaults de los receptores cuando el dueño edita."""
    shared_with = entry.get("shared_with", [])
    for target_id in shared_with:
        vault_recv = data.get("vault", {}).get(target_id, [])
        for e in vault_recv:
            if e.get("shared") and e.get("shared_by_id") == str(owner.id) and e.get("title") == entry["title"]:
                # Actualizar campos
                e.update({
                    "user": entry.get("user", ""),
                    "password": entry.get("password", ""),
                    "email": entry.get("email", ""),
                    "email_pass": entry.get("email_pass", ""),
                    "url": entry.get("url", ""),
                    "notes": entry.get("notes", ""),
                })
        save_data(data)
        await _refresh_vault_panel_by_uid(target_id, guild)


async def _refresh_vault_panel_by_uid(uid: str, guild: discord.Guild):
    """Refresca el panel del vault dado un uid y un guild."""
    ch_id = get_user_text_channel_id(uid)
    if not ch_id:
        return
    ch = guild.get_channel(ch_id)
    if not ch:
        return
    try:
        member = guild.get_member(int(uid)) or await guild.fetch_member(int(uid))
    except Exception:
        return
    if not member:
        return
    entries = data.get("vault", {}).get(uid, [])
    embed = discord.Embed(
        title="CHEPA'S VAULT",
        description=(
            "Guarda tus cuentas y links de forma privada.\n"
            f"Solo tú puedes ver este canal.\n\n"
            f"**{len(entries)}** entradas guardadas"
        ),
        color=0x1a1a2e
    )
    embed.set_footer(text="Tus datos solo son visibles aquí")
    view = VaultMainView(member)
    existing_id = data.get("vault_msg_ids", {}).get(uid)
    if existing_id:
        try:
            msg = await ch.fetch_message(existing_id)
            await msg.edit(embed=embed, view=view)
            return
        except Exception:
            pass
    msg = await ch.send(embed=embed, view=view)
    data.setdefault("vault_msg_ids", {})[uid] = msg.id
    save_data(data)


class VaultConfirmDeleteView(discord.ui.View):
    def __init__(self, member: discord.Member, idx: int):
        super().__init__(timeout=30)
        self.member = member
        self.idx = idx

        confirm_btn = discord.ui.Button(label="Confirmar", style=discord.ButtonStyle.danger)
        confirm_btn.callback = self.delete_entry
        self.add_item(confirm_btn)

        cancel_btn = discord.ui.Button(label="Cancelar", style=discord.ButtonStyle.secondary)
        cancel_btn.callback = self.cancel
        self.add_item(cancel_btn)

    async def delete_entry(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx < len(entries):
            removed = entries.pop(self.idx)
            save_data(data)
            # Borrar el mensaje con la entrada
            try:
                # Buscar y borrar el mensaje padre (el de la entrada)
                async for msg in interaction.channel.history(limit=10):
                    if msg.author == bot.user and msg.embeds and removed['title'] in (msg.embeds[0].title or ""):
                        await msg.delete()
                        break
            except Exception:
                pass
            await interaction.response.edit_message(
                content=f"**{removed['title']}** borrado.", view=None
            )
            await _refresh_vault_panel(self.member, interaction)
        else:
            await interaction.response.edit_message(content="Entrada no encontrada.", view=None)

    async def cancel(self, interaction: discord.Interaction):
        await interaction.response.edit_message(content="Cancelado.", view=None)


class VaultEditAccountModal(discord.ui.Modal, title="Editar cuenta"):
    user_input     = discord.ui.TextInput(label="Usuario o Email", max_length=100)
    pass_input     = discord.ui.TextInput(label="Contraseña", max_length=100)
    email_input    = discord.ui.TextInput(label="Email (opcional)", max_length=100, required=False)
    email_pass_inp = discord.ui.TextInput(label="Contraseña email (opcional)", max_length=100, required=False)

    def __init__(self, member: discord.Member, idx: int):
        super().__init__()
        self.member = member
        self.idx = idx
        uid = str(member.id)
        entries = data.get("vault", {}).get(uid, [])
        if idx < len(entries):
            e = entries[idx]
            self.user_input.default     = e.get("user", "")
            self.pass_input.default     = e.get("password", "")
            self.email_input.default    = e.get("email", "")
            self.email_pass_inp.default = e.get("email_pass", "")

    async def on_submit(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx < len(entries):
            entries[self.idx].update({
                "user":       self.user_input.value,
                "password":   self.pass_input.value,
                "email":      self.email_input.value,
                "email_pass": self.email_pass_inp.value,
            })
            save_data(data)
            await interaction.response.send_message("Entrada actualizada.", ephemeral=True, delete_after=5)
            await _refresh_vault_panel(self.member, interaction)
            # Actualizar mensajes compartidos
            await _update_shared_messages(self.member, entries[self.idx], interaction.guild)
        else:
            await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)


class VaultEditLinkModal(discord.ui.Modal, title="Editar link"):
    title_input = discord.ui.TextInput(label="Título", max_length=30)
    url_input   = discord.ui.TextInput(label="URL", max_length=200)
    notes_input = discord.ui.TextInput(label="Notas (opcional)", max_length=100, required=False)

    def __init__(self, member: discord.Member, idx: int):
        super().__init__()
        self.member = member
        self.idx = idx
        uid = str(member.id)
        entries = data.get("vault", {}).get(uid, [])
        if idx < len(entries):
            e = entries[idx]
            self.title_input.default = e.get("title", "")
            self.url_input.default   = e.get("url", "")
            self.notes_input.default = e.get("notes", "")

    async def on_submit(self, interaction: discord.Interaction):
        uid = str(self.member.id)
        entries = data.get("vault", {}).get(uid, [])
        if self.idx < len(entries):
            entries[self.idx].update({
                "title": self.title_input.value,
                "url":   self.url_input.value,
                "notes": self.notes_input.value,
            })
            save_data(data)
            await interaction.response.send_message("Link actualizado.", ephemeral=True, delete_after=5)
            await _refresh_vault_panel(self.member, interaction)
        else:
            await interaction.response.send_message("Entrada no encontrada.", ephemeral=True)

# =====================================================================
#  RECORDATORIOS — SISTEMA POR LENGUAJE NATURAL
# =====================================================================

# =====================================================================
#  COMANDOS SLASH
# =====================================================================
@bot.tree.command(name="invite", description="Genera invitación + key de acceso")
async def invite_cmd(interaction: discord.Interaction):
    if interaction.channel_id != PINNED_RESPONSE_CHANNEL_ID:
        return await interaction.response.send_message(
            f"❌ Solo en <#{PINNED_RESPONSE_CHANNEL_ID}>", ephemeral=True
        )
    code = generate_code()
    data["codes"].append(code)
    save_data(data)
    try:
        invite = await interaction.channel.create_invite(max_uses=1, unique=True, max_age=86400)
    except Exception as e:
        return await interaction.response.send_message(f"Error: {e}", ephemeral=True)

    embed = discord.Embed(title="✦ INVITACIÓN CHEPA 3.0 ✦", color=0x2b2d31)
    embed.add_field(name="🔗 Link", value=f"[Click para entrar]({invite.url})", inline=False)
    embed.add_field(name="🔑 Key", value=f"```fix\n{code}\n```", inline=False)
    embed.set_footer(text="Copia la Key. Te la pedirá Bender al entrar.")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="gen_keys", description="Generar keys manualmente (Admin)")
async def gen_keys(interaction: discord.Interaction, amount: int = 5):
    if not interaction.user.guild_permissions.administrator:
        return await interaction.response.send_message("❌ Sin permisos.", ephemeral=True)
    amount = min(max(amount, 1), 50)
    new_codes = [generate_code() for _ in range(amount)]
    data["codes"].extend(new_codes)
    save_data(data)
    await interaction.response.send_message(
        f"✅ {amount} keys:\n```\n" + "\n".join(new_codes) + "\n```", ephemeral=True
    )


@bot.tree.command(name="warn", description="Avisar a un usuario (Admin)")
async def warn_cmd(interaction: discord.Interaction, usuario: discord.Member, razon: str = "Sin razón"):
    if not interaction.user.guild_permissions.moderate_members:
        return await interaction.response.send_message("❌ Sin permisos.", ephemeral=True)
    uid = str(usuario.id)
    data.setdefault("warnings", {})[uid] = data["warnings"].get(uid, 0) + 1
    save_data(data)
    count = data["warnings"][uid]
    await interaction.response.send_message(
        f"⚠️ {usuario.mention} advertido por **{razon}**. Total warns: {count}/3"
    )
    if count >= 3:
        try:
            await usuario.timeout(timedelta(minutes=30), reason=razon)
            await interaction.followup.send(f"⛔ {usuario.mention} timeout 30min por acumular 3 warns.")
        except Exception:
            pass


@bot.tree.command(name="actividad", description="Ver actividad semanal de un usuario")
async def actividad_cmd(interaction: discord.Interaction, usuario: discord.Member = None):
    target = usuario or interaction.user
    uid = str(target.id)
    week = get_week_key()
    act = get_activity(uid)
    embed = discord.Embed(
        title=f"Actividad de {target.display_name}",
        description=f"Semana {week}",
        color=0x1a1a2e
    )
    embed.add_field(name="Mensajes", value=str(act["messages"]), inline=True)
    embed.add_field(name="Tiempo en llamada", value=format_time(act["voice_seconds"]), inline=True)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="normas", description="Refresca el panel de normas (Admin)")
async def normas_cmd(interaction: discord.Interaction):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("⛧ No tienes permiso.", ephemeral=True)
        return
    
    await send_rules_embed(interaction.guild)
    await interaction.response.send_message("✅ Panel de normas actualizado.", ephemeral=True)


@tasks.loop(minutes=5)
async def refresh_activity_panels():
    """Actualiza los paneles de actividad cada 5 minutos y limpia paneles de voz huérfanos."""
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        return
    for uid in list(data.get("activity_msg_ids", {}).keys()):
        try:
            await refresh_activity_panel_for(uid, guild)
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"[ACTIVITY] Error uid {uid}: {e}")

    # Limpiar paneles de control de voz huérfanos: el panel debe borrarse si el
    # usuario NO está en el canal concreto al que pertenece ese panel (no vale con
    # estar en otro canal cualquiera, ni el panel debe sobrevivir si el canal ya no existe).
    for uid in list(data.get("voice_control_messages", {}).keys()):
        try:
            ctrl = data["voice_control_messages"].get(uid) or {}
            panel_vc_id = ctrl.get("voice_channel_id")
            member = guild.get_member(int(uid))
            cur_vc_id = member.voice.channel.id if (member and member.voice and member.voice.channel) else None
            if panel_vc_id != cur_vc_id:
                print(f"[CLEANUP] Panel voz huérfano (uid={uid}, panel={panel_vc_id}, actual={cur_vc_id}), borrando...")
                await delete_control_message(uid, guild)
        except Exception as e:
            print(f"[CLEANUP] Error limpiando panel huérfano uid={uid}: {e}")

    # Salvaguarda: borrar canales ⛧ que se hayan quedado vacíos (p.ej. si alguien se
    # desconectó justo al crearse el canal y el evento de salida no lo limpió).
    for vc in guild.voice_channels:
        if vc.name.startswith("⛧︲") and vc.id != VOICE_CREATOR_ID:
            if len([m for m in vc.members if not m.bot]) == 0:
                try:
                    await cleanup_pending_transfer(vc.id, guild)
                    await vc.delete()
                    await cleanup_voice_data(vc.id, guild)
                    print(f"[CLEANUP] Canal vacío colgado borrado: {vc.name} ({vc.id})", flush=True)
                except Exception as e:
                    print(f"[CLEANUP] Error borrando canal vacío {vc.name} ({vc.id}): {e}", flush=True)

@tasks.loop(minutes=1)
async def check_weekly_leaderboard():
    """Publica el leaderboard el domingo a las 00:00."""
    now = datetime.now()
    if now.weekday() != 6 or now.hour != 23 or now.minute != 59:
        return

    guild = bot.get_guild(GUILD_ID)
    if not guild:
        return

    week = get_week_key()
    week_data = data.get("activity", {}).get(week, {})
    if not week_data:
        return

    # Ordenar por horas en voz, luego por mensajes
    sorted_users = sorted(
        week_data.items(),
        key=lambda x: (x[1].get("voice_seconds", 0), x[1].get("messages", 0)),
        reverse=True
    )[:3]

    if not sorted_users:
        return

    medals = ["⛧", "⛧︲", "⛧︲︲"]
    lines = []
    for i, (uid, act) in enumerate(sorted_users):
        member = guild.get_member(int(uid))
        name = member.display_name if member else f"<@{uid}>"
        lines.append(
            f"{medals[i]} **{name}** — {format_time(act.get('voice_seconds', 0))} en llamada · {act.get('messages', 0)} mensajes"
        )

    ch = guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
    if not ch:
        return

    embed = discord.Embed(
        title="ACTIVIDAD DE LA SEMANA",
        description="\n".join(lines),
        color=0x4a0080
    )
    embed.set_footer(text=f"Semana {week} · Nueva semana, nueva oportunidad de no ser un fantasma")
    await ch.send(embed=embed)

# =====================================================================
#  EVENTOS
# =====================================================================
async def _startup_refresh_panels(guild: discord.Guild):
    """Refresca paneles en segundo plano sin bloquear el startup."""
    await asyncio.sleep(5)  # Esperar a que todo esté estable
    print("[STARTUP] Refrescando paneles en background...")
    for uid, ud in list(data.get("user_channels", {}).items()):
        try:
            ch_id = ud["channel_id"] if isinstance(ud, dict) else ud
            ch = guild.get_channel(ch_id)
            if not ch:
                continue
            try:
                member = await guild.fetch_member(int(uid))
            except Exception:
                continue
            if not member:
                continue
            # Limpiar paneles de clips duplicados (dejar solo el último)
            try:
                clip_msgs = []
                async for msg in ch.history(limit=50):
                    if msg.author == bot.user and msg.embeds and msg.embeds[0].title and msg.embeds[0].title.upper() == "CLIPS":
                        clip_msgs.append(msg)
                if len(clip_msgs) > 1:
                    # Borrar todos menos el último
                    for old_msg in clip_msgs[1:]:
                        try:
                            await old_msg.delete()
                            await asyncio.sleep(0.2)
                        except Exception:
                            pass
                    # Registrar el último como el oficial
                    data.setdefault("clip_panel_msg_ids", {})[uid] = clip_msgs[0].id
                    save_data(data)
                    print(f"[STARTUP] Limpiados {len(clip_msgs)-1} paneles de clips duplicados en #{ch.name}")
            except Exception as e:
                print(f"[STARTUP] Error limpiando clips en #{ch.name}: {e}")
            await send_vault_panel(member, ch)
            await asyncio.sleep(0.5)
            await send_clip_panel(member, ch)
            await asyncio.sleep(0.5)
            await send_activity_panel(member, ch)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[STARTUP] Error uid {uid}: {e}")
    print("[STARTUP] Paneles listos.")


@bot.event
async def on_ready():
    global discord_loop
    discord_loop = bot.loop

    guild = bot.get_guild(GUILD_ID)
    if not guild:
        print("❌ GUILD NO ENCONTRADA")
        return

    if not data.get("codes"):
        data["codes"] = [generate_code() for _ in range(10)]
        save_data(data)

    ch = guild.get_channel(LOGIN_CHANNEL_ID)
    if ch:
        try:
            async for m in ch.history(limit=50):
                if m.author == bot.user:
                    await m.delete()
                    await asyncio.sleep(0.3)
            embed = discord.Embed(
                title="ACCESO AL SISTEMA",
                description=(
                    "Introduce tu **KEY DE ACCESO** en el canal.\n\n"
                    "La key se obtiene a través de un miembro del server.\n"
                    "Tienes **3 intentos** antes del bloqueo temporal."
                ),
                color=0x4a0080
            )
            embed.set_image(url="https://images.guns.lol/5a3415a4ffbed3551ecf589da3452df1e3f682dc/go1aXe.gif")
            embed.set_footer(text="⛧︲ Chepa 3.0  ·  Server para cabrones")
            await ch.send(embed=embed)
        except Exception as e:
            print(f"[ERROR] Login channel: {e}")

    await cleanup_orphaned_channels(guild)
    await restore_voice_panels(guild)
    await purge_orphan_voice_panels(guild)
    
    # ─── NUEVO: Enviar embed de normas ───
    await send_rules_embed(guild)
    
    try:
        bot.add_view(MusicPanelView())
    except Exception:
        pass
    check_weekly_leaderboard.start()
    refresh_activity_panels.start()

    # ─── Registrar views persistentes del vault ───
    for uid in data.get("user_channels", {}).keys():
        try:
            member = await guild.fetch_member(int(uid))
            if member:
                bot.add_view(VaultMainView(member))
        except Exception:
            pass

    # ─── Refrescar paneles en background (no bloquea startup) ───
    bot.loop.create_task(_startup_refresh_panels(guild))

    data["v3_migration_done"] = True
    save_data(data)

    try:
        synced = await bot.tree.sync()
        print(f"[SYNC] {len(synced)} comandos.")
    except Exception as e:
        print(f"[ERROR] Sync: {e}")

    # ─── Registrar miembros ya en voz al arrancar ───
    now_str = datetime.now().isoformat()
    for vc in guild.voice_channels:
        for member in vc.members:
            if not member.bot:
                uid = str(member.id)
                if uid not in data.get("voice_join_times", {}):
                    data.setdefault("voice_join_times", {})[uid] = now_str
    save_data(data)

    # Precargar motores de voz en background (para que la 1ª entrada a voz sea rápida)
    if VOICE_LIBS_OK:
        bot.loop.run_in_executor(None, _load_voice_engines)
        # AUTO-REINGRESO: si un reinicio lo echó de una llamada con gente, vuelve solo.
        bot.loop.create_task(_auto_rejoin_voice(guild))
    # (El servidor po_token se arranca perezosamente solo cuando se pega un link de
    #  música, vía _ensure_pot_server() dentro de _ytdlp_resolve — no 24/7.)

    print(f"🤖 BENDER 3.0 READY — {guild.name} | {guild.member_count} miembros")


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if member.guild.id != GUILD_ID or member.bot:
        return

    user_id = str(member.id)

    # ─── NUEVO: Expulsar de canales de voz si no aceptó normas ───
    # Solo aplica si: NO tiene rol LIMITED_ROLE_ID (?) y NO ha aceptado normas
    if after.channel and after.channel.id != VOICE_CREATOR_ID:
        limited_role = member.guild.get_role(LIMITED_ROLE_ID)
        has_limited = limited_role and limited_role in member.roles
        accepted = data.get("accepted_rules", {}).get(user_id, False)
        
        if not has_limited and not accepted:
            # Expulsar del canal de voz
            try:
                await member.move_to(None)
                print(f"[NORMAS] {member.name} expulsado de voz (no aceptó normas)", flush=True)
            except Exception:
                pass
            
            # Notificar por DM
            try:
                await member.send(
                    f"⛧ **ACCESO RESTRINGIDO** ⛧\n\n"
                    f"No puedes unirte a canales de voz sin aceptar los términos.\n\n"
                    f"Ve a <#{RULES_CHANNEL_ID}> y reacciona con {RULES_ACCEPT_EMOJI}"
                )
            except Exception:
                pass
            return

    # ─── ENTRAR AL CREATOR ───
    # Solo si realmente ENTRÓ al creator (cambio de canal), no por toggles de
    # mute/cámara/stream estando ya dentro — esos también disparan este evento
    # y encolaban creaciones duplicadas.
    if (after.channel and after.channel.id == VOICE_CREATOR_ID
            and (not before.channel or before.channel.id != after.channel.id)):
        if user_id not in data.get("user_channels", {}):
            try:
                await member.move_to(None)
            except Exception:
                pass
            return

        if user_id not in voice_creation_locks:
            voice_creation_locks[user_id] = asyncio.Lock()

        async with voice_creation_locks[user_id]:
            # Re-verificar: after.channel es una propiedad dinámica que consulta
            # el caché del guild. Si el bot reconectó el gateway mientras esperábamos
            # el lock (lock contestado), el canal puede haber desaparecido del caché.
            _ach = after.channel
            if not _ach or _ach.id != VOICE_CREATOR_ID:
                return  # canal desapareció del caché — reconexión en curso, ignorar

            # Guard en memoria: si acabamos de crear un canal para este usuario,
            # úsalo directamente. NO dependas del caché del gateway: bajo carga el
            # CHANNEL_CREATE tarda segundos en llegar, get_channel() devuelve None
            # y acabábamos creando 3-4 canales duplicados.
            _rec = _recent_created_vc.get(user_id)
            if _rec and time.time() - _rec[1] < 60:
                try:
                    await member.move_to(_rec[0])
                    return
                except discord.NotFound:
                    _recent_created_vc.pop(user_id, None)
                except Exception:
                    return

            existing_vc_id = data.get("active_voice_channels", {}).get(user_id)
            if existing_vc_id:
                existing_vc = member.guild.get_channel(existing_vc_id)
                if existing_vc is None:
                    # Caché frío ≠ canal borrado. Confirmar contra la API antes
                    # de dar el canal por muerto y crear otro.
                    try:
                        existing_vc = await member.guild.fetch_channel(existing_vc_id)
                    except discord.NotFound:
                        existing_vc = None
                    except Exception:
                        return  # error transitorio de API: no crees un duplicado
                if existing_vc:
                    try:
                        await member.move_to(existing_vc)
                    except Exception:
                        pass
                    return
                else:
                    del data["active_voice_channels"][user_id]
                    save_data(data)

            cat = _ach.category
            limited_role = member.guild.get_role(LIMITED_ROLE_ID)
            overwrites = {
                member.guild.default_role: discord.PermissionOverwrite(connect=True, view_channel=True),
                member: discord.PermissionOverwrite(connect=True, view_channel=True, move_members=True),
                member.guild.me: discord.PermissionOverwrite(connect=True, view_channel=True, manage_channels=True, move_members=True),
            }
            if limited_role:
                overwrites[limited_role] = discord.PermissionOverwrite(connect=False, view_channel=False)

            try:
                vc = await member.guild.create_voice_channel(
                    f"⛧︲{member.display_name}", category=cat, overwrites=overwrites
                )
                _recent_created_vc[user_id] = (vc, time.time())
                data["active_voice_channels"][user_id] = vc.id
                data["voice_channel_owners"][str(vc.id)] = member.id
                data["channel_modes"][str(vc.id)] = "public"
                data.setdefault("member_join_times", {}).setdefault(str(vc.id), {})[user_id] = datetime.now().isoformat()
                # Registrar entrada a voz para contar tiempo
                data.setdefault("voice_join_times", {})[user_id] = datetime.now().isoformat()
                save_data(data)
                await asyncio.sleep(0.5)
                await member.move_to(vc)
                await create_voice_control_panel(member, vc, member.guild)
                add_voice_seconds(user_id, 0)  # voz: se cuenta en on_voice_state_update
                # asyncio.create_task(_join_and_record(vc))  # DESACTIVADO
            except Exception as e:
                print(f"[ERROR] Creando canal: {e}")

    # ─── TRACKING TIEMPO EN VOZ ───
    if before.channel and not after.channel:
        join_str = data.get("voice_join_times", {}).pop(user_id, None)
        if join_str:
            try:
                seconds = int((datetime.now() - datetime.fromisoformat(join_str)).total_seconds())
                if seconds > 0:
                    add_voice_seconds(user_id, seconds)
                    asyncio.create_task(refresh_activity_panel_for(user_id, member.guild))
            except Exception:
                pass
            save_data(data)

    if after.channel and not before.channel:
        data.setdefault("voice_join_times", {})[user_id] = datetime.now().isoformat()
        save_data(data)

    # ─── GRABACIÓN: DESACTIVADO ───
    # if (after.channel
    #         and after.channel.id != VOICE_CREATOR_ID
    #         and after.channel.name.startswith("⛧︲")):
    #     rec_ch = current_rec_ch.get(member.guild.id)
    #     if rec_ch != after.channel.id:
    #         asyncio.create_task(_join_and_record(after.channel))

    # Anunciar en WhatsApp SOLO si "crea" la llamada (primer no-bot en canal vacío, excluyendo creador)
    if after.channel and (not before.channel or before.channel.id != after.channel.id):
        if after.channel.id != VOICE_CREATOR_ID:
            others = [m for m in after.channel.members if not m.bot and m.id != member.id]
            if len(others) == 0:
                last_notify = data.get("last_voice_notify", {}).get(user_id)
                now = datetime.now()
                can_notify = True
                if last_notify:
                    try:
                        last_dt = datetime.fromisoformat(last_notify)
                        if (now - last_dt).total_seconds() < 30:
                            can_notify = False
                    except Exception:
                        pass
                if can_notify:
                    data.setdefault("last_voice_notify", {})[user_id] = now.isoformat()
                    save_data(data)
                    try:
                        name = member.display_name
                        await send_whatsapp(f"⛧︲*{name}* ha creado *llamada*")
                    except Exception as e:
                        print(f"[WA] Error anunciando: {e}")

    # ─── SALIR DE CANAL PERSONAL ───
    if (before.channel
            and before.channel.id != VOICE_CREATOR_ID
            and before.channel.name.startswith("⛧︲")):

        vc = before.channel
        cid_str = str(vc.id)
        is_owner = (data.get("voice_channel_owners", {}).get(cid_str) == member.id)
        remaining = [m for m in vc.members if not m.bot]

        if len(remaining) == 0:
            await cleanup_pending_transfer(vc.id, member.guild)
            # Borrar panel del dueño siempre, sin importar quién salió último
            owner_id = data.get("voice_channel_owners", {}).get(cid_str)
            if owner_id:
                await delete_control_message(owner_id, member.guild)
            # Mover grabación a otro canal activo o salir
            asyncio.create_task(_maybe_move_recording(vc, member.guild))
            try:
                await vc.delete()
                print(f"[VOZ] Canal personal borrado: {vc.name} ({vc.id})", flush=True)
            except Exception as _del_err:
                print(f"[VOZ-WARN] No pude borrar {vc.name} ({vc.id}): {_del_err} — se limpiará en el sweep de 5 min", flush=True)
            await cleanup_voice_data(vc.id, member.guild)
        elif is_owner:
            await cleanup_pending_transfer(vc.id, member.guild)
            await handle_owner_left(vc, member.id, member.guild)

    # ─── OWNER REGRESA ───
    if (after.channel
            and after.channel.id != VOICE_CREATOR_ID
            and after.channel.name.startswith("⛧︲")):

        cid_str = str(after.channel.id)
        task_id = f"{cid_str}_{member.id}"

        if task_id in data.get("owner_left_tasks", {}):
            t_data = data["owner_left_tasks"][task_id]
            data["voice_channel_owners"][cid_str] = member.id
            try:
                ch = member.guild.get_channel(t_data["temp_message_channel_id"])
                if ch:
                    msg = await ch.fetch_message(t_data["temp_message_id"])
                    await msg.delete()
            except Exception:
                pass
            data["owner_left_tasks"].pop(task_id, None)
            save_data(data)
            await create_voice_control_panel(member, after.channel, member.guild)

        # XP por tiempo en voz
        pass  # voz: tiempo contado al salir


@bot.event
async def on_member_join(member: discord.Member):
    if member.guild.id != GUILD_ID:
        return
    role = member.guild.get_role(LIMITED_ROLE_ID)
    if role:
        try:
            await member.add_roles(role)
        except Exception as e:
            print(f"[ERROR] Rol limitado: {e}")


async def handle_dm(message: discord.Message):
    """Responde a mensajes privados (DM) a Bender, sabiendo quién le habla y con
    memoria a corto plazo propia del privado."""
    uid = str(message.author.id)
    txt = (message.content or "").strip()
    if not txt:
        return
    profile = detect_profile(message.author.display_name, uid) or ""
    mood = get_mood()
    system = SERVER_CONTEXT + f"\n\nHoy es {_fecha_es()}. Estado de ánimo: {mood}."
    system += build_identity_anchor(message.author, uid, profile)
    system += build_live_games_context(bot.get_guild(GUILD_ID))
    system += "\n(Esto es un mensaje PRIVADO directo, 1 a 1. Responde corto y a tu bola.)"

    dm_all = data.setdefault("dm_history", {})
    history = clean_history(dm_all.get(uid, []))
    history.append({"role": "user", "content": txt})
    history = history[-12:]
    msgs = [{"role": "system", "content": system}] + history
    try:
        async with message.channel.typing():
            reply = await call_ai(msgs, max_tokens=700, use_web=needs_web_search(txt))
    except Exception:
        reply = error_fallback()
    if is_error_reply(reply):
        reply = error_fallback()
    else:
        history.append({"role": "assistant", "content": reply})
        dm_all[uid] = history[-12:]
        save_data(data)
    try:
        await message.channel.send(reply)
    except Exception as e:
        print(f"[DM] Error enviando: {e}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    # ─── NUEVO: Restricción de normas ───
    # Si NO tiene rol LIMITED_ROLE_ID (?) y NO ha aceptado normas → bloquear
    if message.guild and message.guild.id == GUILD_ID:
        member = message.guild.get_member(message.author.id)
        if member:
            limited_role = message.guild.get_role(LIMITED_ROLE_ID)
            has_limited = limited_role and limited_role in member.roles
            accepted = data.get("accepted_rules", {}).get(str(message.author.id), False)
            
            # Sin rol ? (libre) y sin aceptar normas → bloquear
            if not has_limited and not accepted:
                # Borrar mensaje
                try:
                    await message.delete()
                except Exception:
                    pass
                
                # Enviar DM
                try:
                    await message.author.send(
                        f"⛧ **ACCESO RESTRINGIDO** ⛧\n\n"
                        f"No has aceptado los términos y condiciones de Chepa 3.0.\n\n"
                        f"Para participar en el servidor, debes aceptar las normas en:\n"
                        f"<#{RULES_CHANNEL_ID}>\n\n"
                        f"Reacciona con {RULES_ACCEPT_EMOJI} para confirmar que aceptas."
                    )
                except Exception:
                    pass
                return
    
    # ─── MENSAJES PRIVADOS (DM) a Bender ───
    if message.guild is None:
        await handle_dm(message)
        return
    if message.guild.id != GUILD_ID:
        return

    uid = str(message.author.id)

    # ─── VOZ: entrar/salir de la llamada por orden de texto ───
    # En el canal del bot vale sin decir "bender" (es donde se le habla); en otros
    # canales requiere "bender" o mención para no activarse por error.
    _ml = message.content.lower()
    _en_canal_bot = message.channel.id == AI_CHANNEL_ID
    if _en_canal_bot or ("bender" in _ml) or (bot.user in message.mentions):
        _author_in_voice = bool(getattr(message.author, "voice", None) and message.author.voice.channel)
        if _is_join_voice_cmd(_ml, author_in_voice=_author_in_voice):
            print(f"[VOZ] Orden de UNIRSE de {message.author.display_name} (en_voz={_author_in_voice})", flush=True)
            async with message.channel.typing():
                resp = await join_voice(message.author)
            if resp:
                await safe_reply(message, resp)
            return
        # Salir de la llamada: si Bender YA está en una, basta un "vete/lárgate/fuera"
        # (no hace falta decir "llamada"). Si no está, requiere el comando completo.
        _sess = voice_sessions.get(message.guild.id)
        _bender_en_voz = bool(_sess and _sess.get("vc") and _sess["vc"].is_connected())
        # --- MUSICA por chat ---
        if any(w in _ml for w in ("panel de música", "panel de musica", "panel música", "panel musica")):
            await send_music_panel(message.guild, message.channel)
            return
        if any(w in _ml for w in _MUSIC_STOP) and _bender_en_voz:
            await safe_reply(message, await stop_music(message.guild))
            return
        if any(w in _ml for w in _MUSIC_SKIP) and _bender_en_voz:
            await safe_reply(message, await skip_music(message.guild))
            return
        _mq = _parse_music_query(_ml)
        if _mq is not None:
            if not _bender_en_voz and _author_in_voice:
                await join_voice(message.author)
            async with message.channel.typing():
                resp = await enqueue_music(message.guild, _mq, message.author.display_name)
            await safe_reply(message, resp)
            return
        _mv = _parse_music_volume(_ml)
        if _mv is not None and _bender_en_voz:
            await safe_reply(message, await set_music_volume(message.guild, _mv))
            return
        _verbos_salir = ("vete", "lárgate", "largate", "pírate", "pirate", "fuera",
                         "márchate", "marchate", "piro", "largo", "chao", "adiós",
                         "adios", "desconect", "sal de", "déjanos", "dejanos")
        if _is_leave_voice_cmd(_ml) or (_bender_en_voz and any(v in _ml for v in _verbos_salir)):
            await leave_voice(message.guild)
            await safe_reply(message, "Vale, me piro de la llamada. Hasta luego.")
            return

    # ─── LOGIN ───
    if message.channel.id == LOGIN_CHANNEL_ID:
        timeout_str = data.get("timeout_until", {}).get(uid)
        if timeout_str:
            try:
                if datetime.now() < datetime.fromisoformat(timeout_str):
                    try:
                        await message.delete()
                    except Exception:
                        pass
                    await message.channel.send(
                        f"⛔ {message.author.mention} Bloqueado temporalmente.", delete_after=5
                    )
                    return
                else:
                    data["timeout_until"].pop(uid, None)
                    data["failed_attempts"].pop(uid, None)
                    save_data(data)
            except ValueError:
                data["timeout_until"].pop(uid, None)
                save_data(data)

        code = message.content.strip()
        try:
            await message.delete()
        except Exception:
            pass

        if code in data.get("codes", []):
            data["codes"].remove(code)
            data["failed_attempts"].pop(uid, None)
            save_data(data)

            limited_role = message.guild.get_role(LIMITED_ROLE_ID)
            if limited_role and limited_role in message.author.roles:
                try:
                    await message.author.remove_roles(limited_role)
                except Exception:
                    pass

            if uid in data.get("user_channels", {}):
                existing_ch_id = get_user_text_channel_id(uid)
                existing_ch = message.guild.get_channel(existing_ch_id) if existing_ch_id else None
                if existing_ch:
                    # Refrescar paneles en el self existente
                    await refresh_self_panels(message.author, message.guild)
                    await message.channel.send(
                        f"✅ **ACCESO CONCEDIDO**\nBienvenido de nuevo {message.author.mention}. "
                        f"Tu terminal: {existing_ch.mention}",
                        delete_after=15
                    )
                    return

            new_ch = await create_self_channel(message.author, message.guild)
            await message.channel.send(
                f"✅ **ACCESO CONCEDIDO**\nBienvenido {message.author.mention}. "
                f"Tu terminal: {new_ch.mention}",
                delete_after=15
            )
        else:
            attempts = data.get("failed_attempts", {}).get(uid, 0) + 1
            data.setdefault("failed_attempts", {})[uid] = attempts
            save_data(data)
            if attempts >= 3:
                data.setdefault("timeout_until", {})[uid] = (datetime.now() + timedelta(hours=1)).isoformat()
                save_data(data)
                await message.channel.send(
                    f"⛔ {message.author.mention} 3 fallos. Bloqueado 1 hora.", delete_after=5
                )
            else:
                await message.channel.send(f"❌ Key inválida. ({attempts}/3)", delete_after=5)
        return

    # ─── ANTI-SPAM (canales generales, no en self ni login) ───
    is_self_channel = False
    for _, ud in data.get("user_channels", {}).items():
        ch_id = ud["channel_id"] if isinstance(ud, dict) else ud
        if message.channel.id == ch_id:
            is_self_channel = True
            break

    if not is_self_channel and message.channel.id != AI_CHANNEL_ID:
        if message.attachments:
            pass  # nunca marcar como spam si hay adjunto
        elif await check_spam(message):
            await bot.process_commands(message)
            return

    # ─── XP POR MENSAJE ───
    add_message(uid)
    msgs_count = data.get("activity", {}).get(get_week_key(), {}).get(uid, {}).get("messages", 0)
    if msgs_count % 10 == 0:
        asyncio.create_task(refresh_activity_panel_for(uid, message.guild))

    # ─── CHAT GENERAL — responde si: lo mencionan, dicen "bender", o RESPONDEN a un mensaje suyo ───
    quiere_respuesta = False
    if message.channel.id == PINNED_RESPONSE_CHANNEL_ID:
        quiere_respuesta = (bot.user in message.mentions) or ("bender" in message.content.lower())
        if not quiere_respuesta and message.reference:
            # ¿Es una respuesta (reply) a un mensaje del propio bot?
            ref = message.reference.resolved
            if isinstance(ref, discord.Message):
                quiere_respuesta = (ref.author.id == bot.user.id)
            elif message.reference.message_id:
                try:
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    quiere_respuesta = (ref_msg.author.id == bot.user.id)
                except Exception:
                    pass
    if message.channel.id == PINNED_RESPONSE_CHANNEL_ID and quiere_respuesta:
        async with message.channel.typing():
            try:
                txt = message.content.strip()
                # Quitar la mención al bot del texto
                txt = txt.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()

                # Contexto del mensaje al que se responde
                reply_ctx = ""
                if message.reference and message.reference.message_id:
                    try:
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                        ref_author = ref_msg.author.display_name
                        ref_content = ref_msg.content[:300] if ref_msg.content else "[imagen o adjunto]"
                        reply_ctx = f"\nEl usuario responde al mensaje de {ref_author}: \"{ref_content}\""
                    except Exception:
                        pass

                # Imagen adjunta
                has_image = False
                img_url = None
                for att in message.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        has_image = True
                        img_url = att.url
                        break
                # También buscar imagen en el mensaje referenciado
                if not has_image and message.reference and message.reference.message_id:
                    try:
                        ref_msg2 = await message.channel.fetch_message(message.reference.message_id)
                        for att in ref_msg2.attachments:
                            if att.content_type and att.content_type.startswith("image/"):
                                has_image = True
                                img_url = att.url
                                break
                    except Exception:
                        pass

                profile = data.get("user_profiles", {}).get(uid) or detect_profile(message.author.display_name, uid) or ""
                mood = get_mood()
                system = SERVER_CONTEXT + f"\n\nHoy es {_fecha_es()}. Estado de ánimo: {mood}."
                system += build_identity_anchor(message.author, uid, profile)
                system += game_jab_hint(message.author)
                system += build_live_games_context(message.guild)

                full_txt = txt + reply_ctx

                conv = data.get("conversation_history", {})
                if isinstance(conv, list):
                    conv = {}
                    data["conversation_history"] = conv
                history = conv.get(uid, [])

                if has_image and img_url:
                    user_content = full_txt if full_txt.strip() else "¿Qué ves en esta imagen? Identifícalo."
                else:
                    user_content = full_txt

                history = clean_history(history)
                history.append({"role": "user", "content": user_content})
                history = history[-15:]
                msgs = [{"role": "system", "content": system}] + history
                reply = await call_ai(msgs, max_tokens=700, has_image=has_image, use_web=needs_web_search(full_txt), image_urls=([img_url] if has_image and img_url else None))
                # Solo guardar en el historial si NO es un error (evita envenenarlo)
                if not is_error_reply(reply):
                    history.append({"role": "assistant", "content": reply})
                    data["conversation_history"][uid] = history
                    save_data(data)
                else:
                    # Fallo técnico: error interno en personaje, sin revelar el modelo
                    reply = error_fallback()

                await safe_reply(message, reply)
            except Exception as e:
                print(f"[DISCORD] Error en chat handler: {e}")
        await bot.process_commands(message)
        return

    # ─── IA CHAT + CONTROL ADMIN ───
    if message.channel.id == AI_CHANNEL_ID:
        async with message.channel.typing():
            try:
                txt = message.content.strip()

                # Resolver menciones en el texto — sustituir <@ID> por nombre display
                for mention in message.mentions:
                    txt = txt.replace(f"<@{mention.id}>", mention.display_name)
                    txt = txt.replace(f"<@!{mention.id}>", mention.display_name)
                # Limpiar @ sueltos que Discord no convirtió en mención
                import re as _re
                txt = _re.sub(r'@(\w+)', r'\1', txt)

                # Buscar canal de voz del usuario
                voice_ctx = "No estás en ningún canal de voz."
                admin_vc = None
                for vc in message.guild.voice_channels:
                    if message.author in vc.members:
                        admin_vc = vc
                        mode = data.get("channel_modes", {}).get(str(vc.id), "public")
                        members_in = [m.display_name for m in vc.members]
                        voice_ctx = f"Canal: {vc.name}, modo: {mode}, miembros: {members_in}"
                        break

                # Solo el dueño del canal O admin del server puede dar órdenes
                is_admin = message.author.guild_permissions.administrator
                if admin_vc:
                    vc_owner_id = data.get("voice_channel_owners", {}).get(str(admin_vc.id))
                    if vc_owner_id is None:
                        # Sin dueño registrado (típico tras un reinicio): el que está
                        # dentro y manda lo reclama automáticamente. Autocura el estado.
                        data.setdefault("voice_channel_owners", {})[str(admin_vc.id)] = message.author.id
                        save_data(data)
                        is_admin = True
                    elif vc_owner_id == message.author.id:
                        is_admin = True
                    # Si NO es el dueño ni admin, ignorar acciones de voz silenciosamente
                    elif not is_admin:
                        admin_vc = None  # anula el canal para que no ejecute nada

                # Detectar perfil por nombre si no está guardado
                profile = data.get("user_profiles", {}).get(uid)
                if not profile:
                    detected = detect_profile(message.author.display_name, uid)
                    if detected:
                        data.setdefault("user_profiles", {})[uid] = detected
                        save_data(data)
                    profile = detected or ""

                # Comando "soy X" para asociar perfil
                if txt.lower().startswith("soy "):
                    name_claim = txt[4:].strip().lower()
                    found = MEMBER_PROFILES.get(name_claim)
                    if found:
                        data.setdefault("user_profiles", {})[uid] = found
                        save_data(data)
                        await message.reply(f"Vale, te guardo como **{name_claim}**. Ya sé quién eres.")
                        return

                # Interpretar acciones SOLO si el usuario controla un canal de voz
                # (comandos de voz) o pide explícitamente un anuncio. El resto va
                # directo a chat: más rápido y sin riesgo de que se trague el mensaje.
                wants_action = bool(admin_vc) or any(
                    w in txt.lower() for w in ("anuncia", "announce", "comunica", "anuncio")
                )
                if wants_action:
                    action_data = await call_ai_action(txt, voice_ctx, uid)
                    action = action_data.get("action", "desconocido")
                    params = action_data.get("params", {})
                    print(f"[AI ACTION] user={message.author.display_name} vc={admin_vc} action={action} params={params}")
                else:
                    action = "desconocido"
                    params = {}

                # ── Anuncio — TODOS los usuarios ──
                # Solo si el usuario PIDE explícitamente un anuncio (evita que el
                # clasificador secuestre mensajes normales y se los trague sin responder).
                if action == "announce" and any(w in txt.lower() for w in ("anuncia", "announce", "comunica", "anuncio")):
                    msg_text = params.get("message", "")
                    if msg_text:
                        ch_ann = message.guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
                        if ch_ann:
                            embed = discord.Embed(
                                title="ANUNCIO",
                                description=msg_text,
                                color=0x4a0080,
                                timestamp=datetime.now()
                            )
                            embed.set_footer(text=f"— {message.author.display_name}")
                            await ch_ann.send(embed=embed)
                            await message.reply("Anuncio enviado.")
                            return
                    # Si no hay texto de anuncio, NO cortar: cae a chat normal

                # ── Acciones de voz — solo el dueño del canal ──
                if is_admin and admin_vc and action != "desconocido":

                    # Transferir admin del canal
                    if action == "transferir":
                        name = params.get("name", "")
                        target = None
                        for mentioned in message.mentions:
                            if mentioned in admin_vc.members:
                                target = mentioned
                                break
                        if not target:
                            target = await find_member(name, admin_vc.members, message.guild)
                        if target and target.id != message.author.id:
                            cid = str(admin_vc.id)
                            data["voice_channel_owners"][cid] = target.id
                            # Actualizar active_voice_channels
                            data["active_voice_channels"].pop(uid, None)
                            data["active_voice_channels"][str(target.id)] = admin_vc.id
                            save_data(data)
                            # Borrar panel del dueño anterior
                            await delete_control_message(message.author.id, message.guild)
                            # Dar panel al nuevo dueño
                            await create_voice_control_panel(target, admin_vc, message.guild)
                            await message.reply(f"Admin del canal transferido a **{target.display_name}**.")
                        else:
                            await message.reply("No encontré a ese usuario en el canal.")
                        return

                    response_msg = await execute_voice_action(action, params, admin_vc, message)
                    if response_msg:
                        await message.reply(response_msg)
                        return

                # Chat normal con personalidad (+ visión si hay imagen)
                conv = data.get("conversation_history", {})
                if isinstance(conv, list):
                    conv = {}
                    data["conversation_history"] = conv
                history = conv.get(uid, [])

                mood = get_mood()
                system = SERVER_CONTEXT + f"\n\nHoy es {_fecha_es()}. Estado de ánimo actual: {mood}."
                system += build_identity_anchor(message.author, uid, profile)
                system += game_jab_hint(message.author)
                system += build_live_games_context(message.guild)

                has_image_bot = False
                img_url_bot = None
                for att in message.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        has_image_bot = True
                        img_url_bot = att.url
                        break

                if has_image_bot and img_url_bot:
                    user_content = txt if txt.strip() else "¿Qué ves en esta imagen? Identifícalo."
                else:
                    user_content = txt

                history = clean_history(history)
                history.append({"role": "user", "content": user_content})
                history = history[-15:]
                msgs = [{"role": "system", "content": system}] + history
                reply = await call_ai(msgs, max_tokens=700, has_image=has_image_bot, use_web=needs_web_search(txt), image_urls=([img_url_bot] if has_image_bot and img_url_bot else None))
                # Solo guardar en el historial si NO es un error (evita envenenarlo)
                if not is_error_reply(reply):
                    history.append({"role": "assistant", "content": reply})
                    data["conversation_history"][uid] = history
                    save_data(data)
                else:
                    # Fallo técnico: error interno en personaje, sin revelar el modelo
                    reply = error_fallback()

                await safe_reply(message, reply)

            except Exception as e:
                print(f"[DISCORD] Error en bot handler: {e}")

    await bot.process_commands(message)


def _edit_dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _member_aliases(m: discord.Member) -> set:
    """Todos los nombres por los que se conoce a un miembro (display, user, canónico
    y los 'También llamado X o Y' de su perfil). En minúsculas, palabras sueltas."""
    al = set()
    al.add(m.display_name.lower())
    al.add(m.name.lower())
    if m.global_name:
        al.add(m.global_name.lower())
    key = MEMBER_ID_MAP.get(str(m.id), "")
    if key:
        al.add(key)
        profile = MEMBER_PROFILES.get(key, "")
        mt = re.search(r"tambi[eé]n llamad[oa]\s+([^.]+)", profile, re.I)
        if mt:
            for part in re.split(r"\s+o\s+|,|/", mt.group(1)):
                part = part.strip().lower()
                if part:
                    al.add(part)
    # añadir palabras sueltas de los alias multi-palabra
    words = set()
    for a in al:
        for w in a.split():
            if len(w) >= 3:
                words.add(w)
    return {a for a in (al | words) if a}


async def find_member(name: str, pool, guild: discord.Guild) -> discord.Member | None:
    """Busca un miembro de forma flexible: nombre, apodo, alias del perfil y FUZZY
    (tolera que la voz destroce el nombre: husk->usk, payor->payer, etc.)."""
    name = name.lower().strip().lstrip("@")
    if not name:
        return None

    # 1. EXACTO: el nombre coincide entero con un alias o palabra de alias (prioridad
    #    máxima -> coincidencias por substring se resuelven por longitud).
    name_words = set(w for w in name.split() if len(w) >= 3) or {name}
    for m in pool:
        aliases = _member_aliases(m)
        if name in aliases or (name_words & aliases):
            return m

    # 2. PARCIAL: substring razonable (alias de >=4 letras contenido)
    for m in pool:
        for a in _member_aliases(m):
            if len(a) >= 4 and (name in a or a in name):
                return m

    # 3. FUZZY contra todos los alias (la voz destroza nombres). Mejor candidato.
    best, best_d = None, 3
    nws = [w for w in name.split() if len(w) >= 3] or [name]
    for m in pool:
        for a in _member_aliases(m):
            for nw in nws:
                d = _edit_dist(nw, a)
                if d < best_d and d <= max(1, len(a) // 3 + 1):
                    best, best_d = m, d
    if best:
        return best

    # 4. Por ID numérico
    if name.isdigit():
        m = guild.get_member(int(name))
        if m and m in pool:
            return m
    return None


async def execute_voice_action(action: str, params: dict,
                               voice_channel: discord.VoiceChannel,
                               message: discord.Message) -> str | None:
    guild = message.guild
    cid = str(voice_channel.id)
    owner_id = data.get("voice_channel_owners", {}).get(cid)

    if action == "modo":
        mode = params.get("mode", "public")
        # Mapear sinónimos
        mode_map = {
            "privado": "ghost", "fantasma": "ghost", "ghost": "ghost", "invisible": "ghost",
            "cristal": "crystal", "crystal": "crystal", "visible": "crystal",
            "publico": "public", "público": "public", "public": "public", "abierto": "public",
        }
        mode = mode_map.get(mode.lower(), mode)
        if mode not in ("public", "ghost", "crystal"):
            mode = "ghost"  # default si no se entiende

        data.setdefault("channel_modes", {})[cid] = mode
        if mode in ("ghost", "crystal"):
            current_ids = [m.id for m in voice_channel.members]
            data.setdefault("crystal_permits", {})[cid] = current_ids
        else:
            data.get("crystal_permits", {}).pop(cid, None)
        save_data(data)
        allowed = data.get("crystal_permits", {}).get(cid, [])
        await update_channel_permissions(voice_channel, mode, allowed, guild)
        mode_names = {"public": "público", "ghost": "fantasma", "crystal": "cristal"}
        return f"Canal puesto en modo **{mode_names.get(mode, mode)}**."

    elif action == "kick":
        name = params.get("name", "")
        # Primero intentar con menciones directas del mensaje
        target = None
        for mentioned in message.mentions:
            if mentioned in voice_channel.members and mentioned.id != owner_id:
                target = mentioned
                break

        # Si no hay mención, buscar por nombre
        if not target:
            target = await find_member(name, voice_channel.members, guild)

        if target:
            if target.id == owner_id:
                return "No puedo expulsarte de tu propio canal."
            if target.id == guild.me.id:
                return "A mí no me echas de ningún lado."
            await target.move_to(None)
            return f"**{target.display_name}** expulsado del canal."
        return f"No encontré a nadie llamado '{name}' en el canal."

    elif action == "rename":
        new_name = f"⛧︲{params.get('name', 'canal')}"
        try:
            await voice_channel.edit(name=new_name)
            return f"Canal renombrado a **{new_name}**."
        except Exception as e:
            return f"Error renombrando: {e}"

    elif action == "allow":
        name = params.get("name", "")
        # Buscar en todo el servidor, no solo en el canal
        target = None
        for mentioned in message.mentions:
            target = mentioned
            break
        if not target:
            target = await find_member(name, guild.members, guild)
        if target:
            current = data.get("crystal_permits", {}).get(cid, [])
            if target.id not in current:
                current.append(target.id)
                data.setdefault("crystal_permits", {})[cid] = current
                save_data(data)
            mode = data.get("channel_modes", {}).get(cid, "public")
            await update_channel_permissions(voice_channel, mode, current, guild)
            return f"**{target.display_name}** puede entrar al canal."
        return f"No encontré a '{name}'."

    elif action == "deny":
        name = params.get("name", "")
        target = None
        for mentioned in message.mentions:
            target = mentioned
            break
        if not target:
            target = await find_member(name, guild.members, guild)
        if target:
            current = data.get("crystal_permits", {}).get(cid, [])
            if target.id in current:
                current = [x for x in current if x != target.id]
                data.setdefault("crystal_permits", {})[cid] = current
                save_data(data)
            mode = data.get("channel_modes", {}).get(cid, "public")
            await update_channel_permissions(voice_channel, mode, current, guild)
            # si está dentro y el canal es privado/fantasma, échalo también
            try:
                if mode in ("ghost", "crystal") and target in voice_channel.members \
                        and target.id != message.author.id:
                    await target.move_to(None)
            except Exception:
                pass
            return f"**{target.display_name}** ya no puede entrar al canal."
        return f"No encontré a '{name}'."

    elif action == "announce":
        msg_text = params.get("message", "")
        ch = guild.get_channel(PINNED_RESPONSE_CHANNEL_ID)
        if ch and msg_text:
            embed = discord.Embed(
                title="ANUNCIO",
                description=msg_text,
                color=0x4a0080,
                timestamp=datetime.now()
            )
            embed.set_footer(text=f"— {message.author.display_name}")
            await ch.send(embed=embed)
            return f"Anuncio enviado."
        return "No pude enviar el anuncio."

    return None



# =====================================================================
#  REACCIONES — IDENTIDAD
# =====================================================================
@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id:
        return
    
    # ─── NUEVO: Sistema de normas obligatorio ───
    if payload.channel_id == RULES_CHANNEL_ID:
        emoji_str = str(payload.emoji)
        if emoji_str == RULES_ACCEPT_EMOJI or emoji_str == "CHEPA":
            guild = bot.get_guild(payload.guild_id)
            if not guild:
                return
            member = guild.get_member(payload.user_id)
            if not member:
                return
            
            # Marcar que ha aceptado las normas
            uid = str(payload.user_id)
            data.setdefault("accepted_rules", {})[uid] = True
            save_data(data)
            
            # Notificar por DM
            try:
                await member.send(
                    "✅ **Has aceptado los términos y condiciones de Chepa 3.0.**\n\n"
                    "Ahora tienes acceso completo al servidor.\n\n"
                    "Bienvenido al lado oscuro."
                )
            except Exception:
                pass
            
            print(f"[NORMAS] {member.name} ha aceptado las normas", flush=True)
        return
    
    # ─── Sistema de roles por reacción (existente) ───
    uid = str(payload.user_id)
    ud = data.get("user_channels", {}).get(uid)
    if not ud:
        return
    cid = ud["channel_id"] if isinstance(ud, dict) else ud
    mid = ud.get("color_msg_id") if isinstance(ud, dict) else None
    if payload.channel_id != cid or payload.message_id != mid:
        return

    guild = bot.get_guild(payload.guild_id)
    if not guild:
        return
    member = guild.get_member(payload.user_id)
    if not member:
        return

    estr = str(payload.emoji)
    channel = guild.get_channel(cid)
    if not channel:
        return

    if estr not in EMOJI_TO_ROLE:
        try:
            msg = await channel.fetch_message(mid)
            await msg.remove_reaction(payload.emoji, member)
        except Exception:
            pass
        return

    # Quitar roles anteriores
    for e, rid in EMOJI_TO_ROLE.items():
        if e != estr:
            role = guild.get_role(rid)
            if role and role in member.roles:
                try:
                    await member.remove_roles(role)
                except Exception:
                    pass

    # Añadir nuevo rol
    new_role = guild.get_role(EMOJI_TO_ROLE[estr])
    if new_role:
        try:
            await member.add_roles(new_role)
        except Exception as e:
            print(f"[ERROR] Rol: {e}")
            return

        try:
            msg = await channel.fetch_message(mid)
            if msg.embeds:
                emb = msg.embeds[0]
                emb.set_field_at(
                    0,
                    name="╔═══ IDENTIDAD ACTUAL ═══╗",
                    value=f"```fix\n{estr} {new_role.name} ACTIVO\n```",
                    inline=False
                )
                await msg.edit(embed=emb)
            for reaction in msg.reactions:
                if str(reaction.emoji) != estr:
                    try:
                        await reaction.remove(member)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ERROR] Embed identidad: {e}")


@bot.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id:
        return
    uid = str(payload.user_id)
    ud = data.get("user_channels", {}).get(uid)
    if not ud:
        return
    cid = ud["channel_id"] if isinstance(ud, dict) else ud
    mid = ud.get("color_msg_id") if isinstance(ud, dict) else None
    if payload.channel_id != cid or payload.message_id != mid:
        return

    guild = bot.get_guild(payload.guild_id)
    if not guild:
        return
    member = guild.get_member(payload.user_id)
    if not member:
        return

    estr = str(payload.emoji)
    if estr in EMOJI_TO_ROLE:
        role = guild.get_role(EMOJI_TO_ROLE[estr])
        if role and role in member.roles:
            try:
                await member.remove_roles(role)
            except Exception:
                pass

        channel = guild.get_channel(cid)
        if channel and mid:
            try:
                msg = await channel.fetch_message(mid)
                if msg.embeds:
                    current_emoji = None
                    for e, rid in EMOJI_TO_ROLE.items():
                        r = guild.get_role(rid)
                        if r and r in member.roles:
                            current_emoji = e
                            break
                    emb = msg.embeds[0]
                    if current_emoji:
                        r = guild.get_role(EMOJI_TO_ROLE[current_emoji])
                        val = f"```fix\n{current_emoji} {r.name if r else ''} ACTIVO\n```"
                    else:
                        val = "```diff\n- SIN IDENTIDAD ASIGNADA\n```"
                    emb.set_field_at(0, name="╔═══ IDENTIDAD ACTUAL ═══╗", value=val, inline=False)
                    await msg.edit(embed=emb)
            except Exception:
                pass


# =====================================================================
#  VOZ NATIVA — Bender escucha y habla en canales de voz (100% LOCAL, sin APIs)
# =====================================================================
import audioop
import types as _types
import wave as _wave
try:
    import numpy as _np
    from discord.ext import voice_recv as _vr
    VOICE_LIBS_OK = True
except Exception as _e:
    print(f"[VOZ] Librerías de voz no disponibles: {_e}")
    VOICE_LIBS_OK = False

# PARCHE CRÍTICO: voice_recv mata TODO el bucle de recepción si llega un paquete
# opus corrupto ("corrupted stream") -> deja de oír para siempre. Lo envolvemos
# para saltar el paquete malo y seguir escuchando.
if VOICE_LIBS_OK:
    try:
        import logging as _logging
        _logging.getLogger("discord.ext.voice_recv.reader").setLevel(_logging.WARNING)
    except Exception:
        pass
    try:
        from discord.ext.voice_recv.opus import PacketDecoder as _PacketDecoder
        _orig_pop_data = _PacketDecoder.pop_data
        _decode_stats = {"ok": 0, "err": 0}
        def _safe_pop_data(self, *args, **kwargs):
            try:
                r = _orig_pop_data(self, *args, **kwargs)
                if r is not None:
                    _decode_stats["ok"] += 1
                return r
            except Exception:
                _decode_stats["err"] += 1
                if (_decode_stats["ok"] + _decode_stats["err"]) % 400 == 0:
                    tot = _decode_stats["ok"] + _decode_stats["err"]
                    print(f"[VOZ-DIAG] paquetes OK={_decode_stats['ok']} err={_decode_stats['err']} "
                          f"({100*_decode_stats['err']//max(tot,1)}% corrupto)", flush=True)
                return None  # paquete corrupto -> saltar, NO matar el bucle
        _PacketDecoder.pop_data = _safe_pop_data
        print("[VOZ] Parche anti-corrupted-stream aplicado.", flush=True)
    except Exception as _e:
        print(f"[VOZ] No se pudo parchear el decoder: {_e}", flush=True)

    # PARCHE DAVE-DECRYPT (EL IMPORTANTE): voice_recv llama a dave.decrypt(data) con 1
    # argumento, pero davey necesita decrypt(user_id, media_type, packet). Por eso ~40%
    # de paquetes E2E NO se descifran -> audio mutilado. Lo corregimos con la firma real.
    try:
        import davey as _davey
        from discord.ext.voice_recv.opus import PacketDecoder as _PD2

        def _patched_dave_decrypt(self, data: bytes) -> bytes:
            try:
                vc = self.sink.voice_client
                conn = getattr(vc, "_connection", None)
                dave = getattr(conn, "dave_session", None)
                if dave is None:
                    return data
                ready = getattr(dave, "ready", None)
                is_ready = ready() if callable(ready) else bool(ready)
                if not is_ready:
                    return data  # sin E2E (passthrough): el dato ya es opus válido
                uid = getattr(self, "_cached_id", None)
                if uid is None:
                    try:
                        uid = vc._get_id_from_ssrc(self.ssrc)
                        self._cached_id = uid
                    except Exception:
                        uid = None
                if uid is None:
                    return data
                dec = dave.decrypt(uid, _davey.MediaType.audio, data)
                return dec if dec is not None else data
            except Exception:
                return data

        _PD2._dave_decrypt = _patched_dave_decrypt
        print("[VOZ] Parche DAVE-decrypt (firma correcta) aplicado.", flush=True)
    except Exception as _e:
        print(f"[VOZ] No se pudo parchear dave_decrypt: {_e}", flush=True)


# Cargar libopus (necesario para enviar/recibir voz). En este contenedor no se autocarga.
if not discord.opus.is_loaded():
    for _lib in ("/usr/lib/x86_64-linux-gnu/libopus.so.0", "libopus.so.0", "opus"):
        try:
            discord.opus.load_opus(_lib)
            if discord.opus.is_loaded():
                print(f"[VOZ] opus cargado: {_lib}", flush=True)
                break
        except Exception:
            continue
    if not discord.opus.is_loaded():
        print("[VOZ] AVISO: no se pudo cargar opus, la voz puede fallar.", flush=True)

PIPER_PATH = "/app/voice_models/bender.onnx"
_whisper_model = None   # alias de _whisper_base (compatibilidad)
_whisper_tiny = None    # rápido: detección de "Bender" (siempre lo conserva)
_whisper_base = None     # preciso: contenido cuando te diriges a Bender
_piper_voice = None

# Palabra de activación. Whisper confunde "Bender" con varias; usamos un regex que
# cubre b/v/p + en/én + d + opcional e/i + r, como palabra suelta (sin falsos positivos).
_WAKE_RE = re.compile(r"\b[bvp][eéá]n+d[aeiou]?r?[aeio]?\b", re.IGNORECASE)


_WAKE_AMBIGUOS = {"vender", "vénder", "bende", "vende", "pender",
                  "vendr", "bendr", "vendar", "bénde", "vénde", "render",
                  "venderte", "bendernos", "benderme", "venderme", "vendernos",
                  "venderos", "bendero", "vendera", "bendeo", "vendeo", "bendito",
                  "vander", "bander", "mender", "guender", "bemder", "vemder", "wender"}


def _fuzzy_bender(tok: str, max_dist: int = 2, lenient: bool = True) -> bool:
    """¿Es 'tok' una versión que el STT podría haber sacado de 'Bender'?
    Tolerancia adaptativa: si el token NI SIQUIERA empieza por b/v/m/w, exigimos
    distancia 1 — con dist 2 colaban 'poder', 'perder' y media lengua española."""
    if len(tok) < 3 or len(tok) > 9:
        return False
    md = max_dist
    if tok[0] not in "bvmw":
        md = min(md, 1)
    for target in ("bender", "vender", "pender", "mender", "bander", "vander",
                   "bénder", "bendel", "vendere", "guender", "ender", "iender",
                   "wender", "render", "gender"):
        a, b = tok, target
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        if prev[-1] <= md:
            return True
    return False


# Patrón fonético del nombre: [b/v/p/m/w opcional] + en/én + d. Capta bender, vender,
# vendel, endel, mender, pender, bende, vende... pero NO "entiendo/entender" (empiezan ent).
_BENDER_RE = re.compile(r"^[bvpmw]?[eéáaio]n+d")


# Palabras españolas normales que se PARECEN a Bender pero NO lo son (evita activarse
# con vuestra charla). OJO: "vende/vender/vendedor" SÍ disparan a propósito (el usuario
# quiere máxima reactividad con esa familia); aquí solo bloqueamos lo que NO es el nombre.
_NOT_BENDER = {
    "vendo", "vendí", "vendia", "vendía", "vendrá", "vendria",
    "vendría", "vendido", "vendida", "vendidos", "vendio", "vendió",
    "vendieron", "vendiera", "vendiendo",
    "venta", "ventas", "vente", "venda", "vendan",
    "pende", "depende", "tiende", "entiende", "vengo", "venga", "vengas",
    "vendiendo", "menudo", "tremendo",
    # comunes que casan con el patrón -nd- pero NO son el nombre:
    "anda", "andas", "andan", "ando", "andar", "andando",
    "onda", "ondas", "ronda", "rondas", "honda", "hondas",
    "manda", "mandas", "mandan", "mandar", "banda", "bandas",
    "linda", "lindo", "lindas", "lindos", "funda", "fundas",
    "blando", "blanda", "brenda", "senda", "sonda", "vanda",
    # familia -ender/-oder: a distancia 1-2 de 'pender/bender' pero comunísimas
    "poder", "perder", "prender", "aprender", "entender", "atender",
    "tender", "extender", "encender", "defender", "ofender", "pretender",
    "sorprender", "comprender", "render", "ceder", "morder", "perde",
    "pierde", "pierden", "puede", "pueden", "poden", "joder",
}


# ── CLASIFICADOR VOCATIVO ────────────────────────────────────────────────────
# En español B y V son el MISMO fonema: ningún STT va a distinguir "Bender" de
# "vender" por el sonido. La señal está en CÓMO se usa la palabra:
#   - VOCATIVO (llamada): al principio de la frase ("Bender, pon música"), tras
#     muletilla ("oye vender..."), aislado ("¡Bender!"), o seguido de 2ª persona
#     ("...vender tienes...").
#   - VERBO (charla): precedido de gramática verbal ("voy A vender", "ES vender",
#     "SE vende"), con clítico ("venderNOS", "venderLA"), o seguido de artículo
#     ("vender LA casa").
_MULETILLAS = {"oye", "eh", "ey", "hola", "pero", "y", "que", "venga", "escucha",
               "mira", "va", "vale", "bueno", "pues", "ah", "o", "a", "e", "uy", "buenas"}
_VERB_CTX_BEFORE = {"a", "de", "para", "por", "se", "me", "te", "nos", "lo", "la",
    "los", "las", "le", "les", "voy", "vas", "va", "vamos", "van", "iba", "ibas",
    "quiero", "quieres", "quiere", "queremos", "quieren", "puedo", "puedes",
    "puede", "podemos", "pueden", "debo", "debes", "debe", "es", "era", "fue",
    "ser", "sería", "estoy", "está", "estás", "sin", "al", "del", "el", "un",
    "mi", "su", "tu", "este", "ese", "suelo", "sueles", "intento", "intenta"}
_ART_AFTER = {"la", "el", "los", "las", "una", "un", "unas", "unos", "mi", "tu",
              "su", "este", "esta", "estos", "estas", "ese", "esa", "eso", "esto",
              "todo", "toda", "algo", "cosas", "mierda", "mierdas"}
_DIRECTED_AFTER = {"tienes", "eres", "sabes", "puedes", "podrías", "podrias",
    "quieres", "dime", "di", "dinos", "pon", "ponme", "ponnos", "pones", "quita",
    "cállate", "callate", "calla", "vete", "sal", "oye", "cuánto", "cuanto",
    "cuánta", "cuanta", "cuántos", "cuantos", "qué", "cómo", "quién", "quien",
    "dónde", "donde", "cuál", "cual", "busca", "búscame", "buscame", "cuéntame",
    "cuentame", "haz", "hazme", "dame", "danos", "responde", "respóndeme",
    "respondeme", "contesta", "contéstame", "contestame", "escucha", "escúchame",
    "escuchame", "mira", "ayuda", "ayúdame", "ayudame", "explica", "explícame",
    "explicame", "dile", "cuenta", "canta", "habla", "háblame", "hablame",
    "juega", "tú", "te", "hola", "buenas", "gracias", "venga", "para", "deja"}
_VEND_CLITICS = ("me", "te", "le", "lo", "la", "nos", "os", "les", "los", "las",
                 "selo", "sela", "selos", "selas", "sele")


def _has_wake_word(text: str) -> bool:
    low = text.lower()
    # El nombre con B es inequívoco (whisper sí lo saca a veces): siempre dispara.
    if "bender" in low or "bénder" in low:
        return True
    tokens = re.findall(r"[a-záéíóúñ]+", low)
    if not tokens:
        return False
    for i, tok in enumerate(tokens[:10]):
        if tok in _NOT_BENDER:
            continue
        is_cand = (tok in _WAKE_AMBIGUOS
                   or (3 <= len(tok) <= 9 and _BENDER_RE.match(tok))
                   or _fuzzy_bender(tok))
        if not is_cand:
            continue
        # Clítico de infinitivo ("vendernos", "venderla") → es el verbo
        if len(tok) > 6 and tok.startswith("vend") and tok[6:] in _VEND_CLITICS:
            continue
        prev = tokens[i - 1] if i > 0 else None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        # Gramática verbal delante ("voy a vender", "es vender", "se vende") → verbo
        if prev in _VERB_CTX_BEFORE:
            continue
        # Artículo/posesivo detrás ("vender la casa", "vender una moto") → verbo
        if nxt in _ART_AFTER:
            continue
        # Posición inicial (saltando muletillas) → vocativo: es una llamada
        j = i
        while j > 0 and tokens[j - 1] in _MULETILLAS:
            j -= 1
        if j == 0:
            return True
        # En medio de frase: solo si hay evidencia de 2ª persona dirigida
        if nxt in _DIRECTED_AFTER or prev in ("oye", "eh", "ey", "hola", "escucha", "mira", "venga"):
            return True
    if len(tokens) >= 2 and _fuzzy_bender(tokens[0] + tokens[1], lenient=False):
        return True
    return False

# Estado de sesión por guild
voice_sessions = {}  # guild_id -> dict

# Diagnóstico: VOICE_DIAG loguea tiempos/descartes; VOICE_SAVE_SAMPLES guarda WAVs
# (apagado: ya tengo corpus suficiente y ahorra I/O en cada frase).
VOICE_DIAG = True
VOICE_SAVE_SAMPLES = False
_utt_counter = 0

# Ventana de conversación: tras decir "Bender" una vez, sigues hablándole sin repetir
# el nombre durante CONVO_WINDOW segundos (natural, como con un humano).
_convo_windows = {}   # (guild_id, uid) -> timestamp de expiración
CONVO_WINDOW = 8
# Última frase dirigida por usuario (anti-repetición): (guild_id, uid) -> (letras, ts)
_last_addressed = {}

# LEGACY (pipeline viejo de Whisper): con Vosk como STT principal ya no hay
# serialización global ni colas de PCM. Se mantienen definidos porque alguna
# ruta de limpieza los referencia, pero están inertes.
_stt_busy = False
_stt_busy_since = 0.0
_stt_pending = {}
_gate_count = 0
_whisper_running = False  # True mientras Whisper (respaldo) ejecuta en su hilo — evita colas
from concurrent.futures import ThreadPoolExecutor
# Executors DEDICADOS de 1 hilo: serializan Whisper y Piper. Evita que transcripciones
# lentas se solapen y acumulen hilos compitiendo por la CPU (la causa de que "cada vez
# tarde más"). Una a la vez, sin cascada de hilos zombis.
_stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
# Executor separado para Vosk (NO comparte hilo con Whisper). 2 workers: con
# 4-5 personas hablando, 1 solo hilo encolaba y las transcripciones salían con
# 8-12s de retraso (respuestas tardías). El modelo Kaldi es thread-safe.
_vosk_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vosk")

# ─── Vosk DESACTIVADO — sustituido por Groq Whisper API ────────────────
_vosk_model = None

def _load_vosk():
    """No-op: Vosk sustituido por Groq Whisper API."""
    print("[VOZ] Vosk desactivado — usando Groq Whisper API.", flush=True)

def _vosk_transcribe(audio_f32):
    """No-op: usar Groq Whisper en su lugar."""
    return None


# Gramática restringida — ya no se usa con Groq pero se mantiene por compatibilidad
_VOSK_WAKE_GRAMMAR = '["vender", "venden", "vende", "bende", "tender", "mente", "[unk]"]'
_VOSK_WAKE_HITS = ("vender", "venden", "vende", "bende", "tender")

def _vosk_wake_check(audio_f32) -> bool:
    """No-op: Groq transcribe todo, no necesita gramática restrictiva."""
    return False


_engines_lock = threading.Lock()

def _load_voice_engines():
    """Carga whisper tiny (detección rápida) + base (contenido) + piper (voz)."""
    global _whisper_model, _whisper_tiny, _whisper_base, _piper_voice
    if _whisper_base is not None and _piper_voice is not None:
        return True
    # Lock: on_ready (precarga) y el auto-reingreso pueden llamar a la vez -> sin esto
    # se cargan DOS modelos en paralelo = pico de CPU de arranque (inferencias de 30s+).
    with _engines_lock:
        if _whisper_base is not None and _piper_voice is not None:
            return True
        return _load_voice_engines_locked()


def _load_voice_engines_locked():
    global _whisper_model, _whisper_tiny, _whisper_base, _piper_voice
    try:
        from piper import PiperVoice
        print("[VOZ] Cargando motores (groq whisper + piper)...", flush=True)
        # Vosk desactivado — Groq Whisper es el motor principal
        _load_vosk()
        # Piper TTS (voz local en español)
        _piper_voice = PiperVoice.load(PIPER_PATH)
        # faster-whisper desactivado — Groq Whisper lo sustituye
        _whisper_base = None
        _whisper_tiny = None
        _whisper_model = None
        if _GROQ_CLIENT:
            print("[VOZ] Groq Whisper API activo como motor STT.", flush=True)
        else:
            print("[VOZ] WARNING: Groq no disponible — falta GROQ_API_KEY.", flush=True)
        print("[VOZ] Motores de voz listos (groq whisper + piper tts).", flush=True)
        return True
    except Exception as e:
        print(f"[VOZ] Error cargando motores: {e}", flush=True)
        return False


def _pcm48_to_whisper(pcm: bytes):
    """Discord da PCM 48kHz 16-bit estéreo -> float32 mono 16kHz para whisper.
    Normaliza el volumen porque el audio de Discord suele llegar bajo y whisper
    lo toma como silencio (transcripción vacía)."""
    try:
        # Alinear a frame estéreo de 16 bits (4 bytes) o audioop revienta
        rem = len(pcm) % 4
        if rem:
            pcm = pcm[:len(pcm) - rem]
        if not pcm:
            return _np.zeros(0, dtype=_np.float32)
        mono = audioop.tomono(pcm, 2, 0.5, 0.5)
        pcm16k, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)
        audio = _np.frombuffer(pcm16k, dtype=_np.int16).astype(_np.float32) / 32768.0
        # Normalización por RMS (volumen MEDIO): sube las voces flojas/lejanas (que
        # tiny transcribe como puré) SIN tocar las que ya llegan bien. Solo amplifica
        # (gain>1), nunca baja. Luego limita picos para no saturar tras el gain.
        if len(audio):
            rms = float(_np.sqrt(_np.mean(audio ** 2)))
            if rms > 1e-4:
                gain = min(0.10 / rms, 3.0)   # boost SUAVE (tope x3): sube voces flojas sin distorsionar
                if gain > 1.0:
                    audio = audio * gain
            peak = float(_np.abs(audio).max())
            if peak > 0.97:
                audio = audio * (0.97 / peak)
        return audio
    except Exception as e:
        print(f"[VOZ] Error convirtiendo audio: {e}")
        return _np.zeros(0, dtype=_np.float32)


def _transcribe_with(model, audio) -> str:
    global _whisper_running
    _whisper_running = True
    try:
        _inf = time.time()
        # SIN hotwords/initial_prompt de "Bender": sesgaban a Whisper a INVENTARSE
        # "Bender" donde no lo había (sobre-detección). Transcripción fiel + detección
        # difusa por nuestra cuenta = capta los churros reales sin falsos Benders.
        segs, _ = model.transcribe(
            audio, language="es", beam_size=1, vad_filter=True,
            no_speech_threshold=0.6, condition_on_previous_text=False,
        )
        txt = " ".join(s.text for s in segs).strip()
        _d = time.time() - _inf
        if _d > 2.5:
            print(f"[VOZ-PROF] inferencia pura whisper={_d:.1f}s para audio de {len(audio)/16000:.1f}s", flush=True)
        low = txt.lower()
        if not re.sub(r"[^\wáéíóúñ]", "", low):
            return ""
        if any(h in low for h in _WHISPER_HALLUCINATIONS):
            return ""
        return txt
    except Exception as e:
        print(f"[VOZ] Error transcribiendo: {e}")
        return ""
    finally:
        _whisper_running = False


def _transcribe_tiny(audio) -> str:
    return _transcribe_with(_whisper_tiny, audio)


def _transcribe(audio) -> str:   # base (preciso)
    return _transcribe_with(_whisper_base, audio)


def _synth_wav(text: str, path: str):
    try:
        with _wave.open(path, "wb") as wf:
            _piper_voice.synthesize_wav(text, wf)
        return True
    except Exception as e:
        print(f"[VOZ] Error sintetizando: {e}")
        return False


_TTS_UNI = ["cero","uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve",
    "diez","once","doce","trece","catorce","quince","dieciséis","diecisiete","dieciocho",
    "diecinueve","veinte","veintiuno","veintidós","veintitrés","veinticuatro","veinticinco",
    "veintiséis","veintisiete","veintiocho","veintinueve"]
_TTS_DEC = ["","","","treinta","cuarenta","cincuenta","sesenta","setenta","ochenta","noventa"]
_TTS_CEN = ["","ciento","doscientos","trescientos","cuatrocientos","quinientos","seiscientos",
    "setecientos","ochocientos","novecientos"]

def _n2w(n):
    n = int(n)
    if n < 0: return "menos " + _n2w(-n)
    if n < 30: return _TTS_UNI[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _TTS_DEC[d] + ("" if u == 0 else " y " + _TTS_UNI[u])
    if n == 100: return "cien"
    if n < 1000:
        c, r = divmod(n, 100)
        return _TTS_CEN[c] + ("" if r == 0 else " " + _n2w(r))
    if n < 1000000:
        m, r = divmod(n, 1000)
        pref = "mil" if m == 1 else _n2w(m) + " mil"
        return pref + ("" if r == 0 else " " + _n2w(r))
    m, r = divmod(n, 1000000)
    pref = "un millón" if m == 1 else _n2w(m) + " millones"
    return pref + ("" if r == 0 else " " + _n2w(r))

# marcas / inglés que espeak deletrea o lee fatal -> escritura fonética
_TTS_FON = {
    "iphone":"aifon","iphones":"aifons","ipad":"aipad","ipod":"aipod","imac":"aimac",
    "macbook":"mac buc","airpods":"érpods","whatsapp":"guasap","wasap":"guasap",
    "youtube":"yutub","wifi":"guifi","google":"gúguel","gmail":"yimeil",
    "playstation":"pleiesteichon","xbox":"éxbox","steam":"estim","discord":"discord",
    "nvidia":"envidia","spotify":"espótifai","tiktok":"tictoc","cpu":"ce pe u",
    "gpu":"ge pe u","fps":"efe pe ese",
}

def _normalize_for_tts(t):
    """Pasa números a palabras y símbolos/marcas a algo que Piper diga bien.
    (Sin esto: 25°C -> 'vigésimo quinto', iPhone deletreado, etc.)"""
    # grados: el símbolo ° (y el ordinal º) hace que espeak lea ordinales
    t = re.sub(r"(\d+)\s*[°º]\s*[cC]?", r"\1 grados", t)
    t = t.replace("°", "").replace("º", "")
    t = re.sub(r"(\d+)\s*%", r"\1 por ciento", t)
    t = re.sub(r"(\d+)\s*€|€\s*(\d+)", lambda m: (m.group(1) or m.group(2)) + " euros", t)
    t = re.sub(r"(\d+)\s*\$|\$\s*(\d+)", lambda m: (m.group(1) or m.group(2)) + " dólares", t)
    # horas 14:30 -> "catorce treinta"
    def _hora(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return _n2w(h) + (" en punto" if mi == 0 else " " + _n2w(mi))
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", _hora, t)
    # decimales 3,5 -> "tres coma cinco"
    def _dec(m):
        return _n2w(m.group(1)) + " coma " + " ".join(_n2w(int(d)) for d in m.group(2))
    t = re.sub(r"\b(\d+)[.,](\d+)\b", _dec, t)
    # enteros sueltos -> palabras
    t = re.sub(r"\d+", lambda m: _n2w(m.group()), t)
    # marcas / inglés
    for a, b in _TTS_FON.items():
        t = re.sub(r"\b" + a + r"\b", b, t, flags=re.I)
    return t


def _clean_for_speech(text: str) -> str:
    """Quita markdown/emojis/links para que la voz suene natural y NO diga fuentes."""
    t = re.sub(r"[*_`#>~]", "", text)
    t = re.sub(r"https?://\S+", "", t)
    # quitar dominios/fuentes sueltos (europapress.es, elpais.es, eltiempo.es...)
    t = re.sub(r"\b[\w\-]+\.(es|com|org|net|io|tv|gg|ca|info|news)\b", "", t, flags=re.I)
    # quitar referencias tipo "según X", "[fuente]", "(fuente: ...)"
    t = re.sub(r"\(\s*fuente[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\[[^\]]*\]", "", t)
    # normalizar números/símbolos/marcas para que Piper los DIGA bien (antes del strip)
    t = _normalize_for_tts(t)
    # quitar emojis y símbolos raros
    t = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,;:!?¡¿()\-\"']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or "No tengo nada que decirte."


if VOICE_LIBS_OK:
    class BenderSink(_vr.AudioSink):
        def __init__(self, guild_id):
            super().__init__()
            self.guild_id = guild_id

        def wants_opus(self):
            return False

        def write(self, user, data):
            sess = voice_sessions.get(self.guild_id)
            if user is None or getattr(user, "bot", False):
                return
            if not sess or not sess.get("active"):
                return
            # Buffer rotatorio para CLIPS (captura SIEMPRE, también si Bender habla)
            _pcm0 = getattr(data, "pcm", None)
            if _pcm0 and not getattr(user, "bot", False):
                _ring = sess.setdefault("clip_ring", {})
                _dq = _ring.setdefault(user.id, deque())
                _nowm = time.monotonic()
                _dq.append((_nowm, bytes(_pcm0)))
                _cut = _nowm - CLIP_BUFFER_SECONDS
                while _dq and _dq[0][0] < _cut:
                    _dq.popleft()
            # OJO: NO cortamos aquí si speaking=True.
            # El audio se acumula en buffers; si el user dice "Bender" mientras
            # Bender habla, se transcribe igual (Vosk es barato) y el TEXTO
            # queda diferido hasta que Bender termine de hablar.
            pcm = getattr(data, "pcm", None)
            if not pcm:
                return
            buf = sess["buffers"].setdefault(
                user.id, {"pcm": bytearray(), "last": time.time(), "last_ts": None}
            )
            # RELLENO DE SILENCIO: si se perdieron paquetes (huecos en el timestamp RTP),
            # insertamos silencio para PRESERVAR EL TIEMPO. Sin esto, el audio se
            # comprime/acelera y Whisper no lo entiende.
            pkt = getattr(data, "packet", None)
            ts = getattr(pkt, "timestamp", None)
            if ts is not None and buf.get("last_ts") is not None:
                gap = ts - buf["last_ts"]
                # cada frame = 960 samples (20ms @48k). Hueco normal = 960.
                if 960 < gap < 48000:   # hasta ~1s de hueco; más = pausa real
                    missing = gap - 960
                    buf["pcm"].extend(b"\x00" * (missing * 2 * 2))  # samples*2bytes*2ch
            buf["last_ts"] = ts
            buf["pcm"].extend(pcm)
            buf["last"] = time.time()

        def cleanup(self):
            pass


async def _voice_listen_loop(guild):
    """Detecta fin de frase (silencio) y lanza el procesado.
    NUEVO PIPELINE: Vosk transcribe TODO al instante (100-500ms), así que ya no
    hay cola de PCM pendiente ni flag global de STT ocupado — cada frase se
    procesa en cuanto termina. Solo queda el diferido de TEXTO (respuestas
    que esperan a que Bender termine de hablar)."""
    BYTES_PER_SEC = 48000 * 2 * 2
    MIN_BYTES = int(BYTES_PER_SEC * 0.20)   # mínimo 200ms (era 350ms; cubría "Bender" justo)
    MAX_BYTES = int(BYTES_PER_SEC * 20)
    sess = voice_sessions.get(guild.id)
    alone_ticks = 0
    _listen_check = 0
    while sess and sess.get("active"):
        await asyncio.sleep(0.08)
        try:
            # Auto-reparar listener caído: si el bot deja de escuchar (vc.stop()
            # desde música/TTS lo puede tirar), re-engancha sin resetear la sesión.
            _listen_check += 1
            if _listen_check >= 50:  # cada ~4s
                _listen_check = 0
                vc = sess.get("vc")
                if vc and vc.is_connected():
                    try:
                        if not vc.is_listening():
                            print("[VOZ] Listener caído, re-enganchando.", flush=True)
                            vc.listen(BenderSink(guild.id))
                    except Exception:
                        pass
            # Auto-reparar speaking trabado: si speaking=True pero no está reproduciendo y
            # hace >3s, forzar a False.
            if sess.get("speaking"):
                vc = sess.get("vc")
                if vc and not vc.is_playing() and not sess.get("busy"):
                    stuck_since = sess.get("_speaking_stuck_since")
                    if not stuck_since:
                        sess["_speaking_stuck_since"] = time.time()
                    elif time.time() - stuck_since > 3:
                        print("[VOZ] Auto-reparando speaking trabado.", flush=True)
                        sess["speaking"] = False
                        sess["buffers"].clear()
                        sess.pop("_speaking_stuck_since", None)
                else:
                    sess.pop("_speaking_stuck_since", None)
            # Auto-reparar busy trabado (>30s sin soltar = algo se colgó).
            if sess.get("busy"):
                busy_since = sess.get("_busy_since")
                if not busy_since:
                    sess["_busy_since"] = time.time()
                elif time.time() - busy_since > 60:
                    print("[VOZ] Auto-reparando busy trabado (>60s).", flush=True)
                    sess["busy"] = False
                    sess["speaking"] = False
                    sess.pop("_busy_since", None)
                    sess.pop("_speaking_stuck_since", None)
            else:
                sess.pop("_busy_since", None)
            # ── DRENADO de respuesta diferida ────────────────────────────────────
            # REGLA DE FRESCURA: un ciclo completo de respuesta (LLM + hablar) dura
            # ~20s, así que el diferido vive 20s — si no, un "Bender" dicho mientras
            # habla caducaba SIEMPRE. Más de 20s sí se descarta (respuesta zombie).
            _FRESH = 20.0
            if not sess.get("busy") and not sess.get("speaking"):
                _def = sess.pop("_deferred_voice", None)
                if _def:
                    _duid, _dtxt, _dts = _def
                    if time.time() - _dts < _FRESH:
                        print(f"[VOZ] Drenando diferido: '{_dtxt}'", flush=True)
                        asyncio.create_task(_bender_voice_respond(guild, _duid, _dtxt))
                    else:
                        print(f"[VOZ] Diferido caducado ({time.time()-_dts:.0f}s), descartado.", flush=True)
            # ────────────────────────────────────────────────────────────────────
            ch = sess["vc"].channel if sess.get("vc") and sess["vc"].is_connected() else None
            if ch is None:
                print("[VOZ] Conexión de voz perdida, limpiando sesión.", flush=True)
                await leave_voice(guild)
                break
            if len([m for m in ch.members if not m.bot]) == 0:
                alone_ticks += 1
                if alone_ticks > 8:
                    print("[VOZ] Solo en la llamada, me salgo.", flush=True)
                    await leave_voice(guild)
                    break
            else:
                alone_ticks = 0
            now = time.time()
            for uid, buf in list(sess["buffers"].items()):
                # 0.40s de silencio = frases más completas. Con Vosk (300ms) el total
                # sigue siendo <1s de reacción; con Whisper había que rascar de aquí.
                silencio = (now - buf["last"]) > 0.40
                demasiado_largo = len(buf["pcm"]) >= MAX_BYTES
                if buf["pcm"] and (silencio or demasiado_largo):
                    pcm = bytes(buf["pcm"])
                    buf["pcm"] = bytearray()
                    buf["last_ts"] = None
                    if len(pcm) > MIN_BYTES:
                        asyncio.create_task(_handle_voice_utterance(guild, uid, pcm))
        except Exception as _e:
            print(f"[VOZ] Error en bucle de escucha (ignorado, sigo vivo): {_e}", flush=True)
            await asyncio.sleep(0.2)
            continue


async def _handle_voice_utterance(guild, uid, pcm, _queued_at: float = 0.0):
    """PIPELINE GROQ: transcribe con Groq Whisper API, detecta wake word, responde.
    Sustituye el pipeline Vosk-primero. Groq transcribe todo en una sola llamada."""
    global _utt_counter
    sess = voice_sessions.get(guild.id)
    if not sess or not sess.get("active"):
        return
    # Frescura: si este audio fue encolado hace >8s, tirarlo directamente.
    if _queued_at and time.time() - _queued_at > 8.0:
        return
    # ── DIAGNÓSTICO: guardar muestras solo si se pide (apagado por defecto)
    if VOICE_SAVE_SAMPLES:
        try:
            _utt_counter += 1
            os.makedirs("/tmp/vsamples", exist_ok=True)
            with _wave.open(f"/tmp/vsamples/utt_{_utt_counter:03d}.wav", "wb") as _wf:
                _wf.setnchannels(2); _wf.setsampwidth(2); _wf.setframerate(48000)
                _wf.writeframes(pcm)
        except Exception:
            pass
    # Tope 15s de audio
    if len(pcm) > int(48000 * 2 * 2 * 15):
        if VOICE_DIAG:
            print(f"[VOZ-SKIP] larga ({len(pcm)//(48000*2*2)}s), descartada", flush=True)
        return
    # VAD: filtrar silencio antes de gastar en Groq
    if not _has_speech_energy(pcm):
        return
    _tt = time.time()
    in_window = time.time() < _convo_windows.get((guild.id, uid), 0)
    # ── CONVERTIR PCM a OGG para Groq ──────────────────────────────────────
    loop = asyncio.get_event_loop()
    audio_ogg = await loop.run_in_executor(_vosk_executor, _pcm48_to_ogg, pcm)
    if not audio_ogg:
        return
    duration = len(pcm) / (48000 * 2 * 2)
    # ── TRANSCRIBIR CON GROQ WHISPER ───────────────────────────────────────
    text = await _groq_transcribe(audio_ogg, duration)
    text = (text or "").strip()
    # Filtrar alucinaciones de Whisper
    if _is_hallucination(text):
        if VOICE_DIAG:
            member = guild.get_member(uid)
            print(f"[VOZ] alucinación filtrada: '{text}'", flush=True)
        return
    _letters = re.sub(r"[^a-záéíóúñ]", "", text.lower())
    if len(_letters) < 2:
        return
    # ── DETECTAR WAKE WORD ─────────────────────────────────────────────────
    wake = bool(text) and _has_wake_word(text)
    addressed = wake or (in_window and len(_letters) >= 4)
    if VOICE_DIAG:
        member = guild.get_member(uid)
        tag = "✓dirigido" if addressed else "·ambiente"
        print(f"[VOZ] ({time.time()-_tt:.1f}s {tag}) {member.display_name if member else uid}: '{text}'", flush=True)
    if not addressed:
        return
    # ── ANTI-REPETICIÓN ────────────────────────────────────────────────────
    _akey = (guild.id, uid)
    _prev = _last_addressed.get(_akey)
    _nowa = time.time()
    if _prev and _nowa - _prev[1] < 12.0:
        _a, _b = _prev[0], _letters
        if _a and _b and min(len(_a), len(_b)) >= 10 and (_a.startswith(_b) or _b.startswith(_a)):
            if VOICE_DIAG:
                print(f"[VOZ-DUP] repetición ignorada: '{text}'", flush=True)
            return
    _last_addressed[_akey] = (_letters, _nowa)
    low = text.lower()
    # Salir de la llamada por voz — lista amplia + raíces
    _LEAVE_WORDS = ("vete", "bete", "lárgate", "largate", "larga", "sal del canal",
                    "sal de la llamada", "sal de aqui", "sal de aquí", "desconect",
                    "pírate", "pirate", "pira", "fuera", "fera", "márchate", "marchate",
                    "marcha", "déjanos", "dejanos", "abandona", "que te vayas", "vete ya",
                    "piro", "chao", "chau", "adiós", "adios", "esfúmate", "esfumate",
                    "ya puedes irte", " vete")
    if any(w in low for w in _LEAVE_WORDS):
        _convo_windows.pop((guild.id, uid), None)
        await _speak(guild, "Vale, me piro. Hasta luego, cabrones.")
        await asyncio.sleep(3.0)
        await leave_voice(guild)
        return
    # Cooldown anti-rayada
    if time.time() < sess.get("cooldown_until", 0):
        return
    # Abrir ventana de conversación SOLO si dijo "Bender" de verdad
    if wake:
        _convo_windows[(guild.id, uid)] = time.time() + CONVO_WINDOW
    # ── RESPUESTA O DIFERIDO ─────────────────────────────────────────────────
    # Si Bender está respondiendo a otro (busy) o hablando: diferimos esta
    # respuesta para cuando termine, en lugar de tirarla a la basura.
    if sess.get("busy") or sess.get("speaking"):
        sess["_deferred_voice"] = (uid, text, time.time())
        print(f"[VOZ] Diferida (busy/speaking): '{text}'", flush=True)
        return
    try:
        await asyncio.wait_for(_bender_voice_respond(guild, uid, text), timeout=90.0)
    except asyncio.TimeoutError:
        print("[VOZ] respond >30s, liberando (anti-cuelgue).", flush=True)
        _s = voice_sessions.get(guild.id)
        if _s:
            _s["busy"] = False
            _s.pop("_busy_since", None)


async def _bender_voice_respond(guild, uid, text):
    sess = voice_sessions.get(guild.id)
    if not sess:
        return
    # Guardia: no procesar dos cosas a la vez (que no se solape/hable encima)
    if sess.get("busy"):
        return
    sess["busy"] = True
    sess["_busy_since"] = time.time()
    _t0 = time.time()
    try:
        member = guild.get_member(uid)
        vc = sess["vc"]
        channel = vc.channel
        miembros = [m.display_name for m in channel.members if not m.bot]
        _stored_owner = data.get("voice_channel_owners", {}).get(str(channel.id))
        is_owner = (str(_stored_owner) == str(uid) if _stored_owner is not None else False) or \
                   (member.guild_permissions.administrator if member else False)

        # 1) ¿Es un comando de control? Para el DUEÑO/admin, corremos el clasificador
        #    SIEMPRE (el texto de voz suele venir roto y las keywords no casan; el LLM
        #    interpreta mejor). Si no es una orden, devuelve "desconocido" y seguimos a chat.
        low = text.lower()
        # --- CLIPS por voz (cualquiera en la llamada, no solo el owner) ---
        _cfg_secs = _parse_clip_config(low)
        if _cfg_secs:
            data.setdefault("clip_config", {})[str(uid)] = _cfg_secs
            save_data(data)
            await _speak(guild, _clean_for_speech(f"Hecho, clips de {_cfg_secs} segundos."))
            return
        if _is_clip_cmd(low):
            await _voice_make_clip(guild, uid)
            return
        # --- MUSICA por voz ---
        if any(w in low for w in _MUSIC_STOP):
            await _speak(guild, _clean_for_speech(await stop_music(guild)))
            return
        if any(w in low for w in _MUSIC_SKIP):
            await _speak(guild, _clean_for_speech(await skip_music(guild)))
            return
        _mq = _parse_music_query(low)
        if _mq is not None:
            _nom = guild.get_member(uid).display_name if guild.get_member(uid) else "alguien"
            await _speak(guild, _clean_for_speech(await enqueue_music(guild, _mq, _nom)))
            return
        _mv = _parse_music_volume(low)
        if _mv is not None:
            await _speak(guild, _clean_for_speech(await set_music_volume(guild, _mv)))
            return
        _cmd_hint = any(k in low for k in (
            "expulsa", "echa", "saca", "tira", "fuera", "kick", "larga",
            "modo", "fantasma", "privado", "publico", "público", "cristal",
            "renombra", "renombr", "ponle", "llama al canal", "pon el canal",
            "permite", "deja entrar", "mete a", "invita", "agrega", "añade",
            "quita", "bloquea", "prohibe", "prohíbe", "de la lista",
            "transfiere", "transfier", "admin", "dueño"))
        if _cmd_hint:
            stored_owner = data.get("voice_channel_owners", {}).get(str(channel.id))
            print(f"[VOZ-DEBUG] cmd_hint=True is_owner={is_owner} uid={uid} "
                  f"stored_owner={stored_owner} channel={channel.id} "
                  f"admin={member.guild_permissions.administrator if member else 'N/A'} "
                  f"text='{text[:80]}'", flush=True)
        if _cmd_hint:
            if not is_owner:
                # No es owner: avisar por voz de que no tiene permisos
                await _speak(guild, _clean_for_speech(
                    f"No eres el dueño de este canal, {member.display_name if member else 'colega'}. "
                    f"No puedes darme órdenes de control."
                ))
                return
            try:
                ctx = (f"Estáis en un canal de voz llamado '{channel.name}'. "
                       f"Miembros presentes: {miembros}. El usuario habla por voz (puede venir "
                       f"algo mal transcrito; interpreta su intención).")
                action_data = await call_ai_action(text, ctx, str(uid))
                action = action_data.get("action", "desconocido")
                params = action_data.get("params", {})
                print(f"[VOZ-DEBUG] LLM action={action} params={params}", flush=True)
            except Exception as e:
                print(f"[VOZ-DEBUG] call_ai_action exception: {e}", flush=True)
                action, params = "desconocido", {}
            if action in ("kick", "rename", "modo", "allow", "deny", "transferir"):
                print(f"[VOZ] Comando de voz: {action} {params}", flush=True)
                try:
                    fake_msg = _types.SimpleNamespace(guild=guild, mentions=[])
                    resp = await execute_voice_action(action, params, channel, fake_msg)
                    if resp:
                        await _speak(guild, _clean_for_speech(resp))
                        return
                except Exception as e:
                    print(f"[VOZ] Error ejecutando acción: {e}")

        # 2) Conversación normal con datos reales del canal
        perfil = detect_profile(member.display_name if member else "", uid) or ""
        nombre = member.display_name if member else "alguien"
        mood = get_mood()
        _ahora = _spain_now()
        hora_es = _ahora.strftime("%H:%M")
        juegos = build_live_games_context(guild)  # a qué juega cada uno AHORA
        system = (
            SERVER_CONTEXT
            + f"\n\nEstás EN UNA LLAMADA DE VOZ. Te hablan por voz y respondes por voz (te van a OÍR)."
            + f"\nHoy es {_fecha_es()}. Hora actual en España: {hora_es}. Estado de ánimo (según la hora): {mood}. Si preguntan el día o la hora, eso es; para mañana u otro día, cuéntalo a partir de hoy."
            + juegos
            + f"\n\n━━━ DATOS REALES DE ESTA LLAMADA (no inventes) ━━━"
            + f"\nQuien te habla AHORA mismo y a quien DEBES responder: {nombre}."
            + (f"\nLo que sabes de {nombre}: {perfil}" if perfil else f"\nNo tienes información sobre {nombre}. NO le pongas el nombre de otra persona ni le atribuyas cosas que sabes de otros. {nombre} es {nombre}, punto.")
            + f"\nGente en la llamada ahora mismo: {', '.join(miembros) if miembros else 'solo tú'}."
            + f"\nIMPORTANTE: Quien te habla AHORA es {nombre} y NADIE MÁS. No le confundas con nadie. A él le respondes. Pero ESCUCHAS a todos: esto es una TERTULIA de grupo."
            + "\n\n━━━ CÓMO ESTAR EN LA TERTULIA (IMPORTANTE) ━━━\n"
            "1. Eres LISTO y ÚTIL: contesta DE VERDAD y bien (un dato, una duda, una cuenta, la hora, lo que sea). "
            "Demuestra cabeza, no sueltes gilipolleces.\n"
            "2. Eres borde pero PARTICIPATIVO, NO cerrado: entra al trapo, opina, sigue el hilo y métete en la "
            "conversación como uno más. Reacciona a lo que se está hablando (política, juegos, familia, enfermedades, "
            "movidas, lo que sea). La mala leche es el ADEREZO, no un muro para no aportar. Borde SÍ, seco y cerrado NO.\n"
            "3. SIGUE EL CONTEXTO de la llamada: arriba va lo que se ha ido diciendo, ETIQUETADO con quién lo dijo. "
            "Tenlo en cuenta y enlaza. Prepárate para CAMBIOS DE TEMA constantes y síguelos sin rayarte.\n"
            "4. Si algo viene MUY roto de la transcripción y no tiene NINGÚN sentido, di solo '¿qué? repite'. Pero si "
            "se pilla la idea aunque esté regular, RESPONDE — no te cierres por una palabra mal oída.\n"
            "5. 1 o 2 frases cortas habladas (máx ~25 palabras), sin markdown ni listas. Es una llamada: fluido y natural.\n"
            f"6. BÚSQUEDAS: solo TIEMPO/clima/lluvia o NOTICIAS sin lugar = **{DEFAULT_CITY}**. "
            "Para TODO lo demás (juegos, series, películas, personas, datos, historia), busca tal cual SIN añadir ciudad. "
            "precios de juegos, búscalos. Da el dato concreto y corto."
        )
        web = needs_web_search(text)  # call_ai limpia la consulta (quita Bender, fija lugar)
        # Historial COMPARTIDO de la llamada (tertulia), etiquetado por quién habla:
        # así Bender ve TODA la conversación del grupo Y sabe quién dijo cada cosa.
        convo = clean_history(sess.setdefault("convo", []))[-12:]
        convo.append({"role": "user", "content": f"{nombre}: {text}"})
        msgs = [{"role": "system", "content": system}] + convo
        _tllm = time.time()
        try:
            if web:
                # Buscar EN PARALELO mientras suelta un "espera que lo busco" -> sin mudez
                search_task = asyncio.create_task(call_ai(msgs, max_tokens=120, use_web=True))
                await _speak(guild, search_filler())   # se reproduce mientras busca
                reply = await search_task
            else:
                reply = await call_ai(msgs, max_tokens=80, use_web=False)
        except Exception:
            reply = error_fallback()
        _llm_dt = time.time() - _tllm
        if is_error_reply(reply):
            reply = error_fallback()
        else:
            convo.append({"role": "assistant", "content": reply})
            sess["convo"] = convo[-14:]
        await _speak(guild, _clean_for_speech(reply))
        print(f"[VOZ-T] LLM={_llm_dt:.1f}s total_respuesta={time.time()-_t0:.1f}s", flush=True)
    finally:
        sess["busy"] = False
        # Cooldown anti-rayada: ignora nuevas activaciones durante 1.5s tras responder
        sess["cooldown_until"] = time.time() + 0.8


async def _speak(guild, text):
    """Genera la voz con Piper y la reproduce. Espera hasta que termine de hablar."""
    sess = voice_sessions.get(guild.id)
    if not sess:
        return
    vc = sess["vc"]
    path = f"/tmp/bender_voice_{guild.id}.wav"
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(_tts_executor, _synth_wav, text, path)
    if not ok or not vc or not vc.is_connected():
        sess["speaking"] = False
        return
    _mx = sess.get("mixer")
    if _mx and vc.is_playing():
        try:
            sess["speaking"] = True
            sess["buffers"].clear()
            _dm = asyncio.Event()
            _mx.set_tts(discord.FFmpegPCMAudio(path), _dm)
            try:
                await asyncio.wait_for(_dm.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                print("[VOZ] Timeout mixer esperando audio TTS.", flush=True)
        finally:
            sess["speaking"] = False
            sess["buffers"].clear()
        return
    try:
        if vc.is_playing():
            vc.stop()
        sess["speaking"] = True
        sess["buffers"].clear()
        done_event = asyncio.Event()
        def _after(err):
            try:
                sess["speaking"] = False
                sess["buffers"].clear()
            except Exception:
                pass
            loop.call_soon_threadsafe(done_event.set)
        src = discord.FFmpegPCMAudio(path)
        vc.play(src, after=_after)
        # Esperar hasta que termine el audio (máx 60s; respuestas largas o búsquedas)
        try:
            await asyncio.wait_for(done_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            print("[VOZ] Timeout esperando audio, liberando speaking.", flush=True)
            sess["speaking"] = False
            sess["buffers"].clear()
    except BaseException as e:
        # BaseException captura también CancelledError (asyncio) — nunca dejar speaking=True
        sess["speaking"] = False
        sess["buffers"].clear()
        if not isinstance(e, asyncio.CancelledError):
            print(f"[VOZ] Error reproduciendo: {e}")
        raise


# ===================================================================
#  MUSICA - Audius (canciones completas, sin login, desde el VPS)
# ===================================================================
_AUDIUS_HOST = None
def _audius_host():
    global _AUDIUS_HOST
    if _AUDIUS_HOST:
        return _AUDIUS_HOST
    try:
        import urllib.request, json as _j
        _rq = urllib.request.Request("https://api.audius.co", headers={"User-Agent": "Mozilla/5.0"})
        hs = _j.load(urllib.request.urlopen(_rq, timeout=8)).get("data", [])
        _AUDIUS_HOST = hs[0] if hs else "https://discoveryprovider.audius.co"
    except Exception:
        _AUDIUS_HOST = "https://discoveryprovider.audius.co"
    return _AUDIUS_HOST


class DJMixer(discord.AudioSource):
    """Mezcla música + voz de Bender en UNA fuente (ducking + volumen en vivo = modo DJ)."""
    FRAME = 3840
    SILENCE = b"\x00" * FRAME
    def __init__(self, loop):
        self.loop = loop
        self.music = None
        self.music_vol = 0.7   # volumen estándar; cada usuario ajusta el bot en su control
        self.duck = 0.16
        self.tts = None
        self._tts_done = None
        self._lock = threading.Lock()
        self.keep_alive = False
    def is_opus(self):
        return False
    def set_music(self, src):
        with self._lock:
            old = self.music
            self.music = src
        if old:
            try: old.cleanup()
            except Exception: pass
    def set_tts(self, src, ev):
        with self._lock:
            self.tts = src
            self._tts_done = ev
    def read(self):
        with self._lock:
            m, t = self.music, self.tts
        md = m.read() if m else b""
        if m and not md:
            with self._lock:
                if self.music is m:
                    try: self.music.cleanup()
                    except Exception: pass
                    self.music = None
            md = b""
        td = t.read() if t else b""
        if t and not td:
            ev = None
            with self._lock:
                if self.tts is t:
                    try: self.tts.cleanup()
                    except Exception: pass
                    self.tts = None
                    ev, self._tts_done = self._tts_done, None
            if ev is not None:
                self.loop.call_soon_threadsafe(ev.set)
            td = b""
        if not md and not td:
            return self.SILENCE if self.keep_alive else b""
        if md and len(md) < self.FRAME: md += b"\x00" * (self.FRAME - len(md))
        if td and len(td) < self.FRAME: td += b"\x00" * (self.FRAME - len(td))
        vol = self.music_vol * (self.duck if td else 1.0)
        if md and td:
            return audioop.add(audioop.mul(md, 2, vol), td, 2)
        if md:
            return audioop.mul(md, 2, vol)
        return td
    def cleanup(self):
        with self._lock:
            for x in (self.music, self.tts):
                try:
                    if x: x.cleanup()
                except Exception: pass
            self.music = None
            self.tts = None


def _parse_music_volume(t):
    t = t.lower()
    if not any(k in t for k in ("música", "musica", "volumen", "canción", "cancion", "tema",
                                "súbela", "subela", "bájala", "bajala")):
        return None
    if any(k in t for k in ("ambiente", "ambiental", "de fondo", "bajita", "más baja", "mas baja",
                            "baja la", "bájala", "bajala", "más bajo", "mas bajo", "flojit", "flojo")):
        return 0.22
    if any(k in t for k in ("más alta", "mas alta", "sube la", "súbela", "subela", "más fuerte",
                            "mas fuerte", "a tope", "más alto", "mas alto", "más dura", "mas dura")):
        return 0.95
    return None


async def set_music_volume(guild, vol):
    sess = voice_sessions.get(guild.id)
    mixer = sess.get("mixer") if sess else None
    if not mixer:
        return "No hay música sonando."
    mixer.music_vol = max(0.05, min(1.0, vol))
    return "Música de fondo, modo ambiente." if vol <= 0.4 else "Música a tope."


async def adjust_volume(guild, delta):
    sess = voice_sessions.get(guild.id)
    mixer = sess.get("mixer") if sess else None
    if not mixer:
        return "No hay música sonando."
    mixer.music_vol = max(0.10, min(1.0, mixer.music_vol + delta))
    return f"Volumen: {int(mixer.music_vol * 100)}%"


def _parse_music_query(t):
    m = re.search(r"\b(pon|ponme|reproduce|reproducir|play|suena|pincha)\s+(.+)", t.strip(), re.I)
    if not m:
        return None
    q = m.group(2).strip(" .,¿?¡!")
    q = re.sub(r"^(música|musica|la canción|la cancion|canción|cancion|el tema|tema|la|el)\s+(de\s+)?", "", q, flags=re.I).strip()
    if re.match(r"(el canal|modo|en (fantasma|privado|public|público|cristal)|fantasma|privado|public|público|cristal|el nombre|de nombre)\b", q, re.I):
        return None
    if len(q) < 2:
        return None
    return q


_MUSIC_STOP = ("para la música", "para la musica", "quita la música", "quita la musica",
               "calla la música", "calla la musica", "apaga la música", "apaga la musica",
               "stop música", "stop musica", "para la music", "corta la música", "corta la musica")


_YT_NOISE_RE = re.compile(
    r"\((?:[^)]*\b(?:official|oficial|video|audio|lyric[s]?|letra|music\s*video|"
    r"visualizer|hd|4k|mv|remaster[^)]*)\b[^)]*)\)"
    r"|\[(?:[^\]]*\b(?:official|oficial|video|audio|lyric[s]?|letra|music\s*video|"
    r"visualizer|hd|4k|mv|remaster[^\]]*)\b[^\]]*)\]"
    r"|\b(?:official\s*(?:music\s*)?video|official\s*audio|lyric\s*video|"
    r"video\s*oficial|audio\s*oficial|visualizer)\b",
    re.I)


def _clean_track_title(t):
    """Limpia ruido típico de títulos de YouTube para buscar mejor."""
    t = _YT_NOISE_RE.sub("", t or "")
    t = re.sub(r"\s+", " ", t).strip(" -·|—")
    return t or (t or "").strip()


_POT_SERVER_JS = "/app/bgutil/server/build/main.js"
# Si algún día se dejan cookies de una cuenta de YouTube aquí, yt-dlp las usa y entonces
# SÍ funciona (es lo único que rompe el muro de esta IP de datacenter). Mientras no exista
# el fichero, ni se intenta YouTube en links (sería 15s de espera para nada).
_YT_COOKIES = "/app/yt_cookies.txt"

# ── SOUNDCLOUD ──────────────────────────────────────────────────────────────
_SC_CLIENT_ID = None
_SC_CLIENT_ID_TS = 0

def _sc_get_client_id():
    """Scrapea el client_id de SoundCloud desde los JS bundles. Cachea 1h."""
    global _SC_CLIENT_ID, _SC_CLIENT_ID_TS
    import time as _t
    if _SC_CLIENT_ID and _t.time() - _SC_CLIENT_ID_TS < 3600:
        return _SC_CLIENT_ID
    try:
        from curl_cffi import requests as _sc_req
        import re as _re
        r = _sc_req.get('https://soundcloud.com', impersonate='chrome131', timeout=10)
        scripts = _re.findall(r'src="([^"]+\.js)"', r.text)
        for js_url in scripts[:10]:
            full = js_url if js_url.startswith('http') else 'https://soundcloud.com' + js_url
            try:
                r2 = _sc_req.get(full, impersonate='chrome131', timeout=8)
                for pat in [r'client_id="([a-zA-Z0-9]{32})"', r'client_id:"([a-zA-Z0-9]{32})"']:
                    cids = _re.findall(pat, r2.text)
                    if cids:
                        _SC_CLIENT_ID = cids[0]
                        _SC_CLIENT_ID_TS = _t.time()
                        return _SC_CLIENT_ID
            except Exception:
                continue
    except Exception:
        pass
    # Fallback: client_id conocido
    return "cRTw6GjgH7WJ4vlUCvSb0TfMay14HuXK"

def _sc_search(query, limit=5):
    """Busca canciones en SoundCloud. Devuelve lista de tracks."""
    from curl_cffi import requests as _sc_req
    import urllib.parse as _up
    cid = _sc_get_client_id()
    if not cid:
        return []
    try:
        url = f'https://api-v2.soundcloud.com/search/tracks?q={_up.quote(query)}&limit={limit}&client_id={cid}'
        r = _sc_req.get(url, impersonate='chrome131', timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get('collection', [])
    except Exception:
        return []

def _sc_stream_url(track):
    """Obtiene la URL de stream de un track de SoundCloud."""
    from curl_cffi import requests as _sc_req
    cid = _sc_get_client_id()
    if not cid:
        return None
    transcodings = track.get('media', {}).get('transcodings', [])
    for tr in transcodings:
        if tr.get('format', {}).get('protocol') == 'progressive':
            turl = tr['url'] + f'?client_id={cid}'
            try:
                r = _sc_req.get(turl, impersonate='chrome131', timeout=10)
                if r.status_code == 200:
                    u = r.json().get('url', '')
                    if u:
                        return u
            except Exception:
                pass
    for tr in transcodings:
        if tr.get('format', {}).get('protocol') == 'hls':
            mime = tr.get('format', {}).get('mime_type', '')
            if 'mp4' in mime or 'mpegurl' in mime:
                turl = tr['url'] + f'?client_id={cid}'
                try:
                    r = _sc_req.get(turl, impersonate='chrome131', timeout=10)
                    if r.status_code == 200:
                        u = r.json().get('url', '')
                        if u:
                            return u
                except Exception:
                    pass
    return None

def _sc_resolve(query):
    """Busca en SoundCloud y devuelve (url, title, artist, art) o None."""
    tracks = _sc_search(query, limit=5)
    if not tracks:
        return None
    for t in tracks:
        stream = _sc_stream_url(t)
        if stream:
            art = t.get('artwork_url') or (t.get('user', {}).get('avatar_url'))
            return (stream, t.get('title', '?'), t.get('user', {}).get('username', '?'), art)
    return None


def _audius_search(query, strict=False):
    """Busca una canción en Audius por texto. Devuelve (url, title, artist, art) o None."""
    import urllib.parse, urllib.request, json as _j
    if not query or not query.strip():
        return None
    h = _audius_host()
    ref = query.lower()
    rw = set(w for w in ref.split() if len(w) > 2)
    try:
        req = urllib.request.Request(
            h + "/v1/tracks/search?query=" + urllib.parse.quote(query) + "&app_name=bender",
            headers={"User-Agent": "Mozilla/5.0"})
        d = _j.load(urllib.request.urlopen(req, timeout=12))
    except Exception:
        return None
    items = [t for t in d.get("data", []) if not t.get("is_delete")]
    if not items:
        return None

    def _art(tr):
        aw = tr.get("artwork") or {}
        return aw.get("480x480") or aw.get("150x150") or None

    def score(tr):
        ti = (tr.get("title", "") + " " + (tr.get("user") or {}).get("name", "")).lower()
        return (len(rw & set(ti.split())), tr.get("play_count", 0))

    tr = max(items[:8], key=score)
    if strict:
        need = max(2, (len(rw) + 1) // 2) if len(rw) >= 2 else 1
        if score(tr)[0] < need:
            return None
    return (h + "/v1/tracks/" + str(tr["id"]) + "/stream?app_name=bender",
            tr.get("title", "?"), (tr.get("user") or {}).get("name", "?"), _art(tr))
# Credenciales de la Spotify Web API (client credentials flow — gratis, sin cuenta premium).
# Fichero: {"client_id": "...", "client_secret": "..."}
# Si el fichero no existe, las canciones individuales siguen funcionando (vía oEmbed + YT),
# pero las playlists y álbumes de Spotify no se pueden leer.
_SPOTIFY_CREDS = "/app/spotify_creds.json"
_spotify_token_cache = {"token": None, "expires": 0.0}


def _spotify_get_token():
    """Client credentials flow. No requiere login de usuario."""
    import base64 as _b64, json as _j, time as _t
    if _spotify_token_cache["token"] and _t.time() < _spotify_token_cache["expires"] - 60:
        return _spotify_token_cache["token"]
    if not os.path.exists(_SPOTIFY_CREDS):
        return None
    try:
        import urllib.request as _ur, urllib.parse as _up
        creds = _j.loads(open(_SPOTIFY_CREDS).read())
        cid, sec = creds["client_id"], creds["client_secret"]
        auth = _b64.b64encode(f"{cid}:{sec}".encode()).decode()
        req = _ur.Request(
            "https://accounts.spotify.com/api/token",
            data=b"grant_type=client_credentials",
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/x-www-form-urlencoded"})
        d = _j.load(_ur.urlopen(req, timeout=8))
        _spotify_token_cache["token"] = d["access_token"]
        _spotify_token_cache["expires"] = _t.time() + d.get("expires_in", 3600)
        return d["access_token"]
    except Exception as e:
        print(f"[SPOTIFY] Error token: {e}", flush=True)
        return None


def _spotify_get_list_tracks(url):
    """Extrae pistas de playlist o álbum de Spotify usando Googlebot UA en el embed público.
    Con UA de navegador normal la IP del VPS recibe una página vacía (sin __NEXT_DATA__),
    pero con Googlebot sí llega el JSON completo con trackList.
    Devuelve lista de dicts {title, artist, art, query, lazy, url} o None si falla."""
    import json as _j, re as _re, urllib.request as _ur
    try:
        m = _re.search(r"/(playlist|album)/([A-Za-z0-9]+)", url)  # Funciona con /intl-es/... etc.
        if not m:
            return None
        kind, sid = m.group(1), m.group(2)
        embed_url = f"https://open.spotify.com/embed/{kind}/{sid}"
        req = _ur.Request(embed_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        })
        html = _ur.urlopen(req, timeout=14).read().decode("utf-8", errors="ignore")
        nd_m = _re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, _re.S)
        if not nd_m:
            print("[SPOTIFY] No __NEXT_DATA__ en embed (¿bloqueado?)", flush=True)
            return None
        nd = _j.loads(nd_m.group(1))
        # trackList puede estar anidado profundamente — búsqueda recursiva
        def _deep_find(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    r = _deep_find(v, key)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for item in obj:
                    r = _deep_find(item, key)
                    if r is not None:
                        return r
            return None
        track_list = _deep_find(nd, "trackList") or []
        tracks = []
        for tr in track_list:
            name = tr.get("title", "")
            artist = tr.get("subtitle", "")
            if not name:
                continue
            tracks.append({"title": name, "artist": artist, "art": None,
                            "query": f"{name} {artist}", "lazy": True, "url": None})
        print(f"[SPOTIFY] {kind} cargada: {len(tracks)} pistas", flush=True)
        return tracks if tracks else None
    except Exception as e:
        print(f"[SPOTIFY] Error embed: {e}", flush=True)
        return None


def _spotify_get_track_meta(url):
    """Para links de track de Spotify: devuelve (title, artist) via Twitterbot UA + og: tags.
    Funciona desde la IP del VPS donde la Spotify API y el embed normal están bloqueados."""
    import re as _re, urllib.request as _ur
    try:
        m = _re.search(r"/track/([A-Za-z0-9]+)", url)  # Funciona con /intl-es/track/ etc.
        if not m:
            print(f"[SPOTIFY] track_meta: no track ID en URL: {url[:60]}", flush=True)
            return None, None
        tid = m.group(1)
        req = _ur.Request(f"https://open.spotify.com/track/{tid}",
            headers={"User-Agent": "Twitterbot/1.0", "Accept": "text/html"})
        html = _ur.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
        og_title = _re.search(r'<meta property="og:title" content="([^"]+)"', html)
        og_desc  = _re.search(r'<meta property="og:description" content="([^"]+)"', html)
        title = og_title.group(1) if og_title else None
        artist = ""
        if og_desc:
            # og:description formato: "Artista · Álbum · Song · Año"
            parts = og_desc.group(1).split(" · ")
            if parts:
                artist = parts[0].strip()
        print(f"[SPOTIFY] track_meta OK: {title!r} / {artist!r} (html={len(html)})", flush=True)
        return title, artist
    except Exception as _e:
        print(f"[SPOTIFY] track_meta FAIL: {_e}", flush=True)
        return None, None


def _ensure_pot_server():
    """Arranca (si no está ya) el servidor local de po_token (bgutil) que ayuda a
    yt-dlp a saltarse el muro anti-bot de YouTube. Idempotente (mira el puerto 4416)."""
    import subprocess, urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:4416/ping", timeout=2)
        return True
    except Exception:
        pass
    if not os.path.exists(_POT_SERVER_JS):
        return False
    try:
        subprocess.Popen(["nice", "-n", "15", "node", _POT_SERVER_JS],
                         stdout=open("/tmp/bgutil.log", "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print("[MUSIC] servidor po_token (bgutil) arrancado.", flush=True)
        return True
    except Exception as e:
        print(f"[MUSIC] no pude arrancar po_token: {e}", flush=True)
        return False


def _ytdlp_resolve(target):
    """Intenta sacar audio REAL de YouTube (link directo o ytsearch1:...).
    Devuelve (stream_url, title, artist, art) o None si falla / muro anti-bot."""
    _ensure_pot_server()
    import subprocess, json as _j
    # Flags clave para esta IP de datacenter: po_token (servidor bgutil local) rompe el
    # muro anti-bot en los vídeos que se dejan, y el runtime JS de Node + el solucionador
    # EJS descifran las firmas. Sin esto, YouTube bloquea casi todo.
    cmd = ["nice", "-n", "15", "yt-dlp", "-f", "bestaudio/best", "--no-playlist",
           "--no-warnings", "--quiet",
           "--js-runtimes", "node",
           "--remote-components", "ejs:github",
           "--extractor-args", "youtube:player_client=default,android_vr,tv,web_safari"]
    if os.path.exists(_YT_COOKIES):
        cmd += ["--cookies", _YT_COOKIES]
    cmd += ["-J", target]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=25, text=True)
    except Exception:
        return None
    out = (p.stdout or "").strip()
    if not out:
        return None
    try:
        d = _j.loads(out)
    except Exception:
        return None
    if d.get("_type") == "playlist" and d.get("entries"):
        d = next((e for e in d["entries"] if e), None)
        if not d:
            return None
    url = d.get("url")
    if not url:
        cands = [f for f in (d.get("formats") or [])
                 if f.get("url") and f.get("acodec") not in (None, "none")]
        if cands:
            url = cands[-1]["url"]
    if not url:
        return None
    title = d.get("track") or d.get("title") or "?"
    artist = d.get("artist") or d.get("uploader") or "?"
    art = d.get("thumbnail")
    return (url, title, artist, art)


async def _music_resolve(query):
    import urllib.parse, urllib.request, json as _j
    q = query.strip()
    yt_link = "youtube.com" in q or "youtu.be" in q
    sp_link = "open.spotify.com" in q
    # 1) Texto legible (limpio) a partir del link, para buscar bien.
    clean_q = q
    try:
        if sp_link:
            if "/track/" in q:
                # Track individual: Twitterbot UA → og:title + og:description (título + artista)
                # La Spotify Web API está bloqueada desde la IP del VPS, pero og: sí funciona.
                _sp_title, _sp_artist = _spotify_get_track_meta(q)
                print(f"[SPOTIFY] track_meta: title={_sp_title!r} artist={_sp_artist!r}", flush=True)
                if _sp_title:
                    _ct = _clean_track_title(_sp_title)
                    clean_q = (f"{_ct} {_sp_artist}".strip() if _sp_artist else _ct) or _sp_title
                else:
                    # Fallback: oEmbed solo da el título, sin artista
                    o = _j.load(urllib.request.urlopen(
                        "https://open.spotify.com/oembed?url=" + urllib.parse.quote(q), timeout=8))
                    clean_q = _clean_track_title(o.get("title", "")) or o.get("title", "")
            else:
                # Playlist/álbum: oEmbed da el nombre del conjunto
                o = _j.load(urllib.request.urlopen(
                    "https://open.spotify.com/oembed?url=" + urllib.parse.quote(q), timeout=8))
                clean_q = _clean_track_title(o.get("title", "")) or o.get("title", "")
        elif yt_link:
            o = _j.load(urllib.request.urlopen(
                "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(q), timeout=8))
            clean_q = _clean_track_title(o.get("title", q)) or q
    except Exception as _e:
        print(f"[MUSIC] resolve meta error: {_e}", flush=True)

    # Si clean_q quedó vacío o con la URL original, no se puede buscar bien
    if not clean_q or clean_q == q:
        if sp_link:
            print(f"[MUSIC] WARN: clean_q vacío/igual al link ({q[:60]}), abortando búsqueda YT", flush=True)
        # Para links sin limpieza útil, no buscar en YouTube (devolvería trending aleatorio)

    loop = asyncio.get_event_loop()
    ref = clean_q.lower()
    is_link = yt_link or sp_link

    def _art(tr):
        aw = tr.get("artwork") or {}
        return aw.get("480x480") or aw.get("150x150") or None
    def _search(strict):
        h = _audius_host()
        req = urllib.request.Request(h + "/v1/tracks/search?query=" + urllib.parse.quote(clean_q) + "&app_name=bender",
                                     headers={"User-Agent": "Mozilla/5.0"})
        d = _j.load(urllib.request.urlopen(req, timeout=12))
        items = [t for t in d.get("data", []) if not t.get("is_delete")]
        if not items:
            return None
        rw = set(w for w in ref.split() if len(w) > 2)
        def score(tr):
            ti = (tr.get("title", "") + " " + (tr.get("user") or {}).get("name", "")).lower()
            return (len(rw & set(ti.split())), tr.get("play_count", 0))
        tr = max(items[:8], key=score)
        if strict:
            # Para LINKS exigimos parecido real (Audius casi no tiene mainstream): mejor
            # decir que no que poner una canción distinta (el bug de "saca cosas diferentes").
            need = max(2, (len(rw) + 1) // 2) if len(rw) >= 2 else 1
            if score(tr)[0] < need:
                return None
        return (h + "/v1/tracks/" + str(tr["id"]) + "/stream?app_name=bender",
                tr.get("title", "?"), (tr.get("user") or {}).get("name", "?"), _art(tr))

    if is_link:
        # LINK: YouTube falla (PO Token). Usar el título del oEmbed y buscar en SoundCloud + Audius.
        if clean_q and clean_q != q:
            print(f"[MUSIC] Link → buscando: {clean_q!r}", flush=True)
            # 1. SoundCloud (tiene música mainstream)
            try:
                res = await loop.run_in_executor(None, _sc_resolve, clean_q)
            except Exception:
                res = None
            if res:
                print(f"[MUSIC] resuelto por SoundCloud: {res[1]}", flush=True)
                return res
            # 2. Audius fallback
            try:
                res = await loop.run_in_executor(None, _audius_search, clean_q, False)
            except Exception:
                res = None
            if res:
                print(f"[MUSIC] resuelto por Audius: {res[1]}", flush=True)
                return res
            return None, "No he encontrado esa canción ni en SoundCloud ni en Audius. Prueba con el nombre.", None, None
        return None, "Ese link no lo puedo sacar. Dime el nombre y lo busco.", None, None

    # TEXTO: SoundCloud primero (mainstream), Audius segundo (independiente)
    try:
        res = await loop.run_in_executor(None, _sc_resolve, clean_q)
    except Exception:
        res = None
    if res:
        print(f"[MUSIC] resuelto por SoundCloud: {res[1]}", flush=True)
        return res
    try:
        res = await loop.run_in_executor(None, _audius_search, clean_q, False)
    except Exception:
        return None, "No he podido buscar la canción.", None, None
    if not res:
        return None, "No he encontrado esa canción.", None, None
    print(f"[MUSIC] resuelto por Audius: {res[1]}", flush=True)
    return res


async def play_music(guild, query, requester="alguien"):
    return await enqueue_music(guild, query, requester)


async def stop_music(guild):
    sess = voice_sessions.get(guild.id)
    if not sess:
        return "No hay música."
    vc = sess.get("vc")
    sess["queue"] = []
    sess.pop("now", None)
    sess.pop("music", None)
    m = sess.pop("mixer", None)
    if m:
        try: m.keep_alive = False
        except Exception: pass
    try:
        if vc and vc.is_playing():
            vc.stop()
    except Exception:
        pass
    return "Vale, quito la música."


MUSIC_PANEL_CHANNEL_ID = PINNED_RESPONSE_CHANNEL_ID

_MUSIC_SKIP = ("siguiente", "salta", "sáltala", "saltala", "otra canción", "otra cancion",
               "cambia de canción", "cambia de cancion", "skip", "quita esta canción",
               "quita esta cancion", "la siguiente", "pon otra", "pasa de canción", "pasa de cancion")


def _is_music_admin(member):
    try:
        gp = member.guild_permissions
        return bool(gp.administrator or gp.manage_channels)
    except Exception:
        return False


async def enqueue_music(guild, query, requester="alguien"):
    sess = voice_sessions.get(guild.id)
    vc = sess.get("vc") if sess else None
    if not vc or not vc.is_connected():
        return "No estoy en ninguna llamada. Méteme primero."

    # ── Playlist / álbum de Spotify ──────────────────────────────────────────
    import re as _re
    _sp_list_match = _re.search(r"open\.spotify\.com(?:/intl-[a-z]+)?/(playlist|album)/", query.strip())
    if _sp_list_match:
        loop = asyncio.get_event_loop()
        sp_tracks = await loop.run_in_executor(None, _spotify_get_list_tracks, query.strip())
        if sp_tracks is None:
            return "No he podido leer esa playlist/álbum de Spotify (puede ser privada o Spotify la ha bloqueado)."
        for t in sp_tracks:
            t["by"] = requester
        sess.setdefault("queue", []).extend(sp_tracks)
        kind = "playlist" if "playlist" in query else "álbum"
        n = len(sp_tracks)
        started = False
        if not sess.get("mixer"):
            sess["speaking"] = False
            mx = DJMixer(asyncio.get_event_loop())
            mx.keep_alive = True
            sess["mixer"] = mx
            try:
                if vc.is_playing():
                    vc.stop()
            except Exception:
                pass
            vc.play(mx, after=lambda e: sess.pop("mixer", None))
            asyncio.create_task(_dj_loop(guild))
            started = True
        await _refresh_music_panel(guild)
        return f"{'Reproduciendo' if started else 'Añadida'} {kind} de Spotify: {n} canciones en cola."
    # ────────────────────────────────────────────────────────────────────────

    url, title, artist, art = await _music_resolve(query)
    if not url:
        return title
    sess.setdefault("queue", []).append({"url": url, "title": title, "artist": artist, "art": art, "by": requester})
    started = False
    if not sess.get("mixer"):
        sess["speaking"] = False
        mx = DJMixer(asyncio.get_event_loop())
        mx.keep_alive = True
        sess["mixer"] = mx
        try:
            if vc.is_playing():
                vc.stop()
        except Exception:
            pass
        vc.play(mx, after=lambda e: sess.pop("mixer", None))
        asyncio.create_task(_dj_loop(guild))
        started = True
    await _refresh_music_panel(guild)
    n = len(sess.get("queue", []))
    if started or not sess.get("now"):
        return f"Va, reproduciendo: {title}"
    return f"Añadida a la cola: {title} (hay {n} en espera)"


async def skip_music(guild):
    sess = voice_sessions.get(guild.id)
    mx = sess.get("mixer") if sess else None
    if not mx:
        return "No hay música sonando."
    mx.set_music(None)
    return "Siguiente."


async def _dj_loop(guild):
    idle = 0
    while True:
        await asyncio.sleep(0.5)
        sess = voice_sessions.get(guild.id)
        if not sess:
            break
        mx = sess.get("mixer")
        if not mx:
            break
        if mx.music is None:
            q = sess.get("queue") or []
            if q:
                nxt = q.pop(0)
                # Resolución LAZY: pistas de playlists de Spotify se resuelven justo
                # antes de reproducirse. Intento YouTube, fallback Audius.
                if nxt.get("lazy") and not nxt.get("url"):
                    # SoundCloud primero, Audius fallback
                    try:
                        res = await asyncio.get_event_loop().run_in_executor(
                            None, _sc_resolve, nxt["query"])
                        if res:
                            nxt["url"], nxt["title"], nxt["artist"], nxt["art"] = res
                        else:
                            ares = await asyncio.get_event_loop().run_in_executor(
                                None, _audius_search, nxt["query"])
                            if ares:
                                nxt["url"], nxt["title"], nxt["artist"], nxt["art"] = ares
                            else:
                                print(f"[DJ] Sin resultado SC/Audius: {nxt['query']}", flush=True)
                                continue
                    except Exception as e:
                        print(f"[DJ] Lazy resolve error: {e}", flush=True)
                        try:
                            ares = await asyncio.get_event_loop().run_in_executor(
                                None, _audius_search, nxt["query"])
                            if ares:
                                nxt["url"], nxt["title"], nxt["artist"], nxt["art"] = ares
                            else:
                                continue
                        except Exception:
                            continue
                try:
                    ff = discord.FFmpegPCMAudio(
                        nxt["url"],
                        before_options="-user_agent Mozilla/5.0 -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
                        options="-vn")
                    mx.set_music(ff)
                    sess["now"] = nxt
                    idle = 0
                    await _refresh_music_panel(guild)
                except Exception as e:
                    print(f"[DJ] {e}", flush=True)
            else:
                if sess.get("now") is not None:
                    sess.pop("now", None)
                    await _refresh_music_panel(guild)
                idle += 1
                if idle > 120:
                    m = sess.pop("mixer", None)
                    if m:
                        try: m.keep_alive = False
                        except Exception: pass
                    break
        else:
            idle = 0


def _build_music_embed(guild):
    sess = voice_sessions.get(guild.id) or {}
    now = sess.get("now")
    q = sess.get("queue") or []
    lines = []
    if now:
        a = now.get("artist", "")
        lines.append(f"**Sonando:** {now.get('title','?')}" + (f" — {a}" if a else ""))
        lines.append(f"pedida por {now.get('by','?')}")
    else:
        lines.append("Nada sonando ahora mismo.")
    if q:
        lines.append("")
        lines.append("**En cola:**")
        for i, it in enumerate(q[:10], 1):
            lines.append(f"`{i}.` {it.get('title','?')}  ·  {it.get('by','?')}")
        if len(q) > 10:
            lines.append(f"y {len(q)-10} más")
    color = 0x9B30FF
    footer_extra = ""
    mstate = data.get("music_panel", {}).get("state", "public")
    if mstate == "private":
        color = 0x3BA55D
        footer_extra = "   ·   🔒 Cabina (solo el admin del canal)"
    e = discord.Embed(title="♪  Música", description="\n".join(lines), color=color)
    if now and now.get("art"):
        e.set_thumbnail(url=now["art"])
    e.set_footer(text=f"Ajusta el volumen del bot en tu control de Discord{footer_extra}   ·   Añade con el botón o di: Bender pon [canción]")
    return e


class MusicAddModal(discord.ui.Modal, title="Añadir canción"):
    cancion = discord.ui.TextInput(
        label="Nombre, link de YouTube o Spotify",
        placeholder="nombre de canción · link YT · canción o playlist de Spotify",
        max_length=300)
    async def on_submit(self, interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        sess = voice_sessions.get(guild.id)
        if not (sess and sess.get("vc") and sess["vc"].is_connected()):
            m = guild.get_member(interaction.user.id)
            if m and m.voice and m.voice.channel:
                await join_voice(m)
            else:
                return await interaction.followup.send("Métete a una llamada de voz primero (o invítame con 'Bender métete al canal').", ephemeral=True)
        msg = await enqueue_music(guild, str(self.cancion.value), interaction.user.display_name)
        await interaction.followup.send(msg, ephemeral=True)


class MusicPanelView(discord.ui.View):
    GATE_MSG = "🔒 Modo cabina: solo el admin del canal controla la música."

    def __init__(self, state="public"):
        super().__init__(timeout=None)
        self.state = state
        # Sin botones de volumen: la música suena a volumen estándar y cada uno
        # ajusta el volumen del bot desde su propio control de usuario en Discord.
        defs = [
            ("Añadir", discord.ButtonStyle.success,   "mus_add",  self._add,  0, "➕"),
            ("Saltar", discord.ButtonStyle.secondary, "mus_skip", self._skip, 0, "⏭️"),
            ("Parar",  discord.ButtonStyle.danger,    "mus_stop", self._stop, 0, "⏹️"),
        ]
        for label, style, cid, cb, row, emoji in defs:
            b = discord.ui.Button(label=label, style=style, custom_id=cid, row=row, emoji=emoji)
            b.callback = cb
            self.add_item(b)

    def _check_access(self, member):
        # El estado REAL vive en data (la vista persistente registrada en add_view
        # siempre nace con state="public", así que NO podemos fiarnos de self.state).
        state = data.get("music_panel", {}).get("state", "public")
        if state != "private":
            return True  # off / public -> todos los de la llamada controlan
        # Cabina: SOLO el dueño del canal de voz donde suena la música (el que abrió
        # el panel del canal de voz). Sin atajo por permisos de Discord.
        try:
            sess = voice_sessions.get(member.guild.id)
            vc = sess.get("vc") if sess else None
            ch = vc.channel if (vc and vc.is_connected()) else None
            ch_id = ch.id if ch else (member.voice.channel.id if (member.voice and member.voice.channel) else None)
            if ch_id is not None and data.get("voice_channel_owners", {}).get(str(ch_id)) == member.id:
                return True
        except Exception:
            pass
        return False

    async def _add(self, interaction):
        if not self._check_access(interaction.user):
            return await interaction.response.send_message(self.GATE_MSG, ephemeral=True, delete_after=4)
        await interaction.response.send_modal(MusicAddModal())
    async def _skip(self, interaction):
        if not self._check_access(interaction.user):
            return await interaction.response.send_message(self.GATE_MSG, ephemeral=True, delete_after=4)
        await interaction.response.send_message(await skip_music(interaction.guild), ephemeral=True, delete_after=4)
    async def _stop(self, interaction):
        if not self._check_access(interaction.user):
            return await interaction.response.send_message(self.GATE_MSG, ephemeral=True, delete_after=4)
        await stop_music(interaction.guild)
        await interaction.response.send_message("Música parada y cola vaciada.", ephemeral=True, delete_after=4)
        await _refresh_music_panel(interaction.guild)


async def _rest_panel(channel_id, msg_id, embed, view):
    """Postea/edita el panel por REST (discord.py channel.send() se cuelga al enviar al
    canal de voz conectado; el REST directo funciona). Devuelve msg_id o None."""
    payload = {"embeds": [embed.to_dict()], "components": view.to_components()}
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}", "Content-Type": "application/json",
               "User-Agent": "DiscordBot (https://chepa.local, 1.0)"}
    try:
        async with aiohttp.ClientSession() as ses:
            if msg_id:
                url = f"https://discord.com/api/v10/channels/{channel_id}/messages/{msg_id}"
                async with ses.patch(url, json=payload, headers=headers) as r:
                    return msg_id if r.status in (200, 201) else None
            url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
            async with ses.post(url, json=payload, headers=headers) as r:
                if r.status in (200, 201):
                    d = await r.json()
                    return d.get("id")
                print(f"[MUSIC] REST POST status {r.status}", flush=True)
                return None
    except Exception as e:
        print(f"[MUSIC] REST panel err: {e}", flush=True)
        return None


async def _rest_panel_delete(channel_id, msg_id):
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}", "User-Agent": "DiscordBot (https://chepa.local, 1.0)"}
    try:
        async with aiohttp.ClientSession() as ses:
            async with ses.delete(f"https://discord.com/api/v10/channels/{channel_id}/messages/{msg_id}", headers=headers) as r:
                return r.status in (200, 204)
    except Exception:
        return False


_music_panel_locks = {}

def _music_panel_lock(guild_id):
    lk = _music_panel_locks.get(guild_id)
    if lk is None:
        lk = asyncio.Lock()
        _music_panel_locks[guild_id] = lk
    return lk


async def send_music_panel(guild, channel=None):
    # Lock por guild: evita que dos eventos a la vez creen paneles duplicados.
    async with _music_panel_lock(guild.id):
        ch = channel or guild.get_channel(MUSIC_PANEL_CHANNEL_ID)
        if not ch:
            return
        embed = _build_music_embed(guild)
        mstate = data.get("music_panel", {}).get("state", "public")
        view = MusicPanelView(state=mstate)
        info = data.get("music_panel", {})
        # 1) Reutilizar el panel existente (editar en sitio).
        if info.get("msg_id") and info.get("ch_id"):
            if await _rest_panel(info["ch_id"], info["msg_id"], embed, view):
                return
            # El PATCH falló -> el mensaje ya no existe. Lo borramos por si acaso
            # para no dejar huérfanos y creamos UNO nuevo.
            await _rest_panel_delete(info["ch_id"], info["msg_id"])
            data.setdefault("music_panel", {}).pop("msg_id", None)
            data.get("music_panel", {}).pop("ch_id", None)
        # 2) Crear uno nuevo.
        mid = await _rest_panel(ch.id, None, embed, view)
        dest = ch.id
        if not mid:
            gen = guild.get_channel(MUSIC_PANEL_CHANNEL_ID)
            if gen:
                mid = await _rest_panel(gen.id, None, embed, view)
                dest = gen.id
        if mid:
            data.setdefault("music_panel", {})["msg_id"] = mid
            data["music_panel"]["ch_id"] = dest
            save_data(data)
            print(f"[MUSIC] panel posteado (REST) en {dest}", flush=True)
        else:
            print("[MUSIC] no pude postear el panel (REST).", flush=True)


async def _refresh_music_panel(guild):
    info = data.get("music_panel", {})
    if not info.get("msg_id") or not info.get("ch_id"):
        return
    mstate = info.get("state", "public")
    await _rest_panel(info["ch_id"], info["msg_id"], _build_music_embed(guild), MusicPanelView(state=mstate))


async def join_voice(member, text_channel=None):
    """Bender entra al canal de voz del que se lo pide."""
    if not VOICE_LIBS_OK:
        return "No tengo el módulo de voz montado."
    if not member.voice or not member.voice.channel:
        return "No estás en ningún canal de voz, payaso."
    guild = member.guild
    channel = member.voice.channel

    # Cargar motores si no están (primera vez tarda ~15s)
    if _piper_voice is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_voice_engines)
        if _piper_voice is None:
            return "Se me ha jodido el oído, no puedo entrar."
        if _GROQ_CLIENT is None:
            return "No tengo acceso al motor de transcripción. Revisa la configuración."

    # Si YA está en un canal de voz y conectado, avisar (no se mueve)
    existing = voice_sessions.get(guild.id)
    if existing and existing.get("vc") and existing["vc"].is_connected():
        ch_actual = existing["vc"].channel
        if ch_actual and ch_actual.id == channel.id:
            return "Que ya estoy aquí dentro, pesado."
        return f"Ya estoy en otro canal de voz (**{ch_actual.name if ch_actual else '?'}**). Échame de ahí primero."

    # Limpiar sesión zombie: si la sesión existe pero vc NO está conectado
    # (conexión perdida sin limpiar), forzar limpieza antes de reconectar
    if existing:
        print("[VOZ] Limpiando sesión de voz anterior (zombie).", flush=True)
        existing["active"] = False
        try:
            if existing.get("vc"):
                await existing["vc"].disconnect(force=True)
        except Exception:
            pass

    # Limpiar cualquier conexión de voz colgada de este guild (evita que "se raye")
    try:
        if guild.voice_client and guild.voice_client.is_connected():
            await guild.voice_client.disconnect(force=True)
    except Exception:
        pass
    voice_sessions.pop(guild.id, None)

    # Conectar con reintentos (a veces el handshake de Discord falla a la primera)
    vc = None
    for intento in range(3):
        try:
            vc = await channel.connect(cls=_vr.VoiceRecvClient, timeout=20, reconnect=False)
            break
        except Exception as e:
            print(f"[VOZ] Intento {intento+1} de conectar falló: {e}", flush=True)
            try:
                if guild.voice_client:
                    await guild.voice_client.disconnect(force=True)
            except Exception:
                pass
            await asyncio.sleep(1.5)
    if vc is None:
        return "No he podido entrar a la llamada, prueba otra vez."

    voice_sessions[guild.id] = {
        "vc": vc, "channel_id": channel.id, "buffers": {},
        "active": True, "speaking": False, "histories": {},
    }
    # Persistir el canal para AUTO-REINGRESO tras un reinicio (un restart tira la
    # conexión de voz y dejaba a Bender sordo fuera de la llamada).
    data["bender_last_voice_channel"] = channel.id
    save_data(data)
    try:
        vc.listen(BenderSink(guild.id))
    except Exception as e:
        print(f"[VOZ] Error al escuchar: {e}")
    asyncio.create_task(_voice_listen_loop(guild))
    await asyncio.sleep(0.5)
    await _speak(guild, "Ya estoy aquí, pringados. Decid Bender y lo que queráis.")
    try:
        if data.get("music_panel", {}).get("on"):
            await send_music_panel(guild, channel)
    except Exception as e:
        print(f"[MUSIC] panel error: {e}", flush=True)
    return f"Va, me meto en **{channel.name}**. Decidme *Bender* y lo que queráis."


async def _auto_rejoin_voice(guild):
    """Tras un reinicio Bender pierde la conexión de voz y se queda FUERA de la llamada
    (= sordo, ignora a todos). Si estaba en un canal con gente, vuelve a entrar solo y
    re-engancha el oído. Silencioso (sin saludo) para no dar la brasa en cada deploy."""
    if not VOICE_LIBS_OK:
        return
    try:
        cid = data.get("bender_last_voice_channel")
        channel = guild.get_channel(cid) if cid else None
        # Sin registro útil o canal vacío -> coge el canal de voz con MÁS gente.
        if not channel or len([m for m in channel.members if not m.bot]) == 0:
            cands = [c for c in guild.voice_channels if len([m for m in c.members if not m.bot]) > 0]
            cands.sort(key=lambda c: len([m for m in c.members if not m.bot]), reverse=True)
            channel = cands[0] if cands else None
        if not channel:
            return
        loop = asyncio.get_event_loop()
        if _piper_voice is None:
            await loop.run_in_executor(None, _load_voice_engines)
        try:
            if guild.voice_client and guild.voice_client.is_connected():
                await guild.voice_client.disconnect(force=True)
        except Exception:
            pass
        voice_sessions.pop(guild.id, None)
        vc = None
        for intento in range(3):
            try:
                vc = await channel.connect(cls=_vr.VoiceRecvClient, timeout=20, reconnect=False)
                break
            except Exception as e:
                print(f"[VOZ] Auto-reingreso intento {intento+1} falló: {e}", flush=True)
                try:
                    if guild.voice_client:
                        await guild.voice_client.disconnect(force=True)
                except Exception:
                    pass
                await asyncio.sleep(1.5)
        if vc is None:
            print("[VOZ] Auto-reingreso: no pude conectar.", flush=True)
            return
        voice_sessions[guild.id] = {
            "vc": vc, "channel_id": channel.id, "buffers": {},
            "active": True, "speaking": False, "histories": {},
        }
        data["bender_last_voice_channel"] = channel.id
        save_data(data)
        try:
            vc.listen(BenderSink(guild.id))
        except Exception as e:
            print(f"[VOZ] Auto-reingreso error al escuchar: {e}", flush=True)
        asyncio.create_task(_voice_listen_loop(guild))
        print(f"[VOZ] Auto-reingreso OK en {channel.name} (oído reenganchado).", flush=True)
    except Exception as e:
        print(f"[VOZ] Auto-reingreso error: {e}", flush=True)


async def leave_voice(guild):
    sess = voice_sessions.get(guild.id)
    if not sess:
        return
    sess["active"] = False
    vc = sess.get("vc")
    # Salida voluntaria -> olvidar el canal para que NO auto-reingrese tras un reinicio.
    data.pop("bender_last_voice_channel", None)
    _stt_pending.pop(guild.id, None)
    try:
        if vc and vc.is_connected():
            vc.stop()
            await vc.disconnect(force=True)
    except Exception:
        pass
    try:
        info = data.get("music_panel", {})
        if info.get("msg_id"):
            _ch = guild.get_channel(info.get("ch_id"))
            if _ch:
                _m = await _ch.fetch_message(info["msg_id"])
                await _m.delete()
        data["music_panel"] = {}
        save_data(data)
    except Exception:
        pass
    voice_sessions.pop(guild.id, None)


async def _voice_selftest():
    """Auto-test: entra solo a un canal de voz y comprueba que el cifrado E2E (DAVE)
    quedó desactivado (dave_protocol_version=0). Valida el fix sin necesidad de nadie."""
    await asyncio.sleep(10)
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        return
    ch = None
    for vc_ in guild.voice_channels:
        if vc_.id != VOICE_CREATOR_ID:
            ch = vc_
            break
    if ch is None:
        print("[VOZ-TEST] No hay canal de voz donde probar.", flush=True)
        return
    try:
        print(f"[VOZ-TEST] Conectando a '{ch.name}' para validar DAVE...", flush=True)
        vc = await ch.connect(cls=_vr.VoiceRecvClient, timeout=20)
        await asyncio.sleep(2)
        st = getattr(vc, "_connection", None)
        dave_v = getattr(st, "dave_protocol_version", "?")
        can_enc = getattr(st, "can_encrypt", "?")
        print(f"[VOZ-TEST] CONECTADO ok. dave_protocol_version={dave_v} can_encrypt={can_enc} "
              f"(0/False = E2E OFF = bien)", flush=True)
        await asyncio.sleep(3)
        await vc.disconnect(force=True)
        print("[VOZ-TEST] Desconectado. Test completado.", flush=True)
    except Exception as e:
        print(f"[VOZ-TEST] ERROR: {e}", flush=True)


_JOIN_VERBS = ("métete", "metete", "entra", "éntrate", "entrate", "únete", "unete",
               "conéctate", "conectate", "conecta", "ven", "vente", "ponte", "súmate",
               "sumate", "métete aquí", "asómate", "asomate", "pásate", "pasate")
_VOZ_WORDS = ("canal", "llamada", "voz", "vc", "call", "aquí", "aqui", "conmigo",
              "con nosotros", "la call", "discord")


def _is_join_voice_cmd(t: str, author_in_voice: bool = False) -> bool:
    t = t.lower()
    tiene_verbo = any(w in t for w in _JOIN_VERBS)
    if not tiene_verbo:
        return False
    # Con un verbo de "venir", basta con que mencione voz O que el que habla
    # esté YA en un canal de voz (entonces es obvio que quiere que entre ahí).
    return author_in_voice or any(w in t for w in _VOZ_WORDS)


def _is_leave_voice_cmd(t: str) -> bool:
    t = t.lower()
    tiene_voz = any(w in t for w in _VOZ_WORDS)
    verbos = ("vete", "sal", "lárgate", "largate", "desconéctate", "desconectate",
              "desconecta", "pírate", "pirate", "márchate", "marchate", "déjanos",
              "dejanos", "fuera", "abandona")
    return any(w in t for w in verbos) and tiene_voz


# =====================================================================
#  ARRANQUE
# =====================================================================
# Iniciar servidor webhook en hilo separado ANTES de bloquear con bot.run()
start_web_server()

# Arrancar bot de Discord (bloqueante)
bot.run(DISCORD_TOKEN)