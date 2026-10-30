"""Genera audio/: las voces de los entrenadores (con internet) y los sonidos y la
musica del reto (por codigo, sin derechos de autor).

Solo hace falta correrlo para cambiar frases. El juego no lo necesita.
Requiere: pip install edge-tts soundfile scipy numpy

Uso, desde la carpeta del proyecto:
    python tools/generar_audio.py            # todo
    python tools/generar_audio.py --solo-voces
    python tools/generar_audio.py --solo-sonidos
    python tools/generar_audio.py --solo-voces --faltantes   # tras agregar frases
"""
import argparse
import asyncio
import pathlib
import sys

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import reto_frases  # noqa: E402

AUDIO = pathlib.Path(__file__).resolve().parent.parent / "audio"
SR = 44100
SIMULTANEAS = 4


def escribir_ogg(ruta, x, sr):
    """Por bloques: la libreria se cierra sin error al escribir un audio largo de una vez."""
    with sf.SoundFile(str(ruta), "w", samplerate=sr, channels=1, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(x), 4096):
            f.write(x[i:i + 4096])


# ------------------------------------------------------------------- voces
def temblor(x, sr, hz=6.2, profundidad_ms=1.3, amplitud=0.24):
    """Voz de persona mayor: vibrato y tremolo algo irregulares, y menos cuerpo."""
    x = signal.sosfilt(signal.butter(2, 190, "high", fs=sr, output="sos"), x)
    t = np.arange(len(x)) / sr
    onda = np.sin(2 * np.pi * hz * t + 0.8 * np.sin(2 * np.pi * 0.7 * t))
    y = np.interp(np.arange(len(x)) + onda * profundidad_ms / 1000 * sr, np.arange(len(x)), x)
    y = y * (1 + amplitud * onda)
    return signal.sosfilt(signal.butter(2, 5200, "low", fs=sr, output="sos"), y)


def sin_silencios(x, sr, umbral=0.02, margen=0.03):
    """La voz viene con silencio al inicio y al final: "Siete" duraba casi 2 s y
    atrasaba el conteo de las repeticiones."""
    sonido = np.flatnonzero(np.abs(x) > umbral * np.max(np.abs(x)))
    extra = int(sr * margen)
    return x[max(0, sonido[0] - extra): sonido[-1] + extra]


async def una_voz(sem, carpeta, clave, texto, voz, efecto):
    import edge_tts

    destino = carpeta / f"{clave}.ogg"
    temporal = carpeta / f"_{clave}.mp3"
    async with sem:
        for intento in range(4):
            try:
                await edge_tts.Communicate(texto, voz["voice"], rate=voz["rate"], pitch=voz["pitch"]).save(str(temporal))
                break
            except Exception:
                if intento == 3:
                    raise
                await asyncio.sleep(1.5)
    x, sr = sf.read(str(temporal), dtype="float64")
    x = x.mean(axis=1) if x.ndim > 1 else x
    x = sin_silencios(x, sr)
    if efecto == "temblor":
        x = temblor(x, sr)
    escribir_ogg(destino, 0.95 * x / np.max(np.abs(x)), sr)
    temporal.unlink()


async def voces(solo_faltantes=False):
    sem = asyncio.Semaphore(SIMULTANEAS)
    tareas = []
    for nombre, datos in reto_frases.ENTRENADORES.items():
        carpeta = AUDIO / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        for clave, texto in reto_frases.textos(nombre).items():
            if solo_faltantes and (carpeta / f"{clave}.ogg").exists():
                continue
            tareas.append(una_voz(sem, carpeta, clave, texto, datos["voz"], datos.get("efecto")))
    await asyncio.gather(*tareas)
    print(f"voces: {len(tareas)} frases")


# ----------------------------------------------------------------- sonidos
def tiempo(seg):
    return np.arange(int(SR * seg)) / SR


def tono(hz, seg, caida=6.0, armonicos=(1.0,)):
    t = tiempo(seg)
    onda = sum(a * np.sin(2 * np.pi * hz * (i + 1) * t) for i, a in enumerate(armonicos))
    return onda * np.exp(-caida * t)


def secuencia(notas):
    return np.concatenate(notas)


def mezclar(destino, sonido, inicio):
    a = int(inicio * SR)
    b = min(len(destino), a + len(sonido))
    if a < len(destino):
        destino[a:b] += sonido[: b - a]


def guardar(nombre, x, volumen=0.9):
    x = volumen * x / max(np.max(np.abs(x)), 1e-9)
    # Fundidos de 5 ms: evitan el "clic" al empezar y al terminar.
    n = int(SR * 0.005)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    escribir_ogg(AUDIO / "sonidos" / f"{nombre}.ogg", x, SR)


def aplausos(seg=3.5, palmas=420):
    rng = np.random.default_rng(1)
    x = np.zeros(int(SR * seg))
    filtro = signal.butter(2, (900, 5000), "band", fs=SR, output="sos")
    for _ in range(palmas):
        n = int(SR * rng.uniform(0.006, 0.016))
        palma = signal.sosfilt(filtro, rng.normal(0, 1, n)) * np.exp(-np.linspace(0, 6, n))
        mezclar(x, palma * rng.uniform(0.4, 1.0), rng.uniform(0, seg - 0.02))
    return x * np.minimum(1.0, 3.0 * (1 - tiempo(seg) / seg))


def sonidos():
    (AUDIO / "sonidos").mkdir(parents=True, exist_ok=True)
    guardar("ding", tono(1318.5, 0.45, 7, (1, 0.3)) + 0.7 * tono(1975.5, 0.45, 9))
    guardar("media", tono(659.3, 0.22, 10, (1, 0.4)), 0.7)
    guardar("fallo", secuencia([tono(196, 0.16, 5, (1, 0.6, 0.4)), tono(146.8, 0.3, 5, (1, 0.6, 0.4))]), 0.8)
    guardar("tick", tono(880, 0.1, 25), 0.8)
    guardar("go", secuencia([tono(f, 0.09, 5, (1, 0.4)) for f in (523.3, 659.3, 784)] + [tono(1046.5, 0.45, 5, (1, 0.4))]))
    guardar("fanfarria", secuencia(
        [tono(f, s, 2.5, (1, 0.7, 0.5, 0.3)) for f, s in ((523.3, 0.16), (523.3, 0.16), (523.3, 0.16), (784, 0.5), (659.3, 0.2), (784, 0.9))]
    ))
    guardar("aplausos", aplausos())
    guardar("musica_reto", musica_reto(), 0.8)
    guardar("musica_estatua", musica_estatua(), 0.8)
    print("sonidos: listos")


def musica_reto(bpm=128):
    """8 compases en bucle exacto: bombo, platillo, bajo y acordes."""
    beat = 60.0 / bpm
    total = beat * 4 * 8
    x = np.zeros(int(SR * total))
    t = tiempo(0.22)
    bombo = np.sin(2 * np.pi * (50 * t + 45 * (1 - np.exp(-28 * t)) / 28)) * np.exp(-14 * t)
    rng = np.random.default_rng(2)
    platillo = signal.sosfilt(signal.butter(2, 7000, "high", fs=SR, output="sos"), rng.normal(0, 1, int(SR * 0.05)))
    platillo *= np.exp(-np.linspace(0, 7, len(platillo)))
    grados = ((110.0, 220.0, 261.6, 329.6), (87.3, 174.6, 220.0, 261.6), (130.8, 261.6, 329.6, 392.0), (98.0, 196.0, 246.9, 293.7))
    for compas in range(8):
        bajo, *acorde = grados[compas % 4]
        for b in range(4):
            inicio = (compas * 4 + b) * beat
            mezclar(x, bombo, inicio)
            mezclar(x, 0.28 * platillo, inicio + beat / 2)
            for corchea in (0, 0.5):
                mezclar(x, 0.42 * tono(bajo, beat * 0.45, 5, (1, 0.5, 0.25)), inicio + corchea * beat)
        for golpe in (0, 1.5, 3):
            for nota in acorde:
                mezclar(x, 0.11 * tono(nota * 2, beat, 6, (1, 0.5, 0.3)), (compas * 4 + golpe) * beat)
    return x


def musica_estatua(seg=16.0):
    """Latido y un fondo grave, para aguantar la respiracion. Bucle exacto."""
    x = np.zeros(int(SR * seg))
    t = tiempo(seg)
    # Frecuencias con un numero entero de ciclos: asi el bucle no hace clic.
    for hz, nivel in ((55.0, 0.10), (82.5, 0.07), (110.0, 0.04)):
        f = round(hz * seg) / seg
        x += nivel * np.sin(2 * np.pi * f * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 / seg * t))
    golpe = tono(60, 0.18, 16, (1, 0.3))
    for i in range(int(seg)):
        mezclar(x, 0.7 * golpe, i)
        mezclar(x, 0.45 * golpe, i + 0.22)
    return x


def main():
    parser = argparse.ArgumentParser(description="Genera las voces y los sonidos del reto.")
    parser.add_argument("--solo-voces", action="store_true")
    parser.add_argument("--solo-sonidos", action="store_true")
    parser.add_argument("--faltantes", action="store_true", help="generar solo las frases que aun no tienen audio")
    args = parser.parse_args()
    if not args.solo_voces:
        sonidos()
    if not args.solo_sonidos:
        asyncio.run(voces(args.faltantes))


if __name__ == "__main__":
    main()
