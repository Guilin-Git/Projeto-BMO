"""
bmo_voice/demo.py
------------------
Script de demonstração do sistema de voz do BMO.
Execute para testar se o pipeline TTS (edge-tts) → RVC (Applio) → Áudio funciona.

Pré-requisito:
    Applio rodando localmente:
        python app.py --port 7865   (dentro do diretório do Applio)

Uso:
    python bmo_voice/demo.py
    python bmo_voice/demo.py --phrase greeting
    python bmo_voice/demo.py --text "Beemo diz olá para você!"
    python bmo_voice/demo.py --tts-only   (testa só o TTS, sem Applio)
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from bmo_voice import VoiceEngine, VoiceConfig


DEMO_TEXT = (
    "Oi! Beemo está aqui. Beemo gosta muito de aventuras e de aprender coisas novas. "
    "Hoje Beemo vai te contar um segredo: a Terra de Ooo é o lugar mais incrível do mundo!"
)


async def test_tts_only(text: str) -> None:
    """Testa apenas o TTS (edge-tts) sem passar pelo Applio."""
    import io
    import soundfile as sf
    import sounddevice as sd
    from bmo_voice.tts_backends.edge_tts_backend import EdgeTTSBackend
    from bmo_voice import text_cleaner

    print("▶  Testando somente TTS (edge-tts, sem RVC)...\n")
    backend = EdgeTTSBackend()
    sentences = text_cleaner.split_sentences(text)

    for sentence in sentences:
        print(f"   Falando: '{sentence}'")
        wav_bytes, sr = await backend.synthesize(sentence)
        audio, sr2 = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sd.play(audio, samplerate=sr2)
        sd.wait()

    print("\n✅ TTS OK!")


async def main(text: str, tts_only: bool) -> None:
    print("=" * 55)
    print("🎙️  BMO Voice — Demo")
    print("=" * 55)
    print(f"Texto: {text[:80]}{'...' if len(text) > 80 else ''}\n")

    if tts_only:
        await test_tts_only(text)
        return

    config = VoiceConfig()

    # Verifica modelos RVC
    rvc_path = Path(config.rvc_model)
    if not rvc_path.exists():
        print(f"⚠️  Modelo RVC não encontrado: {rvc_path}")
        sys.exit(1)

    print(f"🔗 Conectando ao Applio em: {config.applio_url}")
    print("   (Certifique-se que o Applio está rodando!)\n")
    print("▶  Iniciando pipeline TTS → RVC → Áudio...")
    print("   (A primeira frase pode demorar um pouco para carregar o modelo)\n")

    engine = VoiceEngine(config)
    await engine.speak(text)
    await engine.close()

    print("\n✅  Concluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMO Voice Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, default=None)
    group.add_argument("--phrase", type=str, default=None)
    parser.add_argument(
        "--tts-only", action="store_true",
        help="Testa apenas o TTS sem passar pelo Applio RVC"
    )
    args = parser.parse_args()

    if args.phrase:
        from bmo_voice import phrase_bank
        text = phrase_bank.get(args.phrase)
        if text is None:
            print(f"Chave '{args.phrase}' não encontrada.")
            print(f"Disponíveis: {phrase_bank.all_keys()}")
            sys.exit(1)
    else:
        text = args.text or DEMO_TEXT

    asyncio.run(main(text, tts_only=args.tts_only))
