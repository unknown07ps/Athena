from faster_whisper import WhisperModel
from gtts import gTTS
import tempfile
import os
from pathlib import Path


class AthenaVoice:
    """
    Windows-compatible voice interface
    - Speech-to-Text: faster-whisper (no FFmpeg required!)
    - Text-to-Speech: gTTS (requires internet)
    """
    
    def __init__(self, whisper_model="base", tts_lang='en'):
        """Initialize voice engine"""
        self.whisper_model_name = whisper_model
        self.tts_lang = tts_lang
        self.whisper_model = None
        
        print(f" Initializing Athena Voice Engine...")
        self._initialize_whisper()
        print(f"    Voice engine ready!")
    
    def _initialize_whisper(self):
        """Load faster-whisper model (no FFmpeg needed!)"""
        try:
            print(f"    Loading Whisper '{self.whisper_model_name}' model...")
            
            # faster-whisper uses different model names
            # Recommended for Windows: tiny, base, small
            self.whisper_model = WhisperModel(
                self.whisper_model_name,
                device="cpu",  # Use CPU (works on all systems)
                compute_type="int8"  # Faster and smaller
            )
            
            print(f"    Whisper ready (faster-whisper, no FFmpeg needed!)")
        except Exception as e:
            print(f"   ❌ Error loading Whisper: {e}")
            print(f"    Install: pip install faster-whisper")
            raise
    
    def transcribe_audio(self, audio_file_path: str, language='en') -> dict:
        """Convert speech to text using faster-whisper"""
        try:
            print(f" Transcribing audio: {Path(audio_file_path).name}")
            
            # Check file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                raise ValueError("Audio file is empty (0 bytes)")
            
            print(f"    File size: {file_size:,} bytes")
            
            # Transcribe with faster-whisper
            segments, info = self.whisper_model.transcribe(
                audio_file_path,
                language=language if language != 'auto' else None,
                task='transcribe',
                beam_size=5,
                vad_filter=True  # Voice activity detection
            )
            
            # Collect all text segments
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)
            
            text = " ".join(text_segments).strip()
            detected_language = info.language
            
            # Estimate confidence from info
            confidence = info.language_probability
            
            print(f"    Transcribed: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"    Language: {detected_language}, Confidence: {confidence:.0%}")
            
            return {
                'text': text,
                'language': detected_language,
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def speak(self, text: str, output_file: str = None, slow: bool = False) -> str:
        """Convert text to speech using gTTS (requires internet)"""
        try:
            print(f" Generating speech with Google TTS...")
            print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if output_file is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_file = temp_file.name
                temp_file.close()
            
            tts = gTTS(text=text, lang=self.tts_lang, slow=slow)
            tts.save(output_file)
            
            print(f"    Audio generated")
            print(f"    Saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"    TTS error: {e}")
            print(f"    Make sure you have internet connection for TTS")
            return None



#  TEST SUITE


if __name__ == "__main__":
    print("=" * 70)
    print(" ATHENA VOICE ENGINE TEST (faster-whisper + gTTS)")
    print("=" * 70)
    
    print(f"\n Python version: {os.sys.version}")
    
    # Initialize
    print("\n Initializing voice engine...")
    try:
        voice = AthenaVoice(whisper_model="tiny")  # tiny is fastest
        print("    Initialization successful!\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print("    Install dependencies:")
        print("      pip install faster-whisper gtts")
        os.sys.exit(1)
    
    # TTS Test
    print(" Testing Text-to-Speech (requires internet)...")
    try:
        test_text = "Hello! This is Athena speaking. Windows voice test successful."
        output = "test_windows_voice.mp3"
        
        result = voice.speak(test_text, output)
        
        if result and os.path.exists(result):
            print(f"    TTS working: {output}")
            print(f"    File size: {os.path.getsize(result):,} bytes")
        else:
            print(f"   ❌ TTS failed")
    except Exception as e:
        print(f"   ❌ TTS test failed: {e}")
    
    # Transcription Test
    print("\n Testing Speech-to-Text...")
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        print(f"   Testing with: {audio_file}")
        if not os.path.exists(audio_file):
            print(f"   ❌ File not found: {audio_file}")
        else:
            result = voice.transcribe_audio(audio_file)
            if result['success']:
                print(f"\n    Transcription:")
                print(f"      Text: {result['text']}")
                print(f"      Language: {result['language']}")
                print(f"      Confidence: {result['confidence']:.0%}")
            else:
                print(f"   ❌ Failed: {result.get('error')}")
    else:
        print("   ℹ️  Skipped (provide audio file to test)")
        print("   Usage: python voice_engine.py test.wav")
    
    print("\n" + "=" * 70)
    print(" TESTS COMPLETE!")
    print("=" * 70)
    
    print("\n Voice Engine Info:")
    print("    Transcription: Offline (faster-whisper)")
    print("    Speech: Online (gTTS)")
    print("    Cost: 100% Free")
    print("    Windows: Full Support (no FFmpeg needed!)")
    
    print("\n Advantages:")
    print("    No FFmpeg installation required")
    print("    Faster transcription")
    print("    Better Windows compatibility")
    print("    Lower memory usage")
    
    print("\n Next Steps:")
    print("   1. Test with audio: python voice_engine.py test.wav")
    print("   2. Run diagnostic: python test_voice.py")
    print("   3. Start Athena: streamlit run app.py")