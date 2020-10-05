import simpleaudio as sa

wave_obj = sa.WaveObject.from_wave_file("imagine.wav")
wave_obj.play()
