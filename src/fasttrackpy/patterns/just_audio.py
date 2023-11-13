import warnings
from pathlib import Path
from typing import Union
try:
    import magic
    no_magic = False
    warnings.warn("libmagic not found. "\
                  "Some audio file types won't be discovered by fasttrack. "\
                  "(mp3, ogg, ...)")
except:
    import sndhdr
    from sndhdr import SndHeaders
    no_magic = True

def create_audio_checker(no_magic = no_magic):

    def magic_checker(path: Union[str, Path]):
        file_mime = magic.from_file(path, mime=True)
        return "audio" in file_mime
    
    def sndhdr_checker(path: Union[str, Path]):
        hdr_info = sndhdr.what(path)
        return isinstance(hdr_info, SndHeaders)
    
    if no_magic:
        return sndhdr_checker
    
    return magic_checker