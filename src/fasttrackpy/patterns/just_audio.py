import warnings
from pathlib import Path
from typing import Union
from collections.abc import Callable

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

def create_audio_checker(no_magic:bool = no_magic) -> Callable:
    """Return an audio checker, dependent on 
       availability of libmagic.

    Args:
        no_magic (bool): is libmagic available

    Returns:
        (Callable): A sound file checker
    """

    def magic_checker(path: str)->bool:
        """Checks whether a file is an audio file using libmagic

        Args:
            path (str): Path to the file in question

        Returns:
            (bool): Whether or not the file is an audio file
        """
        file_mime = magic.from_file(path, mime=True)
        return "audio" in file_mime
    
    def sndhdr_checker(path: str)->bool:
        """Checks whether a file is an audio file using `sndhdr`

        Args:
            path (str): Path to the file

        Returns:
            (bool): Whether or not the file is an audio file.
        """
        hdr_info = sndhdr.what(path)
        return isinstance(hdr_info, SndHeaders)
    
    if no_magic:
        return sndhdr_checker
    
    return magic_checker

is_audio = create_audio_checker(no_magic=no_magic)