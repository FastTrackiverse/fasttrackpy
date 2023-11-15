from fasttrackpy import CandidateTracks,\
                        OneTrack,\
                        Smoother,\
                        Loss,\
                        Agg

class TestImports:
    
    def test_imports(self):
        assert CandidateTracks
        assert OneTrack
        assert Smoother
        assert Loss
        assert Agg